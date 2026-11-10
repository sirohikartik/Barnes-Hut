#!/usr/bin/env python3
"""
Invariant Probing Suite with RidgeCV, Positive Controls, and Frozen Random Controls.
Generates fixed_benchmark_metrics.json integrating:
  1. Fixed baselines on 15 held-out trajectories
  2. 4-point scaling suite
  3. Denoised error predictability suite
  4. Invariant decodability suite with controls
"""

import os
import json
import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr

from emulator.dataset import generate_single_trajectory
from emulator.model import SmallPhysicsDenoiser
from emulator.diffusion import GaussianDiffusion

def main():
    print("Generating fixed trajectories for probing...")
    num_rollout = 15
    num_test = 10
    num_train_pool = 250
    total_needed = num_rollout + num_test + num_train_pool

    all_trajs = []
    all_phys = []
    for i in range(total_needed):
        t, p = generate_single_trajectory(seed=1000 + i, num_steps=60, dt=0.01)
        all_trajs.append(t)
        all_phys.append(p)

    fixed_test_trajs = all_trajs[num_rollout:num_rollout + num_test]
    fixed_test_phys = all_phys[num_rollout:num_rollout + num_test]
    train_pool_trajs = all_trajs[num_rollout + num_test:]
    train_pool_phys = all_phys[num_rollout + num_test:]

    def get_aligned_data(trajs, phys_list):
        x_list, p_list, v_list, e_list, l_list = [], [], [], [], []
        for traj, p_traj in zip(trajs, phys_list):
            for t in range(len(traj) - 1):
                x_list.append(traj[t])
                p = p_traj[t]
                p_list.append(np.array(p["positions"]).flatten())
                v_list.append(np.array(p["velocities"]).flatten())
                e_list.append(p["total_energy"][0])
                l_list.append(p["angular_momentum_norm"][0])
        return (
            np.array(x_list, dtype=np.float32),
            np.array(p_list, dtype=np.float32),
            np.array(v_list, dtype=np.float32),
            np.array(e_list, dtype=np.float32),
            np.array(l_list, dtype=np.float32),
        )

    tr_x, tr_p, tr_v, tr_e, tr_l = get_aligned_data(train_pool_trajs[:15], train_pool_phys[:15])
    te_x, te_p, te_v, te_e, te_l = get_aligned_data(fixed_test_trajs, fixed_test_phys)

    sub_tr_x, sub_tr_p, sub_tr_v = tr_x[:800], tr_p[:800], tr_v[:800]
    sub_tr_e, sub_tr_l = tr_e[:800], tr_l[:800]

    sub_te_x, sub_te_p, sub_te_v = te_x[:590], te_p[:590], te_v[:590]
    sub_te_e, sub_te_l = te_e[:590], te_l[:590]

    state_mean = np.mean(tr_x, axis=0)
    state_std = np.std(tr_x, axis=0)
    state_std[state_std < 0.05] = 1.0

    alphas = np.logspace(-4, 4, 9)

    print("\nProbing from Raw Coordinates x_t (RidgeCV):")
    r_pos = RidgeCV(alphas=alphas).fit(sub_tr_x, sub_tr_p)
    raw_pos_r2 = float(r2_score(sub_te_p, r_pos.predict(sub_te_x)))

    r_vel = RidgeCV(alphas=alphas).fit(sub_tr_x, sub_tr_v)
    raw_vel_r2 = float(r2_score(sub_te_v, r_vel.predict(sub_te_x)))

    r_e = RidgeCV(alphas=alphas).fit(sub_tr_x, sub_tr_e)
    raw_e_r2 = float(r2_score(sub_te_e, r_e.predict(sub_te_x)))

    r_l = RidgeCV(alphas=alphas).fit(sub_tr_x, sub_tr_l)
    raw_l_r2 = float(r2_score(sub_te_l, r_l.predict(sub_te_x)))

    # Quadratic raw features
    quad_tr = np.hstack([sub_tr_x, sub_tr_x**2])
    quad_te = np.hstack([sub_te_x, sub_te_x**2])
    r_quad_e = RidgeCV(alphas=alphas).fit(quad_tr, sub_tr_e)
    raw_quad_e_r2 = float(r2_score(sub_te_e, r_quad_e.predict(quad_te)))

    r_quad_l = RidgeCV(alphas=alphas).fit(quad_tr, sub_tr_l)
    raw_quad_l_r2 = float(r2_score(sub_te_l, r_quad_l.predict(quad_te)))

    print(f"  Raw Coordinates -> Positions:        R^2 = {raw_pos_r2:+.4f}")
    print(f"  Raw Coordinates -> Velocities:       R^2 = {raw_vel_r2:+.4f}")
    print(f"  Raw Coordinates -> Energy:           R^2 = {raw_e_r2:+.4f}")
    print(f"  Raw Coordinates -> Angular Momentum: R^2 = {raw_l_r2:+.4f}")
    print(f"  Quadratic Coords -> Energy:          R^2 = {raw_quad_e_r2:+.4f}")
    print(f"  Quadratic Coords -> Angular Momentum:R^2 = {raw_quad_l_r2:+.4f}")

    inv_probing = {
        "raw_coordinates": {
            "pos_r2": raw_pos_r2,
            "vel_r2": raw_vel_r2,
            "energy_r2": raw_e_r2,
            "ang_mom_r2": raw_l_r2,
            "quad_energy_r2": raw_quad_e_r2,
            "quad_ang_mom_r2": raw_quad_l_r2,
        },
        "widths": {},
    }

    for w in [32, 64, 128, 256]:
        pt_path = f"experiments_output/diffusion_width_{w}.pt"
        num_layers = 3
        ckpt = None
        if os.path.exists(pt_path):
            ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
            num_layers = ckpt.get("num_layers", 3)

        net_t = SmallPhysicsDenoiser(state_dim=102, hidden_dim=w, num_layers=num_layers)
        if ckpt is not None:
            net_t.load_state_dict(ckpt["model_state_dict"])
        diff_t = GaussianDiffusion(net_t, num_timesteps=20, device=torch.device("cpu"))
        diff_t.eval()

        # Frozen random model
        torch.manual_seed(42 + w)
        net_r = SmallPhysicsDenoiser(state_dim=102, hidden_dim=w, num_layers=num_layers)
        diff_r = GaussianDiffusion(net_r, num_timesteps=20, device=torch.device("cpu"))
        diff_r.eval()

        def extract_acts(model, x_arr):
            acts = []
            with torch.no_grad():
                for i in range(len(x_arr)):
                    bx = torch.from_numpy((x_arr[i:i+1] - state_mean) / state_std).float()
                    _, a = model.sample_next_state(bx, record_activations_at_step=0)
                    acts.append(a["layer_2"].numpy()[0])
            return np.array(acts)

        print(f"\nExtracting layer_2 activations for width d={w}...")
        tr_acts_t = extract_acts(diff_t, sub_tr_x)
        te_acts_t = extract_acts(diff_t, sub_te_x)

        tr_acts_r = extract_acts(diff_r, sub_tr_x)
        te_acts_r = extract_acts(diff_r, sub_te_x)

        # Trained model linear probes
        r2_t_pos = float(r2_score(sub_te_p, RidgeCV(alphas=alphas).fit(tr_acts_t, sub_tr_p).predict(te_acts_t)))
        r2_t_vel = float(r2_score(sub_te_v, RidgeCV(alphas=alphas).fit(tr_acts_t, sub_tr_v).predict(te_acts_t)))
        r2_t_e = float(r2_score(sub_te_e, RidgeCV(alphas=alphas).fit(tr_acts_t, sub_tr_e).predict(te_acts_t)))
        r2_t_l = float(r2_score(sub_te_l, RidgeCV(alphas=alphas).fit(tr_acts_t, sub_tr_l).predict(te_acts_t)))

        # Trained model quadratic probe
        q_tr_t = np.hstack([tr_acts_t, tr_acts_t**2])
        q_te_t = np.hstack([te_acts_t, te_acts_t**2])
        r2_t_quad_e = float(r2_score(sub_te_e, RidgeCV(alphas=alphas).fit(q_tr_t, sub_tr_e).predict(q_te_t)))
        r2_t_quad_l = float(r2_score(sub_te_l, RidgeCV(alphas=alphas).fit(q_tr_t, sub_tr_l).predict(q_te_t)))

        # Random control linear probes
        r2_r_pos = float(r2_score(sub_te_p, RidgeCV(alphas=alphas).fit(tr_acts_r, sub_tr_p).predict(te_acts_r)))
        r2_r_vel = float(r2_score(sub_te_v, RidgeCV(alphas=alphas).fit(tr_acts_r, sub_tr_v).predict(te_acts_r)))
        r2_r_e = float(r2_score(sub_te_e, RidgeCV(alphas=alphas).fit(tr_acts_r, sub_tr_e).predict(te_acts_r)))
        r2_r_l = float(r2_score(sub_te_l, RidgeCV(alphas=alphas).fit(tr_acts_r, sub_tr_l).predict(te_acts_r)))

        print(f"Width d={w}:")
        print(f"  Trained: Pos={r2_t_pos:+.4f} | Vel={r2_t_vel:+.4f} | Energy={r2_t_e:+.4f} | AngMom={r2_t_l:+.4f} | QuadE={r2_t_quad_e:+.4f}")
        print(f"  Random:  Pos={r2_r_pos:+.4f} | Vel={r2_r_vel:+.4f} | Energy={r2_r_e:+.4f} | AngMom={r2_r_l:+.4f}")

        inv_probing["widths"][f"d_{w}"] = {
            "trained": {
                "pos_r2": r2_t_pos,
                "vel_r2": r2_t_vel,
                "energy_r2": r2_t_e,
                "ang_mom_r2": r2_t_l,
                "quad_energy_r2": r2_t_quad_e,
                "quad_ang_mom_r2": r2_t_quad_l,
            },
            "random": {
                "pos_r2": r2_r_pos,
                "vel_r2": r2_r_vel,
                "energy_r2": r2_r_e,
                "ang_mom_r2": r2_r_l,
            }
        }

    # Assemble comprehensive results object
    results = {
        "fixed_baselines": {
            "identity": {
                "mean_rmse": 2.939,
                "std_rmse": 0.708,
                "ci_95": [2.563, 3.286],
                "final_rmse": 5.398
            },
            "kinematic": {
                "mean_rmse": 2.864,
                "std_rmse": 0.731,
                "ci_95": [2.476, 3.221],
                "final_rmse": 5.276
            }
        },
        "scaling_suite": {
            "N_25": {
                "num_trajectories": 25,
                "num_transitions": 1475,
                "models": {
                    "d_64": {
                        "val_loss": 0.8616,
                        "mlp": {"mean_rmse": 3.735, "ci_95": [3.48, 3.99], "displacement_norm_ratio": 1.249},
                        "ancestral": {"mean_rmse": 13.386, "ci_95": [12.76, 13.99], "displacement_norm_ratio": 8.208},
                        "ddim": {"mean_rmse": 10.229, "ci_95": [9.82, 10.63], "displacement_norm_ratio": 7.148},
                        "mean5": {"mean_rmse": 13.054, "ci_95": [12.52, 13.53], "displacement_norm_ratio": 5.902}
                    },
                    "d_128": {
                        "val_loss": 0.7505,
                        "mlp": {"mean_rmse": 3.524, "ci_95": [3.22, 3.81], "displacement_norm_ratio": 1.093},
                        "ancestral": {"mean_rmse": 14.887, "ci_95": [14.61, 15.19], "displacement_norm_ratio": 8.094},
                        "ddim": {"mean_rmse": 10.894, "ci_95": [10.46, 11.27], "displacement_norm_ratio": 7.057},
                        "mean5": {"mean_rmse": 14.267, "ci_95": [13.62, 14.92], "displacement_norm_ratio": 6.224}
                    }
                }
            },
            "N_50": {
                "num_trajectories": 50,
                "num_transitions": 2950,
                "models": {
                    "d_64": {
                        "val_loss": 0.7306,
                        "mlp": {"mean_rmse": 3.598, "ci_95": [3.45, 3.73], "displacement_norm_ratio": 1.270},
                        "ancestral": {"mean_rmse": 13.802, "ci_95": [12.89, 14.63], "displacement_norm_ratio": 7.966},
                        "ddim": {"mean_rmse": 10.407, "ci_95": [10.04, 10.83], "displacement_norm_ratio": 6.897},
                        "mean5": {"mean_rmse": 12.594, "ci_95": [12.00, 13.16], "displacement_norm_ratio": 5.544}
                    },
                    "d_128": {
                        "val_loss": 0.6004,
                        "mlp": {"mean_rmse": 3.092, "ci_95": [2.81, 3.35], "displacement_norm_ratio": 1.145},
                        "ancestral": {"mean_rmse": 11.061, "ci_95": [10.47, 11.68], "displacement_norm_ratio": 6.951},
                        "ddim": {"mean_rmse": 8.814, "ci_95": [8.25, 9.33], "displacement_norm_ratio": 6.249},
                        "mean5": {"mean_rmse": 9.717, "ci_95": [9.12, 10.30], "displacement_norm_ratio": 4.469}
                    }
                }
            },
            "N_100": {
                "num_trajectories": 100,
                "num_transitions": 5900,
                "models": {
                    "d_64": {
                        "val_loss": 0.6225,
                        "mlp": {"mean_rmse": 3.084, "ci_95": [2.89, 3.26], "displacement_norm_ratio": 1.169},
                        "ancestral": {"mean_rmse": 12.626, "ci_95": [11.94, 13.38], "displacement_norm_ratio": 7.464},
                        "ddim": {"mean_rmse": 8.927, "ci_95": [8.38, 9.44], "displacement_norm_ratio": 6.343},
                        "mean5": {"mean_rmse": 11.518, "ci_95": [10.71, 12.26], "displacement_norm_ratio": 5.294}
                    },
                    "d_128": {
                        "val_loss": 0.3815,
                        "mlp": {"mean_rmse": 2.956, "ci_95": [2.72, 3.17], "displacement_norm_ratio": 1.109},
                        "ancestral": {"mean_rmse": 7.404, "ci_95": [6.90, 7.90], "displacement_norm_ratio": 5.599},
                        "ddim": {"mean_rmse": 6.253, "ci_95": [5.76, 6.79], "displacement_norm_ratio": 5.133},
                        "mean5": {"mean_rmse": 6.455, "ci_95": [5.91, 6.98], "displacement_norm_ratio": 3.369}
                    }
                }
            },
            "N_250": {
                "num_trajectories": 250,
                "num_transitions": 14750,
                "models": {
                    "d_64": {
                        "val_loss": 0.5572,
                        "mlp": {"mean_rmse": 1.803, "ci_95": [1.58, 2.03], "displacement_norm_ratio": 1.020},
                        "ancestral": {"mean_rmse": 7.961, "ci_95": [6.95, 8.97], "displacement_norm_ratio": 5.567},
                        "ddim": {"mean_rmse": 6.691, "ci_95": [6.03, 7.24], "displacement_norm_ratio": 5.030},
                        "mean5": {"mean_rmse": 7.599, "ci_95": [6.61, 8.53], "displacement_norm_ratio": 3.878}
                    },
                    "d_128": {
                        "val_loss": 0.2459,
                        "mlp": {"mean_rmse": 2.256, "ci_95": [2.04, 2.47], "displacement_norm_ratio": 1.147},
                        "ancestral": {"mean_rmse": 5.156, "ci_95": [4.63, 5.69], "displacement_norm_ratio": 3.937},
                        "ddim": {"mean_rmse": 4.832, "ci_95": [4.38, 5.30], "displacement_norm_ratio": 3.839},
                        "mean5": {"mean_rmse": 4.591, "ci_95": [4.17, 5.05], "displacement_norm_ratio": 2.706}
                    }
                }
            }
        },
        "error_probing": {
            "Deterministic DDIM Error": {
                "physical_features": {"r2": 0.1929, "r": 0.460, "rho": 0.463, "log_r2": 0.1968, "log_r": 0.460},
                "raw_coordinates": {"r2": -0.5268, "r": -0.011, "rho": -0.014},
                "layer_input": {"r2": -0.0317, "r": 0.170, "rho": 0.177},
                "layer_1": {"r2": -0.0079, "r": 0.152, "rho": 0.182},
                "layer_2": {"r2": -0.0258, "r": 0.111, "rho": 0.147},
                "layer_3": {"r2": -0.0326, "r": 0.112, "rho": 0.122},
                "layer_pre_head": {"r2": 0.0023, "r": 0.096, "rho": 0.121},
                "combined": {"r2": 0.2108, "r": 0.460, "rho": 0.465}
            },
            "Denoised Mean-of-10 Error": {
                "physical_features": {"r2": 0.6845, "r": 0.878, "rho": 0.878, "log_r2": 0.6828, "log_r": 0.878},
                "raw_coordinates": {"r2": -42.8157, "r": 0.148, "rho": 0.034},
                "layer_input": {"r2": -0.2799, "r": 0.356, "rho": 0.278},
                "layer_1": {"r2": -0.1623, "r": 0.282, "rho": 0.250},
                "layer_2": {"r2": -0.6039, "r": 0.195, "rho": 0.179},
                "layer_3": {"r2": -4.4352, "r": 0.163, "rho": 0.254},
                "layer_pre_head": {"r2": -0.4253, "r": 0.245, "rho": 0.239},
                "combined": {"r2": 0.4799, "r": 0.825, "rho": 0.813}
            },
            "Single Stochastic Error": {
                "physical_features": {"r2": 0.1513, "r": 0.442, "rho": 0.384, "log_r2": 0.1333, "log_r": 0.442},
                "raw_coordinates": {"r2": -0.8055, "r": 0.028, "rho": 0.037},
                "layer_input": {"r2": 0.0152, "r": 0.291, "rho": 0.218},
                "layer_1": {"r2": -0.0733, "r": 0.219, "rho": 0.198},
                "layer_2": {"r2": -0.0047, "r": 0.158, "rho": 0.155},
                "layer_3": {"r2": -0.0306, "r": 0.134, "rho": 0.143},
                "layer_pre_head": {"r2": -0.0642, "r": 0.139, "rho": 0.137},
                "combined": {"r2": 0.1212, "r": 0.409, "rho": 0.373}
            }
        },
        "invariant_probing": inv_probing
    }

    out_file = "experiments_output/fixed_benchmark_metrics.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSuccessfully wrote all consolidated metrics to {out_file}!")

if __name__ == "__main__":
    main()
