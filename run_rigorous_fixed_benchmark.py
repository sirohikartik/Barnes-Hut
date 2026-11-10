"""
Definitive Rigorous Benchmark on Fixed Held-Out Trajectories.
Addresses all reviewer critique points:
1. Single fixed held-out evaluation set (15 rollout trajectories, 10 single-step test trajectories)
   Never seen during training across all scales, models, and baselines.
2. Bootstrap 95% confidence intervals over held-out trajectories.
3. Multi-point data scaling curve: N_train in [25, 50, 100, 250] trajectories (1,475, 2,950, 5,900, 14,750 transitions).
4. Direct measurement of step displacement shrinkage: ||\Delta \hat{x}|| / ||\Delta x_GT|| across Ancestral, DDIM, Mean-of-5.
5. Denoised error predictability on strictly trajectory-held-out data (DDIM error, Mean-of-10 error, Single-sample error).
6. Activation probing with RidgeCV, positive controls (positions, velocities), and quadratic/nonlinear feature probes.
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr

from emulator.dataset import generate_single_trajectory, compute_physical_quantities
from emulator.model import SmallPhysicsDenoiser
from emulator.diffusion import GaussianDiffusion


class SimpleMLP(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, state_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def bootstrap_ci(values, n_boot=2000, ci=95):
    values = np.array(values)
    if len(values) == 0:
        return 0.0, 0.0
    boot_means = []
    n = len(values)
    rng = np.random.RandomState(42)
    for _ in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (100 - ci) / 2.0)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2.0)
    return float(lower), float(upper)


def extract_physical_features(x_array):
    N = 16
    pos = x_array[:, :N*3].reshape(-1, N, 3)
    vel = x_array[:, N*3:N*6].reshape(-1, N, 3)
    radii = np.linalg.norm(pos, axis=-1)
    speeds = np.linalg.norm(vel, axis=-1)
    r_min = np.min(radii, axis=-1, keepdims=True)
    r_mean = np.mean(radii, axis=-1, keepdims=True)
    v_mean = np.mean(speeds, axis=-1, keepdims=True)
    v_max = np.max(speeds, axis=-1, keepdims=True)
    rs = 2.0 * 1.0 * 1000.0 / (60.0**2)
    r_isco = 3.0 * rs
    d_isco = np.abs(r_min - r_isco)
    L_vec = np.sum(np.cross(pos, vel), axis=1) * 0.1
    L_norm = np.linalg.norm(L_vec, axis=-1, keepdims=True)
    return np.hstack([r_min, r_mean, v_mean, v_max, d_isco, L_norm])


def evaluate_rollout_suite(predict_fn, rollouts, rollout_len=30):
    traj_rmses = []
    step_curves = []
    norm_ratios = []

    for traj in rollouts:
        curr = traj[0].copy()
        preds = [curr]
        pred_deltas = []
        gt_deltas = []

        for step in range(rollout_len):
            next_s = predict_fn(curr)
            pred_delta = next_s - curr
            gt_delta = traj[step + 1] - traj[step]
            pred_deltas.append(np.linalg.norm(pred_delta))
            gt_deltas.append(np.linalg.norm(gt_delta))
            curr = next_s
            preds.append(curr)

        preds = np.array(preds)
        gt = traj[:rollout_len + 1]
        errs = np.sqrt(np.mean((preds - gt)**2, axis=-1))
        step_curves.append(errs)
        traj_rmses.append(np.mean(errs))

        pred_norm_mean = np.mean(pred_deltas)
        gt_norm_mean = np.mean(gt_deltas)
        norm_ratios.append(pred_norm_mean / (gt_norm_mean + 1e-8))

    mean_rmse = float(np.mean(traj_rmses))
    std_rmse = float(np.std(traj_rmses))
    ci_low, ci_high = bootstrap_ci(traj_rmses)
    mean_curve = np.mean(step_curves, axis=0)
    mean_ratio = float(np.mean(norm_ratios))

    return {
        "mean_rmse": mean_rmse,
        "std_rmse": std_rmse,
        "ci_95": [ci_low, ci_high],
        "final_rmse": float(mean_curve[-1]),
        "mean_curve": mean_curve.tolist(),
        "displacement_norm_ratio": mean_ratio,
        "raw_traj_rmses": [float(x) for x in traj_rmses],
    }


def main():
    print("="*80)
    print("STARTING DEFINITIVE RIGOROUS BENCHMARK ON FIXED HELD-OUT SPLIT")
    print("="*80)

    # 1. Generate Master Dataset: 275 Trajectories total
    # 250 Train pool, 10 Test trajectories (single-step), 15 Dedicated Rollout trajectories
    total_traj = 275
    num_rollout = 15
    num_test = 10
    num_train_max = 250

    print(f"Generating {total_traj} trajectories (N=16 bodies, dt=0.01)...")
    all_trajs = []
    all_phys = []
    base_seed = 42

    for i in range(total_traj):
        seed = base_seed + i * 17
        states, phys = generate_single_trajectory(
            num_bodies=16,
            num_steps=60,
            dt=0.01,
            seed=seed,
            disk_radius_min=2.5,
            disk_radius_max=18.0 + (i % 4) * 0.5,
            total_disk_mass=1.6,
        )
        all_trajs.append(states)
        all_phys.append(phys)

    fixed_rollout_trajs = all_trajs[:num_rollout]
    fixed_test_trajs = all_trajs[num_rollout:num_rollout + num_test]
    fixed_test_phys = all_phys[num_rollout:num_rollout + num_test]
    train_pool_trajs = all_trajs[num_rollout + num_test:]
    train_pool_phys = all_phys[num_rollout + num_test:]

    print(f"Fixed Partitioning:")
    print(f"  Dedicated Held-Out Rollout Trajectories: {len(fixed_rollout_trajs)} (15 trajs, 30 steps each)")
    print(f"  Dedicated Held-Out Single-Step Test Trajs: {len(fixed_test_trajs)} (10 trajs, 590 transitions)")
    print(f"  Training Trajectory Pool: {len(train_pool_trajs)} (up to 250 trajs, 14,750 transitions)")

    # Extract test transition pairs
    te_x_list, te_d_list = [], []
    for traj in fixed_test_trajs:
        for t in range(len(traj) - 1):
            te_x_list.append(traj[t])
            te_d_list.append(traj[t+1] - traj[t])
    fixed_test_x = np.array(te_x_list, dtype=np.float32)
    fixed_test_delta = np.array(te_d_list, dtype=np.float32)

    # 2. Evaluate Non-Parametric Baselines on Fixed Rollout Set
    def kin_pred(s):
        N = 16
        pos = s[:N*3]
        vel = s[N*3:N*6]
        new_pos = pos + vel * 0.01
        return np.concatenate([new_pos, vel, s[N*6:]])

    def id_pred(s):
        return s.copy()

    id_results = evaluate_rollout_suite(id_pred, fixed_rollout_trajs)
    kin_results = evaluate_rollout_suite(kin_pred, fixed_rollout_trajs)

    print("\n--- Fixed Baseline Performance (15 Held-Out Rollout Trajectories) ---")
    print(f"Identity Baseline:  Mean RMSE = {id_results['mean_rmse']:.3f} +- {id_results['std_rmse']:.3f} | 95% CI: [{id_results['ci_95'][0]:.3f}, {id_results['ci_95'][1]:.3f}] | Final = {id_results['final_rmse']:.3f}")
    print(f"Kinematic Baseline: Mean RMSE = {kin_results['mean_rmse']:.3f} +- {kin_results['std_rmse']:.3f} | 95% CI: [{kin_results['ci_95'][0]:.3f}, {kin_results['ci_95'][1]:.3f}] | Final = {kin_results['final_rmse']:.3f}")

    results = {
        "fixed_baselines": {
            "identity": id_results,
            "kinematic": kin_results,
        },
        "scaling_suite": {},
    }

    # 3. Multi-Point Data Scaling Curve: N_train in [25, 50, 100, 250]
    scale_points = [25, 50, 100, 250]
    widths = [64, 128]

    # Pre-compute normalization on largest training pool
    largest_trajs = train_pool_trajs[:250]
    all_tx_list, all_td_list = [], []
    for traj in largest_trajs:
        for t in range(len(traj) - 1):
            all_tx_list.append(traj[t])
            all_td_list.append(traj[t+1] - traj[t])
    all_train_x = np.array(all_tx_list, dtype=np.float32)
    all_train_delta = np.array(all_td_list, dtype=np.float32)

    state_mean = np.mean(all_train_x, axis=0)
    state_std = np.std(all_train_x, axis=0)
    state_std[state_std < 0.05] = 1.0

    delta_mean = np.mean(all_train_delta, axis=0)
    delta_std = np.std(all_train_delta, axis=0)
    delta_std[delta_std < 0.05] = 1.0

    norm_test_x = (fixed_test_x - state_mean) / state_std
    norm_test_delta = (fixed_test_delta - delta_mean) / delta_std

    for n_traj in scale_points:
        n_trans = n_traj * 59
        print(f"\n" + "="*70)
        print(f"EVALUATING SCALE: N_train = {n_traj} trajectories ({n_trans} transitions)")
        print("="*70)

        sub_trajs = train_pool_trajs[:n_traj]
        sub_tx_list, sub_td_list = [], []
        for traj in sub_trajs:
            for t in range(len(traj) - 1):
                sub_tx_list.append(traj[t])
                sub_td_list.append(traj[t+1] - traj[t])
        curr_train_x = np.array(sub_tx_list, dtype=np.float32)
        curr_train_delta = np.array(sub_td_list, dtype=np.float32)

        norm_curr_x = (curr_train_x - state_mean) / state_std
        norm_curr_delta = (curr_train_delta - delta_mean) / delta_std

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(norm_curr_x).float(), torch.from_numpy(norm_curr_delta).float()),
            batch_size=64, shuffle=True
        )

        scale_res = {}

        for w in widths:
            print(f"\n--- Model Width d={w} (N_train={n_traj}) ---")
            torch.manual_seed(42)
            np.random.seed(42)

            # Train Matched MLP
            mlp = SimpleMLP(state_dim=102, hidden_dim=w)
            mlp_opt = torch.optim.AdamW(mlp.parameters(), lr=1.5e-3, weight_decay=1e-4)
            epochs = 25 if n_traj <= 50 else 15
            for ep in range(epochs):
                mlp.train()
                for bx, bd in train_loader:
                    mlp_opt.zero_grad()
                    loss = nn.functional.mse_loss(mlp(bx), bd)
                    loss.backward()
                    mlp_opt.step()

            # Train Diffusion
            net = SmallPhysicsDenoiser(state_dim=102, hidden_dim=w)
            diff = GaussianDiffusion(net, num_timesteps=20, schedule="cosine", device=torch.device("cpu"))
            diff_opt = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=1e-4)
            for ep in range(epochs):
                net.train()
                for bx, bd in train_loader:
                    diff_opt.zero_grad()
                    loss = diff.compute_loss(bd, bx)
                    loss.backward()
                    diff_opt.step()

            # Single-step Validation Loss
            net.eval()
            with torch.no_grad():
                val_losses = []
                for i in range(0, len(norm_test_x), 64):
                    bx = torch.from_numpy(norm_test_x[i:i+64]).float()
                    bd = torch.from_numpy(norm_test_delta[i:i+64]).float()
                    val_losses.append(diff.compute_loss(bd, bx).item())
                val_loss = float(np.mean(val_losses))

            # Rollout Predictors
            def mlp_pred(s):
                mlp.eval()
                with torch.no_grad():
                    bx = torch.from_numpy((s - state_mean) / state_std).float().unsqueeze(0)
                    pred_norm = mlp(bx).numpy()[0]
                    return s + pred_norm * delta_std + delta_mean

            def diff_ancestral(s):
                bx = torch.from_numpy((s - state_mean) / state_std).float().unsqueeze(0)
                sample_norm, _ = diff.sample_next_state(bx)
                pred_norm = sample_norm.detach().numpy()[0]
                return s + pred_norm * delta_std + delta_mean

            def diff_ddim(s):
                bx = torch.from_numpy((s - state_mean) / state_std).float().unsqueeze(0)
                sample_norm = diff.sample_next_state_ddim(bx, eta=0.0)
                pred_norm = sample_norm.detach().numpy()[0]
                return s + pred_norm * delta_std + delta_mean

            def diff_mean5(s):
                bx = torch.from_numpy((s - state_mean) / state_std).float().unsqueeze(0)
                samples = [diff.sample_next_state(bx)[0].detach().numpy()[0] for _ in range(5)]
                pred_norm = np.mean(samples, axis=0)
                return s + pred_norm * delta_std + delta_mean

            mlp_res = evaluate_rollout_suite(mlp_pred, fixed_rollout_trajs)
            anc_res = evaluate_rollout_suite(diff_ancestral, fixed_rollout_trajs)
            ddim_res = evaluate_rollout_suite(diff_ddim, fixed_rollout_trajs)
            m5_res = evaluate_rollout_suite(diff_mean5, fixed_rollout_trajs)

            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  MLP:        Mean={mlp_res['mean_rmse']:.3f} | 95% CI: [{mlp_res['ci_95'][0]:.2f}, {mlp_res['ci_95'][1]:.2f}] | Norm Ratio={mlp_res['displacement_norm_ratio']:.3f}")
            print(f"  Ancestral:  Mean={anc_res['mean_rmse']:.3f} | 95% CI: [{anc_res['ci_95'][0]:.2f}, {anc_res['ci_95'][1]:.2f}] | Norm Ratio={anc_res['displacement_norm_ratio']:.3f}")
            print(f"  DDIM:       Mean={ddim_res['mean_rmse']:.3f} | 95% CI: [{ddim_res['ci_95'][0]:.2f}, {ddim_res['ci_95'][1]:.2f}] | Norm Ratio={ddim_res['displacement_norm_ratio']:.3f}")
            print(f"  Mean-of-5:  Mean={m5_res['mean_rmse']:.3f} | 95% CI: [{m5_res['ci_95'][0]:.2f}, {m5_res['ci_95'][1]:.2f}] | Norm Ratio={m5_res['displacement_norm_ratio']:.3f}")

            scale_res[f"d_{w}"] = {
                "val_loss": val_loss,
                "mlp": mlp_res,
                "ancestral": anc_res,
                "ddim": ddim_res,
                "mean5": m5_res,
            }

        results["scaling_suite"][f"N_{n_traj}"] = {
            "num_trajectories": n_traj,
            "num_transitions": n_trans,
            "models": scale_res,
        }

    # 4. Strictly Trajectory-Held-Out Denoised Error Predictability
    print("\n" + "="*80)
    print("DENOISED ERROR PREDICTABILITY ON TRAJECTORY-HELD-OUT DATA")
    print("="*80)

    # Use the flagship d=128 model trained on N=100
    flagship_net = SmallPhysicsDenoiser(state_dim=102, hidden_dim=128)
    flagship_diff = GaussianDiffusion(flagship_net, num_timesteps=20, device=torch.device("cpu"))
    # Train flagship for 20 epochs on 100 trajectories
    fl_tx = all_train_x[:5900]
    fl_td = all_train_delta[:5900]
    fl_opt = torch.optim.AdamW(flagship_net.parameters(), lr=1.5e-3, weight_decay=1e-4)
    fl_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy((fl_tx - state_mean)/state_std).float(),
            torch.from_numpy((fl_td - delta_mean)/delta_std).float()
        ),
        batch_size=64, shuffle=True
    )
    for ep in range(20):
        flagship_net.train()
        for bx, bd in fl_loader:
            fl_opt.zero_grad()
            l = flagship_diff.compute_loss(bd, bx)
            l.backward()
            fl_opt.step()
    flagship_net.eval()

    # Collect predictions and activations on training subset vs fixed test set
    def compute_errors_and_acts(x_arr, d_arr, n_samples=1000):
        sub_n = min(len(x_arr), n_samples)
        acts_dict = {f"layer_{k}": [] for k in ["input", "1", "2", "3", "pre_head"]}
        err_ddim_list = []
        err_mean10_list = []
        err_single_list = []

        with torch.no_grad():
            for i in range(sub_n):
                bx = torch.from_numpy((x_arr[i:i+1] - state_mean) / state_std).float()
                # 1. DDIM prediction (deterministic)
                ddim_norm = flagship_diff.sample_next_state_ddim(bx, eta=0.0).numpy()[0]
                pred_ddim = ddim_norm * delta_std + delta_mean
                err_ddim = float(np.linalg.norm(pred_ddim - d_arr[i]))
                err_ddim_list.append(err_ddim)

                # 2. Stochastic samples (10 samples for denoised mean)
                stoch_samples = []
                sample_acts = None
                for s_i in range(10):
                    s_norm, acts = flagship_diff.sample_next_state(bx, record_activations_at_step=0)
                    stoch_samples.append(s_norm.numpy()[0] * delta_std + delta_mean)
                    if s_i == 0:
                        sample_acts = acts

                pred_single = stoch_samples[0]
                pred_mean10 = np.mean(stoch_samples, axis=0)

                err_single = float(np.linalg.norm(pred_single - d_arr[i]))
                err_mean10 = float(np.linalg.norm(pred_mean10 - d_arr[i]))

                err_single_list.append(err_single)
                err_mean10_list.append(err_mean10)

                for k in ["input", "1", "2", "3", "pre_head"]:
                    acts_dict[f"layer_{k}"].append(sample_acts[f"layer_{k}"].numpy()[0])

        return {
            "acts": {k: np.array(v) for k, v in acts_dict.items()},
            "err_ddim": np.array(err_ddim_list),
            "err_mean10": np.array(err_mean10_list),
            "err_single": np.array(err_single_list),
        }

    print("Extracting representations and denoised error targets on train and fixed held-out test...")
    train_eval = compute_errors_and_acts(fl_tx, fl_td, n_samples=800)
    test_eval = compute_errors_and_acts(fixed_test_x, fixed_test_delta, n_samples=590)

    # Physical features
    tr_phys = extract_physical_features(fl_tx[:800])
    te_phys = extract_physical_features(fixed_test_x[:590])

    error_probing_results = {}
    alphas = np.logspace(-4, 4, 9)

    for target_name, tr_y, te_y in [
        ("Deterministic DDIM Error", train_eval["err_ddim"], test_eval["err_ddim"]),
        ("Denoised Mean-of-10 Error", train_eval["err_mean10"], test_eval["err_mean10"]),
        ("Single Stochastic Error", train_eval["err_single"], test_eval["err_single"]),
    ]:
        print(f"\n--- Target: {target_name} ---")
        target_dict = {}

        # 1. Physical features
        probe = RidgeCV(alphas=alphas).fit(tr_phys, tr_y)
        pred = probe.predict(te_phys)
        r2 = float(r2_score(te_y, pred))
        r = float(pearsonr(te_y, pred)[0]) if np.std(pred) > 1e-8 else 0.0
        rho = float(spearmanr(te_y, pred)[0]) if np.std(pred) > 1e-8 else 0.0

        # Log target
        probe_log = RidgeCV(alphas=alphas).fit(tr_phys, np.log(tr_y + 1e-6))
        pred_log = probe_log.predict(te_phys)
        log_r2 = float(r2_score(np.log(te_y + 1e-6), pred_log))
        log_r = float(pearsonr(np.log(te_y + 1e-6), pred_log)[0]) if np.std(pred_log) > 1e-8 else 0.0

        print(f"  Physical Features:     Test R^2={r2:+.4f} | Pearson r={r:+.3f} | Spearman rho={rho:+.3f} | Log R^2={log_r2:+.4f}")
        target_dict["physical_features"] = {"r2": r2, "r": r, "rho": rho, "log_r2": log_r2, "log_r": log_r}

        # 2. Raw State Coordinates
        probe = RidgeCV(alphas=alphas).fit(fl_tx[:800], tr_y)
        pred = probe.predict(fixed_test_x[:590])
        r2 = float(r2_score(te_y, pred))
        r = float(pearsonr(te_y, pred)[0]) if np.std(pred) > 1e-8 else 0.0
        rho = float(spearmanr(te_y, pred)[0]) if np.std(pred) > 1e-8 else 0.0
        print(f"  Raw Coordinates x_t:   Test R^2={r2:+.4f} | Pearson r={r:+.3f} | Spearman rho={rho:+.3f}")
        target_dict["raw_coordinates"] = {"r2": r2, "r": r, "rho": rho}

        # 3. Layers
        for l_name in ["layer_input", "layer_1", "layer_2", "layer_3", "layer_pre_head"]:
            tr_act = train_eval["acts"][l_name]
            te_act = test_eval["acts"][l_name]
            probe = RidgeCV(alphas=alphas).fit(tr_act, tr_y)
            pred = probe.predict(te_act)
            r2 = float(r2_score(te_y, pred))
            r = float(pearsonr(te_y, pred)[0]) if np.std(pred) > 1e-8 else 0.0
            rho = float(spearmanr(te_y, pred)[0]) if np.std(pred) > 1e-8 else 0.0
            print(f"  {l_name:<21}: Test R^2={r2:+.4f} | Pearson r={r:+.3f} | Spearman rho={rho:+.3f}")
            target_dict[l_name] = {"r2": r2, "r": r, "rho": rho}

        # 4. Combined: Physical Features + layer_pre_head
        comb_tr = np.hstack([tr_phys, train_eval["acts"]["layer_pre_head"]])
        comb_te = np.hstack([te_phys, test_eval["acts"]["layer_pre_head"]])
        probe = RidgeCV(alphas=alphas).fit(comb_tr, tr_y)
        pred = probe.predict(comb_te)
        r2 = float(r2_score(te_y, pred))
        r = float(pearsonr(te_y, pred)[0]) if np.std(pred) > 1e-8 else 0.0
        rho = float(spearmanr(te_y, pred)[0]) if np.std(pred) > 1e-8 else 0.0
        print(f"  Combined (Phys + Act): Test R^2={r2:+.4f} | Pearson r={r:+.3f} | Spearman rho={rho:+.3f}")
        target_dict["combined"] = {"r2": r2, "r": r, "rho": rho}

        error_probing_results[target_name] = target_dict

    results["error_probing"] = error_probing_results

    # 5. Activation Probing with RidgeCV & Positive Controls
    print("\n" + "="*80)
    print("ACTIVATION PROBING WITH RIDGECV, POSITIVE CONTROLS, AND QUADRATIC PROBES")
    print("="*80)

    # Extract physical targets for train subset and fixed test set
    def get_phys_targets(phys_list):
        flat_p = [item for sub in phys_list for item in sub]
        pos = np.array([p["positions"] for p in flat_p])
        vel = np.array([p["velocities"] for p in flat_p])
        e = np.array([p["total_energy"][0] for p in flat_p])
        l = np.array([p["angular_momentum_norm"][0] for p in flat_p])
        return pos, vel, e, l

    tr_p, tr_v, tr_e, tr_l = get_phys_targets(train_pool_phys[:15])
    te_p, te_v, te_e, te_l = get_phys_targets(fixed_test_phys)

    sub_tr_p = tr_p[:800]
    sub_tr_v = tr_v[:800]
    sub_tr_e = tr_e[:800]
    sub_tr_l = tr_l[:800]

    sub_te_p = te_p[:590]
    sub_te_v = te_v[:590]
    sub_te_e = te_e[:590]
    sub_te_l = te_l[:590]

    # Raw coordinates baseline
    r_pos = RidgeCV(alphas=alphas).fit(fl_tx[:800], sub_tr_p)
    raw_pos_r2 = float(r2_score(sub_te_p, r_pos.predict(fixed_test_x[:590])))

    r_vel = RidgeCV(alphas=alphas).fit(fl_tx[:800], sub_tr_v)
    raw_vel_r2 = float(r2_score(sub_te_v, r_vel.predict(fixed_test_x[:590])))

    r_e = RidgeCV(alphas=alphas).fit(fl_tx[:800], sub_tr_e)
    raw_e_r2 = float(r2_score(sub_te_e, r_e.predict(fixed_test_x[:590])))

    r_l = RidgeCV(alphas=alphas).fit(fl_tx[:800], sub_tr_l)
    raw_l_r2 = float(r2_score(sub_te_l, r_l.predict(fixed_test_x[:590])))

    # Quadratic feature probe on raw coordinates
    quad_tr = np.hstack([fl_tx[:800], fl_tx[:800]**2])
    quad_te = np.hstack([fixed_test_x[:590], fixed_test_x[:590]**2])
    r_quad_e = RidgeCV(alphas=alphas).fit(quad_tr, sub_tr_e)
    raw_quad_e_r2 = float(r2_score(sub_te_e, r_quad_e.predict(quad_te)))

    r_quad_l = RidgeCV(alphas=alphas).fit(quad_tr, sub_tr_l)
    raw_quad_l_r2 = float(r2_score(sub_te_l, r_quad_l.predict(quad_te)))

    print("Raw Coordinates Probing Baseline (RidgeCV):")
    print(f"  Positive Control Positions (p):     R^2 = {raw_pos_r2:+.4f}")
    print(f"  Positive Control Velocities (v):    R^2 = {raw_vel_r2:+.4f}")
    print(f"  Linear Target Total Energy (E):     R^2 = {raw_e_r2:+.4f}")
    print(f"  Linear Target Angular Momentum (|L|): R^2 = {raw_l_r2:+.4f}")
    print(f"  Quadratic Features Total Energy:    R^2 = {raw_quad_e_r2:+.4f}")
    print(f"  Quadratic Features Angular Momentum: R^2 = {raw_quad_l_r2:+.4f}")

    inv_probing = {
        "raw_coordinates": {
            "pos_r2": raw_pos_r2, "vel_r2": raw_vel_r2,
            "energy_r2": raw_e_r2, "ang_mom_r2": raw_l_r2,
            "quad_energy_r2": raw_quad_e_r2, "quad_ang_mom_r2": raw_quad_l_r2,
        },
        "widths": {},
    }

    # Evaluate across widths d in [32, 64, 128, 256]
    for w in [32, 64, 128, 256]:
        # Trained model
        net_t = SmallPhysicsDenoiser(state_dim=102, hidden_dim=w)
        diff_t = GaussianDiffusion(net_t, num_timesteps=20, device=torch.device("cpu"))
        pt_path = f"experiments_output/diffusion_width_{w}.pt"
        if os.path.exists(pt_path):
            ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
            net_t.load_state_dict(ckpt["model_state_dict"])
        diff_t.eval()

        # Frozen random model
        net_r = SmallPhysicsDenoiser(state_dim=102, hidden_dim=w)
        diff_r = GaussianDiffusion(net_r, num_timesteps=20, device=torch.device("cpu"))
        diff_r.eval()

        def extract_acts_l2(d_model, x_arr):
            acts = []
            with torch.no_grad():
                for i in range(len(x_arr)):
                    bx = torch.from_numpy((x_arr[i:i+1] - state_mean) / state_std).float()
                    _, a = d_model.sample_next_state(bx, record_activations_at_step=0)
                    acts.append(a["layer_2"].numpy()[0])
            return np.array(acts)

        tr_acts_t = extract_acts_l2(diff_t, fl_tx[:800])
        te_acts_t = extract_acts_l2(diff_t, fixed_test_x[:590])

        tr_acts_r = extract_acts_l2(diff_r, fl_tx[:800])
        te_acts_r = extract_acts_l2(diff_r, fixed_test_x[:590])

        # Trained: Linear probes
        r2_t_pos = float(r2_score(sub_te_p, RidgeCV(alphas=alphas).fit(tr_acts_t, sub_tr_p).predict(te_acts_t)))
        r2_t_vel = float(r2_score(sub_te_v, RidgeCV(alphas=alphas).fit(tr_acts_t, sub_tr_v).predict(te_acts_t)))
        r2_t_e = float(r2_score(sub_te_e, RidgeCV(alphas=alphas).fit(tr_acts_t, sub_tr_e).predict(te_acts_t)))
        r2_t_l = float(r2_score(sub_te_l, RidgeCV(alphas=alphas).fit(tr_acts_t, sub_tr_l).predict(te_acts_t)))

        # Trained: Quadratic feature probe on activations (acts + acts^2)
        q_tr_acts_t = np.hstack([tr_acts_t, tr_acts_t**2])
        q_te_acts_t = np.hstack([te_acts_t, te_acts_t**2])
        r2_t_quad_e = float(r2_score(sub_te_e, RidgeCV(alphas=alphas).fit(q_tr_acts_t, sub_tr_e).predict(q_te_acts_t)))
        r2_t_quad_l = float(r2_score(sub_te_l, RidgeCV(alphas=alphas).fit(q_tr_acts_t, sub_tr_l).predict(q_te_acts_t)))

        # Random: Linear probes
        r2_r_pos = float(r2_score(sub_te_p, RidgeCV(alphas=alphas).fit(tr_acts_r, sub_tr_p).predict(te_acts_r)))
        r2_r_vel = float(r2_score(sub_te_v, RidgeCV(alphas=alphas).fit(tr_acts_r, sub_tr_v).predict(te_acts_r)))
        r2_r_e = float(r2_score(sub_te_e, RidgeCV(alphas=alphas).fit(tr_acts_r, sub_tr_e).predict(te_acts_r)))
        r2_r_l = float(r2_score(sub_te_l, RidgeCV(alphas=alphas).fit(tr_acts_r, sub_tr_l).predict(te_acts_r)))

        print(f"\nWidth d={w} (RidgeCV):")
        print(f"  Trained: Pos={r2_t_pos:+.3f} | Vel={r2_t_vel:+.3f} | Energy={r2_t_e:+.3f} | AngMom={r2_t_l:+.3f} | QuadEnergy={r2_t_quad_e:+.3f}")
        print(f"  Random:  Pos={r2_r_pos:+.3f} | Vel={r2_r_vel:+.3f} | Energy={r2_r_e:+.3f} | AngMom={r2_r_l:+.3f}")

        inv_probing["widths"][f"d_{w}"] = {
            "trained": {
                "pos_r2": r2_t_pos, "vel_r2": r2_t_vel,
                "energy_r2": r2_t_e, "ang_mom_r2": r2_t_l,
                "quad_energy_r2": r2_t_quad_e, "quad_ang_mom_r2": r2_t_quad_l,
            },
            "random": {
                "pos_r2": r2_r_pos, "vel_r2": r2_r_vel,
                "energy_r2": r2_r_e, "ang_mom_r2": r2_r_l,
            }
        }

    results["invariant_probing"] = inv_probing

    # Save to JSON
    out_file = "experiments_output/fixed_benchmark_metrics.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved all results cleanly to {out_file}!")


if __name__ == "__main__":
    main()
