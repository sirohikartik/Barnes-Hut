import os
import time
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emulator.dataset import create_physics_dataset, compute_physical_quantities
from emulator.train import train_diffusion_emulator, extract_dataset_activations, get_device
from emulator.prober import PhysicalProbeSuite
from emulator.model import SmallPhysicsDenoiser
from emulator.diffusion import GaussianDiffusion


def run_model_width_sweep(
    data_dict: dict,
    widths: list = [32, 64, 128, 256],
    epochs: int = 25,
    num_layers: int = 3,
    num_timesteps: int = 20,
    output_dir: str = "experiments_output",
    reuse_cached: bool = True,
):
    print("\n" + "=" * 85)
    print(" 🚀 EXPERIMENT 1: DIFFUSION MODEL WIDTH SCALING STUDY")
    print(f" Testing Denoiser Widths (hidden_dim): {widths}")
    print("=" * 85)

    json_path = os.path.join(output_dir, "width_and_failure_metrics.json")
    if reuse_cached and os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                cached = json.load(f)
            if "widths" in cached and all(str(w) in cached["widths"] for w in widths):
                print("--> Found existing width sweep metrics in JSON, reusing cached results and re-plotting...")
                plot_width_comparison(cached["widths"], widths, output_dir)
                return cached["widths"]
        except Exception as e:
            print(f"--> Notice: could not load cache ({e}), retraining...")

    device = get_device()
    results = {}

    state_mean = data_dict["state_mean"]
    state_std = data_dict["state_std"]
    delta_mean = data_dict["delta_mean"]
    delta_std = data_dict["delta_std"]
    test_x = data_dict["test_x"]
    test_y = data_dict["test_y"]
    test_traj = data_dict["rollout_trajectories"][0]
    rollout_len = min(30, len(test_traj) - 1)
    gt_rollout = test_traj[: rollout_len + 1]

    probe_suite = PhysicalProbeSuite(alpha=1.0)
    train_targets = probe_suite.extract_target_arrays(data_dict["train_phys"])
    test_targets = probe_suite.extract_target_arrays(data_dict["test_phys"])
    tgt_train = data_dict["train_delta"]
    tgt_test = data_dict["test_delta"]

    for width in widths:
        print(f"\n---> Training Model Width: hidden_dim = {width} ({num_layers} ResBlocks)...")
        model_save_path = os.path.join(output_dir, f"diffusion_width_{width}.pt")

        t0 = time.time()
        diffusion, history = train_diffusion_emulator(
            dataset_dict=data_dict,
            hidden_dim=width,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            epochs=epochs,
            batch_size=64,
            learning_rate=1.5e-3,
            target_type="delta",
            save_path=model_save_path,
        )
        train_time = time.time() - t0
        param_count = sum(p.numel() for p in diffusion.model.parameters())

        # Measure 30-step autoregressive rollout
        rollout_pred = [test_traj[0]]
        curr_state = np.expand_dims(test_traj[0], 0)
        step_times = []

        for step in range(rollout_len):
            t_step_start = time.time()
            next_s, _ = diffusion.predict_future_state(
                curr_state,
                state_mean=state_mean,
                state_std=state_std,
                delta_mean=delta_mean,
                delta_std=delta_std,
                target_type="delta",
            )
            step_times.append((time.time() - t_step_start) * 1000.0)
            rollout_pred.append(next_s[0])
            curr_state = next_s

        rollout_pred = np.array(rollout_pred)
        rollout_rmse_per_step = np.sqrt(np.mean((rollout_pred - gt_rollout) ** 2, axis=1))
        mean_rollout_rmse = float(np.mean(rollout_rmse_per_step))
        final_rollout_rmse = float(rollout_rmse_per_step[-1])
        avg_step_ms = float(np.mean(step_times))

        # Physical conservation drift along rollout
        # Compute energy and angular momentum along rollout
        e_drifts = []
        l_drifts = []
        initial_phys = None

        n_bodies = 16
        masses = np.full(n_bodies, 1.6 / n_bodies, dtype=np.float64)

        for s_idx, state_vec in enumerate(rollout_pred):
            # Parse state vector: bodies (N*6) + BH (6)
            pos = state_vec[: n_bodies * 3].reshape(n_bodies, 3)
            vel = state_vec[n_bodies * 3 : n_bodies * 6].reshape(n_bodies, 3)
            bh_pos = state_vec[n_bodies * 6 : n_bodies * 6 + 3]
            bh_vel = state_vec[n_bodies * 6 + 3 : n_bodies * 6 + 6]
            phys = compute_physical_quantities(
                pos=pos, vel=vel, masses=masses,
                bh_pos=bh_pos, bh_vel=bh_vel, bh_mass=1000.0,
                rs=0.5555, softening=0.08
            )
            if s_idx == 0:
                initial_phys = phys
            e_rel_drift = abs(phys["total_energy"][0] - initial_phys["total_energy"][0]) / abs(initial_phys["total_energy"][0])
            l_rel_drift = abs(phys["angular_momentum_norm"][0] - initial_phys["angular_momentum_norm"][0]) / (abs(initial_phys["angular_momentum_norm"][0]) + 1e-6)
            e_drifts.append(float(e_rel_drift))
            l_drifts.append(float(l_rel_drift))

        # Probe representation at layer_input and layer_2
        train_acts = extract_dataset_activations(
            diffusion, data_dict["train_x"], tgt_train,
            state_mean, state_std, delta_mean, delta_std, diffusion_step=0
        )
        test_acts = extract_dataset_activations(
            diffusion, data_dict["test_x"], tgt_test,
            state_mean, state_std, delta_mean, delta_std, diffusion_step=0
        )
        probe_res_layer2 = probe_suite.evaluate_representation(
            train_acts["layer_2"], test_acts["layer_2"], train_targets, test_targets
        )

        results[str(width)] = {
            "width": width,
            "params": param_count,
            "train_time_s": float(train_time),
            "step_time_ms": avg_step_ms,
            "final_train_loss": float(history["train_losses"][-1]),
            "final_val_loss": float(history["test_losses"][-1]) if history["test_losses"] else float(history["train_losses"][-1]),
            "mean_rollout_rmse": mean_rollout_rmse,
            "final_rollout_rmse": final_rollout_rmse,
            "mean_energy_drift": float(np.mean(e_drifts)),
            "mean_l_drift": float(np.mean(l_drifts)),
            "rollout_rmse_curve": rollout_rmse_per_step.tolist(),
            "energy_drift_curve": e_drifts,
            "l_drift_curve": l_drifts,
            "train_loss_curve": history["train_losses"],
            "probing_r2": {
                "total_energy": float(probe_res_layer2["total_energy"]["r2"]),
                "angular_momentum": float(probe_res_layer2["angular_momentum_norm"]["r2"]),
                "positions": float(probe_res_layer2["positions"]["r2"]),
                "velocities": float(probe_res_layer2["velocities"]["r2"]),
                "potential_energy": float(probe_res_layer2["potential_energy"]["r2"]),
            },
        }

        print(f"  Width {width:>3} Summary: Params={param_count:,} | Train Time={train_time:.1f}s | Rollout RMSE={mean_rollout_rmse:.4f} | E-Drift={np.mean(e_drifts)*100:.2f}% | Latency={avg_step_ms:.1f}ms/step")

    # Save visualization of width sweep
    plot_width_comparison(results, widths, output_dir)
    return results


def plot_width_comparison(results: dict, widths: list, output_dir: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {32: "#377eb8", 64: "#4daf4a", 128: "#ff7f00", 256: "#984ea3"}

    # Subplot 1: Train Loss Curves
    ax1 = axes[0, 0]
    for w in widths:
        ax1.plot(results[str(w)]["train_loss_curve"], label=f"Width {w} ({results[str(w)]['params']:,} params)", color=colors[w], linewidth=2)
    ax1.set_title("Training Loss Convergence vs Model Width (MPS)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Diffusion MSE Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Subplot 2: Rollout Error Accumulation over Timesteps
    ax2 = axes[0, 1]
    for w in widths:
        ax2.plot(results[str(w)]["rollout_rmse_curve"], label=f"Width {w} (Mean RMSE: {results[str(w)]['mean_rollout_rmse']:.3f})", color=colors[w], linewidth=2)
    ax2.set_title("30-Step Autoregressive Rollout Error Accumulation", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Rollout Step $t$")
    ax2.set_ylabel("State Coordinate RMSE")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    # Subplot 3: Physical Conservation Drift (Energy Drift %)
    ax3 = axes[1, 0]
    for w in widths:
        e_drift_pct = np.array(results[str(w)]["energy_drift_curve"]) * 100.0
        ax3.plot(e_drift_pct, label=f"Width {w} (Avg: {results[str(w)]['mean_energy_drift']*100:.1f}%)", color=colors[w], linewidth=2)
    ax3.set_title("Total Energy Conservation Violation (|ΔE| / E₀ %)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Rollout Step $t$")
    ax3.set_ylabel("Energy Drift (%)")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend()

    # Subplot 4: Linear Probe R^2 vs Model Width
    ax4 = axes[1, 1]
    w_vals = widths
    e_r2 = [results[str(w)]["probing_r2"]["total_energy"] for w in w_vals]
    l_r2 = [results[str(w)]["probing_r2"]["angular_momentum"] for w in w_vals]
    p_r2 = [results[str(w)]["probing_r2"]["positions"] for w in w_vals]
    v_r2 = [results[str(w)]["probing_r2"]["velocities"] for w in w_vals]

    ax4.plot(w_vals, l_r2, "s-", color="#7570b3", label="Angular Momentum |L|", linewidth=2, markersize=8)
    ax4.plot(w_vals, p_r2, "d-", color="#e7298a", label="Positions", linewidth=2, markersize=8)
    ax4.plot(w_vals, v_r2, "v-", color="#e6ab02", label="Velocities", linewidth=2, markersize=8)
    ax4.plot(w_vals, e_r2, "o-", color="#d95f02", label="Total Energy E", linewidth=2, markersize=8)

    ax4.set_xticks(w_vals)
    ax4.set_xticklabels([f"d={w}" for w in w_vals])
    ax4.set_title("Physical Invariant Decodability ($R^2$) vs Model Width", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Denoiser Hidden Dimension")
    ax4.set_ylabel("Linear Probe $R^2$ Score (Test)")
    ax4.set_ylim(-0.05, 1.0)
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(loc="lower right")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "model_width_comparison.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"\n--> Saved Model Width Scaling plot to {save_path}")


def run_failure_mode_probing_experiment(
    data_dict: dict,
    model_path: str,
    hidden_dim: int = 128,
    num_layers: int = 3,
    num_timesteps: int = 20,
    output_dir: str = "experiments_output",
):
    print("\n" + "=" * 85)
    print(" 🔍 EXPERIMENT 2: PROBING MODEL FAILURE MODES & REPRESENTATION COLLAPSE")
    print(" Probing: When the model is wrong, can internal states self-diagnose failure?")
    print("=" * 85)

    device = get_device()
    state_mean = data_dict["state_mean"]
    state_std = data_dict["state_std"]
    delta_mean = data_dict["delta_mean"]
    delta_std = data_dict["delta_std"]
    test_x = data_dict["test_x"]
    test_y = data_dict["test_y"]
    test_delta = data_dict["test_delta"]
    train_x = data_dict["train_x"]
    train_delta = data_dict["train_delta"]

    # Load trained model
    denoiser = SmallPhysicsDenoiser(state_dim=test_x.shape[1], hidden_dim=hidden_dim, num_layers=num_layers)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    denoiser.load_state_dict(ckpt["model_state_dict"])
    diffusion = GaussianDiffusion(model=denoiser, num_timesteps=num_timesteps, device=device)

    # 1. EVALUATE TRANSITION PREDICTION ERRORS ACROSS TEST SET
    print("\n[Part 1] Computing ground truth transition errors & ensemble uncertainty across test set...")
    n_test = len(test_x)
    pred_deltas = []
    ensemble_stds = []

    # Process in mini-batches to remain memory-friendly on M1 Air
    batch_size = 64
    for i in range(0, n_test, batch_size):
        bx = test_x[i : i + batch_size]
        cond_norm = (bx - state_mean) / state_std
        cond_tensor = torch.from_numpy(cond_norm).float().to(device)

        # Generate K=8 ensemble samples
        ens_samples = []
        for _ in range(8):
            s_norm, _ = diffusion.sample_next_state(cond_tensor)
            s_raw = s_norm.detach().cpu().numpy() * delta_std + delta_mean
            ens_samples.append(np.expand_dims(s_raw, 0))
        ens_arr = np.concatenate(ens_samples, axis=0) # (8, B, D)
        pred_mean = np.mean(ens_arr, axis=0)
        pred_std = np.std(ens_arr, axis=0)

        pred_deltas.append(pred_mean)
        ensemble_stds.append(np.mean(pred_std, axis=1))

    pred_deltas = np.concatenate(pred_deltas, axis=0)
    ensemble_stds = np.concatenate(ensemble_stds, axis=0)

    # Calculate true transition errors
    true_deltas = test_delta
    transition_rmse = np.sqrt(np.mean((pred_deltas - true_deltas) ** 2, axis=1))
    
    # Calculate energy error for each transition
    n_bodies = 16
    masses = np.full(n_bodies, 1.6 / n_bodies, dtype=np.float64)
    energy_errors = []
    proximity_to_bh = []

    for idx in range(n_test):
        # Ground truth next state
        gt_next = test_y[idx]
        pos_gt = gt_next[: n_bodies * 3].reshape(n_bodies, 3)
        vel_gt = gt_next[n_bodies * 3 : n_bodies * 6].reshape(n_bodies, 3)
        bh_pos = gt_next[n_bodies * 6 : n_bodies * 6 + 3]
        bh_vel = gt_next[n_bodies * 6 + 3 : n_bodies * 6 + 6]
        phys_gt = compute_physical_quantities(pos_gt, vel_gt, masses, bh_pos, bh_vel, bh_mass=1000.0, rs=0.5555)

        # Predicted next state
        pred_next = test_x[idx] + pred_deltas[idx]
        pos_pr = pred_next[: n_bodies * 3].reshape(n_bodies, 3)
        vel_pr = pred_next[n_bodies * 3 : n_bodies * 6].reshape(n_bodies, 3)
        phys_pr = compute_physical_quantities(pos_pr, vel_pr, masses, bh_pos, bh_vel, bh_mass=1000.0, rs=0.5555)

        e_gt = phys_gt["total_energy"][0]
        e_pr = phys_pr["total_energy"][0]
        rel_e_err = abs(e_pr - e_gt) / (abs(e_gt) + 1e-4)
        energy_errors.append(float(rel_e_err))

        # Proximity: minimum distance to SMBH across all bodies
        r_min = float(np.min(np.linalg.norm(pos_gt - bh_pos[None, :], axis=1)))
        proximity_to_bh.append(r_min)

    energy_errors = np.array(energy_errors)
    proximity_to_bh = np.array(proximity_to_bh)

    # 2. EXTRACT INTERNAL ACTIVATIONS
    print("[Part 2] Extracting layer representations h_l to probe failure mechanisms...")
    train_acts = extract_dataset_activations(
        diffusion, train_x, train_delta, state_mean, state_std, delta_mean, delta_std, diffusion_step=0
    )
    test_acts = extract_dataset_activations(
        diffusion, test_x, test_delta, state_mean, state_std, delta_mean, delta_std, diffusion_step=0
    )

    probe_suite = PhysicalProbeSuite(alpha=1.0)
    train_phys_targets = probe_suite.extract_target_arrays(data_dict["train_phys"])
    test_phys_targets = probe_suite.extract_target_arrays(data_dict["test_phys"])

    # 3. SELF-DIAGNOSIS PROBING: CAN INTERNAL LAYERS PREDICT MODEL ERROR?
    print("\n[Part 3] Self-Diagnosis Probing: Can internal activations h_l linearly predict model error?")
    # Generate train errors for probe training
    # For training set, compute fast 1-step predicted error
    # To avoid overfitting error probe, compute train transition RMSE using 1-step sample
    train_batch_cond = (train_x - state_mean) / state_std
    with torch.no_grad():
        tr_cond_t = torch.from_numpy(train_batch_cond).float().to(device)
        tr_sample_norm, _ = diffusion.sample_next_state(tr_cond_t)
        tr_pred_delta = tr_sample_norm.detach().cpu().numpy() * delta_std + delta_mean
        train_rmse = np.sqrt(np.mean((tr_pred_delta - train_delta) ** 2, axis=1))

    error_targets_train = {
        "trajectory_rmse": train_rmse,
    }
    error_targets_test = {
        "trajectory_rmse": transition_rmse,
    }

    # State-controlled Baselines for Error Predictability
    def extract_phys_baseline_feats(states, rs=0.5555):
        n_bodies = 16
        r_isco = 3.0 * rs
        feats = []
        for s in states:
            pos = s[:n_bodies*3].reshape(n_bodies, 3)
            vel = s[n_bodies*3:n_bodies*6].reshape(n_bodies, 3)
            bh_pos = s[n_bodies*6:n_bodies*6+3]
            bh_vel = s[n_bodies*6+3:n_bodies*6+6]
            r_dists = np.linalg.norm(pos - bh_pos[None, :], axis=1)
            r_min = np.min(r_dists)
            r_mean = np.mean(r_dists)
            v_mags = np.linalg.norm(vel - bh_vel[None, :], axis=1)
            v_mean = np.mean(v_mags)
            v_max = np.max(v_mags)
            dist_isco = np.min(np.abs(r_dists - r_isco))
            r_min_over_rs = r_min / rs
            L_vec = np.sum(np.cross(pos, vel), axis=0)
            L_norm = np.linalg.norm(L_vec)
            feats.append([r_min, r_mean, v_mean, v_max, dist_isco, r_min_over_rs, L_norm])
        return np.array(feats, dtype=np.float32)

    phys_tr = extract_phys_baseline_feats(train_x)
    phys_te = extract_phys_baseline_feats(test_x)

    res_phys = probe_suite.evaluate_error_predictability(phys_tr, phys_te, error_targets_train, error_targets_test)
    res_raw = probe_suite.evaluate_error_predictability(train_x, test_x, error_targets_train, error_targets_test)
    print(f"  • {'Physical Feats':<15} -> Baseline Error Predictability: R^2 = {res_phys['trajectory_rmse']['r2']:6.4f}, Pearson r = {res_phys['trajectory_rmse']['corr']:6.4f}")
    print(f"  • {'Raw State (x_t)':<15} -> Baseline Error Predictability: R^2 = {res_raw['trajectory_rmse']['r2']:6.4f}, Pearson r = {res_raw['trajectory_rmse']['corr']:6.4f}")

    layer_names = list(train_acts.keys())
    error_probe_results = {
        "physical_baseline": res_phys["trajectory_rmse"],
        "raw_state_baseline": res_raw["trajectory_rmse"],
    }
    for layer in layer_names:
        h_tr = train_acts[layer]
        h_te = test_acts[layer]
        res = probe_suite.evaluate_error_predictability(
            h_tr, h_te, error_targets_train, error_targets_test
        )
        error_probe_results[layer] = res["trajectory_rmse"]
        print(f"  • {layer:<15} -> Predicts Test Error Magnitude: R^2 = {res['trajectory_rmse']['r2']:6.4f}, Pearson r = {res['trajectory_rmse']['corr']:6.4f}")

    # Residualized error test: e_res = e_actual - e_hat(x_t)
    from sklearn.linear_model import Ridge
    from scipy.stats import pearsonr, spearmanr
    e_base_tr = Ridge(alpha=1.0).fit(train_x, train_rmse).predict(train_x)
    e_base_te = Ridge(alpha=1.0).fit(train_x, train_rmse).predict(test_x)
    e_res_tr = train_rmse - e_base_tr
    e_res_te = transition_rmse - e_base_te
    clf_res = Ridge(alpha=1.0).fit(train_acts["layer_pre_head"], e_res_tr)
    pred_res_te = clf_res.predict(test_acts["layer_pre_head"])
    r_res, _ = pearsonr(pred_res_te, e_res_te)
    rho_res, _ = spearmanr(pred_res_te, e_res_te)
    error_probe_results["residualized_pre_head"] = {
        "pearson_r": float(r_res),
        "spearman_rho": float(rho_res),
    }
    print(f"  • {'Residualized (h_pre_head -> e_res)':<35} -> Pearson r = {r_res:6.4f}, Spearman rho = {rho_res:6.4f}")

    # 4. REGIME DISSECTION: SUCCESS VS FAILURE REGIMES
    print("\n[Part 4] Representation Dissection: Probing Success (Lowest 25% error) vs Failure (Highest 25% error)...")
    regime_results = probe_suite.evaluate_success_vs_failure_regimes(
        train_acts["layer_2"], test_acts["layer_2"],
        train_phys_targets, test_phys_targets,
        test_errors=transition_rmse,
        quantile=0.25,
    )

    print("  Physical Target      | Success Regime R^2 | Failure Regime R^2 | Degradation ΔR^2")
    print("  -----------------------------------------------------------------------------")
    regime_summary = {}
    for key, name in [
        ("total_energy", "Total Energy (E)"),
        ("angular_momentum_norm", "Angular Momentum (|L|)"),
        ("positions", "Positions (p)"),
        ("velocities", "Velocities (v)"),
        ("potential_energy", "Potential Energy (U)"),
    ]:
        r2_succ = regime_results["success"][key]["r2"]
        r2_fail = regime_results["failure"][key]["r2"]
        delta_r2 = r2_succ - r2_fail
        regime_summary[key] = {
            "success_r2": float(r2_succ),
            "failure_r2": float(r2_fail),
            "delta_r2": float(delta_r2),
        }
        print(f"  {name:<20} | {r2_succ:18.4f} | {r2_fail:18.4f} | {delta_r2:16.4f}")

    # 5. UNCERTAINTY CALIBRATION AS INTRINSIC FAILURE DETECTOR
    print("\n[Part 5] Evaluating Ensemble Epistemic Uncertainty vs True Error...")
    unc_res = probe_suite.evaluate_uncertainty_correlation(ensemble_stds, transition_rmse)
    print(f"  Pearson Correlation (Uncertainty vs True Error): r = {unc_res['pearson_r']:.4f}")
    print(f"  Spearman Rank Correlation:                      ρ = {unc_res['spearman_rho']:.4f}")

    # Quantile failure spread: compare top 10% highest error transitions vs bottom 10%
    q90 = np.quantile(transition_rmse, 0.90)
    q10 = np.quantile(transition_rmse, 0.10)
    unc_fail = float(np.mean(ensemble_stds[transition_rmse >= q90]))
    unc_succ = float(np.mean(ensemble_stds[transition_rmse <= q10]))
    uncertainty_ratio = unc_fail / (unc_succ + 1e-7)
    print(f"  Ensemble Spread during Failure (Top 10% error): {unc_fail:.4f}")
    print(f"  Ensemble Spread during Success (Bottom 10%):    {unc_succ:.4f}")
    print(f"  Uncertainty Spike Ratio:                         {uncertainty_ratio:.2f}x")

    # 6. VISUALIZE FAILURE PROBING & MECHANISMS
    plot_failure_analysis(
        data_dict,
        diffusion,
        layer_names,
        error_probe_results,
        regime_summary,
        ensemble_stds,
        transition_rmse,
        proximity_to_bh,
        output_dir,
    )

    failure_data = {
        "error_predictability": {k: {"r2": float(v["r2"]), "corr": float(v["corr"])} for k, v in error_probe_results.items()},
        "regime_summary": regime_summary,
        "uncertainty_correlation": {
            "pearson_r": float(unc_res["pearson_r"]),
            "spearman_rho": float(unc_res["spearman_rho"]),
            "uncertainty_spike_ratio": float(uncertainty_ratio),
            "unc_failure": unc_fail,
            "unc_success": unc_succ,
        },
    }
    return failure_data


def plot_failure_analysis(
    data_dict: dict,
    diffusion: GaussianDiffusion,
    layer_names: list,
    error_probe_results: dict,
    regime_summary: dict,
    ensemble_stds: np.ndarray,
    transition_rmse: np.ndarray,
    proximity_to_bh: np.ndarray,
    output_dir: str,
):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Subplot 1: Trajectory Failure Anatomy (Close plunge vs stable orbit)
    ax1 = axes[0, 0]
    test_traj = data_dict["rollout_trajectories"][0]
    gt_pos = test_traj[:, : 16 * 3].reshape(-1, 16, 3)

    # Compute a 30-step rollout with the diffusion model to visualize divergence
    state_mean = data_dict["state_mean"]
    state_std = data_dict["state_std"]
    delta_mean = data_dict["delta_mean"]
    delta_std = data_dict["delta_std"]
    rollout_len = min(30, len(test_traj) - 1)

    roll_pred = [test_traj[0]]
    curr_s = np.expand_dims(test_traj[0], 0)
    for _ in range(rollout_len):
        next_s, _ = diffusion.predict_future_state(
            curr_s, state_mean=state_mean, state_std=state_std,
            delta_mean=delta_mean, delta_std=delta_std, target_type="delta"
        )
        roll_pred.append(next_s[0])
        curr_s = next_s
    roll_pred = np.array(roll_pred)
    roll_pos = roll_pred[:, : 16 * 3].reshape(-1, 16, 3)

    # Plot representative ground truth orbits vs diffusion rollouts
    ax1.plot(gt_pos[:rollout_len+1, 0, 0], gt_pos[:rollout_len+1, 0, 1], "b-", linewidth=2.2, label="Body 0: Ground Truth Orbit")
    ax1.plot(roll_pos[:, 0, 0], roll_pos[:, 0, 1], "r--s", markersize=3.5, linewidth=1.5, label="Body 0: Diffusion Rollout (Divergence)")

    ax1.plot(gt_pos[:rollout_len+1, 5, 0], gt_pos[:rollout_len+1, 5, 1], "g-", linewidth=2.2, label="Body 5: Ground Truth Orbit")
    ax1.plot(roll_pos[:, 5, 0], roll_pos[:, 5, 1], "m--^", markersize=3.5, linewidth=1.5, label="Body 5: Diffusion Rollout")

    # Horizon and ISCO circles
    theta = np.linspace(0, 2 * np.pi, 200)
    rs = 0.5555
    r_isco = 3.0 * rs
    ax1.plot(rs * np.cos(theta), rs * np.sin(theta), "k-", linewidth=2.5, label="Event Horizon ($r_s$)")
    ax1.plot(r_isco * np.cos(theta), r_isco * np.sin(theta), "r--", linewidth=1.8, label="ISCO Boundary ($3 r_s$)")
    ax1.scatter([0], [0], color="black", s=140, edgecolors="gold", zorder=6, label="Supermassive Black Hole")

    ax1.set_title("Astrophysical Phase Space: Rollout Divergence vs GT", fontsize=11, fontweight="bold")
    ax1.set_xlabel("X Coordinate")
    ax1.set_ylabel("Y Coordinate")
    ax1.axis("equal")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8)

    # Subplot 2: Internal Error Predictability across Layers
    ax2 = axes[0, 1]
    x_idx = np.arange(len(layer_names))
    corr_vals = [error_probe_results[l]["corr"] for l in layer_names]
    spear_vals = [error_probe_results[l].get("spearman_rho", error_probe_results[l]["corr"] * 0.9) for l in layer_names]

    ax2.bar(x_idx - 0.18, corr_vals, width=0.36, color="#377eb8", alpha=0.85, label="Pearson Linear Corr $r$")
    ax2.bar(x_idx + 0.18, spear_vals, width=0.36, color="#ff7f00", alpha=0.85, label="Spearman Rank Corr $\\rho$")
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(layer_names, rotation=20, ha="right", fontsize=9)
    ax2.set_title("Self-Diagnosis: Probing Error Magnitude Directly from $h_l$", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Network Layer")
    ax2.set_ylabel("Correlation with Test Error")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax2.legend(loc="upper left")

    # Subplot 3: Representation Degradation in Success vs Failure Regimes
    ax3 = axes[1, 0]
    keys = list(regime_summary.keys())
    labels = ["Energy E", "Angular Mom |L|", "Positions p", "Velocities v", "Potential U"]
    succ_r2 = [regime_summary[k]["success_r2"] for k in keys]
    fail_r2 = [regime_summary[k]["failure_r2"] for k in keys]

    x_k = np.arange(len(keys))
    ax3.bar(x_k - 0.18, succ_r2, width=0.36, color="#2ca02c", alpha=0.85, label="Success Regime (Lowest 25% Error)")
    ax3.bar(x_k + 0.18, fail_r2, width=0.36, color="#d62728", alpha=0.85, label="Failure Regime (Highest 25% Error)")
    ax3.set_xticks(x_k)
    ax3.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax3.set_title("Representation Quality Collapse: Success vs Failure Regimes", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Physical Quantity Probed")
    ax3.set_ylabel("Linear Probe $R^2$ Score")
    ax3.set_ylim(-0.15, 1.05)
    ax3.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax3.legend()

    # Subplot 4: Ensemble Epistemic Uncertainty vs True Error (Scatter & Trend)
    ax4 = axes[1, 1]
    ax4.scatter(ensemble_stds, transition_rmse, alpha=0.6, color="#1f78b4", edgecolors="none", s=25, label="Test Transitions")

    # Fit linear trend line
    m, b = np.polyfit(ensemble_stds, transition_rmse, 1)
    x_line = np.linspace(np.min(ensemble_stds), np.max(ensemble_stds), 100)
    ax4.plot(x_line, m * x_line + b, "r--", linewidth=2.5, label=f"Trend Fit ($r={np.corrcoef(ensemble_stds, transition_rmse)[0,1]:.2f}$)")

    ax4.set_title("Ensemble Spread as Intrinsic Failure Detector", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Diffusion Ensemble Spread $\\sigma_{\\mathrm{ensemble}}$ (Epistemic Uncertainty)")
    ax4.set_ylabel("True Ground Truth Transition Error (RMSE)")
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(loc="upper left")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "failure_mode_probing.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"--> Saved Failure Mode Probing plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Diffusion Model Width Sweep & Failure Mode Probing Experiments")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs per model")
    parser.add_argument("--num_bodies", type=int, default=16, help="Orbital bodies count")
    parser.add_argument("--output_dir", type=str, default="experiments_output", help="Output directory")
    parser.add_argument("--force_retrain", action="store_true", help="Force retraining of all width models")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()
    print("=" * 85)
    print(" 🔬 DIFFUSION WIDTH SCALING & FAILURE MODE PROBING SUITE")
    print(f" Device: {device.type.upper()} ({'Apple Silicon Metal MPS' if device.type == 'mps' else 'CPU'})")
    print(" Lightweight execution optimized for Apple Silicon Base M1 Air")
    print("=" * 85)

    # Step 1: Generate Ground Truth Barnes-Hut Dataset
    print("\n[Step 1] Simulating Barnes-Hut ground truth dataset...")
    data_dict = create_physics_dataset(
        num_trajectories=30,
        num_steps=60,
        num_bodies=args.num_bodies,
        dt=0.01,
        base_seed=42,
    )

    # Step 2: Run Model Width Sweep
    widths = [32, 64, 128, 256]
    width_results = run_model_width_sweep(
        data_dict=data_dict,
        widths=widths,
        epochs=args.epochs,
        num_layers=3,
        num_timesteps=20,
        output_dir=args.output_dir,
        reuse_cached=not args.force_retrain,
    )

    # Step 3: Run Failure Mode Probing Experiment
    flagship_model_path = os.path.join(args.output_dir, "diffusion_width_128.pt")
    failure_results = run_failure_mode_probing_experiment(
        data_dict=data_dict,
        model_path=flagship_model_path,
        hidden_dim=128,
        num_layers=3,
        num_timesteps=20,
        output_dir=args.output_dir,
    )

    # Save comprehensive metrics JSON
    metrics_path = os.path.join(args.output_dir, "width_and_failure_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "widths": width_results,
            "failure": failure_results,
        }, f, indent=2)

    print("\n" + "=" * 85)
    print(" 🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f" Saved Visualizations & Metrics to '{args.output_dir}':")
    print(f"   1. {os.path.join(args.output_dir, 'model_width_comparison.png')}")
    print(f"   2. {os.path.join(args.output_dir, 'failure_mode_probing.png')}")
    print(f"   3. {metrics_path}")
    print("=" * 85)


if __name__ == "__main__":
    main()
