import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emulator.dataset import create_physics_dataset, generate_single_trajectory
from emulator.train import train_diffusion_emulator, extract_dataset_activations, get_device
from emulator.prober import PhysicalProbeSuite


def main():
    parser = argparse.ArgumentParser(description="Barnes-Hut Diffusion Physics Emulator & Probing Experiment")
    parser.add_argument("--num_bodies", type=int, default=16, help="Number of orbiting bodies")
    parser.add_argument("--num_trajectories", type=int, default=30, help="Number of simulation trajectories")
    parser.add_argument("--num_steps", type=int, default=60, help="Timesteps per trajectory")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs for diffusion model")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension of denoiser (small for M1)")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of ResNet layers")
    parser.add_argument("--num_timesteps", type=int, default=20, help="Diffusion schedule timesteps")
    parser.add_argument("--ensemble_size", type=int, default=12, help="Ensemble samples K")
    parser.add_argument("--target_type", type=str, default="delta", choices=["delta", "state"], help="Diffusion target")
    parser.add_argument("--output_dir", type=str, default="experiments_output", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()
    print("=" * 85)
    print(" 🌌 BARNES-HUT DIFFUSION PHYSICS EMULATOR & INTERNAL PROBING EXPERIMENT")
    print(f" Compute: {device.type.upper()} ({'Apple Silicon Metal MPS' if device.type == 'mps' else 'CPU'})")
    print(f" Config : Bodies={args.num_bodies}, Target={args.target_type}, Dim={args.hidden_dim}, Layers={args.num_layers}, Timesteps={args.num_timesteps}")
    print("=" * 85)

    # -------------------------------------------------------------------------
    # STEP 1: GENERATE GROUND TRUTH TRAJECTORIES WITH BARNES-HUT SIMULATOR
    # -------------------------------------------------------------------------
    print("\n[Step 1/4] Simulating Barnes-Hut ground truth trajectories...")
    data_dict = create_physics_dataset(
        num_trajectories=args.num_trajectories,
        num_steps=args.num_steps,
        num_bodies=args.num_bodies,
        dt=0.01,
        base_seed=42,
    )

    # -------------------------------------------------------------------------
    # STEP 2: TRAIN SMALL DIFFUSION EMULATOR ON APPLE SILICON MPS
    # -------------------------------------------------------------------------
    print("\n[Step 2/4] Training small diffusion emulator...")
    save_model_path = os.path.join(args.output_dir, "diffusion_emulator.pt")
    diffusion, history = train_diffusion_emulator(
        dataset_dict=data_dict,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_timesteps=args.num_timesteps,
        epochs=args.epochs,
        batch_size=64,
        learning_rate=1.5e-3,
        target_type=args.target_type,
        save_path=save_model_path,
    )

    # -------------------------------------------------------------------------
    # STEP 3: ENSEMBLE GENERATION & MULTI-STEP ROLLOUT
    # -------------------------------------------------------------------------
    print(f"\n[Step 3/4] Generating ensemble predictions with K={args.ensemble_size} samples...")
    test_x = data_dict["test_x"]
    test_y = data_dict["test_y"]
    state_mean = data_dict["state_mean"]
    state_std = data_dict["state_std"]
    delta_mean = data_dict["delta_mean"]
    delta_std = data_dict["delta_std"]

    # Select a test sample to evaluate one-step ensemble
    sample_idx = 5
    test_cond_raw = test_x[sample_idx : sample_idx + 1]
    true_next_raw = test_y[sample_idx]

    # Sample K ensemble paths for x_{t+1}
    ensemble_preds_raw = diffusion.sample_physical_ensemble(
        x_raw=test_cond_raw,
        state_mean=state_mean,
        state_std=state_std,
        delta_mean=delta_mean,
        delta_std=delta_std,
        ensemble_size=args.ensemble_size,
        target_type=args.target_type,
    )[:, 0, :]  # Shape: (K, State_dim)

    ens_mean = np.mean(ensemble_preds_raw, axis=0)
    ens_std = np.std(ensemble_preds_raw, axis=0)
    ens_rmse = float(np.sqrt(np.mean((ens_mean - true_next_raw) ** 2)))
    print(f"  Ensemble prediction for state_{sample_idx + 1}:")
    print(f"    Ground Truth State Norm: {np.linalg.norm(true_next_raw):.4f}")
    print(f"    Ensemble Mean vs Ground Truth RMSE: {ens_rmse:.6f}")
    print(f"    Mean Ensemble Spread (Epistemic Uncertainty): {np.mean(ens_std):.6f}")

    # Autoregressive rollout for 30 steps comparing with ground truth continuous trajectory
    print("  Running multi-step autoregressive rollout on held-out test trajectory...")
    test_traj = data_dict["rollout_trajectories"][0]
    rollout_len = min(35, len(test_traj) - 1)
    
    rollout_pred = [test_traj[0]]
    curr_state = np.expand_dims(test_traj[0], 0)

    for step in range(rollout_len):
        next_s, _ = diffusion.predict_future_state(
            curr_state,
            state_mean=state_mean,
            state_std=state_std,
            delta_mean=delta_mean,
            delta_std=delta_std,
            target_type=args.target_type,
        )
        rollout_pred.append(next_s[0])
        curr_state = next_s

    rollout_pred = np.array(rollout_pred)
    gt_rollout = test_traj[: rollout_len + 1]

    # One-step ensemble on step 1 of the test trajectory
    ens_test_cond = np.expand_dims(test_traj[0], 0)
    ens_true_next = test_traj[1]
    ensemble_preds_raw = diffusion.sample_physical_ensemble(
        x_raw=ens_test_cond,
        state_mean=state_mean,
        state_std=state_std,
        delta_mean=delta_mean,
        delta_std=delta_std,
        ensemble_size=args.ensemble_size,
        target_type=args.target_type,
    )[:, 0, :]

    ens_mean = np.mean(ensemble_preds_raw, axis=0)
    ens_std = np.std(ensemble_preds_raw, axis=0)
    ens_rmse = float(np.sqrt(np.mean((ens_mean - ens_true_next) ** 2)))
    print(f"  Ensemble prediction at t=0 -> t=1:")
    print(f"    Ensemble Mean vs Ground Truth RMSE: {ens_rmse:.6f}")
    print(f"    Mean Ensemble Spread (Uncertainty): {np.mean(ens_std):.6f}")

    # Plot ensemble & rollout comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Body 0 (X, Y) orbital position
    axes[0].plot(gt_rollout[:, 0], gt_rollout[:, 1], "k-o", label="Ground Truth Orbit (Barnes-Hut)", linewidth=2, markersize=4)
    axes[0].plot(rollout_pred[:, 0], rollout_pred[:, 1], "r--s", label="Diffusion Emulator Rollout", linewidth=1.5, markersize=3)
    axes[0].scatter(ensemble_preds_raw[:, 0], ensemble_preds_raw[:, 1], color="#2b5c8f", alpha=0.8, s=40, zorder=5, label=f"Ensemble Samples (K={args.ensemble_size})")
    # Central Black hole marker
    axes[0].scatter([0], [0], color="black", s=120, edgecolors="gold", linewidths=1.5, zorder=6, label="Black Hole")
    axes[0].set_title(f"Orbital Rollout & Diffusion Ensemble (K={args.ensemble_size})", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("X Position")
    axes[0].set_ylabel("Y Position")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Training loss curve
    axes[1].plot(history["train_losses"], label="Train MSE Loss", color="#1b9e77", linewidth=2)
    axes[1].set_title("Small Diffusion Emulator Loss on Apple Silicon MPS", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Diffusion Loss (MSE)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    rollout_plot_path = os.path.join(args.output_dir, "ensemble_rollout_comparison.png")
    plt.savefig(rollout_plot_path, dpi=200)
    plt.close()
    print(f"  Saved rollout & ensemble plot to {rollout_plot_path}")

    # -------------------------------------------------------------------------
    # STEP 4: PHYSICAL INFORMATION PROBING EXPERIMENT
    # -------------------------------------------------------------------------
    print("\n[Step 4/4] Probing internal representations h_l for physical information...")
    probe_suite = PhysicalProbeSuite(alpha=1.0)
    train_targets = probe_suite.extract_target_arrays(data_dict["train_phys"])
    test_targets = probe_suite.extract_target_arrays(data_dict["test_phys"])

    # Determine targets used during extraction
    if args.target_type == "delta":
        tgt_train = data_dict["train_delta"]
        tgt_test = data_dict["test_delta"]
        t_mean, t_std = delta_mean, delta_std
    else:
        tgt_train = data_dict["train_y"]
        tgt_test = data_dict["test_y"]
        t_mean, t_std = state_mean, state_std

    # Extract layer activations h_l at clean diffusion step (k=0)
    train_acts = extract_dataset_activations(
        diffusion,
        data_dict["train_x"],
        tgt_train,
        state_mean,
        state_std,
        t_mean,
        t_std,
        diffusion_step=0,
    )
    test_acts = extract_dataset_activations(
        diffusion,
        data_dict["test_x"],
        tgt_test,
        state_mean,
        state_std,
        t_mean,
        t_std,
        diffusion_step=0,
    )

    layer_names = list(train_acts.keys())
    print(f"  Extracted activations across {len(layer_names)} layers: {layer_names}")

    target_keys = [
        ("total_energy", "Total Energy (E)"),
        ("angular_momentum_norm", "Angular Momentum (|L|)"),
        ("linear_momentum_norm", "Linear Momentum (|P|)"),
        ("positions", "Positions (p_1..p_N)"),
        ("velocities", "Velocities (v_1..v_N)"),
        ("kinetic_energy", "Kinetic Energy (K)"),
        ("potential_energy", "Potential Energy (U)"),
    ]

    layer_probe_results = {}
    for layer in layer_names:
        h_tr = train_acts[layer]
        h_te = test_acts[layer]
        res = probe_suite.evaluate_representation(h_tr, h_te, train_targets, test_targets, probe_type="linear")
        layer_probe_results[layer] = res

    print("\n  " + "=" * 88)
    print("  PHYSICAL INFORMATION PROBING RESULTS (Linear Probe R^2 on Held-Out Test Set)")
    print("  " + "=" * 88)
    print(f"  {'Physical Target':<25} | " + " | ".join([f"{l:<11}" for l in layer_names]))
    print("  " + "-" * (27 + 14 * len(layer_names)))

    for tgt_key, tgt_name in target_keys:
        row_str = f"  {tgt_name:<25} | "
        for layer in layer_names:
            r2_val = layer_probe_results[layer][tgt_key]["r2"]
            row_str += f"{r2_val:11.4f} | "
        print(row_str)
    print("  " + "=" * (27 + 14 * len(layer_names)))

    # Also evaluate RMSE / Relative error for Total Energy and Angular Momentum
    print("\n  PROBE PREDICTION ERROR (RMSE & Relative Error on Test Set):")
    for key, name in [("total_energy", "Total Energy"), ("angular_momentum_norm", "Angular Momentum"), ("linear_momentum_norm", "Linear Momentum")]:
        best_layer = max(layer_names, key=lambda l: layer_probe_results[l][key]["r2"])
        best_r2 = layer_probe_results[best_layer][key]["r2"]
        best_rmse = layer_probe_results[best_layer][key]["rmse"]
        best_nrmse = layer_probe_results[best_layer][key]["nrmse"]
        print(f"    • {name:<18} (Best Layer: {best_layer:<12}): R^2 = {best_r2:7.4f}, RMSE = {best_rmse:9.5f}, NRMSE = {best_nrmse * 100:6.2f}%")

    # 4B: Probe across Diffusion Timesteps (Reverse Denoising Schedule)
    steps_to_check = [args.num_timesteps - 1, args.num_timesteps // 2, 0]
    time_probe_results = {}
    print(f"\n  Probing intermediate layer ('layer_2') across diffusion steps {steps_to_check}...")
    for k_step in steps_to_check:
        k_tr = extract_dataset_activations(diffusion, data_dict["train_x"], tgt_train, state_mean, state_std, t_mean, t_std, diffusion_step=k_step)
        k_te = extract_dataset_activations(diffusion, data_dict["test_x"], tgt_test, state_mean, state_std, t_mean, t_std, diffusion_step=k_step)
        res_k = probe_suite.evaluate_representation(k_tr["layer_2"], k_te["layer_2"], train_targets, test_targets, probe_type="linear")
        time_probe_results[k_step] = res_k

    # -------------------------------------------------------------------------
    # VISUALIZATION OF PROBING RESULTS
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    quantities_to_plot = [
        ("total_energy", "Total Energy E", "#d95f02", "o-"),
        ("angular_momentum_norm", "Angular Momentum |L|", "#7570b3", "s-"),
        ("linear_momentum_norm", "Linear Momentum |P|", "#1b9e77", "^-"),
        ("positions", "Positions", "#e7298a", "d-"),
        ("velocities", "Velocities", "#e6ab02", "v-"),
    ]

    # Plot 1: Layer-wise R^2 for each physical quantity
    x_indices = np.arange(len(layer_names))
    for q_key, q_label, color, marker in quantities_to_plot:
        r2_vals = [layer_probe_results[l][q_key]["r2"] for l in layer_names]
        ax1.plot(x_indices, r2_vals, marker, label=q_label, color=color, linewidth=2, markersize=7)

    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(layer_names, rotation=25, ha="right", fontsize=9)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("Layer-wise Physical Information Probing ($R^2$ Score)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Internal Layer ($h_l$)")
    ax1.set_ylabel("Linear Probe $R^2$ Score (Held-Out Test)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    # Plot 2: Diffusion Timestep vs Physical Recoverability
    timesteps_ordered = sorted(steps_to_check)
    for q_key, q_label, color, marker in quantities_to_plot:
        vals = [time_probe_results[ts][q_key]["r2"] for ts in timesteps_ordered]
        ax2.plot(timesteps_ordered, vals, marker, label=q_label, color=color, linewidth=2, markersize=7)

    ax2.set_title("Physical Representation vs Reverse Diffusion Timestep", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Diffusion Step $k$ (0 = Clean, " + f"{args.num_timesteps - 1} = Pure Noise)")
    ax2.set_ylabel("Linear Probe $R^2$ Score")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left")

    plt.tight_layout()
    probing_plot_path = os.path.join(args.output_dir, "probing_results_layers.png")
    plt.savefig(probing_plot_path, dpi=200)
    plt.close()
    print(f"\n  Saved probing analysis plot to {probing_plot_path}")

    print("\n" + "=" * 85)
    print(" 🎉 PROBING EXPERIMENT COMPLETE!")
    print(f" Summary artifacts generated in '{args.output_dir}':")
    print(f"   1. {rollout_plot_path}")
    print(f"   2. {probing_plot_path}")
    print(f"   3. {save_model_path}")
    print("=" * 85)


if __name__ == "__main__":
    main()
