import os
import sys
import time
import json
import math
from typing import Tuple, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emulator.dataset import create_physics_dataset, compute_physical_quantities
from emulator.train import get_device, extract_dataset_activations
from emulator.prober import PhysicalProbeSuite
from emulator.model import SmallPhysicsDenoiser
from emulator.diffusion import GaussianDiffusion


class DeterministicMLP(nn.Module):
    """
    Deterministic feedforward MLP baseline matching denoiser parameter budget.
    Directly predicts displacement Delta x from state x_t.
    """
    def __init__(self, state_dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, state_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp_baseline(
    train_x: np.ndarray,
    train_delta: np.ndarray,
    test_x: np.ndarray,
    test_delta: np.ndarray,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    delta_mean: np.ndarray,
    delta_std: np.ndarray,
    hidden_dim: int,
    epochs: int = 25,
    device: torch.device = torch.device("cpu"),
) -> DeterministicMLP:
    norm_tx = (train_x - state_mean) / state_std
    norm_td = (train_delta - delta_mean) / delta_std
    tensor_x = torch.from_numpy(norm_tx).float()
    tensor_d = torch.from_numpy(norm_td).float()

    dataset = TensorDataset(tensor_x, tensor_d)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    mlp = DeterministicMLP(state_dim=train_x.shape[1], hidden_dim=hidden_dim, num_layers=3).to(device)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=1.5e-3, weight_decay=1e-4)

    mlp.train()
    for epoch in range(epochs):
        for bx, bd in loader:
            bx, bd = bx.to(device), bd.to(device)
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(mlp(bx), bd)
            loss.backward()
            optimizer.step()
    return mlp


def evaluate_rollout_rmse(
    predict_fn,
    rollout_trajectories: list,
    rollout_len: int = 30,
) -> Tuple[float, float, list]:
    all_traj_rmse = []
    step_rmses = np.zeros(rollout_len + 1)

    for traj in rollout_trajectories:
        T = min(rollout_len, len(traj) - 1)
        gt = traj[: T + 1]
        pred = [traj[0]]
        curr = traj[0]
        for step in range(T):
            next_state = predict_fn(curr)
            pred.append(next_state)
            curr = next_state
        pred = np.array(pred)
        per_step = np.sqrt(np.mean((pred - gt) ** 2, axis=1))
        step_rmses[: T + 1] += per_step
        all_traj_rmse.append(float(np.mean(per_step)))

    step_rmses /= len(rollout_trajectories)
    mean_rmse = float(np.mean(all_traj_rmse))
    final_rmse = float(step_rmses[rollout_len])
    return mean_rmse, final_rmse, step_rmses.tolist()


def run_full_suite(
    output_dir: str = "experiments_output",
    num_seeds: int = 5,
    epochs: int = 25,
):
    os.makedirs(output_dir, exist_ok=True)
    device = get_device()
    print("=" * 85)
    print(f" 🚀 RUNNING RIGOROUS RE-EXPERIMENTATION SUITE (Seeds: {num_seeds})")
    print(f" Compute: {device.type.upper()} | Cosine Diffusion Schedule (alpha_bar_T -> 0)")
    print("=" * 85)

    # 1. Dataset Generation (Strict Trajectory-Level Partitioning)
    print("\n[Step 1/6] Simulating 35 Barnes-Hut trajectories with strict trajectory partitioning...")
    data_dict = create_physics_dataset(
        num_trajectories=35,
        num_steps=60,
        num_bodies=16,
        dt=0.01,
        base_seed=100,
    )

    train_x = data_dict["train_x"]
    train_y = data_dict["train_y"]
    train_delta = data_dict["train_delta"]
    test_x = data_dict["test_x"]
    test_y = data_dict["test_y"]
    test_delta = data_dict["test_delta"]
    state_mean = data_dict["state_mean"]
    state_std = data_dict["state_std"]
    delta_mean = data_dict["delta_mean"]
    delta_std = data_dict["delta_std"]
    rollout_trajs = data_dict["rollout_trajectories"]

    # 2. Baselines (Identity & Kinematic)
    print("\n[Step 2/6] Evaluating Non-Parametric Baselines (Identity & Kinematic)...")
    id_mean_rmse, id_final_rmse, id_curve = evaluate_rollout_rmse(
        predict_fn=lambda s: s,
        rollout_trajectories=rollout_trajs,
        rollout_len=30,
    )
    dt = 0.01
    def kin_pred(s):
        ns = s.copy()
        ns[:48] += s[48:96] * dt
        return ns
    kin_mean_rmse, kin_final_rmse, kin_curve = evaluate_rollout_rmse(
        predict_fn=kin_pred,
        rollout_trajectories=rollout_trajs,
        rollout_len=30,
    )
    print(f"  Identity Baseline (Zero Motion)   -> 30-step Mean RMSE: {id_mean_rmse:.4f}, Final RMSE: {id_final_rmse:.4f}")
    print(f"  Kinematic Baseline (Delta p=v*dt) -> 30-step Mean RMSE: {kin_mean_rmse:.4f}, Final RMSE: {kin_final_rmse:.4f}")

    # Clamped scale statistics to avoid microscopic-variance amplification (e.g. static BH coordinates)
    state_std_safe = np.copy(state_std)
    state_std_safe[state_std_safe < 0.05] = 1.0
    delta_std_safe = np.copy(delta_std)
    delta_std_safe[delta_std_safe < 0.05] = 1.0

    widths = [32, 64, 128, 256]
    width_metrics = {str(w): {"seeds": []} for w in widths}
    mlp_metrics = {str(w): {"seeds": []} for w in widths}
    random_probing_metrics = {str(w): {} for w in widths}

    probe_suite = PhysicalProbeSuite(alpha=1.0)
    train_targets = probe_suite.extract_target_arrays(data_dict["train_phys"])
    test_targets = probe_suite.extract_target_arrays(data_dict["test_phys"])

    # Sanity Check: Decode positions and velocities directly from linear input projection
    print("\n  [Sanity Check] Evaluating input decodability on raw coordinate projections...")
    sanity_denoiser = SmallPhysicsDenoiser(state_dim=train_x.shape[1], hidden_dim=128, num_layers=3)
    with torch.no_grad():
        x_tr_norm = torch.from_numpy((train_x - state_mean) / state_std_safe).float()
        x_te_norm = torch.from_numpy((test_x - state_mean) / state_std_safe).float()
        h_in_tr = sanity_denoiser.input_proj(torch.cat([torch.zeros_like(x_tr_norm), x_tr_norm], dim=-1)).numpy()
        h_in_te = sanity_denoiser.input_proj(torch.cat([torch.zeros_like(x_te_norm), x_te_norm], dim=-1)).numpy()
    ridge_pos_sanity = Ridge(alpha=1.0).fit(h_in_tr, train_targets["positions"])
    pos_sanity_r2 = float(ridge_pos_sanity.score(h_in_te, test_targets["positions"]))
    ridge_vel_sanity = Ridge(alpha=1.0).fit(h_in_tr, train_targets["velocities"])
    vel_sanity_r2 = float(ridge_vel_sanity.score(h_in_te, test_targets["velocities"]))
    print(f"  Input Sanity Decodability -> Positions R^2: {pos_sanity_r2:.4f} | Velocities R^2: {vel_sanity_r2:.4f}")

    # 3. Model Width Sweep across 5 Seeds
    print(f"\n[Step 3/6] Training Diffusion & MLP models across {num_seeds} seeds...")
    flagship_model = None

    for seed in range(1, num_seeds + 1):
        print(f"\n---> Running Seed {seed}/{num_seeds}...")
        torch.manual_seed(seed * 42)
        np.random.seed(seed * 42)

        for width in widths:
            t0 = time.time()
            # Initialize denoiser and diffusion with cosine schedule
            denoiser = SmallPhysicsDenoiser(state_dim=train_x.shape[1], hidden_dim=width, num_layers=3)
            diffusion = GaussianDiffusion(model=denoiser, num_timesteps=20, schedule="cosine", device=device)

            # Train diffusion emulator with safe standardization
            norm_tx = (train_x - state_mean) / state_std_safe
            norm_td = (train_delta - delta_mean) / delta_std_safe
            loader = DataLoader(
                TensorDataset(torch.from_numpy(norm_tx).float(), torch.from_numpy(norm_td).float()),
                batch_size=64, shuffle=True,
            )
            opt = torch.optim.AdamW(diffusion.model.parameters(), lr=1.5e-3, weight_decay=1e-4)

            train_loss = 0.0
            for ep in range(epochs):
                diffusion.model.train()
                for bx, bd in loader:
                    bx, bd = bx.to(device), bd.to(device)
                    opt.zero_grad()
                    loss = diffusion.compute_loss(target=bd, cond=bx)
                    loss.backward()
                    opt.step()
                    train_loss = loss.item()

            # Validation loss on held-out test transitions
            diffusion.model.eval()
            with torch.no_grad():
                val_x = torch.from_numpy((test_x - state_mean) / state_std_safe).float().to(device)
                val_d = torch.from_numpy((test_delta - delta_mean) / delta_std_safe).float().to(device)
                val_loss = diffusion.compute_loss(target=val_d, cond=val_x).item()

            # Predict function for rollouts
            def diff_pred(curr_s):
                with torch.no_grad():
                    cs = np.expand_dims(curr_s, 0)
                    ns, _ = diffusion.predict_future_state(
                        cs, state_mean=state_mean, state_std=state_std_safe,
                        delta_mean=delta_mean, delta_std=delta_std_safe, target_type="delta"
                    )
                    return ns[0]

            m_rmse, f_rmse, r_curve = evaluate_rollout_rmse(diff_pred, rollout_trajs, rollout_len=30)

            # Invariant probing on intermediate layer_2
            train_acts = extract_dataset_activations(
                diffusion, train_x, train_delta, state_mean, state_std_safe, delta_mean, delta_std_safe,
                diffusion_step=0, use_inference_sampling=True
            )
            test_acts = extract_dataset_activations(
                diffusion, test_x, test_delta, state_mean, state_std_safe, delta_mean, delta_std_safe,
                diffusion_step=0, use_inference_sampling=True
            )
            probe_res = probe_suite.evaluate_representation(
                train_acts["layer_2"], test_acts["layer_2"], train_targets, test_targets
            )

            width_metrics[str(width)]["seeds"].append({
                "seed": seed,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "mean_rollout_rmse": float(m_rmse),
                "final_rollout_rmse": float(f_rmse),
                "rollout_curve": r_curve,
                "probing_r2": {k: float(v["r2"]) for k, v in probe_res.items()},
            })

            # Train matched MLP baseline
            mlp = train_mlp_baseline(
                train_x, train_delta, test_x, test_delta,
                state_mean, state_std_safe, delta_mean, delta_std_safe,
                hidden_dim=width, epochs=epochs, device=device
            )
            def mlp_pred(curr_s):
                with torch.no_grad():
                    nx = (curr_s - state_mean) / state_std_safe
                    bx = torch.from_numpy(nx).float().unsqueeze(0).to(device)
                    pred_d = mlp(bx).squeeze(0).cpu().numpy()
                    pred_delta = pred_d * delta_std_safe + delta_mean
                    return curr_s + pred_delta

            mlp_m_rmse, mlp_f_rmse, mlp_curve = evaluate_rollout_rmse(mlp_pred, rollout_trajs, rollout_len=30)
            mlp_metrics[str(width)]["seeds"].append({
                "seed": seed,
                "mean_rollout_rmse": float(mlp_m_rmse),
                "final_rollout_rmse": float(mlp_f_rmse),
                "rollout_curve": mlp_curve,
            })

            if width == 128 and seed == 1:
                flagship_model = diffusion

            print(f"    d={width:3d} (Seed {seed}) -> Diff Val Loss: {val_loss:.4f} | Diff Rollout RMSE: {m_rmse:.3f} | MLP Rollout RMSE: {mlp_m_rmse:.3f}")

    # Randomly-initialized Network Probing Control (Feature Count Effect)
    print("\n[Step 4/6] Evaluating Randomly Initialized Network Control across widths...")
    for width in widths:
        rand_denoiser = SmallPhysicsDenoiser(state_dim=train_x.shape[1], hidden_dim=width, num_layers=3)
        rand_diff = GaussianDiffusion(model=rand_denoiser, num_timesteps=20, schedule="cosine", device=device)
        rand_train_acts = extract_dataset_activations(
            rand_diff, train_x, train_delta, state_mean, state_std, delta_mean, delta_std,
            diffusion_step=0, use_inference_sampling=True
        )
        rand_test_acts = extract_dataset_activations(
            rand_diff, test_x, test_delta, state_mean, state_std, delta_mean, delta_std,
            diffusion_step=0, use_inference_sampling=True
        )
        rand_res = probe_suite.evaluate_representation(
            rand_train_acts["layer_2"], rand_test_acts["layer_2"], train_targets, test_targets
        )
        random_probing_metrics[str(width)] = {k: float(v["r2"]) for k, v in rand_res.items()}
        print(f"    Random Init d={width:3d} -> Energy R2: {rand_res['total_energy']['r2']:.4f} | Ang Mom R2: {rand_res['angular_momentum_norm']['r2']:.4f} | Pos R2: {rand_res['positions']['r2']:.4f}")

    # 4. Error Predictability with Controlled Baselines on Flagship Model
    print("\n[Step 5/6] Probing Error Predictability with Non-linear and Physical Controls...")
    test_acts = extract_dataset_activations(
        flagship_model, test_x, test_delta, state_mean, state_std, delta_mean, delta_std,
        diffusion_step=0, use_inference_sampling=True
    )
    train_acts = extract_dataset_activations(
        flagship_model, train_x, train_delta, state_mean, state_std, delta_mean, delta_std,
        diffusion_step=0, use_inference_sampling=True
    )

    # Compute actual test errors from inference sampling
    test_pred_deltas = []
    for i in range(len(test_x)):
        s_curr = test_x[i]
        with torch.no_grad():
            cs = np.expand_dims(s_curr, 0)
            ns, _ = flagship_model.predict_future_state(
                cs, state_mean=state_mean, state_std=state_std,
                delta_mean=delta_mean, delta_std=delta_std, target_type="delta"
            )
            test_pred_deltas.append(ns[0] - s_curr)
    test_pred_deltas = np.array(test_pred_deltas)
    actual_test_errors = np.linalg.norm(test_pred_deltas - test_delta, axis=1)
    log_actual_errors = np.log(np.maximum(actual_test_errors, 1e-6))

    # Hand-crafted physical features baseline
    def extract_phys_features(x_array):
        feats = []
        n_bodies = 16
        for vec in x_array:
            pos = vec[: n_bodies * 3].reshape(n_bodies, 3)
            vel = vec[n_bodies * 3 : n_bodies * 6].reshape(n_bodies, 3)
            bh_pos = vec[n_bodies * 6 : n_bodies * 6 + 3]
            r_bh = np.linalg.norm(pos - bh_pos[None, :], axis=1)
            v_mags = np.linalg.norm(vel, axis=1)
            r_min = float(np.min(r_bh))
            v_mean = float(np.mean(v_mags))
            v_max = float(np.max(v_mags))
            L_tot = np.linalg.norm(np.sum(np.cross(pos, vel), axis=0))
            d_isco = abs(r_min - 3.0 * 0.5555)
            feats.append([r_min / 0.5555, v_mean, v_max, L_tot, d_isco])
        return np.array(feats, dtype=np.float32)

    phys_train = extract_phys_features(train_x)
    phys_test = extract_phys_features(test_x)

    # Compute train errors to fit error probes
    train_pred_deltas = []
    for i in range(len(train_x)):
        s_curr = train_x[i]
        with torch.no_grad():
            cs = np.expand_dims(s_curr, 0)
            ns, _ = flagship_model.predict_future_state(
                cs, state_mean=state_mean, state_std=state_std,
                delta_mean=delta_mean, delta_std=delta_std, target_type="delta"
            )
            train_pred_deltas.append(ns[0] - s_curr)
    train_pred_deltas = np.array(train_pred_deltas)
    actual_train_errors = np.linalg.norm(train_pred_deltas - train_delta, axis=1)

    # Baselines: Physical features, Raw coordinates linear, Raw coordinates non-linear MLP
    ridge_phys = Ridge(alpha=1.0).fit(phys_train, actual_train_errors)
    pred_phys = ridge_phys.predict(phys_test)

    ridge_raw = Ridge(alpha=1.0).fit(train_x, actual_train_errors)
    pred_raw_lin = ridge_raw.predict(test_x)

    mlp_raw = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=200, random_state=42).fit(train_x, actual_train_errors)
    pred_raw_nonlin = mlp_raw.predict(test_x)

    def calc_metrics(y_true, y_pred):
        p_r, _ = pearsonr(y_true, y_pred)
        s_rho, _ = spearmanr(y_true, y_pred)
        # R2 score
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        r2 = 1.0 - (ss_res / max(ss_tot, 1e-12))
        # Log correlation
        p_r_log, _ = pearsonr(np.log(np.maximum(y_true, 1e-6)), np.log(np.maximum(np.clip(y_pred, 1e-6, None), 1e-6)))
        return float(p_r), float(s_rho), float(r2), float(p_r_log)

    layer_error_metrics = {}
    layer_names = ["layer_input", "layer_1", "layer_2", "layer_3", "layer_pre_head"]
    for l_name in layer_names:
        h_tr = train_acts[l_name]
        h_te = test_acts[l_name]
        r_probe = Ridge(alpha=1.0).fit(h_tr, actual_train_errors)
        pred_err = r_probe.predict(h_te)
        pr, srho, r2, pr_log = calc_metrics(actual_test_errors, pred_err)
        layer_error_metrics[l_name] = {
            "pearson_r": pr,
            "spearman_rho": srho,
            "test_r2": r2,
            "log_pearson_r": pr_log,
        }

    # Residualized error prediction beyond non-linear raw state baseline
    train_res_nonlin = actual_train_errors - mlp_raw.predict(train_x)
    test_res_nonlin = actual_test_errors - pred_raw_nonlin
    res_probe = Ridge(alpha=1.0).fit(train_acts["layer_pre_head"], train_res_nonlin)
    pred_res = res_probe.predict(test_acts["layer_pre_head"])
    pr_res, srho_res, _, _ = calc_metrics(test_res_nonlin, pred_res)

    baseline_metrics = {
        "physical_features": calc_metrics(actual_test_errors, pred_phys),
        "raw_coordinates_linear": calc_metrics(actual_test_errors, pred_raw_lin),
        "raw_coordinates_nonlinear_mlp": calc_metrics(actual_test_errors, pred_raw_nonlin),
        "residualized_pre_head": (pr_res, srho_res),
    }

    # 5. Failure Regime Dissection (Honest Quartile & Full-Spectrum Analysis)
    print("\n[Step 6/6] Analyzing Failure Regimes & Active Error Intervention...")
    q25 = np.percentile(actual_test_errors, 25)
    q75 = np.percentile(actual_test_errors, 75)
    succ_idx = np.where(actual_test_errors <= q25)[0]
    fail_idx = np.where(actual_test_errors >= q75)[0]

    # Probing across regimes
    regime_results = {}
    for tgt_name, y_tgt in test_targets.items():
        if len(y_tgt.shape) == 1:
            y_tgt = y_tgt[:, None]
        # Fit on all training, evaluate separately on success and failure
        probe = Ridge(alpha=1.0).fit(train_acts["layer_2"], train_targets[tgt_name])
        y_pred = probe.predict(test_acts["layer_2"])
        def get_r2(true_sub, pred_sub):
            ss_tot = np.sum((true_sub - np.mean(true_sub, axis=0)) ** 2)
            ss_res = np.sum((true_sub - pred_sub) ** 2)
            return float(1.0 - ss_res / max(ss_tot, 1e-12))

        regime_results[tgt_name] = {
            "success_r2": get_r2(y_tgt[succ_idx], y_pred[succ_idx]),
            "failure_r2": get_r2(y_tgt[fail_idx], y_pred[fail_idx]),
        }

    # Active Error Intervention Evaluation with Controls
    print("\n  [Intervention Suite] Evaluating Probe-Guided vs Controlled Damping...")
    pre_head_probe = Ridge(alpha=1.0).fit(train_acts["layer_pre_head"], actual_train_errors)
    err_threshold = float(np.median(actual_train_errors))

    # 1. Unconstrained
    def diff_pred_unconstrained(curr_s):
        with torch.no_grad():
            cs = np.expand_dims(curr_s, 0)
            ns, _ = flagship_model.predict_future_state(
                cs, state_mean=state_mean, state_std=state_std_safe,
                delta_mean=delta_mean, delta_std=delta_std_safe, target_type="delta"
            )
            return ns[0]

    uncon_mean, uncon_final, _ = evaluate_rollout_rmse(diff_pred_unconstrained, rollout_trajs, rollout_len=30)

    # 2. Constant Damping (always 0.7)
    def diff_pred_const_damp(curr_s):
        with torch.no_grad():
            cs = np.expand_dims(curr_s, 0)
            ns, _ = flagship_model.predict_future_state(
                cs, state_mean=state_mean, state_std=state_std_safe,
                delta_mean=delta_mean, delta_std=delta_std_safe, target_type="delta"
            )
            pred_delta = (ns[0] - curr_s) * 0.7
            return curr_s + pred_delta

    const_mean, const_final, _ = evaluate_rollout_rmse(diff_pred_const_damp, rollout_trajs, rollout_len=30)

    # 3. Random Damping (50% probability)
    rng_int = np.random.RandomState(42)
    def diff_pred_rand_damp(curr_s):
        with torch.no_grad():
            cs = np.expand_dims(curr_s, 0)
            ns, _ = flagship_model.predict_future_state(
                cs, state_mean=state_mean, state_std=state_std_safe,
                delta_mean=delta_mean, delta_std=delta_std_safe, target_type="delta"
            )
            pred_delta = ns[0] - curr_s
            if rng_int.rand() < 0.5:
                pred_delta = pred_delta * 0.7
            return curr_s + pred_delta

    rand_mean, rand_final, _ = evaluate_rollout_rmse(diff_pred_rand_damp, rollout_trajs, rollout_len=30)

    # 4. Probe-Guided Intervention
    def diff_pred_with_intervention(curr_s):
        with torch.no_grad():
            cs = np.expand_dims(curr_s, 0)
            norm_cs = (cs - state_mean) / state_std_safe
            bx = torch.from_numpy(norm_cs).float().to(device)
            ns, acts = flagship_model.sample_next_state(bx, record_activations_at_step=0)
            h_pre = acts["layer_pre_head"].cpu().numpy()
            pred_err = pre_head_probe.predict(h_pre)[0]

            delta_norm = ns.cpu().numpy()[0]
            pred_delta = delta_norm * delta_std_safe + delta_mean
            if pred_err > err_threshold:
                pred_delta = pred_delta * 0.7
            return curr_s + pred_delta

    interv_mean_rmse, interv_final_rmse, _ = evaluate_rollout_rmse(
        diff_pred_with_intervention, rollout_trajs, rollout_len=30
    )

    print(f"  Flagship Rollout RMSE (Unconstrained):             {uncon_mean:.3f}")
    print(f"  Flagship Rollout RMSE (Constant Damping 0.7):      {const_mean:.3f}")
    print(f"  Flagship Rollout RMSE (Random 50% Damping):        {rand_mean:.3f}")
    print(f"  Flagship Rollout RMSE (Probe-Guided Damping):      {interv_mean_rmse:.3f}")

    # Aggregate Width Scaling Statistics (Mean +- Std)
    summary_width = {}
    for w in widths:
        s_data = width_metrics[str(w)]["seeds"]
        m_rmses = [s["mean_rollout_rmse"] for s in s_data]
        val_losses = [s["val_loss"] for s in s_data]
        mlp_rmses = [s["mean_rollout_rmse"] for s in mlp_metrics[str(w)]["seeds"]]
        e_r2 = [s["probing_r2"]["total_energy"] for s in s_data]
        l_r2 = [s["probing_r2"]["angular_momentum_norm"] for s in s_data]
        p_r2 = [s["probing_r2"]["positions"] for s in s_data]

        summary_width[str(w)] = {
            "val_loss_mean": float(np.mean(val_losses)),
            "val_loss_std": float(np.std(val_losses)),
            "diff_rollout_rmse_mean": float(np.mean(m_rmses)),
            "diff_rollout_rmse_std": float(np.std(m_rmses)),
            "mlp_rollout_rmse_mean": float(np.mean(mlp_rmses)),
            "mlp_rollout_rmse_std": float(np.std(mlp_rmses)),
            "energy_r2_mean": float(np.mean(e_r2)),
            "energy_r2_std": float(np.std(e_r2)),
            "ang_mom_r2_mean": float(np.mean(l_r2)),
            "ang_mom_r2_std": float(np.std(l_r2)),
            "pos_r2_mean": float(np.mean(p_r2)),
            "pos_r2_std": float(np.std(p_r2)),
            "random_energy_r2": random_probing_metrics[str(w)]["total_energy"],
            "random_ang_mom_r2": random_probing_metrics[str(w)]["angular_momentum_norm"],
            "random_pos_r2": random_probing_metrics[str(w)]["positions"],
        }

    # Save comprehensive metrics JSON
    out_json = {
        "identity_baseline": {"mean_rmse": id_mean_rmse, "final_rmse": id_final_rmse},
        "kinematic_baseline": {"mean_rmse": kin_mean_rmse, "final_rmse": kin_final_rmse},
        "sanity_check": {"positions_r2": pos_sanity_r2, "velocities_r2": vel_sanity_r2},
        "width_scaling_summary": summary_width,
        "width_raw_seeds": width_metrics,
        "mlp_raw_seeds": mlp_metrics,
        "random_init_probing": random_probing_metrics,
        "layer_error_metrics": layer_error_metrics,
        "error_baselines": {
            "physical_features": {
                "pearson_r": baseline_metrics["physical_features"][0],
                "spearman_rho": baseline_metrics["physical_features"][1],
                "test_r2": baseline_metrics["physical_features"][2],
            },
            "raw_coordinates_linear": {
                "pearson_r": baseline_metrics["raw_coordinates_linear"][0],
                "spearman_rho": baseline_metrics["raw_coordinates_linear"][1],
                "test_r2": baseline_metrics["raw_coordinates_linear"][2],
            },
            "raw_coordinates_nonlinear_mlp": {
                "pearson_r": baseline_metrics["raw_coordinates_nonlinear_mlp"][0],
                "spearman_rho": baseline_metrics["raw_coordinates_nonlinear_mlp"][1],
                "test_r2": baseline_metrics["raw_coordinates_nonlinear_mlp"][2],
            },
            "residual_pre_head": {
                "pearson_r": baseline_metrics["residualized_pre_head"][0],
                "spearman_rho": baseline_metrics["residualized_pre_head"][1],
            },
        },
        "regime_results": regime_results,
        "intervention_suite": {
            "identity_rmse": float(id_mean_rmse),
            "kinematic_rmse": float(kin_mean_rmse),
            "unconstrained_rmse": float(uncon_mean),
            "constant_damp_rmse": float(const_mean),
            "random_damp_rmse": float(rand_mean),
            "probe_intervened_rmse": float(interv_mean_rmse),
        },
    }

    metrics_path = os.path.join(output_dir, "rigorous_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\n--> Successfully saved rigorous metrics to {metrics_path}")

    # Generate Publication Plots
    plot_rigorous_figures(out_json, output_dir, data_dict, flagship_model, widths)


def plot_rigorous_figures(metrics: dict, output_dir: str, data_dict: dict, flagship_model, widths: list):
    # Figure 1: Model Width Scaling with Baselines & Random Controls
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    w_str = [str(w) for w in widths]
    w_num = np.array(widths)

    # 1. Validation Loss vs Width
    ax1 = axes[0, 0]
    val_means = [metrics["width_scaling_summary"][w]["val_loss_mean"] for w in w_str]
    val_stds = [metrics["width_scaling_summary"][w]["val_loss_std"] for w in w_str]
    ax1.errorbar(w_num, val_means, yerr=val_stds, fmt="o-", color="#1f77b4", linewidth=2, capsize=5, label="Diffusion Val Loss")
    ax1.set_title("Validation Loss Convergence (Cosine Schedule)", fontweight="bold")
    ax1.set_xlabel("Denoiser Hidden Width $d$")
    ax1.set_ylabel("Validation MSE Loss")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(widths)
    ax1.set_xticklabels(w_str)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # 2. Rollout RMSE: Diffusion vs Deterministic MLP vs Baselines
    ax2 = axes[0, 1]
    diff_means = [metrics["width_scaling_summary"][w]["diff_rollout_rmse_mean"] for w in w_str]
    diff_stds = [metrics["width_scaling_summary"][w]["diff_rollout_rmse_std"] for w in w_str]
    mlp_means = [metrics["width_scaling_summary"][w]["mlp_rollout_rmse_mean"] for w in w_str]
    mlp_stds = [metrics["width_scaling_summary"][w]["mlp_rollout_rmse_std"] for w in w_str]
    id_val = metrics["identity_baseline"]["mean_rmse"]
    kin_val = metrics["kinematic_baseline"]["mean_rmse"]

    ax2.errorbar(w_num, diff_means, yerr=diff_stds, fmt="s-", color="#d62728", linewidth=2, capsize=5, label="Diffusion (5 Seeds)")
    ax2.errorbar(w_num, mlp_means, yerr=mlp_stds, fmt="^-", color="#2ca02c", linewidth=2, capsize=5, label="Deterministic MLP (5 Seeds)")
    ax2.axhline(id_val, color="black", linestyle="--", linewidth=1.8, label=f"Identity Baseline (No Motion: {id_val:.2f})")
    ax2.axhline(kin_val, color="#1f77b4", linestyle=":", linewidth=1.8, label=f"Kinematic Baseline ($v\\Delta t$: {kin_val:.2f})")
    ax2.set_title("30-Step Autoregressive Rollout RMSE", fontweight="bold")
    ax2.set_xlabel("Model Hidden Width $d$")
    ax2.set_ylabel("Mean Trajectory RMSE")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(widths)
    ax2.set_xticklabels(w_str)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    # 3. Invariant Decodability: Trained vs Random Weights (Feature Count Control)
    ax3 = axes[1, 0]
    ang_trained = [metrics["width_scaling_summary"][w]["ang_mom_r2_mean"] for w in w_str]
    ang_trained_std = [metrics["width_scaling_summary"][w]["ang_mom_r2_std"] for w in w_str]
    ang_random = [metrics["width_scaling_summary"][w]["random_ang_mom_r2"] for w in w_str]

    ax3.errorbar(w_num - 2, ang_trained, yerr=ang_trained_std, fmt="o-", color="#ff7f0e", linewidth=2, capsize=5, label="Trained Diffusion ($\\|\\vec{L}\\|$)")
    ax3.plot(w_num + 2, ang_random, "k--^", linewidth=1.8, label="Random Weights Control ($\\|\\vec{L}\\|$)")
    ax3.set_title("Invariant Probing vs Random-Weight Control", fontweight="bold")
    ax3.set_xlabel("Denoiser Hidden Width $d$")
    ax3.set_ylabel("Linear Probe $R^2$ Score")
    ax3.set_xscale("log", base=2)
    ax3.set_xticks(widths)
    ax3.set_xticklabels(w_str)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend()

    # 4. Energy Decodability: Trained vs Random
    ax4 = axes[1, 1]
    e_trained = [metrics["width_scaling_summary"][w]["energy_r2_mean"] for w in w_str]
    e_trained_std = [metrics["width_scaling_summary"][w]["energy_r2_std"] for w in w_str]
    e_random = [metrics["width_scaling_summary"][w]["random_energy_r2"] for w in w_str]

    ax4.errorbar(w_num - 2, e_trained, yerr=e_trained_std, fmt="o-", color="#9467bd", linewidth=2, capsize=5, label="Trained Diffusion ($E_{\\mathrm{total}}$)")
    ax4.plot(w_num + 2, e_random, "k--s", linewidth=1.8, label="Random Weights Control ($E_{\\mathrm{total}}$)")
    ax4.set_title("Total Energy Decodability vs Random Control", fontweight="bold")
    ax4.set_xlabel("Denoiser Hidden Width $d$")
    ax4.set_ylabel("Linear Probe $R^2$ Score")
    ax4.set_xscale("log", base=2)
    ax4.set_xticks(widths)
    ax4.set_xticklabels(w_str)
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend()

    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "model_width_comparison.png")
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"--> Saved Rigorous Fig 1 to {fig1_path}")

    # Figure 2: Error Probing & Controls
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))

    # Bar chart of layer correlations vs baselines
    ax_b = axes2[0]
    layer_names = ["layer_input", "layer_1", "layer_2", "layer_3", "layer_pre_head"]
    p_rs = [metrics["layer_error_metrics"][l]["pearson_r"] for l in layer_names]
    s_rhos = [metrics["layer_error_metrics"][l]["spearman_rho"] for l in layer_names]

    x_pos = np.arange(len(layer_names))
    ax_b.bar(x_pos - 0.18, p_rs, width=0.36, color="#377eb8", label="Pearson $r$")
    ax_b.bar(x_pos + 0.18, s_rhos, width=0.36, color="#ff7f00", label="Spearman $\\rho$")
    ax_b.axhline(metrics["error_baselines"]["physical_features"]["pearson_r"], color="green", linestyle="--", label="Physical Feat. Baseline")
    ax_b.axhline(metrics["error_baselines"]["raw_coordinates_nonlinear_mlp"]["pearson_r"], color="purple", linestyle=":", label="Nonlinear MLP Baseline")
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(layer_names, rotation=20, ha="right")
    ax_b.set_title("Error Predictability Across Layers & Baselines", fontweight="bold")
    ax_b.set_ylabel("Correlation with Transition Error")
    ax_b.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax_b.legend(loc="upper left", fontsize=8)

    # Active intervention demonstration with full controls
    ax_int = axes2[1]
    u_rmse = metrics["intervention_suite"]["unconstrained_rmse"]
    c_rmse = metrics["intervention_suite"]["constant_damp_rmse"]
    r_rmse = metrics["intervention_suite"]["random_damp_rmse"]
    i_rmse = metrics["intervention_suite"]["probe_intervened_rmse"]
    id_rmse = metrics["identity_baseline"]["mean_rmse"]
    kin_rmse = metrics["kinematic_baseline"]["mean_rmse"]

    ctrl_names = ["Identity", "Kinematic", "Unconstrained", "Const Damp", "Rand Damp", "Probe Guided"]
    ctrl_vals = [id_rmse, kin_rmse, u_rmse, c_rmse, r_rmse, i_rmse]
    ctrl_colors = ["#7f7f7f", "#1f77b4", "#d62728", "#9467bd", "#e377c2", "#2ca02c"]

    bars = ax_int.bar(ctrl_names, ctrl_vals, color=ctrl_colors, width=0.55)
    ax_int.set_title("Runtime Intervention & Controls", fontweight="bold")
    ax_int.set_ylabel("30-Step Mean Rollout RMSE")
    ax_int.set_xticklabels(ctrl_names, rotation=25, ha="right")
    for bar in bars:
        h = bar.get_height()
        ax_int.text(bar.get_x() + bar.get_width()/2., h + 0.05, f"{h:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax_int.grid(True, linestyle="--", alpha=0.4, axis="y")

    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "failure_mode_probing.png")
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"--> Saved Rigorous Fig 2 to {fig2_path}")


if __name__ == "__main__":
    run_full_suite()
