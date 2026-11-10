import json
import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
import os

from emulator.dataset import create_physics_dataset
from emulator.model import SmallPhysicsDenoiser
from emulator.diffusion import GaussianDiffusion

def get_cv_ridge(X_tr, y_tr, X_te, y_te, alphas=None):
    if alphas is None:
        alphas = np.logspace(-4, 4, 9)
    model = RidgeCV(alphas=alphas)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    r2 = float(r2_score(y_te, y_pred))
    best_alpha = float(model.alpha_)
    return r2, best_alpha, y_pred

def extract_features(x_array):
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

print("Generating 1x baseline dataset...")
data_1x = create_physics_dataset(num_trajectories=35, num_steps=60, num_bodies=16, dt=0.01, base_seed=100)

train_phys = data_1x["train_phys"]
test_phys = data_1x["test_phys"]

tr_com = np.array([p["com_pos"] for p in train_phys])
te_com = np.array([p["com_pos"] for p in test_phys])

tr_e = np.array([p["total_energy"][0] for p in train_phys])
te_e = np.array([p["total_energy"][0] for p in test_phys])

tr_l = np.array([p["angular_momentum_norm"][0] for p in train_phys])
te_l = np.array([p["angular_momentum_norm"][0] for p in test_phys])

raw_tx = data_1x["train_x"]
raw_te = data_1x["test_x"]

sm, ss = data_1x["state_mean"], data_1x["state_std"]
dm, ds = data_1x["delta_mean"], data_1x["delta_std"]
ss_safe = np.where(ss < 0.05, 1.0, ss)
ds_safe = np.where(ds < 0.05, 1.0, ds)

# 1. Raw State Controls
r2_raw_com, a_com, _ = get_cv_ridge(raw_tx, tr_com, raw_te, te_com)
r2_raw_e, a_e, _ = get_cv_ridge(raw_tx, tr_e, raw_te, te_e)
r2_raw_l, a_l, _ = get_cv_ridge(raw_tx, tr_l, raw_te, te_l)

print("\n--- Raw Input State x_t Control Ceiling (RidgeCV) ---")
print(f"Positive Control (Center of Mass):  R^2 = {r2_raw_com:.4f} (alpha={a_com})")
print(f"Total Energy E (Nonlinear target): R^2 = {r2_raw_e:.4f} (alpha={a_e})")
print(f"Angular Momentum |L| (Nonlinear):  R^2 = {r2_raw_l:.4f} (alpha={a_l})")

probing_results = {
    "raw_ceiling": {"com_r2": r2_raw_com, "energy_r2": r2_raw_e, "ang_mom_r2": r2_raw_l}
}

for width in [32, 64, 128, 256]:
    net_tr = SmallPhysicsDenoiser(102, width)
    diff_tr = GaussianDiffusion(net_tr, num_timesteps=20, device=torch.device("cpu"))
    pt_path = f"experiments_output/diffusion_width_{width}.pt"
    if os.path.exists(pt_path):
        net_tr.load_state_dict(torch.load(pt_path, map_location="cpu"))
    diff_tr.eval()

    net_rand = SmallPhysicsDenoiser(102, width)
    diff_rand = GaussianDiffusion(net_rand, num_timesteps=20, device=torch.device("cpu"))
    diff_rand.eval()

    def extract_layer2(d_model, x_arr):
        acts_list = []
        with torch.no_grad():
            for i in range(len(x_arr)):
                bx = torch.from_numpy((x_arr[i:i+1] - sm) / ss_safe).float()
                _, acts = d_model.sample_next_state(bx, record_activations_at_step=0)
                acts_list.append(acts["layer_2"].numpy()[0])
        return np.array(acts_list)

    tr_acts_tr = extract_layer2(diff_tr, raw_tx[:500])
    te_acts_tr = extract_layer2(diff_tr, raw_te)

    tr_acts_rd = extract_layer2(diff_rand, raw_tx[:500])
    te_acts_rd = extract_layer2(diff_rand, raw_te)

    r2_tr_com, _, _ = get_cv_ridge(tr_acts_tr, tr_com[:500], te_acts_tr, te_com)
    r2_tr_e, _, _ = get_cv_ridge(tr_acts_tr, tr_e[:500], te_acts_tr, te_e)
    r2_tr_l, _, _ = get_cv_ridge(tr_acts_tr, tr_l[:500], te_acts_tr, te_l)

    r2_rd_com, _, _ = get_cv_ridge(tr_acts_rd, tr_com[:500], te_acts_rd, te_com)
    r2_rd_e, _, _ = get_cv_ridge(tr_acts_rd, tr_e[:500], te_acts_rd, te_e)
    r2_rd_l, _, _ = get_cv_ridge(tr_acts_rd, tr_l[:500], te_acts_rd, te_l)

    print(f"\nWidth d={width} Invariant Decodability (RidgeCV):")
    print(f"  Trained: COM R^2={r2_tr_com:.3f} | Energy R^2={r2_tr_e:.3f} | AngMom R^2={r2_tr_l:.3f}")
    print(f"  Random:  COM R^2={r2_rd_com:.3f} | Energy R^2={r2_rd_e:.3f} | AngMom R^2={r2_rd_l:.3f}")

    probing_results[f"d_{width}"] = {
        "trained": {"com_r2": r2_tr_com, "energy_r2": r2_tr_e, "ang_mom_r2": r2_tr_l},
        "random": {"com_r2": r2_rd_com, "energy_r2": r2_rd_e, "ang_mom_r2": r2_rd_l},
    }

print("\n--- Incremental Error Predictability (Physical vs Activations vs Combined) ---")
net128 = SmallPhysicsDenoiser(102, 128)
diff128 = GaussianDiffusion(net128, num_timesteps=20, device=torch.device("cpu"))
if os.path.exists("experiments_output/diffusion_width_128.pt"):
    net128.load_state_dict(torch.load("experiments_output/diffusion_width_128.pt", map_location="cpu"))
diff128.eval()

def get_step_errs(x_arr, d_arr):
    acts_pre = []
    actual_errs = []
    with torch.no_grad():
        for i in range(len(x_arr)):
            bx = torch.from_numpy((x_arr[i:i+1] - sm) / ss_safe).float()
            sample_norm, acts = diff128.sample_next_state(bx, record_activations_at_step=0)
            pred_d = sample_norm.numpy()[0] * ds_safe + dm
            err = float(np.linalg.norm(pred_d - d_arr[i]))
            acts_pre.append(acts["layer_pre_head"].numpy()[0])
            actual_errs.append(err)
    return np.array(acts_pre), np.array(actual_errs)

tr_pre, tr_err = get_step_errs(raw_tx[:600], data_1x["train_delta"][:600])
te_pre, te_err = get_step_errs(raw_te, data_1x["test_delta"])

tr_phys_feats = extract_features(raw_tx[:600])
te_phys_feats = extract_features(raw_te)

tr_comb = np.hstack([tr_phys_feats, tr_pre])
te_comb = np.hstack([te_phys_feats, te_pre])

# Predict Raw Error
r2_phys, _, y_pred_phys = get_cv_ridge(tr_phys_feats, tr_err, te_phys_feats, te_err)
r2_act, _, y_pred_act = get_cv_ridge(tr_pre, tr_err, te_pre, te_err)
r2_comb, _, y_pred_comb = get_cv_ridge(tr_comb, tr_err, te_comb, te_err)

r_phys = float(pearsonr(te_err, y_pred_phys)[0])
rho_phys = float(spearmanr(te_err, y_pred_phys)[0])
r_act = float(pearsonr(te_err, y_pred_act)[0])
rho_act = float(spearmanr(te_err, y_pred_act)[0])
r_comb = float(pearsonr(te_err, y_pred_comb)[0])
rho_comb = float(spearmanr(te_err, y_pred_comb)[0])

# Predict Log Error
tr_log_err = np.log(tr_err + 1e-6)
te_log_err = np.log(te_err + 1e-6)

r2_log_phys, _, y_pred_log_phys = get_cv_ridge(tr_phys_feats, tr_log_err, te_phys_feats, te_log_err)
r2_log_act, _, y_pred_log_act = get_cv_ridge(tr_pre, tr_log_err, te_pre, te_log_err)
r2_log_comb, _, y_pred_log_comb = get_cv_ridge(tr_comb, tr_log_err, te_comb, te_log_err)

r_log_phys = float(pearsonr(te_log_err, y_pred_log_phys)[0])
r_log_act = float(pearsonr(te_log_err, y_pred_log_act)[0])
r_log_comb = float(pearsonr(te_log_err, y_pred_log_comb)[0])

print(f"Physical Features Alone:  R^2={r2_phys:.4f}, r={r_phys:.3f}, rho={rho_phys:.3f} | Log R^2={r2_log_phys:.4f}, Log r={r_log_phys:.3f}")
print(f"Activations Alone:        R^2={r2_act:.4f}, r={r_act:.3f}, rho={rho_act:.3f} | Log R^2={r2_log_act:.4f}, Log r={r_log_act:.3f}")
print(f"Combined (Phys + Acts):   R^2={r2_comb:.4f}, r={r_comb:.3f}, rho={rho_comb:.3f} | Log R^2={r2_log_comb:.4f}, Log r={r_log_comb:.3f}")

# Save results
out_data = {
    "probing_results": probing_results,
    "incremental_error": {
        "physical_features": {"r2": r2_phys, "r": r_phys, "rho": rho_phys, "log_r2": r2_log_phys, "log_r": r_log_phys},
        "activations_alone": {"r2": r2_act, "r": r_act, "rho": rho_act, "log_r2": r2_log_act, "log_r": r_log_act},
        "combined": {"r2": r2_comb, "r": r_comb, "rho": rho_comb, "log_r2": r2_log_comb, "log_r": r_log_comb},
    }
}
with open("experiments_output/probing_and_incremental_metrics.json", "w") as f:
    json.dump(out_data, f, indent=2)
print("Saved to experiments_output/probing_and_incremental_metrics.json")
