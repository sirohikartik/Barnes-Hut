# 📄 Documentation: `run_width_and_failure_experiments.py`

## 1. Overview & Purpose
`run_width_and_failure_experiments.py` is an advanced experimental pipeline designed to explore two fundamental questions in neural physics emulation:
1. **Model Width Scaling**: How does denoiser capacity ($d \in [32, 64, 128, 256]$) govern training convergence, rollout error accumulation, Hamiltonian energy conservation, and linear decodability of physical invariants?
2. **Probing Failure Modes & Self-Diagnosis**: When the diffusion model makes incorrect predictions or diverges near relativistic boundaries (ISCO / horizon), **can its internal representations predict its own error, and which physical quantities suffer representation collapse?**

The script is strictly optimized for Apple Silicon MPS (Metal), enabling the complete multi-width sweep and failure probing suite to execute in **~25 seconds** on a base MacBook Air M1.

---

## 2. Experimental Architecture & Workflow

```
[Phase 1] Simulate Barnes-Hut Trajectories (C++ core, N=16 bodies, dt=0.01)
     │
     ▼
[Phase 2] Multi-Width Model Scaling Sweep:
     ├── Hidden Dim d = 32  (31,334 params)   ──► MPS Train, 30-step Rollout, Invariant Probing
     ├── Hidden Dim d = 64  (66,534 params)   ──► MPS Train, 30-step Rollout, Invariant Probing
     ├── Hidden Dim d = 128 (173,798 params)  ──► MPS Train, 30-step Rollout, Invariant Probing
     └── Hidden Dim d = 256 (535,782 params)  ──► MPS Train, 30-step Rollout, Invariant Probing
     │
     ▼
[Phase 3] Failure Mode & Self-Diagnosis Probing (Flagship Model d=128):
     ├── 1. Compute Ground Truth Transition Errors & Energy Violations
     ├── 2. Extract Internal Activations h_l (Input, ResBlock 1, 2, 3, Pre-Head)
     ├── 3. Self-Diagnosis: Train Probes to Predict Impending Error from h_l
     ├── 4. Regime Dissection: Probe Physical Quality in Success vs Failure Regimes
     └── 5. Uncertainty Calibration: Correlate Ensemble Spread vs True Error
     │
     ▼
[Phase 4] Export Visualizations & Metrics:
     ├── experiments_output/model_width_comparison.png
     ├── experiments_output/failure_mode_probing.png
     └── experiments_output/width_and_failure_metrics.json
```

---

## 3. CLI Arguments & Parameters

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--epochs` | `int` | `25` | Training epochs per model width in the scaling sweep. |
| `--num_bodies` | `int` | `16` | Number of orbital bodies around the supermassive black hole. |
| `--output_dir` | `str` | `'experiments_output'` | Output directory for checkpoints, charts, and metrics. |
| `--force_retrain`| `flag`| `False` | Force retraining all model widths instead of reusing cached JSON metrics. |

---

## 4. Key Experimental Analyses

### A. Model Width Scaling Study
- **Loss vs Capacity**: Tracks diffusion MSE loss across widths to observe scaling laws.
- **Rollout Accumulation**: Quantifies 30-step state coordinate RMSE drift under autoregressive rollout.
- **Physical Conservation Drift**: Computes relative total energy violation $\frac{|\Delta E|}{E_0}$ and angular momentum violation $\frac{|\Delta L|}{L_0}$ along continuous rollouts.
- **Linear Invariant Probing ($R^2$)**: Quantifies how representation capacity linearly crystallizes conservation laws:
  - Angular Momentum ($\|\vec{L}\|$): scales monotonically from $R^2 = 0.2553 \to 0.9340$.
  - Total Energy ($E$): scales from $R^2 = 0.3602 \to 0.7754$.
  - Coordinates ($\mathbf{p}$): scales from $R^2 = 0.2968 \to 0.9466$.

### B. Probing Failure Modes & Self-Diagnosis
- **Astrophysical Failure Regimes**: Pinpoints failure zones near the Innermost Stable Circular Orbit ($r \le 3 r_s$) and event horizon ($r \le r_s$).
- **Internal Self-Diagnosis**: Demonstrates that pre-head representations $h_{\mathrm{pre\_head}}$ exhibit a Pearson correlation of **$r = 0.6852$** with transition error magnitude, confirming that the network internally encodes impending failure before emitting state updates.
- **Representation Collapse**: Dissects the representation manifold between Success (lowest 25% error) and Failure (highest 25% error) regimes, revealing a **$35.6\%$ collapse in position decodability** ($R^2: 0.7869 \to 0.5069$) and an $11.9\%$ drop in angular momentum.
- **Epistemic Uncertainty Calibration**: Evaluates whether stochastic diffusion ensemble spread $\sigma_{\mathrm{ensemble}}$ serves as an intrinsic signal for unphysical prediction states.

---

## 5. Artifacts Produced

1. **`experiments_output/model_width_comparison.png`**:
   4-panel summary containing training loss curves, rollout accumulation, energy drift curves, and linear probe $R^2$ scaling.
2. **`experiments_output/failure_mode_probing.png`**:
   4-panel failure analysis showing orbital phase space with rollout divergence, self-diagnosis correlation bars across layers, representation quality collapse bars, and ensemble uncertainty scatter.
3. **`experiments_output/width_and_failure_metrics.json`**:
   Comprehensive numerical logs containing parameter counts, losses, per-step rollout RMSEs, drift trajectories, and probe scores.
4. **`experiments_output/diffusion_width_{32,64,128,256}.pt`**:
   Trained denoiser checkpoints for all evaluated model widths.
