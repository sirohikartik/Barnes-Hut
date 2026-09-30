# 📄 Documentation: `run_emulator_experiment.py`

## 1. Overview & Architecture
`run_emulator_experiment.py` is the top-level orchestration script that executes the complete 4-step physics emulation and probing workflow:
1. **Simulation**: Generates Barnes-Hut ground truth orbital trajectories.
2. **Diffusion Training**: Trains the small diffusion emulator on Apple Silicon Metal (MPS).
3. **Ensemble & Rollout**: Generates $K$-sample stochastic ensembles and multi-step autoregressive rollouts.
4. **Internal Representation Probing**: Decodes physical quantities from internal layers $h_l$ and outputs tables and comparative diagnostic charts.

---

## 2. Execution Pipeline & Internal Steps

```
[Step 1/4] create_physics_dataset()
     │
     ▼
[Step 2/4] train_diffusion_emulator() (MPS device)
     │
     ▼
[Step 3/4] sample_physical_ensemble() & Autoregressive Rollout
     │
     ▼
[Step 4/4] extract_dataset_activations() ──► PhysicalProbeSuite.evaluate_representation()
     │
     ▼
Generate Artifacts:
     ├── experiments_output/ensemble_rollout_comparison.png
     ├── experiments_output/probing_results_layers.png
     └── experiments_output/diffusion_emulator.pt
```

---

## 3. CLI Arguments & Parameters

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--num_bodies` | `int` | `16` | Number of orbiting bodies in the simulation system. |
| `--num_trajectories` | `int` | `30` | Number of independent simulation trajectories generated. |
| `--num_steps` | `int` | `60` | Number of timesteps per trajectory. |
| `--epochs` | `int` | `50` | Training epochs for the diffusion model. |
| `--hidden_dim` | `int` | `128` | Width of denoiser hidden representations (tuned for M1 Air). |
| `--num_layers` | `int` | `3` | Number of ResNet blocks in the denoiser. |
| `--num_timesteps` | `int` | `20` | Number of discrete diffusion schedule timesteps. |
| `--ensemble_size` | `int` | `12` | Number of stochastic ensemble samples $K$. |
| `--target_type` | `str` | `'delta'` | Transition target: `'delta'` ($\Delta x_t$) or `'state'` ($x_{t+1}$). |
| `--output_dir` | `str` | `'experiments_output'`| Output directory for charts and checkpoints. |

---

## 4. Output Visualizations Generated

### 1. `experiments_output/probing_results_layers.png`
- **Subplot 1 (Left)**: Layer-wise $R^2$ scores across $h_{\mathrm{input}} \to h_1 \to h_2 \to h_3 \to h_{\mathrm{out}}$ for:
  - Total Energy $E$
  - Angular Momentum $\|\vec{L}\|$
  - Linear Momentum $\|\vec{P}\|$
  - Positions $\mathbf{p}_1 \ldots \mathbf{p}_N$
  - Velocities $\mathbf{v}_1 \ldots \mathbf{v}_N$
- **Subplot 2 (Right)**: Physical representation decodability across reverse diffusion timesteps ($k = 19 \to 10 \to 0$), demonstrating information crystallization as noise is removed.

### 2. `experiments_output/ensemble_rollout_comparison.png`
- **Subplot 1 (Left)**: 2D orbital trajectory plot comparing the Barnes-Hut ground truth orbit against the diffusion emulator rollout, $K$-sample ensemble points, and central black hole singularity.
- **Subplot 2 (Right)**: Diffusion training loss curve demonstrating stable convergence on Apple Silicon MPS.

### 3. `experiments_output/diffusion_emulator.pt`
PyTorch checkpoint (< 700 KB) containing:
- Denoiser model weights
- Normalization statistics (`state_mean`, `state_std`, `delta_mean`, `delta_std`)
- Hyperparameters and schedule configurations.
