# 📄 Documentation: `emulator/train.py`

## 1. Overview & Architecture
`emulator/train.py` manages the PyTorch training loop and hardware acceleration for the physics diffusion emulator.

It is specifically tuned for **Apple Silicon (M-series)** hardware, leveraging Metal Performance Shaders (`torch.device("mps")`) for GPU-accelerated tensor math and low power consumption.

---

## 2. Core Functions & Variables

### Function: `get_device()`
Auto-detects available compute backend:
- Returns `torch.device("mps")` if `torch.backends.mps.is_available()` is `True`.
- Falls back to `torch.device("cpu")` on non-Mac environments.

---

### Function: `train_diffusion_emulator(...)`
Executes mini-batch training with cosine learning rate scheduling and gradient clipping.

#### Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dataset_dict` | `dict` | Required | Output dictionary from `create_physics_dataset`. |
| `hidden_dim` | `int` | `128` | Model hidden layer width. |
| `num_layers` | `int` | `4` | Number of stacked ResNet blocks. |
| `num_timesteps`| `int` | `25` | Discrete diffusion schedule steps. |
| `epochs` | `int` | `60` | Number of complete passes over the training dataset. |
| `batch_size` | `int` | `64` | Mini-batch size. |
| `learning_rate`| `float` | `1e-3` | Initial AdamW learning rate. |
| `target_type` | `str` | `'delta'` | Target formulation: `'delta'` ($\Delta x_t$) or `'state'` ($x_{t+1}$). |
| `save_path` | `str` | `'experiments_output/diffusion_model.pt'` | Checkpoint save path. |

#### Internal Training Components
- **Optimizer**: `torch.optim.AdamW(..., lr=learning_rate, weight_decay=1e-4)` providing adaptive moment estimation with weight decay regularization.
- **LR Scheduler**: `CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)` smoothly decaying learning rate to prevent late-epoch instability.
- **Gradient Clipping**: `clip_grad_norm_(model.parameters(), 1.0)` preventing exploding gradients during close orbital encounters.

---

### Function: `extract_dataset_activations(...)`
Iterates across the entire training or test dataset in batches, passes states through the denoiser, and aggregates intermediate layer representations:
$$h_l \in \left\{ h_{\text{input}}, h_1, h_2, \ldots, h_{\text{pre\_head}} \right\}$$
into NumPy arrays for probing experiments.

#### Parameters
- `diffusion`: Trained `GaussianDiffusion` model instance.
- `states_x`: Raw condition states $x_t$.
- `targets`: Raw targets ($\Delta x_t$ or $x_{t+1}$).
- `state_mean`, `state_std`: Condition normalization statistics.
- `target_mean`, `target_std`: Target normalization statistics.
- `diffusion_step`: Diffusion timestep $k$ at which to inspect activations (e.g. $k=0$ for clean state, $k=K-1$ for noisy prior).
- `batch_size`: Batch size for feature extraction (default: 128).

#### Returns
- Dictionary mapping layer names (`"layer_input"`, `"layer_1"`, `"layer_2"`, `"layer_3"`, `"layer_pre_head"`) to feature matrices of shape $(N_{\text{samples}}, d_{\text{hidden}})$.
