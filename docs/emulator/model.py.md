# 📄 Documentation: `emulator/model.py`

## 1. Overview & Architecture
`emulator/model.py` defines the neural network architecture for the physics diffusion emulator.

To maintain ultra-fast inference and training on **Apple Silicon Mac M1 Air**, the architecture avoids heavy transformers and instead implements an optimized **Residual MLP with Sinusoidal Time Conditioning**.

### Model Parameters & Footprint
- Hidden dimension: $d_{\text{hidden}} = 128$
- ResNet blocks: 3 blocks
- Parameter count: ~173,798 parameters
- Memory footprint: < 700 KB on disk
- Training throughput: > 50 epochs in ~7 seconds on Apple Silicon MPS.

---

## 2. Modules & Classes

### 2.1 `SinusoidalPosEmb`
Computes continuous frequency positional embeddings for diffusion step $k \in [0, K-1]$:
$$\text{emb}_{2i} = \sin\left(k \cdot \omega_i\right), \quad \text{emb}_{2i+1} = \cos\left(k \cdot \omega_i\right)$$
where $\omega_i = \exp\left(-\frac{\ln 10000}{d/2 - 1} \cdot i\right)$.
- `dim`: Embedding dimensionality (default: 64).

### 2.2 `ResBlock`
Lightweight residual block conditioning on time embeddings.

#### Architecture
```
Input x ──► Linear(d, d) ──► LayerNorm ──+──► GELU ──► Linear(d, d) ──► LayerNorm ──+──► GELU ──► Output
                                         ▲                                          ▲
Timestep Emb ──► Linear(time_dim, d) ────┘                                          │
                                                                                    │
Residual Skip Connection ───────────────────────────────────────────────────────────┘
```

#### Member Variables
- `fc1`, `fc2`: Linear transformations with weight matrices $W_1, W_2 \in \mathbb{R}^{d \times d}$.
- `ln1`, `ln2`: LayerNorm regularizing activation scales across layers.
- `time_proj`: Linear layer projecting time embedding $t_{\text{emb}}$ into the hidden dimension.
- `act`: Smooth non-linear GELU activation function.

---

### 2.3 `SmallPhysicsDenoiser`

The complete denoiser network estimating the noise $\epsilon_\theta(y_k, k, c)$ given noisy future state $y_k$, diffusion timestep $k$, and conditioning physical state $c = x_t$.

#### Constructor Parameters
- `state_dim` (`int`): Dimensionality of physical state vector $D$ (e.g. 102 for $N=16$ bodies + black hole).
- `hidden_dim` (`int`): Width of internal representations $d$ (default: 128).
- `num_layers` (`int`): Number of stacked `ResBlock` layers (default: 3).
- `time_dim` (`int`): Dimension of sinusoidal time projection (default: 64).

#### Activation Capture Mechanism (`return_activations=True`)
The forward pass records internal representations $h_l$ at every layer for probing experiments:
1. `layer_input`: Initial projected representation after input layer norm:
   $$h_{\text{input}} = \text{GELU}(\text{LayerNorm}(W_{\text{in}} [y_k; c])) \in \mathbb{R}^{B \times d}$$
2. `layer_1`, `layer_2`, `layer_3`: Activations exiting each respective `ResBlock`:
   $$h_l = \text{ResBlock}_l(h_{l-1}, t_{\text{emb}}) \in \mathbb{R}^{B \times d}$$
3. `layer_pre_head`: Pre-projection representation exiting the final LayerNorm:
   $$h_{\mathrm{out}} = \text{LayerNorm}(h_{\mathrm{last}}) \in \mathbb{R}^{B \times d}$$
4. Final noise prediction:
   $$\hat{\epsilon} = W_{\mathrm{out}} h_{\mathrm{out}} \in \mathbb{R}^{B \times D}$$
