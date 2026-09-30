# 📄 Documentation: `emulator/diffusion.py`

## 1. Overview & Architecture
`emulator/diffusion.py` implements the discrete-time **Gaussian Diffusion Framework** for physical trajectory generation.

Given the current physical state $c = x_t$, the diffusion emulator models the transition distribution $p(x_{t+1} \mid x_t)$ (or displacement $p(\Delta x_t \mid x_t)$). By repeatedly sampling the stochastic reverse process:
$$\hat{x}_{t+1}^{(1)}, \hat{x}_{t+1}^{(2)}, \ldots, \hat{x}_{t+1}^{(K)}$$
the emulator constructs an ensemble quantifying epistemic uncertainty in the simulated dynamics.

---

## 2. Class: `GaussianDiffusion`

### Mathematical Variables & Buffers
Precomputes diffusion schedule coefficients on the target device:

| Buffer Name | Math Symbol | Formula | Purpose |
| :--- | :--- | :--- | :--- |
| `betas` | $\beta_k$ | $\text{linspace}(\beta_{\text{start}}, \beta_{\text{end}}, K)$ | Variance schedule for noise addition. |
| `alphas` | $\alpha_k$ | $1 - \beta_k$ | Single-step variance retention. |
| `alphas_cumprod` | $\bar{\alpha}_k$ | $\prod_{s=0}^k \alpha_s$ | Cumulative variance retention from step 0 to $k$. |
| `sqrt_alphas_cumprod` | $\sqrt{\bar{\alpha}_k}$ | $\sqrt{\bar{\alpha}_k}$ | Signal scaling coefficient in $q(y_k \mid y_0)$. |
| `sqrt_one_minus_alphas_cumprod` | $\sqrt{1 - \bar{\alpha}_k}$ | $\sqrt{1 - \bar{\alpha}_k}$ | Noise scaling coefficient in $q(y_k \mid y_0)$. |
| `posterior_variance` | $\tilde{\beta}_k$ | $\frac{1 - \bar{\alpha}_{k-1}}{1 - \bar{\alpha}_k} \beta_k$ | Variance of reverse transition $q(y_{k-1} \mid y_k, y_0)$. |
| `posterior_mean_coef1` | $c_1$ | $\frac{\sqrt{\bar{\alpha}_{k-1}}\beta_k}{1 - \bar{\alpha}_k}$ | Weight for predicted clean target $\hat{y}_0$. |
| `posterior_mean_coef2` | $c_2$ | $\frac{\sqrt{\alpha_k}(1 - \bar{\alpha}_{k-1})}{1 - \bar{\alpha}_k}$ | Weight for noisy state $y_k$. |

---

## 3. Core Methods

### 3.1 `q_sample(x_0, t, noise)`
Direct closed-form forward diffusion process:
$$q(y_k \mid y_0) = \sqrt{\bar{\alpha}_k} y_0 + \sqrt{1 - \bar{\alpha}_k} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

### 3.2 `compute_loss(target, cond)`
Evaluates the Mean Squared Error (MSE) between true noise $\epsilon$ and predicted noise $\hat{\epsilon} = \epsilon_\theta(y_k, k, c)$:
$$\mathcal{L}(\theta) = \mathbb{E}_{k, y_0, \epsilon} \left[ \|\epsilon - \epsilon_\theta(y_k, k, c)\|^2 \right]$$

### 3.3 `p_sample(x_k, k, cond, return_activations)`
Computes a single reverse step from $y_k \to y_{k-1}$:
1. Predicts clean state estimate:
   $$\hat{y}_0 = \frac{1}{\sqrt{\bar{\alpha}_k}} \left( y_k - \frac{\sqrt{1 - \bar{\alpha}_k}}{\sqrt{\bar{\alpha}_k}} \epsilon_\theta(y_k, k, c) \right)$$
2. Computes posterior mean:
   $$\mu_k = c_1 \hat{y}_0 + c_2 y_k$$
3. Samples next reverse state:
   $$y_{k-1} = \mu_k + \sqrt{\tilde{\beta}_k} z, \quad z \sim \mathcal{N}(0, \mathbf{I})$$

### 3.4 `sample_next_state(cond, record_activations_at_step)`
Executes full reverse chain from $k = K-1 \to 0$ starting from pure Gaussian noise $y_{K-1} \sim \mathcal{N}(0, \mathbf{I})$.

### 3.5 `predict_future_state(x_raw, state_mean, state_std, delta_mean, delta_std, target_type)`
High-level physical transition method:
- Normalizes raw input physical state $x_t$.
- Executes reverse diffusion.
- Denormalizes and returns predicted future physical state $\hat{x}_{t+1} = x_t + \widehat{\Delta x}_t$.

### 3.6 `sample_physical_ensemble(x_raw, ..., ensemble_size=10)`
Samples $K$ independent stochastic reverse trajectories to construct an ensemble:
$$\left[ \hat{x}_{t+1}^{(1)}, \hat{x}_{t+1}^{(2)}, \ldots, \hat{x}_{t+1}^{(K)} \right]$$
Returns array of shape $(K, B, D)$ where $B$ is batch size and $D$ is state dimension.
