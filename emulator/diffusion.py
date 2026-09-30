import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from emulator.model import SmallPhysicsDenoiser


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion framework for physics trajectory emulation.
    Predicts next physical state conditioned on current state x_t.
    """
    def __init__(
        self,
        model: SmallPhysicsDenoiser,
        num_timesteps: int = 25,
        beta_start: float = 1e-4,
        beta_end: float = 0.028,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.model = model.to(device)
        self.num_timesteps = num_timesteps
        self.device = device

        # Linear beta schedule
        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Calculations for diffusion q(x_t | x_0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance", torch.log(torch.clamp(posterior_variance, min=1e-20)))

        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples noisy target at timestep t:
        q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return sqrt_alpha * x_0 + sqrt_one_minus * noise, noise

    def compute_loss(self, target: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Computes the standard DDPM MSE loss on predicted noise.
        """
        batch_size = target.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(target)

        noisy_target, _ = self.q_sample(target, t, noise)
        pred_noise, _ = self.model(noisy_target, t, cond, return_activations=False)

        return nn.functional.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def p_sample(
        self,
        x_k: torch.Tensor,
        k: int,
        cond: torch.Tensor,
        return_activations: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Single reverse diffusion step k -> k-1.
        """
        batch_size = x_k.shape[0]
        k_tensor = torch.full((batch_size,), k, device=self.device, dtype=torch.long)

        pred_noise, activations = self.model(x_k, k_tensor, cond, return_activations=return_activations)

        # Estimate x_0 from pred_noise
        sqrt_recip_alpha_bar = 1.0 / self.sqrt_alphas_cumprod[k]
        sqrt_recip_m1_alpha_bar = self.sqrt_one_minus_alphas_cumprod[k] / self.sqrt_alphas_cumprod[k]
        x_0_pred = sqrt_recip_alpha_bar * x_k - sqrt_recip_m1_alpha_bar * pred_noise

        # Posterior mean
        mean = self.posterior_mean_coef1[k] * x_0_pred + self.posterior_mean_coef2[k] * x_k

        if k == 0:
            return mean, activations
        else:
            noise = torch.randn_like(x_k)
            variance = torch.sqrt(self.posterior_variance[k])
            return mean + variance * noise, activations

    @torch.no_grad()
    def sample_next_state(
        self,
        cond: torch.Tensor,
        record_activations_at_step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Runs the full reverse diffusion process to sample x_{t+1} conditioned on cond = x_t.
        Optionally records layer activations at a specific diffusion step k.
        """
        batch_size = cond.shape[0]
        state_dim = cond.shape[1]
        x_k = torch.randn((batch_size, state_dim), device=self.device)

        recorded_activations = None

        for k in reversed(range(self.num_timesteps)):
            need_act = (record_activations_at_step == k)
            x_k, acts = self.p_sample(x_k, k, cond, return_activations=need_act)
            if need_act and acts is not None:
                recorded_activations = acts

        return x_k, recorded_activations

    @torch.no_grad()
    def sample_ensemble(
        self,
        cond: torch.Tensor,
        ensemble_size: int = 10,
    ) -> torch.Tensor:
        """
        Generates K ensemble samples for future state/delta:
        [y^{(1)}, ..., y^{(K)}]
        Returns:
            ensemble: (K, batch_size, state_dim) tensor
        """
        ensemble_list = []
        for _ in range(ensemble_size):
            sample, _ = self.sample_next_state(cond)
            ensemble_list.append(sample.unsqueeze(0))
        return torch.cat(ensemble_list, dim=0)

    @torch.no_grad()
    def predict_future_state(
        self,
        x_raw: np.ndarray,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        delta_mean: np.ndarray,
        delta_std: np.ndarray,
        target_type: str = "delta",
        record_activations_at_step: Optional[int] = None,
    ) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """
        Takes raw physical state x_t, runs reverse diffusion emulator, and returns predicted x_{t+1}.
        """
        cond_norm = (x_raw - state_mean) / state_std
        cond_tensor = torch.from_numpy(cond_norm).float().to(self.device)

        sample_norm, acts = self.sample_next_state(
            cond_tensor,
            record_activations_at_step=record_activations_at_step,
        )
        sample_np = sample_norm.detach().cpu().numpy()

        if target_type == "delta":
            delta_raw = sample_np * delta_std + delta_mean
            next_state = x_raw + delta_raw
        else:
            next_state = sample_np * state_std + state_mean

        acts_np = None
        if acts is not None:
            acts_np = {k: v.detach().cpu().numpy() for k, v in acts.items()}

        return next_state, acts_np

    @torch.no_grad()
    def sample_physical_ensemble(
        self,
        x_raw: np.ndarray,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        delta_mean: np.ndarray,
        delta_std: np.ndarray,
        ensemble_size: int = 10,
        target_type: str = "delta",
    ) -> np.ndarray:
        """
        Samples K physical future states: [x_{t+1}^{(1)}, ..., x_{t+1}^{(K)}]
        Returns: (K, batch_size, state_dim) array
        """
        samples = []
        for _ in range(ensemble_size):
            next_s, _ = self.predict_future_state(
                x_raw, state_mean, state_std, delta_mean, delta_std, target_type=target_type
            )
            samples.append(np.expand_dims(next_s, 0))
        return np.concatenate(samples, axis=0)
