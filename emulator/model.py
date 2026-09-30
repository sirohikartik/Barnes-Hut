import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResBlock(nn.Module):
    """
    Lightweight residual block with time conditioning and LayerNorm.
    Suitable for fast training on Apple Silicon MPS.
    """
    def __init__(self, hidden_dim: int, time_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.time_proj = nn.Linear(time_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.fc1(x)
        h = self.ln1(h)
        h = h + self.time_proj(t_emb)
        h = self.act(h)
        h = self.fc2(h)
        h = self.ln2(h)
        return self.act(residual + h)


class SmallPhysicsDenoiser(nn.Module):
    """
    Compact conditional diffusion denoiser designed for physics emulation.
    Conditioned on current state x_t to predict noise for future state x_{t+1}.
    Exposes intermediate layer activations h_l for probing experiments.
    """
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        time_dim: int = 64,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Sinusoidal time embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Input projection: takes [noisy_state_k, condition_state_t]
        self.input_proj = nn.Linear(state_dim * 2, hidden_dim)
        self.input_ln = nn.LayerNorm(hidden_dim)
        self.input_act = nn.GELU()

        # Stack of ResBlocks
        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim, time_dim) for _ in range(num_layers)
        ])

        # Output projection back to state dimension
        self.out_ln = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, state_dim)

    def forward(
        self,
        noisy_target: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        return_activations: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass.
        Args:
            noisy_target: (B, state_dim) noisy future state at diffusion step t
            t: (B,) diffusion timestep integers
            cond: (B, state_dim) conditioning state x_t
            return_activations: whether to return internal layer activations h_l
        Returns:
            eps_pred: (B, state_dim) predicted noise
            activations: dict of layer names to activations (if return_activations=True)
        """
        t_emb = self.time_emb(t)
        x_in = torch.cat([noisy_target, cond], dim=-1)
        h = self.input_act(self.input_ln(self.input_proj(x_in)))

        activations = {}
        if return_activations:
            activations["layer_input"] = h.clone()

        for idx, block in enumerate(self.blocks):
            h = block(h, t_emb)
            if return_activations:
                activations[f"layer_{idx + 1}"] = h.clone()

        out_features = self.out_ln(h)
        if return_activations:
            activations["layer_pre_head"] = out_features.clone()

        eps_pred = self.out_proj(out_features)

        if return_activations:
            return eps_pred, activations
        return eps_pred, None
