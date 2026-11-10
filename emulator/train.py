import os
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Dict, Tuple, Optional
from emulator.model import SmallPhysicsDenoiser
from emulator.diffusion import GaussianDiffusion


def get_device() -> torch.device:
    """
    Selects Apple Silicon Metal Performance Shaders (MPS) if available,
    otherwise falls back to CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_diffusion_emulator(
    dataset_dict: Dict,
    hidden_dim: int = 128,
    num_layers: int = 4,
    num_timesteps: int = 25,
    epochs: int = 60,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    target_type: str = "delta",
    save_path: str = "experiments_output/diffusion_model.pt",
) -> Tuple[GaussianDiffusion, Dict]:
    """
    Trains the lightweight physics diffusion model using Apple Silicon MPS.
    """
    device = get_device()
    print(f"--> Using compute device: {device.type.upper()} ({'Apple Silicon Metal' if device.type == 'mps' else 'CPU'})")

    state_dim = dataset_dict["train_x"].shape[1]
    x_mean = dataset_dict["state_mean"]
    x_std = dataset_dict["state_std"]
    delta_mean = dataset_dict["delta_mean"]
    delta_std = dataset_dict["delta_std"]

    # Normalize condition states
    train_x_norm = (dataset_dict["train_x"] - x_mean) / x_std
    test_x_norm = (dataset_dict["test_x"] - x_mean) / x_std

    if target_type == "delta":
        train_tgt_norm = (dataset_dict["train_delta"] - delta_mean) / delta_std
        test_tgt_norm = (dataset_dict["test_delta"] - delta_mean) / delta_std
    else:
        train_tgt_norm = (dataset_dict["train_y"] - x_mean) / x_std
        test_tgt_norm = (dataset_dict["test_y"] - x_mean) / x_std

    train_tensor_x = torch.from_numpy(train_x_norm).float()
    train_tensor_tgt = torch.from_numpy(train_tgt_norm).float()
    test_tensor_x = torch.from_numpy(test_x_norm).float()
    test_tensor_tgt = torch.from_numpy(test_tgt_norm).float()

    train_loader = DataLoader(
        TensorDataset(train_tensor_x, train_tensor_tgt),
        batch_size=batch_size,
        shuffle=True,
    )

    # Initialize small denoiser and diffusion
    denoiser = SmallPhysicsDenoiser(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )
    total_params = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
    print(f"--> Model initialized: {num_layers} ResBlocks, hidden_dim={hidden_dim}, total params={total_params:,}")

    diffusion = GaussianDiffusion(
        model=denoiser,
        num_timesteps=num_timesteps,
        device=device,
    )

    optimizer = torch.optim.AdamW(diffusion.model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    train_losses = []
    test_losses = []

    print(f"--> Starting training for {epochs} epochs on {len(train_tensor_x)} transitions...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        diffusion.model.train()
        epoch_loss = 0.0
        num_batches = 0

        for cond_b, target_b in train_loader:
            cond_b = cond_b.to(device)
            target_b = target_b.to(device)

            optimizer.zero_grad()
            loss = diffusion.compute_loss(target=target_b, cond=cond_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_train_loss = epoch_loss / max(num_batches, 1)
        train_losses.append(avg_train_loss)

        # Validation loss evaluation
        if epoch % 10 == 0 or epoch == epochs:
            diffusion.model.eval()
            with torch.no_grad():
                val_cond = test_tensor_x.to(device)
                val_target = test_tensor_tgt.to(device)
                val_loss = diffusion.compute_loss(target=val_target, cond=val_cond).item()
                test_losses.append(val_loss)

            elapsed = time.time() - start_time
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {avg_train_loss:.5f} | Val Loss: {val_loss:.5f} | Time: {elapsed:.1f}s")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "model_state_dict": diffusion.model.state_dict(),
        "state_mean": x_mean,
        "state_std": x_std,
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "target_type": target_type,
        "state_dim": state_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_timesteps": num_timesteps,
    }, save_path)
    print(f"--> Training completed in {time.time() - start_time:.2f}s! Model saved to {save_path}")

    history = {
        "train_losses": train_losses,
        "test_losses": test_losses,
    }
    return diffusion, history


@torch.no_grad()
def extract_dataset_activations(
    diffusion: GaussianDiffusion,
    states_x: np.ndarray,
    targets: Optional[np.ndarray] = None,
    state_mean: np.ndarray = None,
    state_std: np.ndarray = None,
    target_mean: Optional[np.ndarray] = None,
    target_std: Optional[np.ndarray] = None,
    diffusion_step: int = 0,
    batch_size: int = 64,
    use_inference_sampling: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Passes conditioning states x_t through the denoiser and collects activations h_l.
    If use_inference_sampling=True (default), activations are collected during actual
    reverse ancestral diffusion sampling conditioned ONLY on x_t, with ZERO ground-truth leakage.
    """
    device = diffusion.device
    diffusion.model.eval()

    norm_x = (states_x - state_mean) / state_std
    tensor_x = torch.from_numpy(norm_x).float()

    collected = {}
    n_samples = len(tensor_x)

    for i in range(0, n_samples, batch_size):
        bx = tensor_x[i : i + batch_size].to(device)
        b_size = bx.shape[0]

        if use_inference_sampling:
            # Run genuine ancestral reverse sampling conditioned on bx without target leakage
            _, acts = diffusion.sample_next_state(bx, record_activations_at_step=diffusion_step)
        else:
            norm_tgt = (targets[i : i + batch_size] - target_mean) / target_std
            by = torch.from_numpy(norm_tgt).float().to(device)
            t_tensor = torch.full((b_size,), diffusion_step, device=device, dtype=torch.long)
            noisy_target, _ = diffusion.q_sample(by, t_tensor)
            _, acts = diffusion.model(noisy_target, t_tensor, bx, return_activations=True)

        for layer_name, act_tensor in acts.items():
            act_np = act_tensor.detach().cpu().numpy()
            if layer_name not in collected:
                collected[layer_name] = []
            collected[layer_name].append(act_np)

    return {k: np.concatenate(v, axis=0) for k, v in collected.items()}
