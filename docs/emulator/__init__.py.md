# 📄 Documentation: `emulator/__init__.py`

## 1. Overview & Architecture
`emulator/__init__.py` is the package initialization file for the `emulator` module. It exposes the public API for the physics dataset generator, small neural denoiser model, Gaussian diffusion process, probing suite, and Apple Silicon training utilities.

---

## 2. Exported Symbols & API

| Symbol | Module Source | Type | Description |
| :--- | :--- | :--- | :--- |
| `create_physics_dataset` | `emulator.dataset` | Function | Runs Barnes-Hut simulator and packages trajectories and physical invariants. |
| `compute_physical_quantities` | `emulator.dataset` | Function | Evaluates energy, angular momentum, linear momentum, and center of mass. |
| `SmallPhysicsDenoiser` | `emulator.model` | `nn.Module` | Lightweight conditional denoiser with activation extraction hooks. |
| `GaussianDiffusion` | `emulator.diffusion` | `nn.Module` | Gaussian diffusion engine with ancestral sampling and ensemble generation. |
| `PhysicalProbeSuite` | `emulator.prober` | Class | Ridge/MLP linear probing framework for measuring internal layer $R^2$ scores. |
| `train_diffusion_emulator` | `emulator.train` | Function | Trains diffusion model using Apple Silicon Metal Performance Shaders (MPS). |
| `extract_dataset_activations` | `emulator.train` | Function | Passes dataset through model and extracts internal activations $h_l$. |
| `get_device` | `emulator.train` | Function | Auto-selects `torch.device("mps")` or falls back to CPU. |
