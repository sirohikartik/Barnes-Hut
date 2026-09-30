# 📚 Comprehensive Codebase Documentation

This directory mirrors the structure of the Barnes-Hut Black Hole N-body Simulation & Diffusion Physics Emulator repository. Every code file has a dedicated documentation markdown file explaining its architecture, classes, functions, and internal variables in detail.

---

## 🗂️ Mirrored Documentation Index

### 1. Root Level Scripts & Build Configurations
- [`setup.py.md`](file:///Users/kartiksirohi/black_hole/docs/setup.py.md): Setuptools extension compilation script for building the C++ pybind11 module with hardware optimization flags (`-O3`, `-march=native`, `-ffast-math`).
- [`CMakeLists.txt.md`](file:///Users/kartiksirohi/black_hole/docs/CMakeLists.txt.md): CMake configuration for compiling `black_hole_core`.
- [`benchmark.py.md`](file:///Users/kartiksirohi/black_hole/docs/benchmark.py.md): Performance benchmarking script measuring $O(N \log N)$ Barnes-Hut Octree throughput up to 8,000+ bodies.
- [`test_sim.py.md`](file:///Users/kartiksirohi/black_hole/docs/test_sim.py.md): Verification test suite validating Keplerian stability, Paczyński–Wiita potential, and event horizon accretions.
- [`animate.py.md`](file:///Users/kartiksirohi/black_hole/docs/animate.py.md): High-performance visualizer rendering 3D orbits, color-coded velocity vectors, and dynamic octree wireframe cubes via Matplotlib/FFmpeg.
- [`run_emulator_experiment.py.md`](file:///Users/kartiksirohi/black_hole/docs/run_emulator_experiment.py.md): Master CLI runner executing data generation, MPS-accelerated diffusion training, ensemble sampling, and internal layer representation probing.
- [`run_width_and_failure_experiments.py.md`](file:///Users/kartiksirohi/black_hole/docs/run_width_and_failure_experiments.py.md): Master experiment runner executing the multi-width scaling study ($d=32, 64, 128, 256$), failure mode analysis, self-diagnosis error probing, and epistemic uncertainty calibration.

### 2. C++ Engine Core (`include/` and `src/`)
- [`include/vec3.hpp.md`](file:///Users/kartiksirohi/black_hole/docs/include/vec3.hpp.md): High-performance 3D vector arithmetic struct, vector products, and Euclidean norms.
- [`include/body.hpp.md`](file:///Users/kartiksirohi/black_hole/docs/include/body.hpp.md): Particle data structure, active flags, state vectors, and event horizon flags.
- [`include/octree.hpp.md`](file:///Users/kartiksirohi/black_hole/docs/include/octree.hpp.md): Memory-pooled 3D Barnes-Hut Octree with multipole center-of-mass aggregation and wireframe extraction.
- [`include/simulation.hpp.md`](file:///Users/kartiksirohi/black_hole/docs/include/simulation.hpp.md): Physics engine orchestrator, Paczyński–Wiita pseudo-Newtonian potential, event horizon capture, and symplectic Velocity-Verlet integrator.
- [`src/bindings.cpp.md`](file:///Users/kartiksirohi/black_hole/docs/src/bindings.cpp.md): Pybind11 module bindings exposing zero-copy NumPy array views of positions, velocities, masses, and octree bounding boxes.

### 3. Diffusion Physics Emulator & Probing Suite (`emulator/`)
- [`emulator/__init__.py.md`](file:///Users/kartiksirohi/black_hole/docs/emulator/__init__.py.md): Package initialization and public API exports.
- [`emulator/dataset.py.md`](file:///Users/kartiksirohi/black_hole/docs/emulator/dataset.py.md): Trajectory generation engine, state vector packaging ($x_t = [\mathbf{p}_1, \mathbf{v}_1, \dots, \mathbf{p}_N, \mathbf{v}_N]$), and exact calculation of physical invariants ($E, \vec{L}, \vec{P}, \mathbf{p}, \mathbf{v}, K, U$).
- [`emulator/model.py.md`](file:///Users/kartiksirohi/black_hole/docs/emulator/model.py.md): Lightweight conditional diffusion denoiser network (~173k params) with Sinusoidal Positional Embeddings and intermediate activation hooks.
- [`emulator/diffusion.py.md`](file:///Users/kartiksirohi/black_hole/docs/emulator/diffusion.py.md): Gaussian diffusion framework (forward noising schedule, DDPM ancestral sampling, physical displacement inversion, and multi-path ensemble generation).
- [`emulator/train.py.md`](file:///Users/kartiksirohi/black_hole/docs/emulator/train.py.md): PyTorch training loop on Apple Silicon Metal Performance Shaders (MPS) and dataset activation extraction.
- [`emulator/prober.py.md`](file:///Users/kartiksirohi/black_hole/docs/emulator/prober.py.md): Physical probing suite testing layer representations $h_l$ with cross-validated Ridge regression, self-diagnosis error predictability, success vs failure regime dissection, and epistemic uncertainty calibration.
