# 🌌 Barnes-Hut Octree Black Hole N-Body Simulation

[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B)](https://en.cppreference.com/w/cpp/17)
[![Python 3](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)](https://python.org)
[![pybind11](https://img.shields.io/badge/binding-pybind11-blue)](https://github.com/pybind/pybind11)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance astrophysical N-body simulator written in **C++17**, bound seamlessly to Python via **pybind11**, and rendered using **Matplotlib** and **FFmpeg**. The engine calculates self-gravity across thousands of stellar bodies using the **3D Barnes-Hut Octree** algorithm ($O(N \log N)$) while modeling strong-field gravitational physics around a central **Supermassive Black Hole (SMBH)** using the **Paczyński–Wiita potential** and relativistic event horizon capture.

---

## 🎬 Visualizations & Simulations

### 1. Relativistic Accretion Disk (Self-Gravitating)
Orbiting plasma disk showing Keplerian shear, spiral density wave formations from inter-stellar self-gravity, and inner matter plunging past the ISCO into the event horizon.

https://github.com/user-attachments/assets/accretion_disk.mp4

<div align="center">
  <video src="accretion_disk.mp4" width="100%" controls autoplay loop muted playsinline></video>
  <p><em>Direct MP4 playback (<a href="accretion_disk.mp4">download MP4</a>). Animated GIF preview below:</em></p>
  <img src="accretion_disk.gif" alt="Accretion Disk Simulation" width="100%" />
</div>

---

### 2. Tidal Disruption Event (TDE)
A dense star cluster on a parabolic plunge towards the supermassive black hole. The cluster undergoes catastrophic tidal shredding; stars passing within the Schwarzschild radius ($r_s$) are swallowed, dynamically increasing the black hole's mass and expanding its event horizon.

https://github.com/user-attachments/assets/tidal_disruption.mp4

<div align="center">
  <video src="tidal_disruption.mp4" width="100%" controls autoplay loop muted playsinline></video>
  <p><em>Direct MP4 playback (<a href="tidal_disruption.mp4">download MP4</a>). Animated GIF preview below:</em></p>
  <img src="tidal_disruption.gif" alt="Tidal Disruption Event" width="100%" />
</div>

---

### 3. Real-Time 3D Barnes-Hut Octree Space Partitioning
Wireframe visualization of the octree cells dynamically subdividing 3D space around the black hole. High-density regions generate deeper tree levels, while distant regions are clustered into single multipole nodes.

https://github.com/user-attachments/assets/octree_demo.mp4

<div align="center">
  <video src="octree_demo.mp4" width="100%" controls autoplay loop muted playsinline></video>
  <p><em>Direct MP4 playback (<a href="octree_demo.mp4">download MP4</a>). Animated GIF preview below:</em></p>
  <img src="octree_demo.gif" alt="3D Octree Partitioning" width="100%" />
</div>

---

## ⚡ Mathematical & Physical Formulation

### 1. Barnes-Hut 3D Octree ($O(N \log N)$)
Direct $N$-body computation requires evaluating $O(N^2)$ pairwise interactions. The Barnes-Hut algorithm reduces this to $O(N \log N)$ by recursively partitioning 3D space into octants.

- **Multipole Acceptance Criterion (MAC)**:
  For a query body at position $\vec{r}$ and an octree node with side length $s$ and center of mass $\vec{R}_{\text{com}}$:
  $$\frac{s}{\|\vec{R}_{\text{com}} - \vec{r}\|} < \theta$$
  When this condition is met (standard opening angle $\theta \approx 0.5 - 0.7$), the entire subtree is approximated as a single gravitational source located at the node's center of mass.
- **Plummer Softening Length ($\epsilon$)**:
  To avoid non-physical infinite accelerations during close stellar encounters:
  $$\vec{a}_{ij} = \frac{G \, m_j \, (\vec{r}_j - \vec{r}_i)}{\left(\|\vec{r}_j - \vec{r}_i\|^2 + \epsilon^2\right)^{3/2}}$$

### 2. Paczyński–Wiita Pseudo-Newtonian Black Hole
To incorporate General Relativistic dynamics around a Schwarzschild black hole without the extreme computational overhead of full numerical relativity, the central potential uses the Paczyński–Wiita prescription:

$$\Phi_{\text{PW}}(r) = -\frac{G M_{\text{BH}}}{r - r_s}$$

$$\vec{a}_{\text{PW}}(r) = -\frac{G M_{\text{BH}}}{(r - r_s)^2} \frac{\vec{r}}{r} \quad (r > r_s)$$

This pseudo-Newtonian field reproduces key General Relativity metrics:
- **Schwarzschild Event Horizon**: $r_s = \frac{2 G M_{\text{BH}}}{c^2}$
- **Photon Sphere**: $r_{\text{ph}} = 1.5 \, r_s$
- **Innermost Stable Circular Orbit (ISCO)**: $r_{\text{ISCO}} = 3.0 \, r_s$ (circular orbits inside $r_{\text{ISCO}}$ become dynamically unstable and plunge into the horizon)
- **Perihelion Advance**: Authentic apsidal precession of eccentric orbits

### 3. Event Horizon Capture & Accretion Dynamics
When any particle crosses within the event horizon ($r \le r_s$):
1. The particle is deactivated and flagged as swallowed.
2. Inelastic collision updates the black hole's momentum and mass:
   $$M_{\text{BH}} \leftarrow M_{\text{BH}} + m_i, \qquad \vec{P}_{\text{BH}} \leftarrow \vec{P}_{\text{BH}} + m_i \vec{v}_i$$
3. The Schwarzschild radius expands dynamically:
   $$r_s \leftarrow \frac{2 G M_{\text{BH}}}{c^2}$$

### 4. Symplectic Velocity-Verlet Integrator
Second-order symplectic time-integration guarantees phase-space volume preservation and superior energy conservation over long orbital baselines:
$$\vec{x}(t + \Delta t) = \vec{x}(t) + \vec{v}(t)\Delta t + \frac{1}{2}\vec{a}(t)\Delta t^2$$
$$\vec{v}(t + \Delta t) = \vec{v}(t) + \frac{1}{2}\Big(\vec{a}(t) + \vec{a}(t + \Delta t)\Big)\Delta t$$

---

## 📊 Performance Benchmark

Benchmarked on Apple Silicon (M-series, 100 simulation steps):

| $N$ Bodies | Octree Nodes | Total Time (100 Steps) | Throughput (Steps/sec) |
| :---: | :---: | :---: | :---: |
| **500** | 862 | **0.076 s** | **1,308 steps/s** |
| **1,000** | 1,744 | **0.216 s** | **463 steps/s** |
| **2,000** | 3,328 | **0.490 s** | **204 steps/s** |
| **4,000** | 6,416 | **1.256 s** | **79.6 steps/s** |
| **8,000** | 12,591 | **3.049 s** | **32.8 steps/s** |

---

## 🛠️ Quickstart Guide

### 1. Prerequisites
- C++17 compatible compiler (`clang++` or `g++`)
- Python 3.10+
- `ffmpeg` (for MP4 rendering)

### 2. Setup Virtual Environment & Build Extension
```bash
# Clone the repository
git clone git@github-personal:sirohikartik/Barnes-Hut.git
cd Barnes-Hut

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install pybind11 numpy matplotlib setuptools wheel

# Compile the C++ pybind11 extension module in-place
python setup.py build_ext --inplace
```

### 3. Run Benchmark
```bash
python benchmark.py
```

### 4. Run the Visualizer

#### Interactive 3D Window (Accretion Disk)
```bash
python animate.py --scenario disk --n-bodies 1000
```

#### Render with Live 3D Octree Wireframe Cubes
```bash
python animate.py --scenario disk --n-bodies 600 --show-octree
```

#### Tidal Disruption Simulation (Cluster Plunge)
```bash
python animate.py --scenario tidal_disruption --n-bodies 1000
```

#### Export Video or Animated GIF
```bash
# Export high-definition MP4
python animate.py --scenario disk --n-bodies 1200 --frames 180 --save accretion_disk.mp4 --headless

# Export Tidal Disruption MP4
python animate.py --scenario tidal_disruption --n-bodies 800 --frames 120 --save tidal_disruption.mp4 --headless

# Export Animated GIF
python animate.py --scenario disk --n-bodies 500 --frames 80 --save accretion_disk.gif --headless
```

---

## 💻 Python API Usage

```python
import black_hole_core
import numpy as np

# 1. Initialize simulation with G=1.0, c=60.0, theta=0.6, softening=0.08
sim = black_hole_core.Simulation(G=1.0, c=60.0, theta=0.6, softening=0.08)

# 2. Configure central Supermassive Black Hole
sim.set_black_hole(mass=1500.0, x=0.0, y=0.0, z=0.0)
print(f"Schwarzschild radius: {sim.rs:.3f}")

# 3. Initialize self-gravitating accretion disk
sim.init_accretion_disk(
    n=1000,
    r_min=sim.rs * 2.5,
    r_max=25.0,
    total_disk_mass=75.0,
    thickness_ratio=0.03
)

# 4. Advance physics using Barnes-Hut Octree
sim.steps(num_steps=100, dt=0.01)

# 5. Zero-copy access to NumPy arrays
positions   = sim.get_positions()    # (N, 3) double array
velocities  = sim.get_velocities()   # (N, 3) double array
active_mask = sim.get_active()       # (N,) bool array

# 6. Retrieve Octree cells for 3D visualization
boxes = sim.get_octree_boxes(max_boxes=200) # (K, 5): [cx, cy, cz, half_width, depth]

print(f"Time: {sim.time:.2f} | Active: {sim.active_count} | Swallowed: {sim.swallowed_count}")
```

---

## 🔬 Physics Diffusion Emulator & Internal Representation Probing

Can a generative diffusion model learn relativistic astrophysical dynamics, and **what physical quantities are linearly accessible within its internal neural representations?**

```
Ground Truth Simulator ──► Trajectories x_0, x_1, ..., x_T ──► Train Diffusion Emulator
                                                                     │
                                       ┌─────────────────────────────┴────────────────────────────┐
                                       ▼                                                          ▼
                      Stochastic Ensemble Generation                             Internal Representation Probing
                      x̂_{t+1}^(1), ..., x̂_{t+1}^(K)                              h_l ──► Energy, Angular Momentum, etc.
```

### 1. Conceptual Framework & Pipeline
1. **Ground Truth Trajectories ($x_0, x_1, \ldots, x_T$)**:
   The Barnes-Hut simulator generates authentic relativistic orbital trajectories under the Paczyński–Wiita potential and self-gravity. Each physical state contains coordinates and velocities:
   $$x_t = [\mathbf{p}_1, \mathbf{v}_1, \ldots, \mathbf{p}_N, \mathbf{v}_N, \mathbf{p}_{\text{BH}}, \mathbf{v}_{\text{BH}}] \in \mathbb{R}^D$$
2. **Diffusion Physics Emulator**:
   A lightweight conditional diffusion model is trained using **Apple Silicon Metal Performance Shaders (MPS)** to generate future physical states:
   $$x_t \longrightarrow \text{Diffusion Emulator} \longrightarrow \hat{x}_{t+1}$$
3. **Stochastic Ensemble Generation**:
   Sampling the reverse diffusion chain $K$ times with different random seeds yields an ensemble:
   $$\hat{x}_{t+1}^{(1)}, \hat{x}_{t+1}^{(2)}, \ldots, \hat{x}_{t+1}^{(K)}$$
   The ensemble mean provides the predicted trajectory while the variance measures epistemic uncertainty.
4. **Internal Representation Probing ($h_l$)**:
   Hidden layer activations $h_l$ are extracted across all network layers:
   $$h_l \in \{ h_{\text{input}}, h_1, h_2, h_3, h_{\text{pre\_head}} \}$$
   Linear probes are trained to map representations directly to physical invariants and quantities computed from the ground truth simulator:
   $$h_l \longrightarrow \text{Total Energy } E$$
   $$h_l \longrightarrow \text{Angular Momentum } \|\vec{L}\|$$
   $$h_l \longrightarrow \text{Linear Momentum } \|\vec{P}\|$$
   $$h_l \longrightarrow \text{Positions } \mathbf{p}, \quad \text{Velocities } \mathbf{v}$$

---

### 2. Experimental Results & Visualizations

<div align="center">
  <img src="experiments_output/probing_results_layers.png" alt="Probing Results Across Layers and Diffusion Steps" width="100%" />
  <p><em>Figure 1: (Left) Layer-wise Linear Probe $R^2$ scores across denoiser layers. (Right) Emergence of physical information as reverse diffusion denoises from pure noise ($k=19$) to clean state ($k=0$).</em></p>
</div>

<div align="center">
  <img src="experiments_output/ensemble_rollout_comparison.png" alt="Orbital Rollout and Ensemble Comparison" width="100%" />
  <p><em>Figure 2: (Left) Ground truth Barnes-Hut orbit vs. multi-step autoregressive diffusion emulator rollout and $K=12$ ensemble predictions around the central SMBH. (Right) Fast MPS training loss convergence on Mac M1 Air.</em></p>
</div>

#### Quantitative Probing Scores (Held-Out Test Set)

| Physical Target | Input Representation ($h_{\text{input}}$) | Layer 1 ($h_1$) | Layer 2 ($h_2$) | Layer 3 ($h_3$) | Pre-Head ($h_{\text{pre\_head}}$) | Best $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Angular Momentum ($\|\vec{L}\|$)** | **0.8836** | 0.7711 | 0.7684 | 0.7339 | 0.6649 | **0.8836** |
| **Potential Energy ($U$)** | **0.8855** | 0.8454 | 0.7815 | 0.7457 | 0.8155 | **0.8855** |
| **Positions ($\mathbf{p}_1 \dots \mathbf{p}_N$)** | **0.8668** | 0.7510 | 0.5811 | 0.6342 | 0.6456 | **0.8668** |
| **Velocities ($\mathbf{v}_1 \dots \mathbf{v}_N$)** | **0.8493** | 0.7369 | 0.5406 | 0.5659 | 0.5598 | **0.8493** |
| **Linear Momentum ($\|\vec{P}\|$)** | **0.7041** | 0.6992 | 0.6236 | 0.6334 | 0.6261 | **0.7041** |
| **Total Energy ($E$)** | **0.6542** | 0.6265 | 0.5339 | 0.5823 | 0.5391 | **0.6542** |
| **Kinetic Energy ($K$)** | **0.6527** | 0.6257 | 0.5229 | 0.5778 | 0.5391 | **0.6527** |

---

### 3. Key Findings: What Physical Information is Accessible?

1. **Conservation Laws are Discovered and Retained:**
   - **Angular Momentum** ($\|\vec{L}\|$) achieves the highest decodability ($R^2 = 0.8836$) and remains linearly decodable through the deepest layers ($R^2 > 0.66 - 0.77$). In a central gravitational field, preserving angular momentum is essential to maintain radial stability and prevent unphysical orbital collapse.
   - **Potential Energy** ($U$) is strongly linearly decodable ($R^2 = 0.8855$) because the network must encode proximity to the Schwarzschild event horizon to correctly scale acceleration kicks.
2. **Hierarchical Abstraction:**
   - **Early Layers ($h_{\text{input}}, h_1$)**: Exhibit maximum decodability for raw coordinates (Positions $R^2 = 0.867$, Velocities $R^2 = 0.849$).
   - **Intermediate Layers ($h_2, h_3$)**: Coordinates are transformed into higher-order interaction features, yet physical invariants remain strongly accessible.
3. **Information Crystallization during Reverse Diffusion:**
   - At diffusion step $k=19$ (pure noise prior), physical accessibility is near zero ($R^2 < 0.20$).
   - As reverse denoising steps remove noise ($k = 19 \to 10 \to 0$), physical invariants emerge monotonically, verifying that physical validity is recovered in lockstep with noise removal.

---

### 4. Running the Experiment on Apple Silicon (M1 / M2 / M3)

The model is optimized to train in **~7.7 seconds** on a Mac M1 Air:

```bash
# Execute end-to-end simulation, MPS training, rollout, and probing
./.venv/bin/python run_emulator_experiment.py \
  --num_bodies 16 \
  --num_trajectories 30 \
  --num_steps 50 \
  --epochs 45 \
  --hidden_dim 128 \
  --num_layers 3 \
  --ensemble_size 12 \
  --target_type delta
```

---

## 📁 Repository Structure

```
.
├── docs/                        # Complete mirrored codebase documentation
│   ├── README.md                # Documentation table of contents
│   ├── animate.py.md            # Visualizer and animation documentation
│   ├── benchmark.py.md          # Benchmark script documentation
│   ├── test_sim.py.md           # Test suite documentation
│   ├── setup.py.md              # Setuptools build documentation
│   ├── CMakeLists.txt.md        # CMake build documentation
│   ├── run_emulator_experiment.py.md # Probing runner documentation
│   ├── include/                 # C++ engine header documentation
│   │   ├── vec3.hpp.md
│   │   ├── body.hpp.md
│   │   ├── octree.hpp.md
│   │   └── simulation.hpp.md
│   ├── src/
│   │   └── bindings.cpp.md      # Pybind11 bindings documentation
│   └── emulator/                # Diffusion physics emulator documentation
│       ├── __init__.py.md
│       ├── dataset.py.md
│       ├── model.py.md
│       ├── diffusion.py.md
│       ├── train.py.md
│       └── prober.py.md
├── emulator/                    # Diffusion Physics Emulator & Probing Package
│   ├── __init__.py
│   ├── dataset.py               # Trajectory generator & physical invariant calculator
│   ├── model.py                 # Lightweight ResNet denoiser with activation hooks
│   ├── diffusion.py             # Gaussian diffusion, sampling, and ensembles
│   ├── train.py                 # Apple Silicon MPS training loop
│   └── prober.py                # Linear & non-linear probing suite (R^2, RMSE, correlation)
├── experiments_output/          # Generated experiment artifacts
│   ├── diffusion_emulator.pt    # Trained model weights (< 700 KB)
│   ├── ensemble_rollout_comparison.png # Rollout and ensemble uncertainty plot
│   └── probing_results_layers.png      # Layer-wise R^2 probing plot
├── include/                     # High-performance C++ header core
│   ├── vec3.hpp                 # 3D vector arithmetic, vector products, and Euclidean norms
│   ├── body.hpp                 # Particle data structure, flags, state vectors
│   ├── octree.hpp               # Memory-pooled 3D Barnes-Hut octree & wireframe extractor
│   └── simulation.hpp           # Physics engine, Paczyński-Wiita potential, Verlet integrator
├── src/
│   └── bindings.cpp             # Pybind11 module bindings and NumPy array converters
├── setup.py                     # Extension build script
├── CMakeLists.txt               # CMake build configuration
├── run_emulator_experiment.py   # Master experiment CLI script
├── animate.py                   # Multi-viewport animation & video rendering engine
├── benchmark.py                 # Scaling and performance benchmarking script
├── test_sim.py                  # Automated verification test suite
├── accretion_disk.mp4           # Rendered accretion disk video
├── tidal_disruption.mp4         # Rendered tidal disruption event video
├── octree_demo.mp4              # Rendered octree bounding boxes video
├── accretion_disk.gif           # Rendered accretion disk GIF
├── tidal_disruption.gif         # Rendered tidal disruption event GIF
├── octree_demo.gif              # Rendered octree bounding boxes GIF
└── README.md
```

---

## 📜 License
Released under the [MIT License](LICENSE).
