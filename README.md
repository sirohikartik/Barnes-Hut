# Relativistic Black Hole N-Body Simulation with Barnes-Hut Octree

A high-performance astrophysical N-body simulator written in **C++17**, bound to Python using **pybind11**, and animated with **Matplotlib** (with support for interactive 3D GUI, MP4, and GIF exports).

---

## 🌟 Key Features

1. **Barnes-Hut Octree Algorithm ($O(N \log N)$)**:
   - Dynamic 3D bounding box computation per frame.
   - Recursive spatial partitioning into 8 octants with depth limiting.
   - Multipole Acceptance Criterion (MAC) controlled by opening angle $\theta$ (e.g., $\theta = 0.6$).
   - High-performance memory-pooled node allocation (zero per-step dynamic memory churn).
   - Plummer gravitational softening length $\epsilon$ to prevent singularities during close stellar encounters.
   - Real-time extraction and 3D wireframe visualization of the octree cells.

2. **Astrophysical Black Hole Modeling**:
   - Central Supermassive Black Hole (SMBH) with configurable mass $M_{BH}$ and speed of light $c$.
   - **Paczyński-Wiita Pseudo-Newtonian Potential**:
     $$\Phi_{PW}(r) = -\frac{G M_{BH}}{r - r_s}, \quad \vec{a}_{PW} = -\frac{G M_{BH}}{(r - r_s)^2} \frac{\vec{r}}{r}$$
     Faithfully reproduces general-relativistic Schwarzschild phenomena:
     - Event Horizon at $r_s = \frac{2 G M_{BH}}{c^2}$
     - Photon Sphere at $r_{ph} = 1.5\, r_s$
     - Innermost Stable Circular Orbit (ISCO) at $r_{ISCO} = 3.0\, r_s$
     - Relativistic apsidal precession
   - **Event Horizon Inelastic Capture & Accretion**:
     - Matter crossing $r \le r_s$ is absorbed by the black hole.
     - Conservation of momentum and dynamic growth of black hole mass $M_{BH}$ and Schwarzschild radius $r_s$.
   - **Self-Gravity**: All disk stars and matter interact gravitationally with one another via the Barnes-Hut Octree.

3. **2nd-Order Symplectic Velocity-Verlet Integrator**:
   - Preserves orbital phase space and energy far better than standard Euler/RK methods.

4. **Pybind11 Python Extension**:
   - Zero-copy / direct numpy array views for positions, velocities, masses, and active status flags.
   - Exposes simulation configuration, step execution, and tree introspection directly to Python.

5. **Astrophysical Visualization & Animation**:
   - Dual-viewport (3D orbital perspective + 2D top-down accretion plane).
   - Color mapping based on orbital velocity (relativistic Doppler shift / virial temperature).
   - Dynamic orbital tracers / streak lines.
   - Wireframe overlay of Barnes-Hut 3D octree bounding boxes.
   - Live telemetry HUD (simulation time, active bodies, accreted mass, node count, render FPS).
   - Export directly to **MP4** (via ffmpeg) or **GIF** (via pillow).

---

## 📂 Project Structure

```
black_hole/
├── include/
│   ├── vec3.hpp         # 3D vector algebra (dot, cross, norm, operators)
│   ├── body.hpp         # Body struct (state, mass, flags)
│   ├── octree.hpp       # Barnes-Hut 3D Octree implementation & bounding box extractor
│   └── simulation.hpp   # Physics engine, Paczyński-Wiita potential, Verlet integrator
├── src/
│   └── bindings.cpp     # Pybind11 Python bindings
├── setup.py             # Extension build script
├── CMakeLists.txt       # CMake build configuration
├── test_sim.py          # Unit test verifying physics and bindings
├── benchmark.py         # O(N log N) performance benchmark
├── animate.py           # Visualization script with interactive 3D and export modes
└── README.md
```

---

## 🚀 Quickstart

### 1. Build the C++ Extension

Using the local Python virtual environment:
```bash
source .venv/bin/activate
python setup.py build_ext --inplace
```

### 2. Run the Benchmark

```bash
python benchmark.py
```
*Expected output: ~460+ steps/second for $N = 1000$ bodies; ~33 steps/second for $N = 8000$ bodies on Apple Silicon.*

### 3. Run the Animation

#### Interactive 3D Window:
```bash
python animate.py --scenario disk --n-bodies 1000
```

#### Show Barnes-Hut Octree 3D Wireframe Boxes:
```bash
python animate.py --scenario disk --n-bodies 600 --show-octree
```

#### Tidal Disruption Event (Cluster Plunging into SMBH):
```bash
python animate.py --scenario tidal_disruption --n-bodies 1000
```

#### Render and Save Video (MP4 or GIF):
```bash
# Save to MP4
python animate.py --scenario disk --n-bodies 1000 --frames 180 --save accretion_disk.mp4 --headless

# Save Tidal Disruption to MP4
python animate.py --scenario tidal_disruption --n-bodies 800 --frames 120 --save tidal_disruption.mp4 --headless

# Save to Animated GIF
python animate.py --scenario disk --n-bodies 500 --frames 80 --save accretion_disk.gif --headless
```

---

## 🐍 Python API Example

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

# 4. Step forward using Barnes-Hut octree
sim.steps(num_steps=100, dt=0.01)

# 5. Access numpy arrays directly
positions = sim.get_positions()    # (N, 3) double array
velocities = sim.get_velocities()  # (N, 3) double array
active = sim.get_active()          # (N,) bool array

# 6. Retrieve Octree cells
boxes = sim.get_octree_boxes(max_boxes=100) # (K, 5): [cx, cy, cz, half_width, depth]
print(f"Active bodies: {sim.active_count}, Swallowed: {sim.swallowed_count}")
```
