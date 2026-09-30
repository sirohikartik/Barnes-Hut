# 📄 Documentation: `animate.py`

## 1. Overview & Architecture
`animate.py` is the graphical rendering and animation engine for the simulation. It provides:
1. Interactive real-time 3D Matplotlib window visualization.
2. Headless batch video/GIF rendering using `ffmpeg` (`FFMpegWriter`) or Pillow (`PillowWriter`).
3. Three distinct astrophysical scenarios:
   - **`disk`**: Relativistic self-gravitating accretion disk with Keplerian shear and ISCO plunging.
   - **`tidal_disruption`**: Star cluster on a parabolic collision course disrupted into an accretion flare.
   - **`collision`**: Counter-rotating stellar streams colliding in the SMBH gravitational well.
4. Real-time 3D wireframe rendering of the Barnes-Hut Octree space-partitioning cells.

---

## 2. Core Functions & CLI Arguments

### Function: `create_octree_cube_lines(boxes)`
Transforms raw bounding box coordinates $[c_x, c_y, c_z, \text{half\_width}, \text{depth}]$ into sets of 12 line segments per cube for rendering via `Line3DCollection`.
- `boxes`: $(K, 5)$ NumPy array from `sim.get_octree_boxes()`.
- Filters deeper subdivisions (`depth > 5`) to prevent visual clutter and maintain high rendering frame rates.

### Function: `setup_simulation(scenario, n_bodies, theta, softening, bh_mass, c)`
Instantiates and populates the C++ `black_hole_core.Simulation`:
- Computes Schwarzschild radius $r_s$, Innermost Stable Circular Orbit $r_{\text{ISCO}} = 3 r_s$, and Photon Sphere $r_{\text{ph}} = 1.5 r_s$.
- Configures selected scenario initial conditions (`disk`, `tidal_disruption`, or `collision`).

### Function: `run_animation(args)`
Configures the Matplotlib Figure, 3D scatter plots, event horizon sphere, camera angles, color mapping, and the `FuncAnimation` loop.
- **Color Mapping**: Uses `plt.cm.plasma` or `plt.cm.magma` based on particle velocity magnitude $\|\vec{v}\|$, illustrating relativistic Doppler shifts and gravitational blue-shift towards the center.
- **HUD (Heads-Up Display)**: Real-time text overlay showing simulation time $t$, FPS, active body count, swallowed particles count, kinetic energy, and octree node count.

### CLI Arguments
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--scenario` | `str` | `'disk'` | Simulation scenario (`disk`, `tidal_disruption`, `collision`). |
| `--n-bodies` | `int` | `1000` | Number of orbiting particles. |
| `--bh-mass` | `float` | `1500.0` | Mass of central black hole. |
| `--theta` | `float` | `0.6` | Barnes-Hut opening angle. |
| `--softening` | `float` | `0.08` | Softening length $\epsilon$. |
| `--dt` | `float` | `0.01` | Physical timestep per frame. |
| `--steps-per-frame` | `int` | `2` | Number of physics sub-steps per rendered frame. |
| `--frames` | `int` | `300` | Maximum number of animation frames. |
| `--save` | `str` | `None` | Output video or GIF file path (e.g. `disk.mp4`, `demo.gif`). |
| `--fps` | `int` | `30` | Animation video frame rate. |
| `--show-octree` | `flag` | `False` | Renders dynamic 3D octree wireframe boxes. |
| `--headless` | `flag` | `False` | Runs without graphical window (required for CI or background rendering). |

---

## 3. Usage Examples
```bash
# Live interactive 3D window
python animate.py --scenario disk --n-bodies 1000

# Live octree space-partitioning wireframe
python animate.py --scenario disk --n-bodies 600 --show-octree

# Export 1080p MP4 video
python animate.py --scenario tidal_disruption --n-bodies 1200 --frames 180 --save tidal_disruption.mp4 --headless
```
