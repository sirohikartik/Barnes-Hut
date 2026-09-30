# 📄 Documentation: `src/bindings.cpp`

## 1. Overview & Architecture
`src/bindings.cpp` defines the **pybind11** Python C-extension interface for `black_hole_core`. It exposes the C++ `BlackHoleSimulation` engine to Python with zero-copy NumPy buffer views for high performance.

---

## 2. Python Module: `black_hole_core`

### Module Initialization: `PYBIND11_MODULE(black_hole_core, m)`
Registers the extension module documentation and binds `BlackHoleSimulation` as `black_hole_core.Simulation`.

---

## 3. Class Binding: `Simulation`

### Constructor
```python
sim = black_hole_core.Simulation(G=1.0, c=60.0, theta=0.6, softening=0.08)
```
- `G` (`float`): Gravitational constant.
- `c` (`float`): Speed of light.
- `theta` (`float`): Barnes-Hut opening angle.
- `softening` (`float`): Gravitational softening parameter $\epsilon$.

### Bound Methods
| Python Method | C++ Target | Description / Arguments |
| :--- | :--- | :--- |
| `set_black_hole(mass, x=0, y=0, z=0, vx=0, vy=0, vz=0)` | `sim.set_black_hole(...)` | Positions and configures the central black hole. |
| `add_body(mass, x, y, z, vx=0, vy=0, vz=0)` | `sim.add_body(...)` | Injects an individual body into the simulation; returns integer `id`. |
| `init_accretion_disk(n, r_min, r_max, total_disk_mass, thickness_ratio=0.03, eccentricity=0.0, seed=42)` | `sim.init_accretion_disk(...)` | Spawns $N$ particles in a relativistic accretion disk. |
| `init_star_cluster(n, cx, cy, cz, cvx, cvy, cvz, radius, total_cluster_mass, seed=123)` | `sim.init_star_cluster(...)` | Spawns a Plummer sphere star cluster. |
| `step(dt=0.01)` | `sim.step(dt)` | Advances simulation by one timestep $\Delta t$. |
| `steps(num_steps, dt=0.01)` | `sim.steps(...)` | Advances simulation by multiple steps in optimized C++ loop. |
| `clear_bodies()` | `sim.clear_bodies()` | Clears all bodies and resets simulation clock. |
| `get_positions()` | Lambda $\to$ NumPy array | Returns $(N, 3)$ float64 array of body coordinates $(x, y, z)$. |
| `get_velocities()` | Lambda $\to$ NumPy array | Returns $(N, 3)$ float64 array of body velocities $(v_x, v_y, v_z)$. |
| `get_masses()` | Lambda $\to$ NumPy array | Returns $(N,)$ float64 array of body masses. |
| `get_active()` | Lambda $\to$ NumPy array | Returns $(N,)$ boolean array indicating whether each particle is alive. |
| `get_black_hole_pos()` | Lambda $\to$ NumPy array | Returns 3-element float64 array for black hole coordinates. |
| `get_black_hole_vel()` | Lambda $\to$ NumPy array | Returns 3-element float64 array for black hole velocity. |
| `get_octree_boxes(max_boxes=500)` | Lambda $\to$ NumPy array | Returns $(K, 5)$ float64 array of octree cells: `[cx, cy, cz, half_width, depth]`. |

### Read-Only Properties
| Property | Type | Description |
| :--- | :--- | :--- |
| `black_hole_mass` | `float` | Current dynamic mass of the black hole $M_{\text{BH}}$. |
| `rs` | `float` | Current Schwarzschild radius $r_s = \frac{2 G M_{\text{BH}}}{c^2}$. |
| `swallowed_count` | `int` | Cumulative count of accreted particles. |
| `total_accreted_mass`| `float` | Cumulative mass captured past event horizon. |
| `active_count` | `int` | Number of currently active bodies in orbit. |
| `total_count` | `int` | Total particle pool size (active + swallowed). |
| `time` | `float` | Cumulative simulation time $t$. |
| `step_count` | `int` | Number of elapsed steps. |
| `kinetic_energy` | `float` | Total kinetic energy of black hole and active bodies: $\sum \frac{1}{2} m v^2$. |
| `octree_node_count` | `int` | Total number of cells generated in the current octree. |

### Read-Write Properties
- `theta` (`float`): Dynamic Barnes-Hut opening angle.
- `softening` (`float`): Softening length $\epsilon$.
- `use_paczynski_wiita` (`bool`): Toggle between Paczyński-Wiita and Newtonian potentials.
- `fixed_black_hole` (`bool`): Toggle central black hole anchoring.
