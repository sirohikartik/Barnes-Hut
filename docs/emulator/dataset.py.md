# 📄 Documentation: `emulator/dataset.py`

## 1. Overview & Architecture
`emulator/dataset.py` serves as the bridge between the C++ Barnes-Hut simulator and the PyTorch machine learning pipeline. It generates ground truth trajectories:
$$x_0, x_1, x_2, \ldots, x_T$$
where each state vector $x_t$ represents the complete physical state:
$$x_t = [\mathbf{p}_1, \mathbf{v}_1, \ldots, \mathbf{p}_N, \mathbf{v}_N, \mathbf{p}_{\text{BH}}, \mathbf{v}_{\text{BH}}] \in \mathbb{R}^{D}$$
It also computes exact physical quantities (Energy, Angular Momentum, Linear Momentum, Center of Mass) at every timestep $t$ for ground truth validation and internal probing.

---

## 2. Core Functions & Physical Formulations

### Function: `compute_physical_quantities(...)`
Evaluates exact physical invariants and observables for the system at any given state.

#### Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `pos` | `np.ndarray` | Required | Shape $(N, 3)$ array of particle coordinates. |
| `vel` | `np.ndarray` | Required | Shape $(N, 3)$ array of particle velocities. |
| `masses` | `np.ndarray` | Required | Shape $(N,)$ array of individual body masses. |
| `bh_pos` | `np.ndarray` | Required | Shape $(3,)$ black hole position vector. |
| `bh_vel` | `np.ndarray` | Required | Shape $(3,)$ black hole velocity vector. |
| `bh_mass`| `float` | Required | Mass of central black hole $M_{\text{BH}}$. |
| `G` | `float` | `1.0` | Gravitational constant. |
| `rs` | `float` | `0.5555` | Schwarzschild radius $r_s$. |
| `softening`| `float` | `0.08` | Softening length $\epsilon$. |

#### Computed Physical Quantities
1. **Kinetic Energy ($K$)**:
   $$K = \frac{1}{2} M_{\text{BH}} \|\vec{v}_{\text{BH}}\|^2 + \sum_{i=1}^N \frac{1}{2} m_i \|\vec{v}_i\|^2$$
2. **Potential Energy ($U$)**:
   - Paczyński–Wiita potential between bodies and central SMBH:
     $$U_{\text{BH}} = -\sum_{i=1}^N \frac{G M_{\text{BH}} m_i}{\max(\|\vec{r}_i - \vec{r}_{\text{BH}}\| - r_s, 10^{-4})}$$
   - Pairwise self-gravity among orbiting bodies:
     $$U_{\text{self}} = -\sum_{i < j} \frac{G m_i m_j}{\sqrt{\|\vec{r}_i - \vec{r}_j\|^2 + \epsilon^2}}$$
   - Total Potential: $U = U_{\text{BH}} + U_{\text{self}}$.
3. **Total Energy ($E$)**:
   $$E = K + U$$
4. **Angular Momentum Vector & Magnitude ($\vec{L}, \|\vec{L}\|$)**:
   $$\vec{L} = M_{\text{BH}} (\vec{r}_{\text{BH}} \times \vec{v}_{\text{BH}}) + \sum_{i=1}^N m_i (\vec{r}_i \times \vec{v}_i)$$
5. **Linear Momentum Vector & Magnitude ($\vec{P}, \|\vec{P}\|$)**:
   $$\vec{P} = M_{\text{BH}} \vec{v}_{\text{BH}} + \sum_{i=1}^N m_i \vec{v}_i$$
6. **Center of Mass ($\vec{R}_{\text{cm}}, \vec{V}_{\text{cm}}$)**:
   $$\vec{R}_{\text{cm}} = \frac{M_{\text{BH}} \vec{r}_{\text{BH}} + \sum m_i \vec{r}_i}{M_{\text{BH}} + \sum m_i}, \quad \vec{V}_{\text{cm}} = \frac{M_{\text{BH}} \vec{v}_{\text{BH}} + \sum m_i \vec{v}_i}{M_{\text{BH}} + \sum m_i}$$

---

### Function: `generate_single_trajectory(...)`
Simulates a single $N$-body trajectory around the SMBH.

#### Parameters
- `num_bodies`: Number of orbiting stellar bodies $N$ (e.g. 16).
- `num_steps`: Length of trajectory $T$ in timesteps.
- `dt`: Numerical integration timestep (e.g. 0.01).
- `seed`: PRNG seed for reproducible orbital phase generation.
- `disk_radius_min`, `disk_radius_max`: Annulus bounds for accretion disk.
- `total_disk_mass`: Summed mass of disk particles.
- `bh_mass`, `c`, `G`, `softening`: Simulation physical constants.

#### Returns
- `states`: Shape $(T, \text{State\_dim})$ NumPy array where $\text{State\_dim} = N \times 6 + 6$.
- `physics_list`: Length-$T$ list of physical property dictionaries.

---

### Function: `create_physics_dataset(...)`
Generates a multi-trajectory dataset and splits it into training and test sets.

#### Trajectory & Step Partitioning Strategy
- Generates `num_trajectories` (e.g. 30 runs with distinct random seeds).
- Dedicates the final 2 trajectories strictly for **continuous multi-step autoregressive rollouts** and ensemble visualization.
- Assembles all transitions $(x_t, x_{t+1}, \Delta x_t)$ from the remaining pool trajectories into a randomized 80% train / 20% test split.
- Evaluates normalization statistics:
  - `state_mean`, `state_std` for $x_t$.
  - `delta_mean`, `delta_std` for $\Delta x_t = x_{t+1} - x_t$.
