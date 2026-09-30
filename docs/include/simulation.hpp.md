# 📄 Documentation: `include/simulation.hpp`

## 1. Overview & Architecture
`include/simulation.hpp` is the core physics simulation engine (`BlackHoleSimulation`). It orchestrates:
1. Strong-field gravity via the relativistic **Paczyński–Wiita potential**.
2. Inter-particle self-gravity through the **Barnes-Hut Octree**.
3. Dynamic event horizon captures ($r \le r_s$), mass accretion, and momentum transfer.
4. Second-order symplectic time integration via the **Velocity-Verlet scheme**.

---

## 2. Class: `BlackHoleSimulation`

### Member Variables
| Variable | Type | Default | Description / Physical Role |
| :--- | :--- | :--- | :--- |
| `G` | `double` | `1.0` | Gravitational constant in simulation units. |
| `c` | `double` | `60.0` | Speed of light in simulation units (sets the Schwarzschild radius scale). |
| `theta` | `double` | `0.6` | Barnes-Hut opening angle parameter $\theta$. |
| `softening` | `double` | `0.08` | Plummer gravitational softening length $\epsilon$ preventing divergent forces at small separation. |
| `current_time` | `double` | `0.0` | Elapsed simulation physical time $t$. |
| `step_count` | `int` | `0` | Number of completed integration steps. |
| `black_hole` | `Body` | Mass=1000 | Central supermassive black hole body. |
| `rs` | `double` | Derived | Schwarzschild radius $r_s = \frac{2 G M_{\text{BH}}}{c^2}$. |
| `use_paczynski_wiita` | `bool` | `true` | If `true`, applies General Relativistic pseudo-Newtonian field; if `false`, applies Newtonian $1/r^2$. |
| `fixed_black_hole` | `bool` | `false` | If `true`, the black hole position is pinned at $(0, 0, 0)$ without reaction recoil. |
| `swallowed_count` | `int` | `0` | Number of stellar bodies that have crossed within $r \le r_s$. |
| `total_accreted_mass` | `double` | `0.0` | Cumulative stellar mass swallowed by the black hole. |
| `bodies` | `std::vector<Body>` | Empty | Array of stellar bodies / particles in orbit. |
| `octree` | `Octree` | Configured | Spatial octree instance evaluating inter-particle gravity. |

---

## 3. Physical Models & Methods

### 3.1 Paczyński–Wiita Pseudo-Newtonian Potential
```cpp
Vec3 compute_bh_acceleration(const Vec3& pos) const
```
Computes the strong-field acceleration exerted by the central black hole:
$$\Phi_{\text{PW}}(r) = -\frac{G M_{\text{BH}}}{r - r_s}$$
$$\vec{a}_{\text{PW}}(\vec{r}) = -\frac{G M_{\text{BH}}}{(r - r_s)^2} \frac{\vec{r}}{r} \quad (r > r_s)$$
Key features reproduced without full numerical relativity tensors:
- Event horizon at $r = r_s$.
- Innermost Stable Circular Orbit (ISCO) at $r_{\text{ISCO}} = 3 r_s$.
- Unstable circular orbits inside $r_{\text{ISCO}}$ leading to natural accretion plunges.
- Perihelion precession of eccentric orbits.

### 3.2 Accretion Disk Initialization
```cpp
void init_accretion_disk(int n, double r_min, double r_max, double total_disk_mass, double thickness_ratio, double eccentricity, unsigned int seed)
```
Generates a realistic Keplerian / relativistic disk:
- Radial distribution: surface density $\Sigma(r) \propto 1/r$.
- Vertical scale height: $z \sim \mathcal{N}(0, h)$ where $h = r \times \mathtt{thickness\_ratio}$.
- Relativistic circular velocity:
  $$v_{\text{circ}}(r) = \sqrt{\frac{G M_{\text{BH}} r}{(r - r_s)^2}}$$
- Tangential velocity vector: $\vec{v} = (-v_{\text{circ}} \sin \phi, v_{\text{circ}} \cos \phi, 0) + \vec{v}_{\text{turb}}$.

### 3.3 Star Cluster Initialization
```cpp
void init_star_cluster(int n, const Vec3& cluster_center, const Vec3& cluster_vel, double cluster_radius, double total_cluster_mass, unsigned int seed)
```
Generates a Plummer sphere cluster undergoing a parabolic plunge for Tidal Disruption Events (TDE):
- Density profile: $\rho(r) = \frac{3 M}{4 \pi a^3} \left(1 + \frac{r^2}{a^2}\right)^{-5/2}$.
- Internal velocity dispersion: $v_{\text{disp}} \sim 0.2 \times v_{\text{esc}}(r)$.

### 3.4 Event Horizon Inelastic Capture
```cpp
void check_accretions()
```
When any particle satisfies $\|\vec{r}_i - \vec{r}_{\text{BH}}\| \le r_s$:
1. Marks `b.active = false` and increments `swallowed_count`.
2. Updates black hole mass and linear momentum via inelastic collision conservation:
   $$M_{\text{BH}} \leftarrow M_{\text{BH}} + m_i$$
   $$\vec{P}_{\text{BH}} \leftarrow \vec{P}_{\text{BH}} + m_i \vec{v}_i$$
   $$\vec{v}_{\text{BH}} = \frac{\vec{P}_{\text{BH}}}{M_{\text{BH}}}$$
3. Dynamically expands the Schwarzschild radius:
   $$r_s \leftarrow \frac{2 G M_{\text{BH}}}{c^2}$$

### 3.5 Velocity-Verlet Symplectic Integrator
```cpp
void step(double dt)
```
Advances the entire multi-body system using a 2nd-order symplectic integrator that preserves phase-space volume:
1. Velocity half-kick:
   $$\vec{v}\left(t + \frac{\Delta t}{2}\right) = \vec{v}(t) + \frac{1}{2} \vec{a}(t) \Delta t$$
2. Position drift:
   $$\vec{x}(t + \Delta t) = \vec{x}(t) + \vec{v}\left(t + \frac{\Delta t}{2}\right) \Delta t$$
3. Accretion check at new positions.
4. Acceleration recalculation $\vec{a}(t + \Delta t)$ using Barnes-Hut octree and BH potential.
5. Velocity second half-kick:
   $$\vec{v}(t + \Delta t) = \vec{v}\left(t + \frac{\Delta t}{2}\right) + \frac{1}{2} \vec{a}(t + \Delta t) \Delta t$$
