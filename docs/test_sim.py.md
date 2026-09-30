# 📄 Documentation: `test_sim.py`

## 1. Overview & Architecture
`test_sim.py` is the automated verification test suite for `black_hole_core`. It tests:
1. Module initialization and parameter validation.
2. Schwarzschild radius formula: $r_s = \frac{2 G M}{c^2}$.
3. Accretion disk particle count and initial conditions.
4. Time integration loop and active body tracking.
5. Octree bounding box retrieval.
6. Kinetic energy computation.

---

## 2. Variables & Functions

### `test_basic_simulation()`
Main test runner containing verification assertions:

#### Internal Variables
- `sim`: `black_hole_core.Simulation(G=1.0, c=60.0, theta=0.6, softening=0.08)`.
- `expected_rs`: Analytical Schwarzschild radius:
  $$\text{expected\_rs} = \frac{2 \times 1.0 \times 1000.0}{60.0^2} \approx 0.555556$$
- `pos`, `vel`: Initial position and velocity NumPy arrays from `sim.get_positions()` and `sim.get_velocities()`.
- `active`: Boolean mask from `sim.get_active()`.
- `dt`: Simulation timestep (`0.02`).
- `boxes`: Retrieved octree wireframe boxes from `sim.get_octree_boxes(max_boxes=50)`.
- `pos_after`, `vel_after`, `active_after`: Post-step state arrays after 100 iterations.

---

## 3. Usage
```bash
python test_sim.py
```
Outputs status checks and exits with code 0 upon passing all assertions.
