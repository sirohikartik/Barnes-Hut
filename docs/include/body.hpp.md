# 📄 Documentation: `include/body.hpp`

## 1. Overview & Architecture
`include/body.hpp` declares the `Body` struct, which models a single point-mass entity in the simulation (e.g., stellar particle, gas tracer parcel, or central supermassive black hole).

It holds phase-space variables (position, velocity, acceleration), physical properties (mass), and simulation flags (active status, black hole designation).

---

## 2. Struct: `Body`

### Member Variables
| Variable | Type | Default | Description / Physical Role |
| :--- | :--- | :--- | :--- |
| `id` | `int` | `0` | Unique integer identifier of the body within the simulation pool. |
| `pos` | `Vec3` | `{0,0,0}` | Current Cartesian position vector $\vec{r} = (x, y, z)$. |
| `vel` | `Vec3` | `{0,0,0}` | Current velocity vector $\vec{v} = (v_x, v_y, v_z)$. |
| `acc` | `Vec3` | `{0,0,0}` | Gravitational acceleration vector $\vec{a} = (a_x, a_y, a_z)$ computed at the current step. |
| `mass` | `double` | `1.0` | Inertial and gravitational mass $m_i$ of the particle. |
| `is_black_hole` | `bool` | `false` | Boolean flag. If `true`, this body is treated as the central singularity source with event horizon mechanics. |
| `active` | `bool` | `true` | Liveness flag. Set to `false` when a body plunges across the event horizon ($r \le r_s$). Deactivated bodies are omitted from octree construction and acceleration sums. |

---

## 3. Constructors

- `Body()`: Default constructor creating an uninitialized body at the origin.
- `Body(int id_, const Vec3& p, const Vec3& v, double m, bool is_bh = false)`:
  Initializes the body with:
  - `id`: `id_`
  - `pos`: `p`
  - `vel`: `v`
  - `acc`: `{0.0, 0.0, 0.0}` (acceleration evaluated on first integration pass)
  - `mass`: `m`
  - `is_black_hole`: `is_bh`
  - `active`: `true`
