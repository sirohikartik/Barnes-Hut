# 📄 Documentation: `include/vec3.hpp`

## 1. Overview & Architecture
`include/vec3.hpp` defines the foundational mathematical structure `Vec3` used throughout the C++ N-body engine. It encapsulates 3D Cartesian coordinates $(x, y, z)$ with overloaded operators, vector products, and Euclidean norms optimized for tight physics inner loops.

All functions are marked `inline` or `constexpr` to allow the compiler to eliminate function call overhead and leverage SIMD auto-vectorization (e.g., NEON on Apple Silicon ARM64, AVX2 on x86_64).

---

## 2. Struct: `Vec3`

### Member Variables
| Variable | Type | Default | Physical Meaning / Description |
| :--- | :--- | :--- | :--- |
| `x` | `double` | `0.0` | Cartesian coordinate along the X-axis (position, velocity, or acceleration component). |
| `y` | `double` | `0.0` | Cartesian coordinate along the Y-axis. |
| `z` | `double` | `0.0` | Cartesian coordinate along the Z-axis (normal to the orbital disk plane). |

---

## 3. Methods & Operator Overloads

### Constructors
- `constexpr Vec3()`: Default constructor initializing coordinates to $(0, 0, 0)$.
- `constexpr Vec3(double x_, double y_, double z_)`: Parameterized constructor setting explicit components.

### Arithmetic Operators
- `Vec3 operator+(const Vec3& o) const`: Vector addition $\vec{a} + \vec{b} = (a_x + b_x, a_y + b_y, a_z + b_z)$.
- `Vec3 operator-(const Vec3& o) const`: Vector subtraction $\vec{a} - \vec{b} = (a_x - b_x, a_y - b_y, a_z - b_z)$.
- `Vec3 operator*(double s) const`: Scalar multiplication $\vec{a} \cdot s = (a_x s, a_y s, a_z s)$.
- `Vec3 operator/(double s) const`: Scalar division. Multiplies by the reciprocal `double inv = 1.0 / s` to avoid repeated costly division instructions.
- `Vec3& operator+=(const Vec3& o)`: In-place addition accumulator (used heavily in acceleration sums).
- `Vec3& operator-=(const Vec3& o)`: In-place subtraction.
- `Vec3& operator*=(double s)`: In-place scalar scaling.

### Norms and Metrics
- `double norm_sq() const`:
  $$\|\vec{v}\|^2 = x^2 + y^2 + z^2$$
  Avoids the square root when computing squared distances $\|\vec{r}_j - \vec{r}_i\|^2$ in gravitational softening.
- `double norm() const`:
  $$\|\vec{v}\| = \sqrt{x^2 + y^2 + z^2}$$
  Euclidean distance / speed magnitude.
- `Vec3 normalized() const`:
  Unit direction vector $\hat{v} = \frac{\vec{v}}{\|\vec{v}\|}$. Guards against divide-by-zero if $\|\vec{v}\| \le 10^{-12}$.

### Vector Products
- `double dot(const Vec3& o) const`:
  $$\vec{a} \cdot \vec{b} = a_x b_x + a_y b_y + a_z b_z$$
  Calculates projections and kinetic energy terms.
- `Vec3 cross(const Vec3& o) const`:
  $$\vec{a} \times \vec{b} = \begin{pmatrix} a_y b_z - a_z b_y \\ a_z b_x - a_x b_z \\ a_x b_y - a_y b_x \end{pmatrix}$$
  Used in computing specific angular momentum $\vec{L} = \vec{r} \times \vec{p}$.

### Non-Member Operators
- `inline Vec3 operator*(double s, const Vec3& v)`: Allows symmetric scalar multiplication ($s \cdot \vec{v}$ in addition to $\vec{v} \cdot s$).
