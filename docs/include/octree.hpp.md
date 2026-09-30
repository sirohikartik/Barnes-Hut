# 📄 Documentation: `include/octree.hpp`

## 1. Overview & Architecture
`include/octree.hpp` implements a high-performance **3D Barnes-Hut Octree** spatial partitioning system. In direct $N$-body simulations, evaluating pairwise gravity requires $O(N^2)$ calculations. The Barnes-Hut octree clusters distant groups of particles into multipole pseudo-particles, reducing time complexity to $O(N \log N)$.

### Key Features
1. **Memory Pooling**: Nodes are stored in a contiguous `std::vector<OctreeNode>` rather than individual heap allocations (`new`/`delete`), eliminating memory fragmentation and cache misses.
2. **Iterative Traversal**: Force computation uses a fixed-size stack array `int stack[256]` instead of recursion, avoiding stack overflows and minimizing call overhead.
3. **Bounding Box Extraction**: Exposes 3D spatial cell coordinates to Python for live rendering.

---

## 2. Struct: `OctreeNode`

Represents an individual cubical cell in the octree.

### Member Variables
| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `center` | `Vec3` | `{0,0,0}` | Geometric centroid $(c_x, c_y, c_z)$ of the cubical cell. |
| `half_width` | `double` | `0.0` | Half-length of the cube's side length $s = 2 \times w_{\mathrm{half}}$. |
| `mass` | `double` | `0.0` | Total aggregate mass enclosed by this subtree: $M = \sum m_i$. |
| `com` | `Vec3` | `{0,0,0}` | Center of mass of the enclosed bodies: $\vec{R}_{\text{com}} = \frac{1}{M}\sum m_i \vec{r}_i$. |
| `body_idx` | `int` | `-1` | Index of the particle if this node is an occupied leaf; `-1` otherwise. |
| `children` | `std::array<int, 8>` | `{-1, ...}` | Indices of 8 sub-octant children in `Octree::nodes`. `-1` denotes an empty octant. |
| `is_leaf` | `bool` | `true` | `true` if this node has not been subdivided into children. |
| `count` | `int` | `0` | Number of active bodies enclosed in this subtree. |
| `depth` | `int` | `0` | Tree depth level (Root = 0). |

---

## 3. Struct: `BoundingBox`

Lightweight struct used for exporting wireframe cell geometry to Python for visualization.

| Variable | Type | Description |
| :--- | :--- | :--- |
| `center` | `Vec3` | Center of the box. |
| `half_width` | `double` | Half side-length. |
| `depth` | `int` | Subdivision depth. |
| `count` | `int` | Number of particles inside. |
| `mass` | `double` | Aggregate mass. |

---

## 4. Class: `Octree`

### Member Variables
| Variable | Type | Default | Meaning / Configuration |
| :--- | :--- | :--- | :--- |
| `nodes` | `std::vector<OctreeNode>` | Reserve 4096 | Contiguous buffer pool storing all octree cells. |
| `theta` | `double` | `0.6` | Multipole Acceptance Criterion (MAC) opening angle $\theta$. |
| `softening`| `double` | `0.05` | Plummer gravitational softening length $\epsilon$. |
| `G` | `double` | `1.0` | Gravitational constant in simulation units. |
| `max_depth`| `int` | `32` | Maximum tree subdivision depth to prevent infinite loops from identical coordinates. |

---

## 5. Core Methods & Mathematical Algorithms

### `void build(const std::vector<Body>& bodies)`
Constructs the tree from active particles:
1. Calculates the minimum and maximum bounding extent of all active bodies:
   $$\vec{r}_{\min}, \vec{r}_{\max}$$
2. Sets root cube center and size:
   $$\vec{c} = \frac{\vec{r}_{\min} + \vec{r}_{\max}}{2}, \quad w_{\mathrm{half}} = \frac{\max(\Delta x, \Delta y, \Delta z)}{2} \times 1.05 + 1.0$$
3. Inserts each active particle into root node `0`.

### `void insert(int node_idx, int body_id, const std::vector<Body>& bodies)`
Inserts particle into the octree iteratively:
- If node is empty: stores particle.
- If node is a leaf with an existing particle: subdivides cell into 8 octants, relocates existing particle to its appropriate child, and places the incoming particle.
- Updates cumulative mass and center of mass along the path:
  $$M_{\text{new}} = M_{\text{old}} + m_{\text{body}}$$
  $$\vec{R}_{\text{com, new}} = \frac{\vec{R}_{\text{com, old}} M_{\text{old}} + \vec{r}_{\text{body}} m_{\text{body}}}{M_{\text{new}}}$$

### `static inline int get_octant(const Vec3& pos, const Vec3& center)`
Determines the child index $(0 \dots 7)$ via bitwise flags:
- Bit 0: $x \ge c_x$
- Bit 1: $y \ge c_y$
- Bit 2: $z \ge c_z$

### `Vec3 compute_force_on(int body_idx, const std::vector<Body>& bodies) const`
Traverses the tree to evaluate the gravitational acceleration $\vec{a}_i$ on body $i$:
1. Uses non-recursive stack `int stack[256]`.
2. For each node, evaluates the **Barnes-Hut Multipole Acceptance Criterion**:
   $$\frac{s}{d} < \theta \iff s < \theta \cdot d$$
   where $s = 2 \times w_{\mathrm{half}}$ is the node side length and $d = \|\vec{R}_{\mathrm{com}} - \vec{r}_i\|$ is the distance to the center of mass.
3. **If MAC is satisfied**: treats the entire subtree as a single point mass at $\vec{R}_{\text{com}}$:
   $$\vec{a} += \frac{G M (\vec{R}_{\text{com}} - \vec{r}_i)}{\left(\|\vec{R}_{\text{com}} - \vec{r}_i\|^2 + \epsilon^2\right)^{3/2}}$$
4. **If MAC fails**: pushes all 8 children onto the stack to resolve higher spatial detail.

### `std::vector<BoundingBox> get_bounding_boxes(int max_boxes) const`
Filters and returns occupied spatial bounding cells up to `max_boxes` for 3D visualization.
