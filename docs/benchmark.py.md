# 📄 Documentation: `benchmark.py`

## 1. Overview & Architecture
`benchmark.py` is a performance profiling script that benchmarks the execution throughput and scaling behavior of the Barnes-Hut C++ simulation engine across varying body counts $N \in [500, 8000]$.

---

## 2. Variables & Functions

### `run_benchmark()`
Executes the benchmark suite and displays a formatted tabular summary.

#### Internal Variables
- `test_counts`: List of particle counts evaluated: `[500, 1000, 2000, 4000, 8000]`.
- `sim`: Instance of `black_hole_core.Simulation(G=1.0, c=60.0, theta=0.6, softening=0.08)`.
- `t0`: High-resolution start timestamp recorded using `time.perf_counter()`.
- `t_elapsed`: Wall-clock execution time for 100 simulation steps: `time.perf_counter() - t0`.
- `sps`: Simulation throughput measured in steps per second:
  $$\text{sps} = \frac{100}{\Delta t}$$
- `node_count`: Number of 3D octree nodes created (`sim.octree_node_count`), illustrating logarithmic tree growth $O(N \log N)$.

---

## 3. Usage
```bash
python benchmark.py
```
Outputs performance metrics (e.g. 1,308 steps/sec at $N=500$, ~33 steps/sec at $N=8,000$).
