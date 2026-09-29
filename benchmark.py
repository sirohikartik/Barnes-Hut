#!/usr/bin/env python3
"""
Benchmark script demonstrating Barnes-Hut Octree scaling vs particle count N.
"""

import time
import black_hole_core

def run_benchmark():
    print("=" * 65)
    print(" BARNES-HUT OCTREE PERFORMANCE BENCHMARK (C++ pybind11)")
    print("=" * 65)
    print(f"{'N Bodies':<10} | {'Octree Nodes':<14} | {'100 Steps Time':<16} | {'Steps/sec':<10}")
    print("-" * 65)

    test_counts = [500, 1000, 2000, 4000, 8000]

    for n in test_counts:
        sim = black_hole_core.Simulation(G=1.0, c=60.0, theta=0.6, softening=0.08)
        sim.set_black_hole(mass=1500.0)
        sim.init_accretion_disk(
            n=n,
            r_min=sim.rs * 3.5,
            r_max=30.0,
            total_disk_mass=50.0,
            seed=42
        )

        # Warm up 5 steps
        sim.steps(5, 0.01)

        # Time 100 steps
        t0 = time.perf_counter()
        sim.steps(100, 0.01)
        t_elapsed = time.perf_counter() - t0

        sps = 100.0 / t_elapsed
        node_count = sim.octree_node_count

        print(f"{n:<10} | {node_count:<14} | {t_elapsed:<14.3f} s | {sps:<10.1f}")

    print("=" * 65)

if __name__ == "__main__":
    run_benchmark()
