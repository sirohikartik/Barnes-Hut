import black_hole_core
import numpy as np

def test_basic_simulation():
    print("Testing black_hole_core module...")
    # Initialize simulation: G=1.0, c=60.0, theta=0.6, softening=0.08
    sim = black_hole_core.Simulation(G=1.0, c=60.0, theta=0.6, softening=0.08)
    sim.set_black_hole(mass=1000.0, x=0.0, y=0.0, z=0.0)

    print(f"Initial Black Hole Mass: {sim.black_hole_mass}, Schwarzschild Radius rs: {sim.rs:.4f}")
    assert sim.black_hole_mass == 1000.0
    expected_rs = 2.0 * 1.0 * 1000.0 / (60.0 ** 2)
    assert abs(sim.rs - expected_rs) < 1e-6

    # Initialize accretion disk with 500 particles
    sim.init_accretion_disk(
        n=500,
        r_min=sim.rs * 3.5,  # Just outside ISCO (3.0 * rs)
        r_max=30.0,
        total_disk_mass=50.0,
        thickness_ratio=0.02,
        seed=42
    )

    pos = sim.get_positions()
    vel = sim.get_velocities()
    active = sim.get_active()

    print(f"Generated {len(pos)} bodies. Active count: {sim.active_count}")
    assert len(pos) == 500
    assert sim.active_count == 500
    assert np.all(active)

    # Step simulation
    dt = 0.02
    print("Running 100 steps...")
    for i in range(100):
        sim.step(dt)

    pos_after = sim.get_positions()
    vel_after = sim.get_velocities()
    active_after = sim.get_active()

    boxes = sim.get_octree_boxes(max_boxes=50)
    print(f"After 100 steps: Time={sim.time:.2f}, Swallowed={sim.swallowed_count}, Active={sim.active_count}, Octree Nodes={sim.octree_node_count}, Boxes queried={len(boxes)}")
    print(f"Kinetic energy: {sim.kinetic_energy:.4f}")

    assert sim.time > 0
    assert sim.octree_node_count > 0
    print("Simulation test passed successfully!")

if __name__ == "__main__":
    test_basic_simulation()
