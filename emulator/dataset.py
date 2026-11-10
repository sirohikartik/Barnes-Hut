import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import black_hole_core


def compute_physical_quantities(
    pos: np.ndarray,
    vel: np.ndarray,
    masses: np.ndarray,
    bh_pos: np.ndarray,
    bh_vel: np.ndarray,
    bh_mass: float,
    G: float = 1.0,
    rs: float = 0.5555,
    softening: float = 0.08,
) -> Dict[str, np.ndarray]:
    """
    Computes exact physical quantities for the N-body + Black Hole system:
    - Kinetic energy (bodies + BH)
    - Potential energy (Paczynski-Wiita with BH + pairwise softened self-gravity)
    - Total energy = KE + PE
    - Angular momentum vector and norm
    - Linear momentum vector and norm
    - Center of mass position & velocity
    """
    # 1. Kinetic energy
    ke_bh = 0.5 * bh_mass * np.sum(bh_vel**2)
    ke_bodies = 0.5 * np.sum(masses[:, None] * (vel**2))
    ke = float(ke_bh + ke_bodies)

    # 2. Potential energy: Paczynski-Wiita potential around central BH
    r_bh = np.linalg.norm(pos - bh_pos[None, :], axis=1)
    pe_bh = float(-np.sum(G * bh_mass * masses / np.maximum(r_bh - rs, 1e-4)))

    # Pairwise self-gravity among orbiting bodies
    n_bodies = len(masses)
    pe_self = 0.0
    for i in range(n_bodies):
        for j in range(i + 1, n_bodies):
            rij = np.linalg.norm(pos[i] - pos[j])
            pe_self -= G * masses[i] * masses[j] / np.sqrt(rij**2 + softening**2)
    pe = float(pe_bh + pe_self)
    total_energy = ke + pe

    # 3. Angular momentum: L = r x p
    L_bh = bh_mass * np.cross(bh_pos, bh_vel)
    L_bodies = np.sum(masses[:, None] * np.cross(pos, vel), axis=0)
    total_L = L_bh + L_bodies
    L_norm = float(np.linalg.norm(total_L))

    # 4. Linear momentum: P = sum(m * v)
    P_bh = bh_mass * bh_vel
    P_bodies = np.sum(masses[:, None] * vel, axis=0)
    total_P = P_bh + P_bodies
    P_norm = float(np.linalg.norm(total_P))

    # 5. Center of mass
    total_mass = bh_mass + np.sum(masses)
    com_pos = (bh_mass * bh_pos + np.sum(masses[:, None] * pos, axis=0)) / total_mass
    com_vel = (bh_mass * bh_vel + np.sum(masses[:, None] * vel, axis=0)) / total_mass

    return {
        "total_energy": np.array([total_energy], dtype=np.float32),
        "kinetic_energy": np.array([ke], dtype=np.float32),
        "potential_energy": np.array([pe], dtype=np.float32),
        "angular_momentum": total_L.astype(np.float32),
        "angular_momentum_norm": np.array([L_norm], dtype=np.float32),
        "linear_momentum": total_P.astype(np.float32),
        "linear_momentum_norm": np.array([P_norm], dtype=np.float32),
        "com_pos": com_pos.astype(np.float32),
        "com_vel": com_vel.astype(np.float32),
        "positions": pos.flatten().astype(np.float32),
        "velocities": vel.flatten().astype(np.float32),
    }


def generate_single_trajectory(
    num_bodies: int = 16,
    num_steps: int = 80,
    dt: float = 0.01,
    seed: int = 42,
    disk_radius_min: Optional[float] = None,
    disk_radius_max: float = 22.0,
    total_disk_mass: float = 1.6,
    bh_mass: float = 1000.0,
    c: float = 60.0,
    G: float = 1.0,
    softening: float = 0.08,
) -> Tuple[np.ndarray, List[Dict[str, np.ndarray]]]:
    """
    Simulates an N-body system around a supermassive black hole.
    Returns:
        states: (T, State_dim) numpy array of flattened [pos_1, vel_1, ..., pos_N, vel_N, bh_pos, bh_vel]
        physics_info: list of physical property dicts for each timestep t
    """
    sim = black_hole_core.Simulation(G=G, c=c, theta=0.6, softening=softening)
    sim.set_black_hole(mass=bh_mass, x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0)

    if disk_radius_min is None:
        disk_radius_min = sim.rs * 4.0  # Safely outside ISCO (3.0 * rs)

    sim.init_accretion_disk(
        n=num_bodies,
        r_min=disk_radius_min,
        r_max=disk_radius_max,
        total_disk_mass=total_disk_mass,
        thickness_ratio=0.02,
        seed=seed,
    )

    masses = sim.get_masses()
    states_list = []
    physics_list = []

    for step_idx in range(num_steps):
        pos = sim.get_positions()
        vel = sim.get_velocities()
        bh_pos = sim.get_black_hole_pos()
        bh_vel = sim.get_black_hole_vel()

        # State representation: [pos, vel] concatenated
        # Shape: (num_bodies * 6 + 6)
        state_vec = np.concatenate([pos.flatten(), vel.flatten(), bh_pos, bh_vel]).astype(np.float32)
        states_list.append(state_vec)

        # Physics quantities
        phys = compute_physical_quantities(
            pos=pos,
            vel=vel,
            masses=masses,
            bh_pos=bh_pos,
            bh_vel=bh_vel,
            bh_mass=bh_mass,
            G=G,
            rs=sim.rs,
            softening=softening,
        )
        physics_list.append(phys)

        sim.step(dt)

    return np.array(states_list, dtype=np.float32), physics_list


def create_physics_dataset(
    num_trajectories: int = 60,
    num_steps: int = 60,
    num_bodies: int = 16,
    dt: float = 0.01,
    base_seed: int = 100,
    train_split: float = 0.8,
):
    """
    Generates a full dataset of trajectories and physical metadata.
    Uses consistent body mass so physical invariants (E, L, P) are consistent functions of (p, v).
    """
    all_trajectories = []
    all_physics = []

    print(f"Generating {num_trajectories} trajectories of {num_steps} steps (N={num_bodies} bodies)...")
    for i in range(num_trajectories):
        seed = base_seed + i * 23
        states, phys = generate_single_trajectory(
            num_bodies=num_bodies,
            num_steps=num_steps,
            dt=dt,
            seed=seed,
            disk_radius_min=2.5, # safely 4.5 * rs, well outside ISCO (3 * rs) to avoid singular horizon plunges
            disk_radius_max=18.0 + (i % 4) * 0.5,
            total_disk_mass=1.6, # fixed total mass so each particle has mass m=0.1
        )
        all_trajectories.append(states)
        all_physics.append(phys)

    # Strict trajectory-level partitioning (no transition leakage):
    num_rollout = 5
    num_test_traj = 5
    num_train_traj = num_trajectories - num_rollout - num_test_traj

    train_trajectories = all_trajectories[:num_train_traj]
    train_physics_traj = all_physics[:num_train_traj]

    test_trajectories = all_trajectories[num_train_traj : num_train_traj + num_test_traj]
    test_physics_traj = all_physics[num_train_traj : num_train_traj + num_test_traj]

    rollout_trajectories = all_trajectories[-num_rollout:]
    rollout_physics = all_physics[-num_rollout:]

    def extract_pairs(trajs, phys_list):
        xs, ys, deltas, phys_steps = [], [], [], []
        for states, phys in zip(trajs, phys_list):
            T = len(states)
            for t in range(T - 1):
                curr_state = states[t]
                next_state = states[t + 1]
                delta = next_state - curr_state
                xs.append(curr_state)
                ys.append(next_state)
                deltas.append(delta)
                phys_steps.append(phys[t])
        return (
            np.array(xs, dtype=np.float32),
            np.array(ys, dtype=np.float32),
            np.array(deltas, dtype=np.float32),
            phys_steps,
        )

    train_x, train_y, train_delta, train_phys = extract_pairs(train_trajectories, train_physics_traj)
    test_x, test_y, test_delta, test_phys = extract_pairs(test_trajectories, test_physics_traj)

    # Compute normalization statistics strictly on training data
    state_mean = np.mean(train_x, axis=0)
    state_std = np.std(train_x, axis=0)
    state_std[state_std < 1e-4] = 1.0

    delta_mean = np.mean(train_delta, axis=0)
    delta_std = np.std(train_delta, axis=0)
    delta_std[delta_std < 1e-4] = 1.0

    # Physical group scaling factors (coordinate-type invariant)
    pos_idx = list(range(num_bodies * 3)) + [num_bodies * 6, num_bodies * 6 + 1, num_bodies * 6 + 2]
    vel_idx = list(range(num_bodies * 3, num_bodies * 6)) + [num_bodies * 6 + 3, num_bodies * 6 + 4, num_bodies * 6 + 5]

    pos_scale = float(np.std(train_x[:, pos_idx]))
    vel_scale = float(np.std(train_x[:, vel_idx]))
    d_pos_scale = float(np.std(train_delta[:, pos_idx]))
    d_vel_scale = float(np.std(train_delta[:, vel_idx]))

    print(f"Dataset summary (Trajectory-level split):")
    print(f"  Train: {num_train_traj} trajectories ({len(train_x)} transitions)")
    print(f"  Test : {num_test_traj} trajectories ({len(test_x)} transitions)")
    print(f"  Rollout : {len(rollout_trajectories)} dedicated held-out trajectories")
    print(f"  State dimension: {train_x.shape[1]}")
    print(f"  Physical group scales: pos={pos_scale:.3f}, vel={vel_scale:.3f}, d_pos={d_pos_scale:.4f}, d_vel={d_vel_scale:.4f}")

    return {
        "train_x": train_x,
        "train_y": train_y,
        "train_delta": train_delta,
        "train_phys": train_phys,
        "test_x": test_x,
        "test_y": test_y,
        "test_delta": test_delta,
        "test_phys": test_phys,
        "state_mean": state_mean,
        "state_std": state_std,
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "pos_scale": pos_scale,
        "vel_scale": vel_scale,
        "d_pos_scale": d_pos_scale,
        "d_vel_scale": d_vel_scale,
        "pos_idx": pos_idx,
        "vel_idx": vel_idx,
        "rollout_trajectories": rollout_trajectories,
        "rollout_physics": rollout_physics,
        "raw_trajectories": all_trajectories,
        "raw_physics": all_physics,
        "num_bodies": num_bodies,
    }
