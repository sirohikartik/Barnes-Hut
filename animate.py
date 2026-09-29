#!/usr/bin/env python3
"""
Black Hole N-body Simulation Animated with Barnes-Hut Octree
Visualizes self-gravitating particles orbiting a relativistic Black Hole (Paczyński-Wiita potential).
"""

import argparse
import sys
import os
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Import compiled pybind11 module
try:
    import black_hole_core
except ImportError:
    # Try looking in build or local directory
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import black_hole_core


def create_octree_cube_lines(boxes):
    """Convert octree bounding boxes (cx, cy, cz, hw) into 3D line segments."""
    segments = []
    for b in boxes:
        cx, cy, cz, hw, depth = b[0], b[1], b[2], b[3], int(b[4])
        # Only show intermediate depths to avoid excessive clutter
        if depth > 5:
            continue
        x0, x1 = cx - hw, cx + hw
        y0, y1 = cy - hw, cy + hw
        z0, z1 = cz - hw, cz + hw

        # 12 edges of the cube
        edges = [
            [(x0, y0, z0), (x1, y0, z0)],
            [(x1, y0, z0), (x1, y1, z0)],
            [(x1, y1, z0), (x0, y1, z0)],
            [(x0, y1, z0), (x0, y0, z0)],
            [(x0, y0, z1), (x1, y0, z1)],
            [(x1, y0, z1), (x1, y1, z1)],
            [(x1, y1, z1), (x0, y1, z1)],
            [(x0, y1, z1), (x0, y0, z1)],
            [(x0, y0, z0), (x0, y0, z1)],
            [(x1, y0, z0), (x1, y0, z1)],
            [(x1, y1, z0), (x1, y1, z1)],
            [(x0, y1, z0), (x0, y1, z1)],
        ]
        segments.extend(edges)
    return segments


def setup_simulation(scenario, n_bodies, theta=0.6, softening=0.08, bh_mass=1500.0, c=60.0):
    """Configures the C++ Barnes-Hut simulation for selected astrophysical scenario."""
    sim = black_hole_core.Simulation(G=1.0, c=c, theta=theta, softening=softening)
    sim.set_black_hole(mass=bh_mass, x=0.0, y=0.0, z=0.0)
    rs = sim.rs

    if scenario == "disk":
        # Relativistic Accretion Disk around SMBH
        r_min = rs * 2.2  # Starts inside ISCO (3.0 rs) so inner edge accretes!
        r_max = 24.0
        total_mass = bh_mass * 0.05  # 5% of BH mass in disk (self-gravitating)
        sim.init_accretion_disk(
            n=n_bodies,
            r_min=r_min,
            r_max=r_max,
            total_disk_mass=total_mass,
            thickness_ratio=0.025,
            eccentricity=0.04,
            seed=42
        )
    elif scenario == "tidal_disruption":
        # Infalling dense star cluster on parabolic encounter with SMBH
        # Cluster starts at (-30, -20, 2) headed towards black hole
        cluster_mass = bh_mass * 0.08
        sim.init_star_cluster(
            n=n_bodies,
            cx=-28.0, cy=-18.0, cz=2.5,
            cvx=4.2, cvy=2.8, cvz=-0.3,
            radius=4.5,
            total_cluster_mass=cluster_mass,
            seed=101
        )
        # Add a light background disk
        sim.init_accretion_disk(
            n=n_bodies // 4,
            r_min=rs * 2.5,
            r_max=20.0,
            total_disk_mass=bh_mass * 0.01,
            thickness_ratio=0.02,
            seed=999
        )
    elif scenario == "collision":
        # Two colliding stellar streams orbiting the black hole
        half_n = n_bodies // 2
        sim.init_star_cluster(
            n=half_n,
            cx=-18.0, cy=0.0, cz=1.0,
            cvx=0.0, cvy=6.5, cvz=0.2,
            radius=3.5,
            total_cluster_mass=bh_mass * 0.03,
            seed=1
        )
        sim.init_star_cluster(
            n=half_n,
            cx=18.0, cy=0.0, cz=-1.0,
            cvx=0.0, cvy=-6.5, cvz=-0.2,
            radius=3.5,
            total_cluster_mass=bh_mass * 0.03,
            seed=2
        )

    return sim


def run_animation(args):
    if args.headless:
        matplotlib.use("Agg")

    # Initialize C++ Simulation
    sim = setup_simulation(
        scenario=args.scenario,
        n_bodies=args.n_bodies,
        theta=args.theta,
        softening=args.softening,
        bh_mass=args.bh_mass,
        c=args.c
    )

    rs = sim.rs
    r_isco = 3.0 * rs
    r_photon = 1.5 * rs

    print("=" * 60)
    print(" BARNES-HUT OCTREE BLACK HOLE SIMULATION")
    print(f" Scenario:            {args.scenario}")
    print(f" Total Bodies:        {sim.total_count}")
    print(f" Black Hole Mass:     {sim.black_hole_mass:.1f} M☉")
    print(f" Schwarzschild (rs):  {rs:.3f}")
    print(f" Photon Sphere:       {r_photon:.3f}")
    print(f" ISCO:                {r_isco:.3f}")
    print(f" Octree θ (MAC):      {sim.theta}")
    print(f" Plummer Softening ε: {sim.softening}")
    print("=" * 60)

    # Set up Matplotlib Figure with sleek dark astrophysical theme
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(13, 8), facecolor="#06060c")

    if args.view == "multi":
        # Dual view: Left 3D Perspective, Right 2D Top-Down accretion plane
        ax3d = fig.add_subplot(1, 2, 1, projection="3d", facecolor="#06060c")
        ax2d = fig.add_subplot(1, 2, 2, facecolor="#06060c")
    else:
        ax3d = fig.add_subplot(1, 1, 1, projection="3d", facecolor="#06060c")
        ax2d = None

    # Style 3D axis
    lim = args.limit
    ax3d.set_xlim(-lim, lim)
    ax3d.set_ylim(-lim, lim)
    ax3d.set_zlim(-lim * 0.5, lim * 0.5)
    ax3d.set_box_aspect([1, 1, 0.5])
    ax3d.grid(False)
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.xaxis.pane.set_edgecolor("#111122")
    ax3d.yaxis.pane.set_edgecolor("#111122")
    ax3d.zaxis.pane.set_edgecolor("#111122")
    ax3d.tick_params(colors="#444466", labelsize=8)
    ax3d.set_title("3D Barnes-Hut Octree Space", color="#99aacc", fontsize=11, pad=10)

    # In 3D: Draw Black Hole Event Horizon sphere & ISCO circle
    u = np.linspace(0, 2 * np.pi, 25)
    v = np.linspace(0, np.pi, 15)
    xs = rs * np.outer(np.cos(u), np.sin(v))
    ys = rs * np.outer(np.sin(u), np.sin(v))
    zs = rs * np.outer(np.ones(np.size(u)), np.cos(v))
    bh_surface = ax3d.plot_surface(xs, ys, zs, color="#000000", edgecolor="#ff5500", alpha=0.9, lw=0.3)

    # Photon sphere ring in 3D
    phi_circ = np.linspace(0, 2 * np.pi, 100)
    ax3d.plot(r_photon * np.cos(phi_circ), r_photon * np.sin(phi_circ), np.zeros(100),
              color="#ffaa00", lw=1.2, ls="-", alpha=0.8, label="Photon Sphere (1.5 rs)")
    # ISCO ring in 3D
    ax3d.plot(r_isco * np.cos(phi_circ), r_isco * np.sin(phi_circ), np.zeros(100),
              color="#00ddff", lw=1.0, ls="--", alpha=0.6, label="ISCO (3.0 rs)")

    # Compute initial speeds for color mapping
    initial_vel = sim.get_velocities()
    init_speeds = np.linalg.norm(initial_vel, axis=1)
    norm_max = np.percentile(init_speeds, 95) if len(init_speeds) > 0 else 1.0
    init_norm = np.clip(init_speeds / (norm_max + 1e-4), 0.0, 1.0)

    # 3D Star particles scatter
    initial_pos = sim.get_positions()
    scat3d = ax3d.scatter(initial_pos[:, 0], initial_pos[:, 1], initial_pos[:, 2],
                          c=init_norm, s=6, alpha=0.85, cmap="inferno", vmin=0.0, vmax=1.0, depthshade=True)

    # 3D Octree Wireframe collection
    octree_coll = None
    if args.show_octree:
        boxes = sim.get_octree_boxes(max_boxes=args.max_octree_boxes)
        initial_segs = create_octree_cube_lines(boxes)
        if not initial_segs:
            # Provide at least one dummy segment if empty
            initial_segs = [[(0,0,0), (0,0,0)]]
        octree_coll = Line3DCollection(initial_segs, colors="#38bdf8", linewidths=0.5, alpha=0.25)
        ax3d.add_collection3d(octree_coll)

    # Trailing paths for select tracer particles
    n_tracers = min(15, sim.total_count)
    tracer_indices = np.linspace(0, sim.total_count - 1, n_tracers, dtype=int)
    tracer_history = [[] for _ in range(n_tracers)]
    tracer_lines = [ax3d.plot([], [], [], color="#ffcc44", alpha=0.4, lw=0.9)[0] for _ in range(n_tracers)]

    # 2D Viewport setup (if multi)
    if ax2d is not None:
        ax2d.set_xlim(-lim, lim)
        ax2d.set_ylim(-lim, lim)
        ax2d.set_aspect("equal")
        ax2d.set_facecolor("#030308")
        ax2d.tick_params(colors="#444466", labelsize=8)
        ax2d.set_title("Accretion Plane (Top-Down)", color="#99aacc", fontsize=11)
        ax2d.grid(True, color="#111126", linestyle=":", alpha=0.5)

        # Black hole disk circles in 2D
        bh_circle = plt.Circle((0, 0), rs, color="#000000", ec="#ff3300", lw=1.5, zorder=5)
        photon_circle = plt.Circle((0, 0), r_photon, fill=False, color="#ffaa00", lw=1.2, ls="-", alpha=0.8, zorder=4)
        isco_circle = plt.Circle((0, 0), r_isco, fill=False, color="#00ddff", lw=1.0, ls="--", alpha=0.6, zorder=4)
        ax2d.add_patch(bh_circle)
        ax2d.add_patch(photon_circle)
        ax2d.add_patch(isco_circle)

        scat2d = ax2d.scatter(initial_pos[:, 0], initial_pos[:, 1], c=init_norm, s=7, alpha=0.8, cmap="inferno", vmin=0.0, vmax=1.0)

    # HUD Overlay Text
    hud_text = fig.text(
        0.02, 0.94, "",
        fontsize=9.5, fontfamily="monospace", color="#90cdf4",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#0b0f19", edgecolor="#2d3748", alpha=0.85)
    )

    t0 = time.time()
    frame_times = []

    def update(frame):
        nonlocal rs

        # Run C++ Barnes-Hut integration sub-steps
        for _ in range(args.substeps):
            sim.step(args.dt)

        # Retrieve updated state directly from C++ into numpy arrays
        positions = sim.get_positions()
        velocities = sim.get_velocities()
        active = sim.get_active()

        # Update event horizon if BH grew
        current_rs = sim.rs

        # Color particles by orbital speed (relativistic Doppler / kinetic temperature)
        speeds = np.linalg.norm(velocities, axis=1)
        # Normalize speeds for colormap
        norm_speeds = np.clip(speeds / (np.percentile(speeds[active], 95) + 1e-4), 0.0, 1.0) if np.any(active) else speeds

        active_indices = np.where(active)[0]
        pos_active = positions[active]
        speed_active = norm_speeds[active]

        # Update 3D scatter
        if len(pos_active) > 0:
            scat3d._offsets3d = (pos_active[:, 0], pos_active[:, 1], pos_active[:, 2])
            scat3d.set_array(speed_active)
        else:
            scat3d._offsets3d = ([], [], [])

        # Camera rotation for 3D cinematic feel
        if args.rotate:
            ax3d.view_init(elev=32 + 8 * np.sin(frame * 0.02), azim=(frame * 0.5) % 360)

        # Update tracer trails
        for idx_t, b_idx in enumerate(tracer_indices):
            if active[b_idx]:
                tracer_history[idx_t].append(positions[b_idx].copy())
                if len(tracer_history[idx_t]) > 40:
                    tracer_history[idx_t].pop(0)
                pts = np.array(tracer_history[idx_t])
                tracer_lines[idx_t].set_data(pts[:, 0], pts[:, 1])
                tracer_lines[idx_t].set_3d_properties(pts[:, 2])
            else:
                tracer_lines[idx_t].set_data([], [])
                tracer_lines[idx_t].set_3d_properties([])

        # Update Octree wireframes if enabled
        if args.show_octree and octree_coll is not None:
            boxes = sim.get_octree_boxes(max_boxes=args.max_octree_boxes)
            segs = create_octree_cube_lines(boxes)
            if segs:
                octree_coll.set_segments(segs)

        # Update 2D viewport (if multi)
        if ax2d is not None:
            if len(pos_active) > 0:
                scat2d.set_offsets(pos_active[:, :2])
                scat2d.set_array(speed_active)
            else:
                scat2d.set_offsets(np.empty((0, 2)))

        # Update HUD stats
        fps = 1.0 / (time.time() - frame_times[-1]) if frame_times else 0.0
        frame_times.append(time.time())
        if len(frame_times) > 30:
            frame_times.pop(0)

        hud_str = (
            f"  [BARNES-HUT OCTREE N-BODY]\n"
            f"  Sim Time:      {sim.time:6.2f}\n"
            f"  Active Bodies: {sim.active_count:4d} / {sim.total_count:4d}\n"
            f"  Swallowed:     {sim.swallowed_count:4d}\n"
            f"  BH Mass:       {sim.black_hole_mass:8.2f} M☉\n"
            f"  Schwarzschild: {current_rs:6.3f} rs\n"
            f"  Octree Nodes:  {sim.octree_node_count:4d}\n"
            f"  Kinetic E:     {sim.kinetic_energy:8.1f}\n"
            f"  Render FPS:    {fps:5.1f}"
        )
        hud_text.set_text(hud_str)

        if frame % 25 == 0 or frame == args.frames - 1:
            elapsed = time.time() - t0
            print(f"Frame {frame:4d}/{args.frames} | Active: {sim.active_count} | Swallowed: {sim.swallowed_count} | BH Mass: {sim.black_hole_mass:.1f} | Octree Nodes: {sim.octree_node_count} | Elapsed: {elapsed:.1f}s")

        return [scat3d, hud_text]

    anim = FuncAnimation(fig, update, frames=args.frames, interval=args.interval, blit=False)

    if args.save:
        save_path = os.path.abspath(args.save)
        print(f"\nRendering animation to: {save_path} ...")
        t_save_start = time.time()

        if save_path.endswith(".mp4"):
            writer = FFMpegWriter(fps=args.fps, metadata=dict(artist="Antigravity"), bitrate=3000)
            anim.save(save_path, writer=writer, dpi=args.dpi)
        elif save_path.endswith(".gif"):
            writer = PillowWriter(fps=min(args.fps, 25))
            anim.save(save_path, writer=writer, dpi=args.dpi)
        else:
            print(f"Unknown extension for {save_path}, defaulting to mp4")
            writer = FFMpegWriter(fps=args.fps, metadata=dict(artist="Antigravity"), bitrate=3000)
            anim.save(save_path + ".mp4", writer=writer, dpi=args.dpi)

        print(f"Animation successfully saved in {time.time() - t_save_start:.1f} seconds!")
    else:
        plt.tight_layout()
        print("\nDisplaying interactive window. Close window to exit.")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Simulate Black Hole N-body dynamics with Barnes-Hut Octree in C++/pybind11")
    parser.add_argument("--scenario", type=str, default="disk", choices=["disk", "tidal_disruption", "collision"],
                        help="Simulation scenario: 'disk' (accretion disk), 'tidal_disruption', or 'collision'")
    parser.add_argument("--n-bodies", type=int, default=1200, help="Number of orbiting bodies")
    parser.add_argument("--bh-mass", type=float, default=1500.0, help="Mass of central supermassive black hole")
    parser.add_argument("--c", type=float, default=60.0, help="Speed of light parameter (controls rs = 2GM/c^2)")
    parser.add_argument("--theta", type=float, default=0.6, help="Barnes-Hut multipole acceptance criterion (MAC)")
    parser.add_argument("--softening", type=float, default=0.08, help="Gravitational Plummer softening length")
    parser.add_argument("--frames", type=int, default=200, help="Number of animation frames")
    parser.add_argument("--substeps", type=int, default=4, help="Simulation sub-steps per animation frame")
    parser.add_argument("--dt", type=float, default=0.015, help="Simulation timestep dt")
    parser.add_argument("--limit", type=float, default=25.0, help="Coordinate axis limit")
    parser.add_argument("--view", type=str, default="multi", choices=["3d", "multi"],
                        help="Viewport mode: '3d' or 'multi' (3D + 2D top-down view)")
    parser.add_argument("--show-octree", action="store_true", help="Visualize 3D Barnes-Hut octree bounding boxes")
    parser.add_argument("--max-octree-boxes", type=int, default=200, help="Max octree wireframes to draw")
    parser.add_argument("--rotate", action="store_true", default=True, help="Slowly rotate 3D camera")
    parser.add_argument("--no-rotate", action="store_false", dest="rotate", help="Disable camera rotation")
    parser.add_argument("--interval", type=int, default=25, help="Frame delay in milliseconds")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for saved video")
    parser.add_argument("--dpi", type=int, default=110, help="Resolution DPI for video export")
    parser.add_argument("--save", type=str, default=None, help="Save animation to file (.mp4 or .gif)")
    parser.add_argument("--headless", action="store_true", help="Run without GUI display")

    args = parser.parse_args()
    run_animation(args)


if __name__ == "__main__":
    main()
