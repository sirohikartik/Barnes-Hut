#!/usr/bin/env python3
"""
Generate an aesthetic, high-taste, scientific research thumbnail
for the Barnes-Hut / Relativistic Diffusion Dynamics paper.
Outputs to diffusion_physics.png (1200x1200).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.patheffects as pe

def main():
    fig = plt.figure(figsize=(10, 10), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#000000")
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.set_aspect("equal")
    ax.axis("off")

    # 1. Subtle 3D-like Octree Wireframe Grid (Faint dark lines)
    grid_lines = [-10, -6, -2, 2, 6, 10]
    for g in grid_lines:
        ax.axvline(g, color="#121312", linewidth=0.7, linestyle="-", alpha=0.7)
        ax.axhline(g, color="#121312", linewidth=0.7, linestyle="-", alpha=0.7)

    # Sub-octree boxes near center
    for bx in [-4, 0]:
        for by in [-4, 0]:
            rect = Rectangle((bx, by), 4, 4, fill=False, edgecolor="#1a1c1a", linewidth=0.9, linestyle="--", alpha=0.6)
            ax.add_patch(rect)

    # 2. Coordinate axes / crosshair ticks
    ax.plot([-11.5, 11.5], [0, 0], color="#202220", linewidth=0.8, alpha=0.8)
    ax.plot([0, 0], [-11.5, 11.5], color="#202220", linewidth=0.8, alpha=0.8)

    for tick in [-8, -4, 4, 8]:
        ax.plot([tick, tick], [-0.3, 0.3], color="#333633", linewidth=0.8)
        ax.plot([-0.3, 0.3], [tick, tick], color="#333633", linewidth=0.8)

    # 3. Relativistic landmarks
    rs = 0.8
    r_isco = 3.0 * rs  # 2.4
    r_init = 4.5 * rs  # 3.6

    # ISCO ring (dashed with subtle glow)
    isco_ring = Circle((0, 0), r_isco, fill=False, edgecolor="#4E5049", linewidth=1.2, linestyle="--", alpha=0.8)
    ax.add_patch(isco_ring)

    # Photon sphere ring (1.5 rs)
    r_photon = 1.5 * rs
    photon_ring = Circle((0, 0), r_photon, fill=False, edgecolor="#6BAEEF", linewidth=1.8, alpha=0.9,
                         path_effects=[pe.Stroke(linewidth=4, foreground=(0.42, 0.68, 0.94, 0.3)), pe.Normal()])
    ax.add_patch(photon_ring)

    # Outer initial orbital boundary (r = 4.5 rs)
    init_ring = Circle((0, 0), r_init, fill=False, edgecolor="#252723", linewidth=0.8, linestyle=":")
    ax.add_patch(init_ring)

    # 4. Numerical Integration of Paczyński-Wiita Precessing Rosette Orbit (Ground Truth)
    dt = 0.003
    steps = 2200
    r0 = 4.8
    v0 = 0.94  # Slight eccentricity to create precessing rosette
    pos = np.array([r0, 0.0])
    vel = np.array([0.0, v0])

    orbit_x = [pos[0]]
    orbit_y = [pos[1]]

    for _ in range(steps):
        r = np.linalg.norm(pos)
        if r <= rs:
            break
        # PW acceleration: a = - GM / (r - rs)^2 * (r / |r|)
        acc = - (15.0 / ((r - rs)**2)) * (pos / r)
        pos = pos + vel * dt + 0.5 * acc * dt**2
        r_new = np.linalg.norm(pos)
        if r_new <= rs:
            break
        acc_new = - (15.0 / ((r_new - rs)**2)) * (pos / r_new)
        vel = vel + 0.5 * (acc + acc_new) * dt
        orbit_x.append(pos[0])
        orbit_y.append(pos[1])

    orbit_x = np.array(orbit_x)
    orbit_y = np.array(orbit_y)

    # Plot Rosette Ground Truth Orbit with gradient alpha
    n_pts = len(orbit_x)
    for seg in range(0, n_pts - 20, 20):
        frac = seg / n_pts
        alpha = 0.15 + 0.8 * frac
        color = "#6BAEEF" if frac > 0.4 else "#3A5C80"
        lw = 0.8 + 1.2 * frac
        ax.plot(orbit_x[seg:seg+22], orbit_y[seg:seg+22], color=color, linewidth=lw, alpha=alpha)

    # Discrete orbital bodies along the rosette
    sample_indices = [300, 750, 1200, 1650, 2100]
    for s_idx in sample_indices:
        if s_idx < n_pts:
            px, py = orbit_x[s_idx], orbit_y[s_idx]
            ax.plot(px, py, "o", color="#EEEEEE", markersize=3.5,
                    path_effects=[pe.Stroke(linewidth=6, foreground=(0.42, 0.68, 0.94, 0.3)), pe.Normal()])

    # 5. Diverging Diffusion Trajectory (The 8.21x Displacement Inflation Scattering)
    diverge_start = 700
    div_pos = np.array([orbit_x[diverge_start], orbit_y[diverge_start]])
    div_vel = np.array([orbit_x[diverge_start] - orbit_x[diverge_start-5],
                        orbit_y[diverge_start] - orbit_y[diverge_start-5]]) / (5 * dt)

    div_x = [div_pos[0]]
    div_y = [div_pos[1]]

    for step in range(35):
        # Step inflation pushes displacement outward
        inflation = 1.0 + step * 0.18
        rand_kick = np.array([np.cos(step*1.3), np.sin(step*1.7)]) * 0.35
        div_pos = div_pos + (div_vel * 0.08 + rand_kick) * inflation
        div_x.append(div_pos[0])
        div_y.append(div_pos[1])
        if np.linalg.norm(div_pos) > 11.5:
            break

    # Plot diverging path in amber
    ax.plot(div_x, div_y, color="#E5A93C", linewidth=1.8, linestyle="--", alpha=0.9,
            path_effects=[pe.Stroke(linewidth=4, foreground=(0.90, 0.66, 0.24, 0.25)), pe.Normal()])

    # Markers along diverging path showing step displacement jumps
    for dx, dy in zip(div_x[::4], div_y[::4]):
        ax.plot(dx, dy, "s", color="#FFAA33", markersize=3, alpha=0.85)

    # Diverging label
    if len(div_x) > 12:
        ax.text(div_x[10] + 0.3, div_y[10] - 0.5, "autoregressive\nmanifold drift",
                color="#E5A93C", fontsize=8.5, fontfamily="monospace", alpha=0.95,
                bbox=dict(boxstyle="square,pad=0.2", facecolor="#000000", edgecolor="none", alpha=0.8))

    # 6. Central Black Hole Event Horizon Disk (Pitch Black Core)
    bh_core = Circle((0, 0), rs, facecolor="#000000", edgecolor="#FFFFFF", linewidth=1.5, zorder=10)
    ax.add_patch(bh_core)

    # 7. Scientific Monospace Technical Typography & Badges
    font_mono = "monospace"

    # Top Left: System Domain
    ax.text(-11.2, 10.7, "[PHYSICS / N-BODY SURROGATE]", color="#6BAEEF", fontsize=9.5, fontweight="bold", fontfamily=font_mono)
    ax.text(-11.2, 9.8, "Paczy\u0144ski\u2013Wiita Field  \u00b7  Velocity-Verlet", color="#9B9D94", fontsize=8.5, fontfamily=font_mono)

    # Top Right: Complexity
    ax.text(11.2, 10.7, "O(N log N)", color="#EEEEEE", fontsize=10, fontweight="bold", fontfamily=font_mono, ha="right")
    ax.text(11.2, 9.8, "Barnes-Hut Octree", color="#9B9D94", fontsize=8.5, fontfamily=font_mono, ha="right")

    # Bottom Left: Relativistic Landmarks
    ax.text(-11.2, -10.0, "r_s = 2GM/c\u00b2  \u00b7  r_ISCO = 3.0", color="#9B9D94", fontsize=8.5, fontfamily=font_mono)
    ax.text(-11.2, -10.9, "\u03a6_PW(r) = -GM / (r - r_s)", color="#4E5049", fontsize=8, fontfamily=font_mono)

    # Bottom Right: Key Finding (Norm Inflation)
    ax.text(11.2, -10.0, "||\u0394x\u0302|| / ||\u0394x*|| = 8.21\u00d7", color="#E5A93C", fontsize=9, fontweight="bold", fontfamily=font_mono, ha="right")
    ax.text(11.2, -10.9, "DDIM \u2192 7.15\u00d7  \u00b7  MLP \u2192 1.02\u00d7", color="#9B9D94", fontsize=8, fontfamily=font_mono, ha="right")

    # Rosette label near stable orbit
    ax.text(orbit_x[200] + 0.3, orbit_y[200] + 0.3, "precessing geodesic", color="#6BAEEF", fontsize=8, fontfamily=font_mono, alpha=0.85)

    # Save high-res PNG to both sirohikartik.github.io and black_hole
    out_paths = [
        "/Users/kartiksirohi/sirohikartik.github.io/diffusion_physics.png",
        "/Users/kartiksirohi/black_hole/diffusion_physics.png"
    ]
    for p in out_paths:
        plt.savefig(p, dpi=120, facecolor="#000000", edgecolor="none")
        print(f"Saved thumbnail to {p}")
    plt.close()

if __name__ == "__main__":
    main()
