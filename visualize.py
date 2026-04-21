"""
visualize.py — Racecourse map viewer.

Modes
-----
  python visualize.py                      # matplotlib: top-down + 3D chart
  python visualize.py --map eval           # same, but for the eval course
  python visualize.py --pybullet           # opens PyBullet GUI (train map)
  python visualize.py --pybullet --map eval  # opens PyBullet GUI (eval map)

matplotlib is the only extra dependency for the default mode.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


# ── Gate data (imported so it stays in sync with the actual env) ──────────────
sys.path.insert(0, ".")
from envs.gate_manager import MAPS, HALF_OPEN_W, HALF_OPEN_H


# ── Shared geometry helpers ───────────────────────────────────────────────────

def gate_corners_3d(gate) -> np.ndarray:
    """Return the 4 corners of a gate's inner opening in world space (4×3)."""
    pos   = gate.position
    right = gate.right                      # horizontal axis in gate plane
    up    = np.array([0.0, 0.0, 1.0])      # always world-Z
    tl = pos + HALF_OPEN_W * right + HALF_OPEN_H * up
    tr = pos - HALF_OPEN_W * right + HALF_OPEN_H * up
    br = pos - HALF_OPEN_W * right - HALF_OPEN_H * up
    bl = pos + HALF_OPEN_W * right - HALF_OPEN_H * up
    return np.array([tl, tr, br, bl])


# ══════════════════════════════════════════════════════════════════════════════
# matplotlib visualisation
# ══════════════════════════════════════════════════════════════════════════════

def show_matplotlib(gates: list, map_name: str) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    START = np.array([0.0, 0.0, 0.30])

    GATE_COLOR  = "#FF7300"
    PATH_COLOR  = "#1E88E5"
    START_COLOR = "#43A047"
    NORM_COLOR  = "#E53935"

    fig = plt.figure(figsize=(14, 6), facecolor="#1a1a2e")
    fig.suptitle(f"Drone Racing — Course Map  [{map_name}]",
                 color="white", fontsize=14, y=0.97)

    # ── Left: top-down (XY) ───────────────────────────────────────────────
    ax2d = fig.add_subplot(1, 2, 1, facecolor="#16213e")
    ax2d.set_title("Top-down view (XY)", color="white", pad=8)
    ax2d.tick_params(colors="white")
    ax2d.xaxis.label.set_color("white")
    ax2d.yaxis.label.set_color("white")
    for spine in ax2d.spines.values():
        spine.set_edgecolor("#444")
    ax2d.set_xlabel("X (m)")
    ax2d.set_ylabel("Y (m)")
    ax2d.set_aspect("equal")
    ax2d.grid(True, color="#2a2a4a", linewidth=0.5)

    # Flight path: start → G1 → … → GN → G1 (closed loop)
    path_pts = [START[:2]] + [g.position[:2] for g in gates] + [gates[0].position[:2]]
    xs, ys   = zip(*path_pts)
    ax2d.plot(xs, ys, "--", color=PATH_COLOR, linewidth=1.2,
              alpha=0.6, zorder=1, label="Flight path")

    # Start marker
    ax2d.scatter(*START[:2], s=120, color=START_COLOR, zorder=5,
                 marker="^", label="Start")
    ax2d.annotate("Start", START[:2], textcoords="offset points",
                  xytext=(6, 4), color=START_COLOR, fontsize=8)

    # Gates
    for gate in gates:
        cx, cy = gate.position[:2]
        right  = gate.right[:2]
        normal = gate.normal[:2]

        # Gate bar (projected inner opening)
        p1 = gate.position[:2] + HALF_OPEN_W * right
        p2 = gate.position[:2] - HALF_OPEN_W * right
        ax2d.plot([p1[0], p2[0]], [p1[1], p2[1]],
                  color=GATE_COLOR, linewidth=3, zorder=3, solid_capstyle="round")

        # Normal arrow (exit direction)
        ax2d.annotate(
            "", xy=(cx + 0.9 * normal[0], cy + 0.9 * normal[1]),
            xytext=(cx, cy),
            arrowprops=dict(arrowstyle="-|>", color=NORM_COLOR,
                            lw=1.4, mutation_scale=12),
            zorder=4,
        )

        # Label
        offset = -normal * 0.55
        ax2d.text(cx + offset[0], cy + offset[1], gate.label,
                  color="white", fontsize=9, fontweight="bold",
                  ha="center", va="center", zorder=5,
                  bbox=dict(boxstyle="round,pad=0.2", fc="#1a1a2e", ec=GATE_COLOR, lw=1))

    # World bounds overlay
    from envs.reward import WORLD_BOUNDS
    bx = [WORLD_BOUNDS[0], WORLD_BOUNDS[3], WORLD_BOUNDS[3],
          WORLD_BOUNDS[0], WORLD_BOUNDS[0]]
    by = [WORLD_BOUNDS[1], WORLD_BOUNDS[1], WORLD_BOUNDS[4],
          WORLD_BOUNDS[4], WORLD_BOUNDS[1]]
    ax2d.plot(bx, by, ":", color="#555", linewidth=0.8, label="World bounds")

    ax2d.legend(facecolor="#1a1a2e", edgecolor="#444",
                labelcolor="white", fontsize=8, loc="lower right")

    # ── Right: 3D perspective ─────────────────────────────────────────────
    ax3d = fig.add_subplot(1, 2, 2, projection="3d", facecolor="#16213e")
    ax3d.set_title("3D view", color="white", pad=8)
    ax3d.tick_params(colors="white")
    ax3d.set_xlabel("X", color="white", labelpad=4)
    ax3d.set_ylabel("Y", color="white", labelpad=4)
    ax3d.set_zlabel("Z", color="white", labelpad=4)
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    for axis in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
        axis.pane.set_edgecolor("#2a2a4a")

    # Axis limits derived from gate positions
    all_pos = np.array([g.position for g in gates])
    pad_xy, pad_z = 1.5, 0.5
    x_lo = min(all_pos[:, 0].min() - pad_xy, START[0] - pad_xy)
    x_hi = all_pos[:, 0].max() + pad_xy
    y_lo = min(all_pos[:, 1].min() - pad_xy, START[1] - pad_xy)
    y_hi = all_pos[:, 1].max() + pad_xy
    z_hi = all_pos[:, 2].max() + pad_z + HALF_OPEN_H

    # Ground plane grid
    gx = np.linspace(x_lo, x_hi, 6)
    gy = np.linspace(y_lo, y_hi, 6)
    for xv in gx:
        ax3d.plot([xv, xv], [gy[0], gy[-1]], [0, 0], color="#2a2a4a", lw=0.5)
    for yv in gy:
        ax3d.plot([gx[0], gx[-1]], [yv, yv], [0, 0], color="#2a2a4a", lw=0.5)

    # Flight path (closed loop)
    path_3d = np.array([START] + [g.position for g in gates] + [gates[0].position])
    ax3d.plot(path_3d[:, 0], path_3d[:, 1], path_3d[:, 2],
              "--", color=PATH_COLOR, linewidth=1.2, alpha=0.7)

    # Start marker
    ax3d.scatter(*START, s=80, color=START_COLOR, marker="^", zorder=5)

    # Gate frames as filled quads
    for gate in gates:
        corners = gate_corners_3d(gate)
        poly = Poly3DCollection(
            [corners],
            alpha=0.25, facecolor=GATE_COLOR, edgecolor=GATE_COLOR, linewidth=1.5,
        )
        ax3d.add_collection3d(poly)

        lx, ly, lz = gate.position
        ax3d.text(lx, ly, lz + HALF_OPEN_H + 0.25, gate.label,
                  color="white", fontsize=8, fontweight="bold", ha="center")

        n = gate.normal * 0.8
        ax3d.quiver(lx, ly, lz, n[0], n[1], n[2],
                    color=NORM_COLOR, linewidth=1.2, arrow_length_ratio=0.3)

    ax3d.set_xlim(x_lo, x_hi)
    ax3d.set_ylim(y_lo, y_hi)
    ax3d.set_zlim(0, z_hi)
    ax3d.view_init(elev=28, azim=-55)

    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PyBullet live visualisation
# ══════════════════════════════════════════════════════════════════════════════

def show_pybullet(gates: list, map_name: str) -> None:
    import time
    from envs.drone_racing_env import DroneRacingEnv
    import pybullet as p

    print(f"Opening PyBullet GUI  [{map_name}]…  Press Ctrl-C to quit.")
    env = DroneRacingEnv(gui=True, map_name=map_name)
    env.reset()
    client = env.CLIENT

    # ── Camera: bird's-eye with slight angle ─────────────────────────────
    all_pos  = np.array([g.position for g in gates])
    cam_target = (all_pos.max(axis=0) + all_pos.min(axis=0)) / 2
    p.resetDebugVisualizerCamera(
        cameraDistance       = 12.0,
        cameraYaw            = 30,
        cameraPitch          = -50,
        cameraTargetPosition = cam_target.tolist(),
        physicsClientId      = client,
    )

    # ── Draw flight path (coloured segments between gates) ───────────────
    START     = [0.0, 0.0, 0.30]
    waypoints = [START] + [g.position.tolist() for g in gates] + [gates[0].position.tolist()]
    colours   = [
        [0.27, 0.53, 0.90],   # blue
        [0.27, 0.90, 0.53],   # green
        [0.90, 0.80, 0.27],   # yellow
        [0.90, 0.53, 0.27],   # orange
        [0.90, 0.27, 0.53],   # pink
        [0.60, 0.27, 0.90],   # purple (loop-back)
    ]
    for i in range(len(waypoints) - 1):
        p.addUserDebugLine(
            lineFromXYZ    = waypoints[i],
            lineToXYZ      = waypoints[i + 1],
            lineColorRGB   = colours[i % len(colours)],
            lineWidth      = 2.0,
            physicsClientId = client,
        )

    # ── Gate labels and normal arrows ────────────────────────────────────
    for gate in gates:
        pos = gate.position.tolist()

        p.addUserDebugText(
            text            = gate.label,
            textPosition    = [pos[0], pos[1], pos[2] + HALF_OPEN_H + 0.35],
            textColorRGB    = [1.0, 0.45, 0.0],
            textSize        = 1.5,
            physicsClientId = client,
        )

        tip = [
            pos[0] + gate.normal[0] * 1.0,
            pos[1] + gate.normal[1] * 1.0,
            pos[2],
        ]
        p.addUserDebugLine(
            lineFromXYZ    = pos,
            lineToXYZ      = tip,
            lineColorRGB   = [0.9, 0.16, 0.22],
            lineWidth      = 2.5,
            physicsClientId = client,
        )

    # ── Start marker ─────────────────────────────────────────────────────
    p.addUserDebugText(
        text            = "START",
        textPosition    = [START[0], START[1], START[2] + 0.3],
        textColorRGB    = [0.26, 0.63, 0.28],
        textSize        = 1.2,
        physicsClientId = client,
    )

    # ── Keep window open ─────────────────────────────────────────────────
    try:
        while True:
            p.stepSimulation(physicsClientId=client)
            import time as _t; _t.sleep(1 / 60)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        print("Done.")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise the drone racing course")
    parser.add_argument(
        "--pybullet",
        action  = "store_true",
        default = False,
        help    = "Open the PyBullet GUI instead of a matplotlib chart",
    )
    parser.add_argument(
        "--map",
        type    = str,
        default = "train",
        choices = list(MAPS),
        help    = "Which course to visualise: train | eval  (default: %(default)s)",
    )
    args = parser.parse_args()

    gates = MAPS[args.map]

    if args.pybullet:
        show_pybullet(gates, args.map)
    else:
        show_matplotlib(gates, args.map)
