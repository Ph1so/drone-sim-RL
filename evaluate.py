"""
evaluate.py — Run a trained PPO policy in the GUI-enabled simulator.

Usage
-----
  python evaluate.py --model ./best_model/best_model.zip
  python evaluate.py --model ./checkpoints/drone_racing_ppo_final.zip \\
                     --episodes 10 --record

The script prints per-episode statistics (total reward, gates passed,
lap time if completed) and optionally renders to video via PyBullet's
built-in recorder.

Notes
-----
  - Checkpoints saved on a different Python version (e.g. Colab Python 3.12
    vs local Python 3.10) may fail to deserialise policy_kwargs.  This script
    passes custom_objects so the correct architecture is always used.
  - The model is loaded without passing env to PPO.load so SB3 does not wrap
    the env in VecTransposeImage (which would double-transpose the image obs).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pybullet as p
import torch.nn as nn
from PIL import Image
from stable_baselines3 import PPO

from envs import DroneRacingEnv
from envs.reward import WORLD_BOUNDS
from train import GateObsExtractor, FEATURES_DIM

# ── Hyper-parameters ──────────────────────────────────────────────────────────

EVAL_DIR = "./eval_trajectories"

# ── Reward breakdown keys (must match reward.py) ──────────────────────────────

_BREAKDOWN_KEYS = [
    ("r_prog",       "progress"),
    ("r_perc",       "perception"),
    ("r_jerk",       "jerk"),
    ("r_body_rate",  "body rate"),
    ("r_gate_bonus", "gate bonus"),
    ("r_collision",  "collision"),
    ("r_oob",        "oob"),
]


# ── Camera control ────────────────────────────────────────────────────────────

def _handle_camera(client: int, pov_mode: bool) -> bool:
    """Adjust the PyBullet debug camera based on keyboard input.

    Controls
    --------
    V            — toggle drone-POV / free-camera mode
    A / D        — orbit yaw left / right   (free-camera only)
    W / S        — tilt pitch up / down     (free-camera only)
    Q / E        — zoom in / out            (free-camera only)
    Arrow keys   — pan the camera target    (free-camera only)

    Returns the (possibly toggled) pov_mode flag.
    """
    keys = p.getKeyboardEvents(physicsClientId=client)
    if not keys:
        return pov_mode

    # V key toggles POV mode (triggered = fires once per press, not per frame)
    if keys.get(ord('v'), 0) & p.KEY_WAS_TRIGGERED:
        pov_mode = not pov_mode
        print(f"[camera] {'Drone POV' if pov_mode else 'Free camera'} mode")
        return pov_mode

    if pov_mode:
        return pov_mode   # orbit controls are inactive in POV mode

    cam    = p.getDebugVisualizerCamera(physicsClientId=client)
    dist   = cam[10]
    yaw    = cam[8]
    pitch  = cam[9]
    target = list(cam[11])

    YAW_STEP   = 2.0    # degrees per frame
    PITCH_STEP = 1.5
    ZOOM_STEP  = 0.15
    PAN_STEP   = 0.06

    DOWN = p.KEY_IS_DOWN

    if keys.get(ord('a'), 0) & DOWN: yaw   -= YAW_STEP
    if keys.get(ord('d'), 0) & DOWN: yaw   += YAW_STEP
    if keys.get(ord('w'), 0) & DOWN: pitch  = min(pitch + PITCH_STEP, 89)
    if keys.get(ord('s'), 0) & DOWN: pitch  = max(pitch - PITCH_STEP, -89)
    if keys.get(ord('q'), 0) & DOWN: dist   = max(0.1, dist - ZOOM_STEP)
    if keys.get(ord('e'), 0) & DOWN: dist  += ZOOM_STEP

    # Arrow keys pan the target in the horizontal plane
    yaw_rad = np.radians(yaw)
    fwd  = np.array([ np.cos(yaw_rad), np.sin(yaw_rad), 0.0])
    left = np.array([-np.sin(yaw_rad), np.cos(yaw_rad), 0.0])
    if keys.get(p.B3G_UP_ARROW,    0) & DOWN: target += PAN_STEP * fwd
    if keys.get(p.B3G_DOWN_ARROW,  0) & DOWN: target -= PAN_STEP * fwd
    if keys.get(p.B3G_LEFT_ARROW,  0) & DOWN: target += PAN_STEP * left
    if keys.get(p.B3G_RIGHT_ARROW, 0) & DOWN: target -= PAN_STEP * left

    p.resetDebugVisualizerCamera(
        cameraDistance       = dist,
        cameraYaw            = yaw,
        cameraPitch          = pitch,
        cameraTargetPosition = target,
        physicsClientId      = client,
    )
    return pov_mode


def _update_drone_pov_camera(env, client: int) -> None:
    """Reposition the debug visualizer to a first-person view from the drone.

    Places the camera at the drone's centre of mass looking along its yaw
    heading.  Drone pitch/roll are intentionally ignored so the view stays
    level and easy to read during aggressive manoeuvres.

    PyBullet's resetDebugVisualizerCamera orbits the camera around a target
    point at a given distance.  Setting the target 0.5 m ahead of the drone
    and the distance to 0.5 m places the camera eye exactly at the drone.
    """
    state = env._getDroneStateVector(0)
    pos   = state[0:3]
    yaw   = float(state[9])          # rpy[2] — drone heading in radians

    _POV_DIST = 0.5                  # metres — target offset and camera distance
    fwd    = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    target = (pos + fwd * _POV_DIST).tolist()

    # PyBullet places the camera AT azimuth cameraYaw relative to the target,
    # so the look direction is -(cam_pos − target), i.e. opposite to cameraYaw.
    # Adding 180° flips the azimuth so the camera looks in the drone's forward
    # direction while remaining positioned at the drone's centre of mass.
    p.resetDebugVisualizerCamera(
        cameraDistance       = _POV_DIST,
        cameraYaw            = float(np.degrees(yaw) + 180),
        cameraPitch          = 0.0,
        cameraTargetPosition = target,
        physicsClientId      = client,
    )


# ── OOB wireframe ────────────────────────────────────────────────────────────

def _draw_oob_wireframe(client: int) -> None:
    """Draw the 12 edges of the out-of-bounds WORLD_BOUNDS box as red debug lines."""
    x0, y0, z0 = WORLD_BOUNDS[0], WORLD_BOUNDS[1], WORLD_BOUNDS[2]
    x1, y1, z1 = WORLD_BOUNDS[3], WORLD_BOUNDS[4], WORLD_BOUNDS[5]

    corners = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),  # bottom
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),  # top
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),   # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),   # top face
        (0, 4), (1, 5), (2, 6), (3, 7),   # verticals
    ]
    color = [1.0, 0.2, 0.0]   # orange-red
    for a, b in edges:
        p.addUserDebugLine(
            lineFromXYZ    = corners[a],
            lineToXYZ      = corners[b],
            lineColorRGB   = color,
            lineWidth      = 2,
            lifeTime       = 0,             # 0 = persist until removed
            physicsClientId = client,
        )


# ── Trajectory diagnostic plot ───────────────────────────────────────────────

def _save_traj_plot(
    traj:      dict,
    ep_num:    int,
    gates:     list,    # List[Gate] snapshot for this episode (offset already applied)
    num_gates: int,
) -> None:
    """
    Save a 4-panel diagnostic PNG for one episode.

    Panel 1 — Bird's-eye XY trajectory
        Flight path coloured by which gate is being targeted.
        Gate openings (thick bars) and exit-normal arrows drawn.
        Flip location marked with a red ×.

    Panel 2 — Speed (‖v‖) over time
        Total speed in m/s with gate passage markers.

    Panel 3 — Roll & Pitch over time
        Both in degrees, with ±FLIP_THRESHOLD dashed lines and
        vertical markers where each gate was passed.

    Panel 4 — Angular velocity magnitude ‖ω‖ over time
        Shows whether instability builds gradually or spikes suddenly.

    Panels 2–4 share the same step axis so they can be read in sync.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")          # non-interactive — works headless
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed — pip install matplotlib")
        return

    from envs.gate_manager import HALF_OPEN_W
    from envs.reward import RewardComputer

    pos      = np.array(traj["pos"],        dtype=float)   # (N, 3)
    rpy      = np.array(traj["rpy"],        dtype=float)   # (N, 3) radians
    lin_vel  = np.array(traj["lin_vel"],    dtype=float)   # (N, 3)
    ang_sq   = np.array(traj["ang_vel_sq"], dtype=float)   # (N,)
    g_idx    = np.array(traj["gate_idx"],   dtype=int)     # (N,)
    steps    = np.arange(len(pos))
    speed    = np.linalg.norm(lin_vel, axis=1)             # (N,) m/s

    flip_thresh_deg = np.rad2deg(RewardComputer.FLIP_THRESHOLD)

    # Steps where the target gate advanced (gate passage events)
    gate_pass_steps = list(np.where(np.diff(g_idx) > 0)[0])
    # First step where the flip penalty fired (None if clean episode)
    flip_step = next((i for i, f in enumerate(traj["flip_fired"]) if f), None)

    seg_colors = ["tab:cyan", "tab:green", "tab:orange", "tab:red", "tab:purple", "tab:brown"]

    fig, axes = plt.subplots(4, 1, figsize=(10, 17))
    fig.suptitle(f"Episode {ep_num} — Diagnostic Trajectory", fontsize=13, fontweight="bold")

    # ── Panel 1: Bird's-eye XY ──────────────────────────────────────────
    ax = axes[0]
    for seg in range(num_gates + 1):
        mask = g_idx == seg
        if not np.any(mask):
            continue
        color = seg_colors[seg % len(seg_colors)]
        label = (f"→ G{seg + 1}" if seg < num_gates else "post-last")
        ax.plot(pos[mask, 0], pos[mask, 1], color=color, lw=1.8, label=label)
    ax.plot(pos[0, 0], pos[0, 1], "g^", ms=8, zorder=5, label="spawn")
    spawn_yaw = rpy[0, 2]   # yaw in radians at first recorded step
    _ARROW_LEN = 0.4
    ax.annotate(
        "",
        xy=(pos[0, 0] + np.cos(spawn_yaw) * _ARROW_LEN,
            pos[0, 1] + np.sin(spawn_yaw) * _ARROW_LEN),
        xytext=(pos[0, 0], pos[0, 1]),
        arrowprops=dict(arrowstyle="->", color="green", lw=2.0),
        zorder=6,
    )
    if flip_step is not None:
        ax.plot(pos[flip_step, 0], pos[flip_step, 1],
                "rx", ms=14, mew=2.5, zorder=6, label=f"flip (step {flip_step})")
    for gate in gates:
        gx, gy = gate.position[0], gate.position[1]
        right = np.array([np.cos(gate.yaw_rad), np.sin(gate.yaw_rad)])
        p1 = np.array([gx, gy]) - right * HALF_OPEN_W
        p2 = np.array([gx, gy]) + right * HALF_OPEN_W
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k-", lw=3, zorder=4)
        ax.annotate(gate.label, (gx, gy), fontsize=9, fontweight="bold",
                    ha="center", va="bottom",
                    xytext=(0, 7), textcoords="offset points")
        ax.annotate("",
                    xy=(gx + gate.normal[0] * 0.6, gy + gate.normal[1] * 0.6),
                    xytext=(gx, gy),
                    arrowprops=dict(arrowstyle="->", color="k", lw=1.5))
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Bird's-eye trajectory — colour = current gate target")
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Speed ─────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(steps, speed, color="tab:blue", lw=1.5, label="‖v‖")
    ax.fill_between(steps, speed, alpha=0.15, color="tab:blue")
    for s in gate_pass_steps:
        ax.axvline(s, color="green", ls=":", lw=1, alpha=0.8)
    if flip_step is not None:
        ax.axvline(flip_step, color="red", lw=1.5, alpha=0.6, label="flip")
    ax.set_xlabel("step")
    ax.set_ylabel("m/s")
    ax.set_title("Speed ‖v‖  (green verticals = gate passages)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Roll & Pitch ───────────────────────────────────────────
    ax = axes[2]
    roll_deg  = np.rad2deg(rpy[:, 0])
    pitch_deg = np.rad2deg(rpy[:, 1])
    ax.plot(steps, roll_deg,  color="tab:blue",   lw=1.5, label="roll")
    ax.plot(steps, pitch_deg, color="tab:orange",  lw=1.5, label="pitch")
    ax.axhline( flip_thresh_deg, color="red", ls="--", lw=1.2,
                label=f"flip threshold ±{flip_thresh_deg:.0f}°")
    ax.axhline(-flip_thresh_deg, color="red", ls="--", lw=1.2)
    ax.axhline(0, color="gray", ls=":", lw=0.7)
    for s in gate_pass_steps:
        ax.axvline(s, color="green", ls=":", lw=1, alpha=0.8)
    if flip_step is not None:
        ax.axvline(flip_step, color="red", lw=1.5, alpha=0.6)
    ax.set_xlabel("step")
    ax.set_ylabel("degrees")
    ax.set_title("Roll & Pitch  (green verticals = gate passages, red vertical = flip)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 4: Angular velocity magnitude ────────────────────────────
    ax = axes[3]
    omega = np.sqrt(np.maximum(ang_sq, 0.0))
    ax.plot(steps, omega, color="tab:red", lw=1.5, label="‖ω‖")
    ax.fill_between(steps, omega, alpha=0.15, color="tab:red")
    for s in gate_pass_steps:
        ax.axvline(s, color="green", ls=":", lw=1, alpha=0.8)
    if flip_step is not None:
        ax.axvline(flip_step, color="red", lw=1.5, alpha=0.6, label="flip")
    ax.set_xlabel("step")
    ax.set_ylabel("rad/s")
    ax.set_title("Angular velocity magnitude ‖ω‖  (green verticals = gate passages)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"{EVAL_DIR}/eval_traj_ep{ep_num}.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Trajectory figure saved → {out_path}")


# ── Statistics helper ─────────────────────────────────────────────────────────

class EpisodeStats:
    def __init__(self, ep_num: int) -> None:
        self.ep_num           = ep_num
        self.total_reward     = 0.0
        self.steps            = 0
        self.gates_passed     = 0
        self.lap_complete     = False
        self.terminated       = False
        self.truncated        = False
        self.breakdown        = {k: 0.0 for k, _ in _BREAKDOWN_KEYS}

    def update(self, reward: float, info: dict, terminated: bool, truncated: bool) -> None:
        self.total_reward += reward
        self.steps        += 1
        self.gates_passed  = info.get("num_gates_passed", self.gates_passed)
        self.lap_complete  = info.get("lap_complete", self.lap_complete)
        self.terminated    = terminated
        self.truncated     = truncated
        for k, _ in _BREAKDOWN_KEYS:
            self.breakdown[k] += info.get(k, 0.0)

    def print_summary(self, ctrl_freq: int = 48) -> None:
        duration   = self.steps / ctrl_freq
        end_reason = (
            "LAP COMPLETE" if self.lap_complete
            else "collision/OOB" if self.terminated
            else "time limit"
        )
        print(
            f"  Ep {self.ep_num:3d} | "
            f"reward: {self.total_reward:8.1f} | "
            f"gates: {self.gates_passed}/{5} | "
            f"steps: {self.steps:5d} ({duration:.1f}s) | "
            f"{end_reason}"
        )
        print("           Reward breakdown:")
        for k, label in _BREAKDOWN_KEYS:
            v = self.breakdown[k]
            if v != 0.0:
                print(f"             {label:<18} {v:+10.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation loop
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(args: argparse.Namespace) -> None:
    print(f"\n{'='*60}")
    print(f"  DroneRacing — Policy Evaluation")
    print(f"  Model     : {args.model}")
    print(f"  Episodes  : {args.episodes}")
    print(f"  Render Hz : {args.render_fps} fps")
    print(f"{'='*60}\n")

    # custom_objects ensures the correct architecture is reconstructed even
    # when policy_kwargs can't be deserialised (e.g. cross-Python-version
    # checkpoints saved on Colab Python 3.12 and loaded on Python 3.10).
    custom_objects = {
        "policy_kwargs": dict(
            features_extractor_class  = GateObsExtractor,
            features_extractor_kwargs = {"features_dim": FEATURES_DIM},
            net_arch      = dict(pi=[256, 128], vf=[256, 128]),
            activation_fn = nn.ReLU,
        )
    }

    # Load model WITHOUT passing env so SB3 does not wrap it.
    # We use the raw DroneRacingEnv directly in the loop; the
    # MultimodalExtractor's internal permute handles HWC→CHW correctly.
    model = PPO.load(args.model, device="cpu", custom_objects=custom_objects)
    print(f"[evaluate] Policy loaded from: {args.model}\n")

    step_sleep = 1.0 / args.render_fps

    all_rewards:      list[float] = []
    all_gates_passed: list[int]   = []
    laps_completed    = 0
    gif_frames:       list        = []
    all_breakdowns:   list[dict]  = []

    # Single env for all episodes — the window stays open the whole time.
    gate_offset = args.gate_offset if args.gate_offset else None
    env = DroneRacingEnv(gui=True, record=args.record, num_gates=args.num_gates,
                         gate_pos_offset=gate_offset,
                         spawn_mid_course_prob=args.spawn_mid_course_prob)

    _draw_oob_wireframe(env.CLIENT)

    print("[camera] V — toggle drone POV / free-camera")

    pov_mode = False   # persists across episodes; toggled by V key

    # ── Episode loop ──────────────────────────────────────────────────
    for ep in range(1, args.episodes + 1):
        obs, _info = env.reset()
        stats = EpisodeStats(ep_num=ep)
        done  = False

        # Snapshot active gates after reset (gate_offset already applied).
        snap_gates = list(env._gate_manager.gates)

        # Show spawn info and mark position in the GUI.
        spawn_state = env._getDroneStateVector(0)
        spawn_pos   = spawn_state[0:3]
        gate_label  = env._gate_manager.current_gate.label if env._gate_manager.current_gate else "?"
        print(f"  [spawn] ep {ep} — pos ({spawn_pos[0]:.2f}, {spawn_pos[1]:.2f}, {spawn_pos[2]:.2f})"
              f"  targeting {gate_label}")
        # Yellow debug sphere at spawn location (radius 0.15 m, persists for 30 s).
        p.addUserDebugLine(
            lineFromXYZ     = (spawn_pos[0] - 0.15, spawn_pos[1], spawn_pos[2]),
            lineToXYZ       = (spawn_pos[0] + 0.15, spawn_pos[1], spawn_pos[2]),
            lineColorRGB    = [1.0, 0.9, 0.0],
            lineWidth       = 4,
            lifeTime        = 30,
            physicsClientId = env.CLIENT,
        )
        p.addUserDebugLine(
            lineFromXYZ     = (spawn_pos[0], spawn_pos[1] - 0.15, spawn_pos[2]),
            lineToXYZ       = (spawn_pos[0], spawn_pos[1] + 0.15, spawn_pos[2]),
            lineColorRGB    = [1.0, 0.9, 0.0],
            lineWidth       = 4,
            lifeTime        = 30,
            physicsClientId = env.CLIENT,
        )
        p.addUserDebugLine(
            lineFromXYZ     = (spawn_pos[0], spawn_pos[1], spawn_pos[2] - 0.15),
            lineToXYZ       = (spawn_pos[0], spawn_pos[1], spawn_pos[2] + 0.15),
            lineColorRGB    = [1.0, 0.9, 0.0],
            lineWidth       = 4,
            lifeTime        = 30,
            physicsClientId = env.CLIENT,
        )
        p.addUserDebugText(
            text            = f"ep{ep} → {gate_label}",
            textPosition    = (spawn_pos[0], spawn_pos[1], spawn_pos[2] + 0.25),
            textColorRGB    = [1.0, 0.9, 0.0],
            textSize        = 1.2,
            lifeTime        = 30,
            physicsClientId = env.CLIENT,
        )

        traj: dict = {
            "pos":        [],
            "rpy":        [],
            "lin_vel":    [],
            "ang_vel_sq": [],
            "gate_idx":   [],
            "flip_fired": [],
        }

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            stats.update(reward, info, terminated, truncated)
            done = terminated or truncated
            if (args.plot or args.spawn_mid_course_prob > 0) and "drone_pos" in info:
                traj["pos"].append(info["drone_pos"])
                traj["rpy"].append(info["drone_rpy"])
                traj["lin_vel"].append(info["drone_lin_vel"])
                ang_vel = info.get("drone_ang_vel", np.zeros(3))
                traj["ang_vel_sq"].append(float(np.dot(ang_vel, ang_vel)))
                traj["gate_idx"].append(info.get("current_gate_idx", 0))
                traj["flip_fired"].append(bool(info.get("flip", False)))
            if ep == 1:
                gif_frames.append(env._render_ego_camera())
            pov_mode = _handle_camera(env.CLIENT, pov_mode)
            if pov_mode:
                _update_drone_pov_camera(env, env.CLIENT)
            time.sleep(step_sleep)

        stats.print_summary(ctrl_freq=env.CTRL_FREQ)
        all_rewards.append(stats.total_reward)
        all_gates_passed.append(stats.gates_passed)
        if stats.lap_complete:
            laps_completed += 1
        all_breakdowns.append(stats.breakdown)

        if (args.plot or args.spawn_mid_course_prob > 0) and traj["pos"]:
            _save_traj_plot(traj, ep, snap_gates, args.num_gates)

        if ep == 1 and gif_frames:
            gif_path = f"{EVAL_DIR}/eval_ego.gif"
            images = [Image.fromarray(f) for f in gif_frames]
            images[0].save(
                gif_path,
                save_all      = True,
                append_images = images[1:],
                loop          = 0,
                duration      = int(1000 / args.render_fps),
            )
            print(f"[evaluate] Ego-camera GIF saved → {gif_path}")

    env.close()

    # ── Aggregate summary ─────────────────────────────────────────────
    n = args.episodes
    print(f"\n{'='*60}")
    print(f"  Summary over {n} episodes:")
    print(f"    Mean reward        : {np.mean(all_rewards):.1f} ± {np.std(all_rewards):.1f}")
    print(f"    Mean gates passed  : {np.mean(all_gates_passed):.2f} / 5")
    print(f"    Laps completed     : {laps_completed} / {n}  ({100*laps_completed/n:.0f}%)")
    print(f"  Mean reward breakdown:")
    for k, label in _BREAKDOWN_KEYS:
        mean_v = np.mean([bd[k] for bd in all_breakdowns])
        if mean_v != 0.0:
            print(f"    {label:<18} {mean_v:+10.4f}")
    print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DroneRacing PPO policy"
    )
    parser.add_argument(
        "--model",
        type    = str,
        default = "./best_model/best_model.zip",
        help    = "Path to the saved .zip policy (default: %(default)s)",
    )
    parser.add_argument(
        "--episodes",
        type    = int,
        default = 5,
        help    = "Number of evaluation episodes (default: %(default)s)",
    )
    parser.add_argument(
        "--render_fps",
        type    = int,
        default = 48,
        help    = "Rendering pace in frames-per-second (default: %(default)s)",
    )
    parser.add_argument(
        "--record",
        action  = "store_true",
        default = False,
        help    = "Ask PyBullet to record the GUI to video frames",
    )
    parser.add_argument(
        "--num_gates",
        type    = int,
        default = 5,
        help    = "Number of active gates (must match the training config, default: %(default)s)",
    )
    parser.add_argument(
        "--gate_offset",
        type    = float,
        nargs   = 3,
        default = None,
        metavar = ("DX", "DY", "DZ"),
        help    = "Shift all gates by (dx, dy, dz) metres — used to test generalisation",
    )
    parser.add_argument(
        "--spawn_mid_course_prob",
        type    = float,
        default = 0.0,
        metavar = "P",
        help    = "Probability [0, 1] of teleporting the drone to a random mid-course gate "
                  "approach at episode start (default: 0 — always spawn at start line)",
    )
    parser.add_argument(
        "--plot",
        action  = "store_true",
        default = False,
        help    = "Save a 3-panel diagnostic figure per episode: XY path, roll/pitch, ang-vel "
                  "(requires matplotlib; saved as eval_traj_ep<N>.png)",
    )
    evaluate(parser.parse_args())
