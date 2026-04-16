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
from stable_baselines3 import PPO

from envs import DroneRacingEnv
from train import GateObsExtractor, FEATURES_DIM


# ── Camera control ────────────────────────────────────────────────────────────

def _handle_camera(client: int) -> None:
    """Adjust the PyBullet debug camera based on keyboard input.

    Controls
    --------
    A / D        — orbit yaw left / right
    W / S        — tilt pitch up / down
    Q / E        — zoom in / out
    Arrow keys   — pan the camera target point
    """
    keys = p.getKeyboardEvents(physicsClientId=client)
    if not keys:
        return

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


# ── Statistics helper ─────────────────────────────────────────────────────────

class EpisodeStats:
    def __init__(self, ep_num: int) -> None:
        self.ep_num      = ep_num
        self.total_reward = 0.0
        self.steps        = 0
        self.gates_passed = 0
        self.lap_complete = False
        self.terminated   = False
        self.truncated    = False

    def update(self, reward: float, info: dict, terminated: bool, truncated: bool) -> None:
        self.total_reward += reward
        self.steps        += 1
        self.gates_passed  = info.get("num_gates_passed", self.gates_passed)
        self.lap_complete  = info.get("lap_complete", self.lap_complete)
        self.terminated    = terminated
        self.truncated     = truncated

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

    # Single env for all episodes — the window stays open the whole time.
    env = DroneRacingEnv(gui=True, record=args.record)

    # ── Episode loop ──────────────────────────────────────────────────
    for ep in range(1, args.episodes + 1):
        obs, _info = env.reset()
        stats = EpisodeStats(ep_num=ep)
        done  = False

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            stats.update(reward, info, terminated, truncated)
            done = terminated or truncated
            _handle_camera(env.CLIENT)
            time.sleep(step_sleep)

        stats.print_summary(ctrl_freq=env.CTRL_FREQ)
        all_rewards.append(stats.total_reward)
        all_gates_passed.append(stats.gates_passed)
        if stats.lap_complete:
            laps_completed += 1

    env.close()

    # ── Aggregate summary ─────────────────────────────────────────────
    n = args.episodes
    print(f"\n{'='*60}")
    print(f"  Summary over {n} episodes:")
    print(f"    Mean reward        : {np.mean(all_rewards):.1f} ± {np.std(all_rewards):.1f}")
    print(f"    Mean gates passed  : {np.mean(all_gates_passed):.2f} / 5")
    print(f"    Laps completed     : {laps_completed} / {n}  ({100*laps_completed/n:.0f}%)")
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
    evaluate(parser.parse_args())
