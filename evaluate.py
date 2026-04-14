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
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from stable_baselines3 import PPO

from envs import DroneRacingEnv


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

    # ── Create GUI environment ────────────────────────────────────────
    env = DroneRacingEnv(
        gui    = True,
        record = args.record,
    )

    # ── Load trained policy ───────────────────────────────────────────
    model = PPO.load(args.model, env=env, device="cpu")
    print(f"[evaluate] Policy loaded from: {args.model}\n")

    step_sleep = 1.0 / args.render_fps

    all_rewards:      list[float] = []
    all_gates_passed: list[int]   = []
    laps_completed    = 0

    # ── Episode loop ──────────────────────────────────────────────────
    for ep in range(1, args.episodes + 1):
        obs, _info = env.reset()
        stats      = EpisodeStats(ep_num=ep)
        done       = False

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            stats.update(reward, info, terminated, truncated)
            done = terminated or truncated
            time.sleep(step_sleep)

        stats.print_summary(ctrl_freq=env.CTRL_FREQ)
        all_rewards.append(stats.total_reward)
        all_gates_passed.append(stats.gates_passed)
        if stats.lap_complete:
            laps_completed += 1

    # ── Aggregate summary ─────────────────────────────────────────────
    n = args.episodes
    print(f"\n{'='*60}")
    print(f"  Summary over {n} episodes:")
    print(f"    Mean reward        : {np.mean(all_rewards):.1f} ± {np.std(all_rewards):.1f}")
    print(f"    Mean gates passed  : {np.mean(all_gates_passed):.2f} / 5")
    print(f"    Laps completed     : {laps_completed} / {n}  ({100*laps_completed/n:.0f}%)")
    print(f"{'='*60}\n")

    env.close()


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
