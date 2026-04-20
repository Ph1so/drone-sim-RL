"""
train.py — PPO training script for DroneRacingEnv.

Architecture  (matches Swift paper, Kaufmann et al. 2023)
------------
  MlpPolicy — flat 31-D observation → 2-layer MLP (128 × 128, LeakyReLU α=0.2)
  No custom feature extractor.  Policy and value heads each get a 128-unit hidden
  layer on top of the shared backbone.

Usage
-----
  python train.py                        # default settings
  python train.py --timesteps 100_000_000
  python train.py --n_envs 100           # match paper's 100 parallel agents

Checkpoints are saved to ./checkpoints/ every SAVE_FREQ steps.
TensorBoard logs go to ./logs/.
"""

from __future__ import annotations

import argparse
import functools
import os

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from envs import DroneRacingEnv
from envs.gate_manager import RACE_GATES
from envs.reward import RewardComputer


# ── Hyper-parameters ──────────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 3_000_000
N_ENVS          = os.cpu_count() or 4   # paper uses 100; scale to local hardware
N_STEPS         = 1500       # matches MAX_EPISODE_STEPS — one rollout = one episode
BATCH_SIZE      = 256
N_EPOCHS        = 10
GAMMA           = 0.99       # matches paper Extended Data Table 1a
GAE_LAMBDA      = 0.95
CLIP_RANGE      = 0.2        # ε in paper = 0.2
LR              = 3e-4       # matches paper Adam lr
SAVE_FREQ       = 50_000     # checkpoint interval (env steps)

CHECKPOINT_DIR  = "./checkpoints"
LOG_DIR         = "./logs"
BEST_MODEL_DIR  = "./best_model"

# LeakyReLU(α=0.2) as used in Swift (Kaufmann et al. 2023, Methods).
# functools.partial is picklable — safe with SubprocVecEnv.
_LeakyReLU02 = functools.partial(nn.LeakyReLU, negative_slope=0.2)


# ══════════════════════════════════════════════════════════════════════════════
# Environment factory
# ══════════════════════════════════════════════════════════════════════════════

def make_env(rank: int = 0, seed: int = 0, num_gates: int = 5, spawn_mid_course_prob: float = 0.0):
    """Return a callable that creates a single DroneRacingEnv."""
    def _init():
        env = DroneRacingEnv(gui=False, num_gates=num_gates, spawn_mid_course_prob=spawn_mid_course_prob)
        env.reset(seed=seed + rank)
        return env
    return _init


# ══════════════════════════════════════════════════════════════════════════════
# Training entry-point
# ══════════════════════════════════════════════════════════════════════════════

def main(args: argparse.Namespace) -> None:
    checkpoint_dir  = args.checkpoint_dir
    log_dir         = args.log_dir
    best_model_dir  = args.best_model_dir

    os.makedirs(checkpoint_dir,  exist_ok=True)
    os.makedirs(log_dir,         exist_ok=True)
    os.makedirs(best_model_dir,  exist_ok=True)

    n_envs = args.n_envs

    num_gates             = args.num_gates
    spawn_mid_course_prob = args.spawn_mid_course_prob

    # ── Vectorised training environments ─────────────────────────────
    if n_envs > 1:
        train_env = SubprocVecEnv(
            [make_env(rank=i, seed=args.seed, num_gates=num_gates,
                      spawn_mid_course_prob=spawn_mid_course_prob)
             for i in range(n_envs)]
        )
    else:
        from stable_baselines3.common.vec_env import DummyVecEnv
        train_env = DummyVecEnv([make_env(rank=0, seed=args.seed, num_gates=num_gates,
                                          spawn_mid_course_prob=spawn_mid_course_prob)])

    train_env = VecMonitor(train_env)

    # ── Evaluation environment — no mid-course spawns for clean metrics ──
    eval_env = DroneRacingEnv(gui=False, num_gates=num_gates, spawn_mid_course_prob=0.0)

    # ── Policy kwargs — Swift: flat 2×128 MLP with LeakyReLU(α=0.2) ────
    # net_arch=[128, 128] → shared backbone only; pi=[] vf=[] means no
    # separate heads beyond the shared layers (matches Swift's 2-layer MLP).
    policy_kwargs = dict(
        net_arch      = [128, 128],
        activation_fn = _LeakyReLU02,
    )

    # ── PPO model ─────────────────────────────────────────────────────
    if args.resume:
        print(f"[train] Resuming from checkpoint: {args.resume}")
        custom_objects = {
            "learning_rate": 5e-5,
            "policy_kwargs": dict(
                net_arch      = [128, 128],
                activation_fn = _LeakyReLU02,
            ),
        }
        model = PPO.load(
            args.resume,
            env             = train_env,
            device          = args.device,
            tensorboard_log = log_dir,
            verbose         = 1,
            custom_objects  = custom_objects,
        )
    else:
        model = PPO(
            policy          = "MlpPolicy",
            env             = train_env,
            n_steps         = N_STEPS,
            batch_size      = BATCH_SIZE,
            n_epochs        = N_EPOCHS,
            gamma           = GAMMA,
            gae_lambda      = GAE_LAMBDA,
            clip_range      = CLIP_RANGE,
            ent_coef        = 0.0,
            learning_rate   = LR,
            policy_kwargs   = policy_kwargs,
            tensorboard_log = log_dir,
            verbose         = 1,
            seed            = args.seed,
            device          = args.device,
        )

    # ── Callbacks ─────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq         = max(SAVE_FREQ // n_envs, 1),
        save_path         = checkpoint_dir,
        name_prefix       = "drone_racing_ppo",
        save_replay_buffer = False,
        save_vecnormalize = False,
        verbose           = 1,
    )

    eval_cb = EvalCallback(
        eval_env              = eval_env,
        best_model_save_path  = best_model_dir,
        log_path              = log_dir,
        eval_freq             = max(SAVE_FREQ // n_envs, 1),
        n_eval_episodes       = 5,
        deterministic         = True,
        render                = False,
        verbose               = 1,
    )

    # ── Train ─────────────────────────────────────────────────────────
    rc = RewardComputer
    lr = 5e-5 if args.resume else LR
    print(f"\n{'='*60}")
    print(f"  DroneRacing PPO  —  Swift architecture (Kaufmann et al. 2023)")
    print(f"  Policy          : MlpPolicy  2×128  LeakyReLU(α=0.2)")
    print(f"  Obs dim         : 31  (pos·vel·rotmat·gate_corners·prev_action)")
    print(f"  Total timesteps : {args.timesteps:,}")
    print(f"  Parallel envs   : {n_envs}  (paper: 100)")
    print(f"  N_steps/worker  : {N_STEPS}  (= episode length)")
    print(f"  Active gates    : {num_gates} / {len(RACE_GATES)}")
    print(f"  Mid-course prob : {spawn_mid_course_prob:.0%}")
    print(f"  Learning rate   : {lr}")
    print(f"  Device          : {args.device}")
    print(f"  {'─'*54}")
    print(f"  Reward  (exact Swift formulation)")
    print(f"    r_prog  = {rc.LAMBDA_1} × (d_prev − d_curr)")
    print(f"    r_perc  = {rc.LAMBDA_2} × exp({rc.LAMBDA_3} × δ_cam⁴)")
    print(f"    r_cmd   = {rc.LAMBDA_4} × ‖a^ω‖²  +  {rc.LAMBDA_5} × ‖Δa‖²")
    print(f"    r_crash = {rc.CRASH_PENALTY}  (collision or OOB, binary)")
    print(f"{'='*60}\n")

    model.learn(
        total_timesteps = args.timesteps,
        callback        = [checkpoint_cb, eval_cb],
        progress_bar    = True,
        reset_num_timesteps = not bool(args.resume),
    )

    save_path = os.path.join(checkpoint_dir, "drone_racing_ppo_final")
    model.save(save_path)
    print(f"\n[train] Final model saved to: {save_path}.zip")

    train_env.close()
    eval_env.close()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DroneRacingEnv with PPO")
    parser.add_argument(
        "--timesteps", type=int, default=TOTAL_TIMESTEPS,
        help="Total environment timesteps (default: %(default)s)",
    )
    parser.add_argument(
        "--n_envs", type=int, default=N_ENVS,
        help="Number of parallel environments (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: %(default)s)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="PyTorch device: auto | cpu | cuda | mps (default: %(default)s)",
    )
    parser.add_argument(
        "--resume", type=str, default="",
        help="Path to a .zip checkpoint to resume from (default: none)",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default=CHECKPOINT_DIR,
        help="Where to save periodic checkpoints (default: %(default)s)",
    )
    parser.add_argument(
        "--log_dir", type=str, default=LOG_DIR,
        help="TensorBoard log directory (default: %(default)s)",
    )
    parser.add_argument(
        "--best_model_dir", type=str, default=BEST_MODEL_DIR,
        help="Where to save the best model (default: %(default)s)",
    )
    parser.add_argument(
        "--num_gates", type=int, default=5,
        help="Number of active gates for curriculum training (1–5, default: %(default)s)",
    )
    parser.add_argument(
        "--spawn_mid_course_prob", type=float, default=0.0,
        help="Probability of spawning mid-course each episode (0–1, default: %(default)s)",
    )
    main(parser.parse_args())
