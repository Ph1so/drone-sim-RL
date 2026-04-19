"""
train.py — PPO training script for DroneRacingEnv.

Architecture
------------
  MultiInputPolicy + custom GateObsExtractor:
    - Telemetry branch : 2-layer MLP (13-D)  →  64-D feature vector
    - Gate-obs branch  : 2-layer MLP (10-D)  →  64-D feature vector
    - Fusion           : concat → Linear(256) → ReLU  →  policy/value heads

Usage
-----
  python train.py                        # default settings
  python train.py --timesteps 5_000_000
  python train.py --headless 0           # run with GUI (slow, for debug)

Checkpoints are saved to ./checkpoints/ every SAVE_FREQ steps.
TensorBoard logs go to ./logs/.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from envs import DroneRacingEnv
from envs.gate_manager import RACE_GATES
from envs.reward import RewardComputer


# ── Hyper-parameters ──────────────────────────────────────────────────────────
TOTAL_TIMESTEPS  = 3_000_000
N_ENVS           = os.cpu_count() or 4   # parallel workers — defaults to CPU count
N_STEPS          = 512        # rollout steps per worker per update
BATCH_SIZE       = 256
N_EPOCHS         = 10
GAMMA            = 0.99
GAE_LAMBDA       = 0.95
CLIP_RANGE       = 0.2
ENT_COEF         = 0.01       # exploration entropy bonus
LR               = 3e-4
FEATURES_DIM     = 256
SAVE_FREQ        = 50_000     # checkpoint interval (env steps, not timesteps)

CHECKPOINT_DIR   = "./checkpoints"   # overridable via --checkpoint_dir
LOG_DIR          = "./logs"           # overridable via --log_dir
BEST_MODEL_DIR   = "./best_model"     # overridable via --best_model_dir


# ══════════════════════════════════════════════════════════════════════════════
# Custom feature extractor
# ══════════════════════════════════════════════════════════════════════════════

class GateObsExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for a Dict observation with keys:
      "telemetry"  : shape (13,)  — float32  [pos, quat, lin_vel, ang_vel]
      "gate_obs"   : shape (10,)  — float32  [curr_rel_x, curr_rel_y, curr_rel_z, curr_dist, curr_yaw_err,
                                              next_rel_x, next_rel_y, next_rel_z, next_dist, next_yaw_err]

    Two small MLPs are fused into *features_dim* for the policy/value heads.
    No CNN — the policy never sees raw pixels.

    Swap note
    ---------
    At competition time, wire the gate_obs input to estimates from a CV pipeline
    that produces the same 5-float format.  This extractor and the policy weights
    require no changes.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = FEATURES_DIM,
    ) -> None:
        super().__init__(observation_space, features_dim)

        tel_dim      = observation_space["telemetry"].shape[0]   # 13
        gate_obs_dim = observation_space["gate_obs"].shape[0]    # 10 (curr 5 + next 5)

        # ── Telemetry branch ──────────────────────────────────────────
        self.tel_net = nn.Sequential(
            nn.Linear(tel_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

        # ── Gate-obs branch ───────────────────────────────────────────
        self.gate_net = nn.Sequential(
            nn.Linear(gate_obs_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        # ── Fusion layer ──────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64, features_dim),
            nn.ReLU(inplace=True),
        )

        print(
            f"[GateObsExtractor] "
            f"tel out=64, gate_obs out=64, "
            f"fusion → {features_dim}"
        )

    def forward(self, obs: dict) -> torch.Tensor:
        tel_feat  = self.tel_net(obs["telemetry"].float())
        gate_feat = self.gate_net(obs["gate_obs"].float())
        return self.fusion(torch.cat([tel_feat, gate_feat], dim=1))


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

    # ── Policy kwargs with custom extractor ──────────────────────────
    policy_kwargs = dict(
        features_extractor_class  = GateObsExtractor,
        features_extractor_kwargs = {"features_dim": FEATURES_DIM},
        # Separate policy and value networks after the shared extractor
        net_arch = dict(pi=[256, 128], vf=[256, 128]),
        activation_fn = nn.ReLU,
    )

    # ── PPO model ─────────────────────────────────────────────────────
    if args.resume:
        print(f"[train] Resuming from checkpoint: {args.resume}")
        # Override hyperparams for pursuit phase:
        #   - lower LR (5e-5) to avoid shattering stable flight physics
        #   - ent_coef (0.01) reinjects exploration under the new reward signal
        custom_objects = {
            "learning_rate": 5e-5,
            "ent_coef": 0.01,
            "policy_kwargs": dict(
                features_extractor_class  = GateObsExtractor,
                features_extractor_kwargs = {"features_dim": FEATURES_DIM},
                net_arch      = dict(pi=[256, 128], vf=[256, 128]),
                activation_fn = nn.ReLU,
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
            policy          = "MultiInputPolicy",
            env             = train_env,
            n_steps         = N_STEPS,
            batch_size      = BATCH_SIZE,
            n_epochs        = N_EPOCHS,
            gamma           = GAMMA,
            gae_lambda      = GAE_LAMBDA,
            clip_range      = CLIP_RANGE,
            ent_coef        = ENT_COEF,
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
    print(f"  DroneRacing PPO training  (Swift reward)")
    print(f"  Total timesteps : {args.timesteps:,}")
    print(f"  Parallel envs   : {n_envs}")
    print(f"  Active gates    : {num_gates} / {len(RACE_GATES)}")
    print(f"  Mid-course prob : {spawn_mid_course_prob:.0%}")
    print(f"  Learning rate   : {lr}")
    print(f"  Device          : {args.device}")
    print(f"  {'─'*54}")
    print(f"  Reward policy (Swift, Kaufmann et al. 2023)")
    print(f"    Rewards")
    print(f"      progress         {rc.LAMBDA_1} × (d_prev − d_curr)")
    print(f"      perception       {rc.LAMBDA_2} × exp(−δ_cam / {rc.SIGMA_PERC})")
    print(f"      gate passage     {rc.GATE_PASS_BONUS} (flat, per gate)")
    print(f"    Penalties")
    print(f"      jerk             {rc.LAMBDA_4} × ‖Δa‖²")
    print(f"      body rate        {rc.LAMBDA_5} × ‖a^ω‖²")
    print(f"      ang vel          {rc.LAMBDA_6} × ‖ω‖²")
    print(f"      crash / OOB      {rc.CRASH_PENALTY} (binary)")
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
