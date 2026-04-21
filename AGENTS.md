# AGENTS.md — Drone Racing RL Environment

AI agent guidance for working in this repository.

## What This Project Does

Trains a PPO policy to race a simulated quadrotor through 5 sequential gates on an oval course using `gym-pybullet-drones`, `Gymnasium`, and `stable-baselines3`. The policy receives noisy gate-relative observations (no pixels) and outputs normalized thrust/roll/pitch/yaw commands. The goal is to complete a lap through all gates in minimum time.

## Key Commands

Always activate the conda environment first:

```bash
source ~/miniforge3/bin/activate && conda activate drone-sim
```

```bash
# Headless training (4 workers, 3M steps)
python train.py --timesteps 3_000_000 --n_envs 4

# Resume from checkpoint
python train.py --resume ./best_model/best_model.zip --timesteps 1_000_000

# Curriculum (recommended for new policies)
python train.py --num_gates 1 --timesteps 1_000_000
python train.py --num_gates 3 --timesteps 2_000_000 --resume <ckpt>
python train.py --num_gates 5 --timesteps 3_000_000 --resume <ckpt>

# Evaluate with GUI
python evaluate.py --model ./best_model/best_model.zip --episodes 5

# Visualize racecourse (no model needed)
python visualize.py

# TensorBoard (separate terminal)
tensorboard --logdir ./logs
```

There are no automated tests. Validate by running `evaluate.py` and checking gates passed and reward breakdown in the printed episode stats.

## File Map

```
envs/drone_racing_env.py   ← Gymnasium env; action/obs/reward/termination
envs/gate_manager.py       ← Gate dataclass, passing detection, lap logic
envs/reward.py             ← RewardComputer (stateless; all 13 components)
train.py                   ← PPO setup, GateObsExtractor, training loop
evaluate.py                ← GUI evaluation, stats, optional GIF recording
visualize.py               ← Racecourse visualization only
assets/gate.urdf           ← 1.4 × 1.4 m gate mesh
```

## Architecture You Must Understand

### Observation

Two-branch Dict observation fed to `GateObsExtractor` in `train.py`:

- `"telemetry"` `Box(13)` — `[pos_rel_gate(3), quat(4), lin_vel(3), ang_vel(3)]`. Position is relative to the **current gate center**, not world origin.
- `"gate_obs"` `Box(10)` — `[curr_gate(5), next_gate(5)]` in the drone's yaw frame. Each 5-tuple: `[rel_x, rel_y, rel_z, dist, gate_yaw_err]`. The `next_*` slice is all zeros on the last gate.

Noise is **distance-scaled**: `σ = gate_noise_std × max(dist / 3.0, 0.2)`. Far gates are noisier. Do not flatten this to a uniform noise model.

### Step Cache

`_get_step_state()` in `drone_racing_env.py` runs exactly once per step and populates `_step_cache`. All of `_computeObs`, `_computeReward`, `_computeTerminated`, and `_computeInfo` read from this shared cache. Do not call `gate_manager.update()` or `getKinematicsVector()` directly inside those methods — go through `_get_step_state()`.

### Perception Swap Point

The policy was designed to accept CV-estimated gate observations at competition time. The only interface is `_compute_gate_obs()` in `drone_racing_env.py:_computeObs()`. Replace that call with a CV module that outputs the same 10-float format. **Policy weights and `GateObsExtractor` require zero changes.**

## Critical Design Decisions — Do Not Undo

### Reward shape

The catapult-and-crash policy (fly fast at gate 1, crash into it for the bonus) was a persistent failure mode. The current reward structure defeats it by design:

- Velocity reward is `tanh`-saturated — unbounded speed has no extra value.
- Gate 1 bonus (80) < collision penalty (100) — crashing at gate 1 is net-negative.
- Gate bonuses escalate (`80 × gates_cleared`) — later gates are worth exponentially more.
- Proximity + heading + velocity-alignment rewards only matter near the gate — no incentive to overshoot.

Do not raise the gate bonus above the collision penalty, remove `tanh` from the velocity term, or revert to a linear velocity reward.

### Vertical velocity penalty is disabled

It was causing asymmetric reward cliffs below gate Z. The altitude alignment penalty (`-0.4 × |drone_z − gate_z|`) provides the necessary vertical correction without this artifact. Do not re-enable it without testing.

### Curriculum training is required

Training from scratch on 5 gates fails to converge. Always use the 1→3→5 gate curriculum when starting fresh policies or making large architecture changes.

## Curriculum Progress Tracker

**Agent instructions:** Whenever the user shares a TensorBoard screenshot, evaluate.py output, or any training update, update the Status column of this table and add the observed metrics in the Notes column. Do not move a phase to ✅ Complete based on `ep_len` alone — `gates_passed > 0` in the evaluate.py summary is required to confirm real progress. A policy that survives 1500 steps with 0 gates passed has found a hover local optimum, not a racing strategy.

Status emoji convention: ✅ Complete · 🔄 In progress · ⏳ Pending · ❌ Stuck/regressed

| Phase | Gates | `--no-obs-noise` | `spawn_prob` | Status | Move on when | Notes |
|---|---|---|---|---|---|---|
| 1 | 1 | yes | 0.0 | ✅ Complete | ep_len → 1500 | ep_len=1500; erratic flight but stable |
| 2a | 3 | yes | 0.0 | ✅ Complete | gates_passed = 3/3 | 3/3 gates, reward +31.4 ±0.0; hover exploit broken with gate bonus (+10); jerkiness acceptable without obs noise |
| 2b | 3 | yes | 0.5 | ⏸ Skipped | — | Exceeded criteria before reaching this phase |
| 3a | 5 | yes | 0.5→0.8 | 🔄 In progress | gates_passed = 5/5 OR lap completed | 4/5 gates @3.5M steps, reward +47.4; crashes into ground at G5 (z=0.5m descent from G4 z=1.8m); yaw oscillation noted but acceptable without obs noise; explained_variance=0.95, entropy declining to −0.7, KL<0.02 — soft local optimum forming; bumped spawn_prob to 0.8 and ent_coef to 0.05 to escape |
| 3b | 5 | no | 0.8 | ⏳ Pending | eval reward > +40 | ROM will penalise yaw oscillation via ‖ω‖ → observation drift; expect smoother flight to emerge |

**Commands for each phase:**

```bash
# Phase 1 (complete)
python train.py --num_gates 1 --timesteps 5_000_000 --no-obs-noise --spawn_mid_course_prob 0.0

# Phase 2a (complete)
python train.py --num_gates 3 --timesteps 10_000_000 --resume best_model/best_model.zip --no-obs-noise --spawn_mid_course_prob 0.0

# Phase 3a (current — local optimum escape attempt)
python train.py --num_gates 5 --timesteps 3_000_000 --resume best_model/best_model.zip --no-obs-noise --spawn_mid_course_prob 0.8 --ent_coef 0.05

# Phase 3a (normal — once G5 is being passed)
python train.py --num_gates 5 --timesteps 15_000_000 --resume best_model/best_model.zip --no-obs-noise --spawn_mid_course_prob 0.5

# Phase 3b (full paper config)
python train.py --num_gates 5 --timesteps 20_000_000 --resume best_model/best_model.zip --spawn_mid_course_prob 0.8
```

**Always match evaluate.py flags to training config:**

```bash
# Example: Phase 2a was trained with --num_gates 3 --no-obs-noise
python evaluate.py --model best_model/best_model.zip --map train --num_gates 3
# obs_noise defaults to False in evaluate.py — matches --no-obs-noise training
# For policies trained WITH noise: add --obs_noise flag
```

Mismatch symptom: training shows reward +X but evaluate.py shows -5. Root cause is almost always `obs_noise` defaulting to `True` in the environment while training used `False`, or `num_gates` mismatch loading extra gate bodies.

**Key diagnostics to check at each update:**

| Metric | Where | Healthy sign |
|---|---|---|
| `rollout/ep_len_mean` | TensorBoard | Rising toward 1500 |
| `eval/mean_reward` | TensorBoard | Rising, eventually > 0 |
| `gates_passed` | evaluate.py summary | Must be > 0 before advancing |
| Reward breakdown | evaluate.py summary | `progress` dominant, `crash` shrinking |

### Mid-course spawning

`spawn_mid_course_prob` teleports the drone to a random gate approach at episode start. This bootstraps value function learning for gate transitions the policy would rarely reach organically early in training. Do not set this to 0 during early curriculum stages.

### Next-gate lookahead

`gate_obs` includes the *next* gate's 5-float obs. Without this, the policy would have no signal after passing the current gate and would crash during the gate transition. The `next_*` slice is zeros only on the final gate, which is correct.

## Reward Components (current values)

Swift formulation — Kaufmann et al. 2023. No escalating gate bonuses or lap-completion
bonus. A small flat gate bonus (5.0) distinguishes "flew through the opening" from
"crashed into the frame" — `r_prog` only sees scalar distance and cannot make this
distinction. Gate transitions reset `_prev_dist` to avoid a negative spike when the
target switches to the next (farther) gate.

| Component | Formula | Weight |
|---|---|---|
| Progress | `d_{t-1} − d_t` (distance delta to next gate) | λ₁ = 1.0 |
| Perception | `exp(−δ_cam / σ)`, δ_cam = angle from body-fwd to gate | λ₂ = 0.02, σ = 0.5 rad |
| Gate passage bonus | flat `+5.0` per gate cleared | — |
| Jerk penalty | `−‖a_t − a_{t-1}‖²` | λ₄ = 2×10⁻⁴ |
| Body-rate penalty | `−‖a_t^ω‖²` (roll/pitch/yaw channels) | λ₅ = 1×10⁻⁴ |
| Crash / Out-of-bounds | `−50` + episode ends | — |

## Racecourse

```
G1: [0.0, 2.5, 1.5]  yaw=0°    → +Y (north)
G2: [2.5, 5.0, 1.5]  yaw=-40°  → NE diagonal
G3: [5.5, 5.5, 1.5]  yaw=-90°  → +X (east)
G4: [7.5, 2.8, 1.5]  yaw=-135° → SE diagonal
G5: [5.0, 0.5, 1.5]  yaw=180°  → -Y (south)
Spawn: [0.0, 0.0, 0.3] facing +Y
```

Gate pass detection: signed distance to gate plane transitions `≤ 0 → > 0` while within `±0.60 m` horizontally and vertically of gate center.

## Where to Make Changes Safely

| Task | Where to edit | Watch out for |
|---|---|---|
| Tune a reward term | `envs/reward.py` — `RewardComputer` | Keep gate bonus < crash penalty |
| Add an obs field | `drone_racing_env.py:_computeObs` + observation space definition + `GateObsExtractor` in `train.py` | Old checkpoints become incompatible |
| Change gate layout | `gate_manager.py` top-level gate list | Update `visualize.py` if it hardcodes positions |
| Adjust noise | `DroneRacingEnv(gate_noise_std=...)` in `train.py` | Distance-scaling is intentional; don't flatten |
| Change PPO hyperparams | `train.py` `model = PPO(...)` block | `n_steps × n_envs` must be divisible by `batch_size` |
| Swap CV perception | `drone_racing_env.py:_computeObs` — replace `_compute_gate_obs()` | Output must match `Box(10)` exact format |

## Common Failure Modes

- **Drone crashes into gate 1 immediately** — catapult-and-crash policy. The velocity reward or gate bonus structure has been broken. Check `tanh` saturation and bonus vs. crash penalty ratio.
- **Policy stops improving after gate 2** — altitude alignment signal may be too weak, or the gate_obs lookahead is missing. Check that `next_gate` obs is non-zero during gate 2 approach.
- **Training collapses to hovering** — entropy coefficient too low or learning rate too high on resume. Default `ent_coef=0.01`; resume `lr=5e-5`.
- **`assert n_steps * n_envs % batch_size == 0`** — SB3 requirement. Keep `n_steps=512`, `batch_size=256`, `n_envs` a power of 2.
- **PyBullet import error on Apple Silicon** — must use conda-forge pybullet, not pip. See README installation steps.

## Notes File

`notes.md` is a development diary. It documents the catapult-and-crash root cause analysis and the five fixes applied. Read it before making large reward or observation changes to avoid re-introducing solved problems.
