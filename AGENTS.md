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

### Mid-course spawning

`spawn_mid_course_prob` teleports the drone to a random gate approach at episode start. This bootstraps value function learning for gate transitions the policy would rarely reach organically early in training. Do not set this to 0 during early curriculum stages.

### Next-gate lookahead

`gate_obs` includes the *next* gate's 5-float obs. Without this, the policy would have no signal after passing the current gate and would crash during the gate transition. The `next_*` slice is zeros only on the final gate, which is correct.

## Reward Components (current values)

Swift formulation — Kaufmann et al. 2023. No per-gate or lap-completion bonuses;
`r_prog` is dense enough to guide the agent through every gate without sparse rewards.
Gate transitions reset `_prev_dist` to avoid a negative spike when the target switches
to the next (farther) gate.

| Component | Formula | Weight |
|---|---|---|
| Progress | `d_{t-1} − d_t` (distance delta to next gate) | λ₁ = 1.0 |
| Perception | `exp(−δ_cam / σ)`, δ_cam = angle from body-fwd to gate | λ₂ = 0.02, σ = 0.5 rad |
| Jerk penalty | `−‖a_t − a_{t-1}‖²` | λ₄ = 2×10⁻⁴ |
| Body-rate penalty | `−‖a_t^ω‖²` (roll/pitch/yaw channels) | λ₅ = 1×10⁻⁴ |
| Ang-vel penalty | `−‖ω‖²` (physical angular velocity) | λ₆ = 0.02 |
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
