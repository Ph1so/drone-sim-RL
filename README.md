# Drone Racing RL — Architecture Overview

Autonomous drone racing environment built on `gym-pybullet-drones` + Gymnasium.
Policy trained with PPO (stable-baselines3), following the Swift system architecture
(Kaufmann et al., *Nature* 2023).

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      DroneRacingEnv                          │
│                                                              │
│  ┌─────────────┐  ┌──────────────────┐  ┌────────────────┐   │
│  │ GateManager │  │ RewardComputer   │  │ Residual Obs   │   │
│  │             │  │                  │  │ Model (ROM)    │   │
│  │ Sequential  │  │ Swift reward:    │  │                │   │
│  │ gate pass   │  │ prog + perc      │  │ 9 GP draws,    │   │
│  │ detection   │  │ + cmd − crash    │  │ per episode    │   │
│  └──────┬──────┘  └────────┬─────────┘  └───────┬────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ Residual Dynamics Model (RDM)                        │    │
│  │ 6 GP draws → [f_x,f_y,f_z, τ_x,τ_y,τ_z]              │    │
│  │ Applied as body-frame force+torque each step         │    │
│  └──────────────────────────────────────────────────────┘    │
│         │                  │                     │           │
│         └──────────────────┴─────────────────────┘           │
│                            ↓                                 │
│          _computeObs() → 31-D flat observation               │
│          [ pos(3) | vel(3) | rot(9) | corners(12) | act(4) ] │
│                            ↓  ↑ action                       │
└────────────────────────────┼──┘──────────────────────────────┘
                             │
              ┌──────────────▼───────────────┐
              │      PPO Policy (SB3)        │
              │   MlpPolicy — 2 × 128 units  │
              │   LeakyReLU(α = 0.2)         │
              │   Output: [T, R, P, Y] ∈ R⁴  │
              └──────────────────────────────┘
                             │
              ┌──────────────▼───────────────┐
              │    Action Smoothing (EMA)    │
              │  applied = α·raw + (1−α)·prev│
              │  α = 0.7                     │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │   X-config quadrotor mixer   │
              │   → per-motor RPMs (1, 4)    │
              │   → PyBullet physics engine  │
              └──────────────────────────────┘
```

---

## Components

### `envs/drone_racing_env.py` — DroneRacingEnv

Subclass of `BaseAviary`. Owns the control loop, observation assembly,
step cache, and mid-course spawn logic.

**Step cache** (`_step_cache`): lazily populated on the first call within a
step and shared across `_computeObs`, `_computeReward`, `_computeTerminated`,
and `_computeInfo`. Guarantees a single PyBullet state query per step and that
`gate_manager.update()` fires exactly once.

**Mid-course spawn** (Swift paper, Methods): at each episode reset, with
probability `spawn_mid_course_prob`, the drone is teleported to just past a
random gate. Two spawn paths:

1. *Buffer spawn* (preferred): sample a previously-recorded gate-crossing state
   from `_gate_buffer`, add bounded noise to pos/vel/attitude/yaw. The buffer
   accumulates crossing states across episodes, ensuring spawns match real
   racing distributions (speed, bank angle, body rates).
2. *Geometric fallback*: used before the buffer fills. Fixed offset past the
   previous gate's exit plane with a small forward velocity.

**Action smoothing**: EMA low-pass filter applied between the raw policy
output and the physics engine. The jerk penalty is computed on raw outputs
so the policy is still penalised for demanding jerky commands even though
the motors never see them.

---

### `envs/gate_manager.py` — GateManager

Maintains the ordered gate sequence and detects passage events.

**Gate normal convention**: `R_z(yaw_deg) × [0, 1, 0] = [−sin θ, cos θ, 0]`.
A gate is passed when the drone's signed distance to the gate plane transitions
`≤ 0 → > 0` while the drone's lateral and vertical projection onto the gate is
within ±0.60 m of centre.

**Racecourse** (5-gate S-curve "train" map, `RACE_GATES`):

Turn sequence: RIGHT, LEFT, RIGHT, RIGHT, LEFT — mixed altitude variation.

| Gate | Position (m)          | Yaw     | Exit direction | Note                 |
|------|-----------------------|---------|----------------|----------------------|
| G1   | `[0.0,  4.0,  1.50]` | 0°      | N              | Straight from spawn  |
| G2   | `[6.0,  6.0,  2.00]` | −75°    | ENE            | Right turn + climb   |
| G3   | `[2.5,  8.5,  1.20]` | +50°    | NNW            | Left turn + dive     |
| G4   | `[8.5,  7.0,  1.80]` | −110°   | ESE            | Right turn + climb   |
| G5   | `[8.0,  1.5,  1.50]` | −160°   | SSE            | Right turn + descent |

Gate normal convention: `normal = [−sin(yaw), cos(yaw), 0]`.

---

### `envs/reward.py` — RewardComputer

Exact Swift reward (Kaufmann et al. 2023, Extended Data Table 1a):

```
r_t = r_prog + r_perc + r_cmd − r_crash

r_prog  = λ₁ [d_{t-1} − d_t]                  λ₁ = 1.0

r_perc  = λ₂ exp(λ₃ · δ_cam⁴)                 λ₂ = 0.02,  λ₃ = −10.0

r_cmd   = λ₄ ‖a_t^ω‖² + λ₅ ‖a_t − a_{t-1}‖²  λ₄ = −2e-4 (body-rate), λ₅ = −1e-4 (jerk)
         (a_t^ω and Δa_t are scaled to rad/s / N before penalty; see reward.py)

r_crash = 5.0  if p_z < 0 OR collision         (terminates episode)
```

`δ_cam` is the angle between the body-forward axis and the vector to the next
gate centre. The perception term rewards keeping the gate in the camera's field
of view without ever referencing image pixels — the same formulation works when
the gate-corners observation is swapped from ground-truth geometry to a real CV
pipeline output.

Gate-transition reset: `_prev_dist` is set to `None` whenever a gate is passed
so that the target switching from the just-passed gate (dist ≈ 0) to the next
gate (dist > 0) does not produce a large negative r_prog spike.

---

### `envs/residual_obs_model.py` — ResidualObservationModel (ROM)

GP-based perception residual model from Kaufmann et al. 2023 (Methods,
"Residual observation model"). Injects state-conditioned drift into the
position, velocity, and attitude observation slots to model VIO degradation
during high-speed and agile flight.

**Architecture**: 9 independent 1-D GPs (one per observation component)
approximated via Random Fourier Features (RFF). Input features `z ∈ R³`:

| Feature | Symbol | Physical driver |
|---------|--------|-----------------|
| Linear speed | `‖v‖` | Motion blur → visual feature tracking loss |
| Angular rate | `‖ω‖` | Rotational blur → IMU-visual misalignment |
| Tilt angle | `θ_tilt` | Horizon feature loss at large bank angles |

**Per-episode sampling**: `reset()` draws new weights `w ~ N(0, I)` for each
GP realization. The weights are held fixed for the entire episode; the drift
at each step is `f(z_t) = w · φ(z_t)`, scaled by a state-dependent amplitude
`σ(speed, ang_rate)`. Same state → same drift value within an episode.

**Temporal consistency**: consecutive time steps share similar states →
similar GP inputs → similar outputs (RBF kernel is smooth). The drift
evolves at the rate the physical state changes, not at the noise injection rate.

**Propagation through gate corners**: gate corners are recomputed from the
drifted position and rotation matrix so that VIO attitude error propagates
consistently into the body-frame gate perception.

Toggle: `DroneRacingEnv(obs_noise=False)` disables the ROM for clean evaluation.

---

### `train.py` — PPO Training

Policy: `MlpPolicy` with `net_arch=[128, 128]` and `activation_fn=LeakyReLU(α=0.2)`.
Matches the Swift paper architecture exactly (2-layer 128-unit MLP, same activation).

Key PPO hyperparameters:

| Parameter | Value | Note |
|-----------|-------|------|
| `n_steps` | 1500 | One rollout ≈ one full episode per worker |
| `batch_size` | 256 | |
| `n_epochs` | 10 | |
| `gamma` | 0.99 | |
| `gae_lambda` | 0.95 | |
| `clip_range` | 0.2 | |
| `learning_rate` | 3e-4 | Lowered to 5e-5 on `--resume` |

---

### `evaluate.py` — Evaluation Loop

Runs the policy in the PyBullet GUI with `obs_noise=False` (clean ground-truth
observations). Records per-episode breakdown of all reward components via the
`info` dict.

---

## Observation Space

Flat `Box(31,)` — identical to the Swift paper input:

| Slice | Content | Why |
|-------|---------|-----|
| `[0:3]` | Position, world frame (m) | Localisation |
| `[3:6]` | Linear velocity, world frame (m/s) | Speed awareness |
| `[6:15]` | Rotation matrix, body→world, row-major | Attitude without quaternion discontinuities (Zhou et al. 2019) |
| `[15:27]` | 4 gate-corner positions in drone body frame (m) | Geometry-complete gate representation; swappable with CV output |
| `[27:31]` | Previous raw policy action at t−1 | Temporal context for smooth control |

When the ROM is enabled, slices `[0:3]`, `[3:6]`, and `[6:15]` carry drifted
estimates; `[15:27]` is recomputed from those estimates so the gate-corner
error is physically consistent with the attitude drift.

---

## Sim-to-Real Design

The environment is structured so that the trained policy requires no changes
at competition time — only the observation source is swapped:

```
Training                           Deployment
──────────────────────────────     ──────────────────────────────
GateManager (ground-truth pos)  →  CV pipeline (gate detection)
ROM (synthetic GP drift)        →  Real VIO + gate-corner detector
_compute_gate_corners_body_frame → same 12-D corners from detector
Policy weights ─────────────────── unchanged
```

The perception reward (`r_perc`) trains the policy to keep the gate in the
forward FOV, which simultaneously trains a behaviour that helps real gate
detectors stay on target during agile maneuvers.

---

## Quick-start

```bash
# Create and activate conda environment (pybullet requires Python 3.10 on Apple Silicon)
conda create -n drone-sim python=3.10 -y
conda activate drone-sim
conda install -c conda-forge pybullet -y
pip install -r requirements.txt
pip install git+https://github.com/utiasDSL/gym-pybullet-drones.git --no-deps
pip install transforms3d "scipy<2.0" "control>=0.10.2,<0.11.0" tqdm rich
```

```bash
# Train (headless)
python train.py --timesteps 3_000_000 --n_envs 4

# Evaluate in GUI
python evaluate.py --model ./best_model/best_model.zip --episodes 5

# Curriculum (start narrow, expand)
python train.py --num_gates 1 --timesteps 1_000_000
python train.py --num_gates 3 --timesteps 2_000_000 --resume best_model/best_model.zip
python train.py --num_gates 5 --timesteps 3_000_000 --resume best_model/best_model.zip
```

Always activate the conda environment first:
```bash
source ~/miniforge3/bin/activate && conda activate drone-sim
```

---

## CLI Reference

### `train.py` — PPO training (headless)

```bash
python train.py [options]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--timesteps` | int | `3_000_000` | Total environment steps to train for (paper: 1×10⁸) |
| `--n_envs` | int | `cpu_count` | Parallel SubprocVecEnv workers (paper: 100) |
| `--num_gates` | int | `5` | Active gates for curriculum: `1`→`3`→`5` |
| `--spawn_mid_course_prob` | float | `0.8` | Probability of mid-course spawn each episode (0–1) |
| `--resume` | str | `""` | Path to `.zip` checkpoint to resume from |
| `--lr` | float | `None` | Learning rate override (`3e-4` fresh · `1e-4` curriculum resume · `5e-5` same-task fine-tune) |
| `--seed` | int | `42` | Global RNG seed |
| `--device` | str | `"auto"` | PyTorch device: `auto` \| `cpu` \| `cuda` \| `mps` |
| `--checkpoint_dir` | str | `./checkpoints` | Directory for periodic checkpoints |
| `--log_dir` | str | `./logs` | TensorBoard log directory |
| `--best_model_dir` | str | `./best_model` | Directory for best-eval-reward checkpoint |
| `--obs_noise` / `--no-obs-noise` | flag | `True` | Enable/disable ROM+RDM noise; use `--no-obs-noise` for Phase 1 curriculum |
| `--ent_coef` | float | `0.01` | PPO entropy coefficient; raise to `0.05` to escape local optima, lower to `0.0` for final convergence |

**Examples:**
```bash
# Full paper scale
python train.py --timesteps 100_000_000 --n_envs 100

# Curriculum (Phase 1: learn stable flight without noise)
python train.py --num_gates 1 --timesteps 5_000_000 --no-obs-noise

# Curriculum (Phase 2+: add noise domain randomisation)
python train.py --num_gates 1 --timesteps 1_000_000
python train.py --num_gates 3 --timesteps 2_000_000 --resume best_model/best_model.zip
python train.py --num_gates 5 --timesteps 3_000_000 --resume best_model/best_model.zip
```

---

### `evaluate.py` — GUI evaluation

```bash
python evaluate.py [options]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model` | str | `./best_model/best_model.zip` | Path to saved `.zip` policy |
| `--map` | str | `"eval"` | Course to use: `eval` (unseen hook-shape) or `train` (S-curve) |
| `--episodes` | int | `5` | Number of episodes to run |
| `--num_gates` | int | `5` | Must match the training config |
| `--render_fps` | int | `48` | Simulation playback pace (fps) |
| `--record` | flag | off | Ask PyBullet to record GUI frames to video |
| `--plot` | flag | off | Save 4-panel diagnostic PNG per episode (requires matplotlib) |
| `--spawn_mid_course_prob` | float | `0.0` | Mid-course spawn probability (0 = always start at start line) |
| `--gate_offset` | 3×float | `None` | Shift all gates by `DX DY DZ` metres — tests position generalisation |

**In-GUI camera controls (keyboard):**

| Key | Action |
|---|---|
| `V` | Toggle drone first-person POV / free camera |
| `A` / `D` | Orbit yaw left / right (free camera) |
| `W` / `S` | Tilt pitch up / down (free camera) |
| `Q` / `E` | Zoom in / out (free camera) |
| Arrow keys | Pan camera target (free camera) |

**Examples:**
```bash
# Basic evaluation on the unseen eval map
python evaluate.py --model ./best_model/best_model.zip

# 10 episodes on the training map with diagnostic plots
python evaluate.py --model ./checkpoints/drone_racing_ppo_final.zip \
                   --map train --episodes 10 --plot

# Test gate position generalisation (+2 m in X)
python evaluate.py --model ./best_model/best_model.zip --gate_offset 2 0 0
```

---

### `visualize.py` — Course map viewer

```bash
python visualize.py [options]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--map` | str | `"train"` | Course to display: `train` (S-curve) or `eval` (hook-shape) |
| `--pybullet` | flag | off | Open PyBullet 3D GUI instead of matplotlib chart |

**Examples:**
```bash
# Matplotlib top-down + 3D chart (no extra dependencies)
python visualize.py
python visualize.py --map eval

# Live 3D PyBullet view (press Ctrl-C to quit)
python visualize.py --pybullet
python visualize.py --pybullet --map eval
```
