# Drone Racing RL Environment

A Python-based Reinforcement Learning environment for autonomous drone racing built on `gym-pybullet-drones`, `Gymnasium`, and `stable-baselines3`.

## Project Layout

```
drone-sim/
├── assets/
│   └── gate.urdf                ← 1.4 × 1.4 m gate frame (XZ-plane opening)
├── envs/
│   ├── __init__.py
│   ├── gate_manager.py          ← Gate dataclass + GateManager
│   ├── reward.py                ← RewardComputer
│   └── drone_racing_env.py      ← DroneRacingEnv(BaseAviary)
├── train.py                     ← PPO + MultimodalExtractor
├── evaluate.py                  ← GUI evaluation loop
└── requirements.txt
```

## Quick-start

### Installation (macOS Apple Silicon)

pybullet cannot be compiled from source on Apple Silicon / Python 3.12.
Use a conda environment with the pre-built conda-forge binary instead.

```bash
# 1. Install Miniforge (skip if conda is already available)
curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh -o miniforge.sh
bash miniforge.sh -b -p ~/miniforge3
source ~/miniforge3/bin/activate

# 2. Create the env with pybullet from conda-forge
conda create -n drone-sim python=3.10 -y
conda activate drone-sim
conda install -c conda-forge pybullet -y

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install gym-pybullet-drones (not on PyPI; --no-deps prevents recompiling pybullet)
pip install git+https://github.com/utiasDSL/gym-pybullet-drones.git --no-deps

# 5. Install transitive deps that --no-deps skipped
pip install transforms3d "scipy<2.0" "control>=0.10.2,<0.11.0" tqdm rich
```

### Running

Always activate the conda env before running any script:

```bash
source ~/miniforge3/bin/activate && conda activate drone-sim
```

```bash
# Train (headless, 3M steps, 4 parallel workers)
python train.py --timesteps 3_000_000 --n_envs 4

# Monitor in TensorBoard (separate terminal, same env)
tensorboard --logdir ./logs

# Evaluate best checkpoint in PyBullet GUI
python evaluate.py --model ./best_model/best_model.zip --episodes 5

# Visualise the racecourse map (no model needed)
python visualize.py
```

## Design Reference

### Action Space

`Box(4,)` normalised `[T, R, P, Y] ∈ [−1, +1]`. `_preprocessAction` maps these through a classical X-config quadrotor mixer centred around `HOVER_RPM`:

```
M0 (FL, CCW) = hover + 0.5·h·T  −  0.2·h·R  −  0.2·h·P  +  0.1·h·Y
M1 (FR, CW)  = hover + 0.5·h·T  +  0.2·h·R  −  0.2·h·P  −  0.1·h·Y
M2 (RL, CW)  = hover + 0.5·h·T  −  0.2·h·R  +  0.2·h·P  −  0.1·h·Y
M3 (RR, CCW) = hover + 0.5·h·T  +  0.2·h·R  +  0.2·h·P  +  0.1·h·Y
```

### Observation Space

Flat `Box(31,)` — matches the exact input used by the Swift policy (Kaufmann et al., 2023).

| Slice | Content | Dim |
|-------|---------|-----|
| `[0:3]` | Drone position in world frame (m) | 3 |
| `[3:6]` | Linear velocity in world frame (m/s) | 3 |
| `[6:15]` | Attitude as **rotation matrix** (row-major, body→world) | 9 |
| `[15:27]` | 4 gate-corner positions in **drone body frame** (m) | 12 |
| `[27:31]` | Previous action `[T, R, P, Y]` applied at t−1 | 4 |

Attitude is encoded as a flattened 3×3 rotation matrix (not quaternion) to avoid gimbal-lock discontinuities (Zhou et al., 2019 — cited as ref. [47] in the Swift paper).

Gate corners are computed from the current target gate's centre and yaw using a ±0.6 m half-opening in the gate plane, then transformed into drone body frame: `R^T @ (corner_world − pos_drone)`. The four corners are ordered top-right, top-left, bottom-left, bottom-right.

> **No raw pixels are passed to the policy.** `_render_ego_camera()` is retained in the env for visualization and future CV pipeline integration.

### Gate Passing Logic

Each gate has a **normal vector** `[−sin θ, cos θ, 0]` derived from its `yaw_deg`. A pass is detected when the drone's signed distance to the gate plane transitions `≤ 0 → > 0` while its projected position is within `±0.60 m` (horizontal) × `±0.60 m` (vertical) of the gate centre.

### Racecourse

Five gates on an oval-ish lap. The drone spawns at `[0, 0, 0.3]` facing `+Y`.

| Gate | Position (m)       | Yaw    | Normal direction |
|------|--------------------|--------|------------------|
| G1   | `[0.0, 2.5, 1.5]` | 0°     | +Y (north)       |
| G2   | `[2.5, 5.0, 1.5]` | −40°   | NE diagonal      |
| G3   | `[5.5, 5.5, 1.5]` | −90°   | +X (east)        |
| G4   | `[7.5, 2.8, 1.5]` | −135°  | SE diagonal      |
| G5   | `[5.0, 0.5, 1.5]` | 180°   | −Y (south)       |

### Reward Components

Exact Swift formulation — Kaufmann et al., *Nature* 2023, Extended Data Table 1a.

```
r_t = r_prog + r_perc + r_cmd − r_crash

r_prog  = λ₁ × [d_{t-1}^Gate − d_t^Gate]                   λ₁ = 1.0
r_perc  = λ₂ × exp(λ₃ × δ_cam⁴)                            λ₂ = 0.02,  λ₃ = −10.0
r_cmd   = λ₄ × ‖a_t^ω‖²  +  λ₅ × ‖a_t − a_{t-1}‖²        λ₄ = −2e-4, λ₅ = −1e-4
r_crash = 5.0  if p_z < 0 OR collision with gate; else 0   (episode terminates)
```

| Component | Formula | Weight |
|---|---|---|
| Progress | `λ₁ [d_{t-1} − d_t]` — distance delta to next gate | λ₁ = 1.0 |
| Perception | `λ₂ exp(λ₃ δ_cam⁴)` — δ_cam = angle body-fwd → gate centre | λ₂ = 0.02, λ₃ = −10.0 |
| Body-rate penalty | `λ₄ ‖a_t^ω‖²` (roll/pitch/yaw action channels) | λ₄ = −2×10⁻⁴ |
| Jerk penalty | `λ₅ ‖a_t − a_{t-1}‖²` | λ₅ = −1×10⁻⁴ |
| Crash / Out-of-bounds | `−5.0` + episode ends (p_z < 0 or collision) | — |

**No gate passage bonus.** Gate transitions reset `_prev_dist` to suppress the distance spike when the target switches to the next (farther) gate.

### Training Architecture

Flat `MlpPolicy` matching the Swift paper (Kaufmann et al., 2023, Methods):

```
Input (31-D) → Linear(128) → LeakyReLU(α=0.2) → Linear(128) → LeakyReLU(α=0.2) → policy / value heads
```

- **No feature extractor**, no CNN, no LayerNorm — just a 2-layer 128-unit MLP shared backbone
- **LeakyReLU(α=0.2)** — paper-exact activation; `functools.partial(nn.LeakyReLU, negative_slope=0.2)` used so it is picklable with `SubprocVecEnv`
- **Policy and value networks** share the backbone (standard SB3 `MlpPolicy`)

PPO hyperparameters:

| Parameter | Value | Note |
|-----------|-------|------|
| `n_steps` | 1500 | One rollout = one full episode per worker |
| `batch_size` | 256 | |
| `n_epochs` | 10 | |
| `gamma` | 0.99 | Matches paper Extended Data Table 1a |
| `gae_lambda` | 0.95 | |
| `clip_range` | 0.2 | ε in paper |
| `ent_coef` | 0.0 | Not used in Swift |
| `lr` | 3×10⁻⁴ | Adam; auto-lowered to 5×10⁻⁵ on `--resume` |

When resuming (`--resume`), `lr` is automatically lowered to `5e-5` to stabilise the value function under the new reward scale.

### Modular Perception Design

The policy never receives raw pixels. Gate corners in body frame are the only perception interface:

```
[Prototype]                              [Competition]
GateManager (ground truth geometry)      CV pipeline on real sim camera
  → corner positions in body frame  →      same 12-D gate_corners slice
             ↓                                         ↓
             policy ─────────────────────────── same policy weights
```

To deploy on the competition sim:
1. Build a CV module that detects the next gate from the onboard camera and outputs the 4 corner positions in the drone body frame (12 floats).
2. Replace the `_compute_gate_corners_body_frame()` call in `_computeObs()` with your CV module's output.
3. The trained policy requires zero changes.

### Step Cache

`_get_step_state()` populates `_step_cache` lazily so `_computeObs`, `_computeReward`, `_computeTerminated`, and `_computeInfo` share a single PyBullet state query and `gate_manager.update()` fires exactly once per control step.

## CLI Reference

### train.py

```
--timesteps INT   Total environment timesteps               (default: 3_000_000)
--n_envs    INT   Number of parallel environments           (default: cpu_count)
--seed      INT   Random seed                               (default: 42)
--device    STR   auto | cpu | cuda | mps                   (default: auto)
--resume    STR   Path to a .zip checkpoint to resume       (default: none)
--num_gates INT   Active gates for curriculum training 1–5  (default: 5)
```

Curriculum workflow:
```bash
python train.py --num_gates 1 --timesteps 1_000_000   # master gate 1 first
python train.py --num_gates 3 --timesteps 2_000_000   # extend to 3 gates
python train.py --num_gates 5 --timesteps 3_000_000   # full course
```

### evaluate.py

```
--model      STR   Path to saved .zip policy                (default: ./best_model/best_model.zip)
--episodes   INT   Number of evaluation episodes            (default: 5)
--render_fps INT   Rendering pace in fps                    (default: 48)
--num_gates  INT   Active gates (match training config)     (default: 5)
--record         Record PyBullet GUI to video frames
```
# drone-sim-RL
