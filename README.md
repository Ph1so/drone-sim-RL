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

```python
Dict{
  "telemetry": Box(13,)  # pos(3) + quat(4) + lin_vel(3) + ang_vel(3)
  "gate_obs":  Box(5,)   # noisy gate-relative obs in drone yaw frame (see below)
}
```

`gate_obs` components — all in the drone's yaw-rotated body frame:

| Index | Name | Description |
|-------|------|-------------|
| 0 | `rel_x` | Distance to next gate along drone forward axis (m) |
| 1 | `rel_y` | Distance to next gate along drone left axis (m) |
| 2 | `rel_z` | Vertical offset to next gate center (m, + = above) |
| 3 | `dist` | Euclidean distance to next gate (m) |
| 4 | `gate_yaw_err` | Signed angle from drone heading to gate normal (rad) |

Gaussian noise (σ=0.3 m on position, σ=0.05 rad on heading) is injected at every step to simulate CV estimation uncertainty. Configurable via `DroneRacingEnv(gate_noise_std=...)`.

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

| Component        | Value                                      |
|------------------|--------------------------------------------|
| Distance shaping | `+3.0 × Δdist_to_gate` per step           |
| Gate passage     | `+100` one-off                             |
| Lap completion   | additional `+500`                          |
| Time penalty     | `−0.05` per step                           |
| Tilt penalty     | `−1.0 × (tilt − 45°)` when over threshold |
| Collision        | `−100` + episode ends                      |
| Out-of-bounds    | `−50` + episode ends                       |

### Training Architecture

`GateObsExtractor` plugged into SB3's `MultiInputPolicy`:

- **Telemetry branch** — `Linear(13→128)` → LayerNorm → ReLU → `Linear(128→64)` → ReLU → 64-D
- **Gate-obs branch** — `Linear(5→64)` → LayerNorm → ReLU → `Linear(64→64)` → ReLU → 64-D
- **Fusion** — concat(128-D) → `Linear(256)` → ReLU → policy / value heads

No CNN. The policy is a pure MLP — fast to train, trivial to run at inference.

PPO hyperparameters: 4 parallel envs, `n_steps=512`, `batch_size=256`, `ent_coef=0.01`, `lr=3e-4`.

### Modular Perception Design

The policy never receives raw pixels. The `gate_obs` vector is the only perception interface:

```
[Prototype]                          [Competition]
GateManager (ground truth)           CV pipeline on real sim camera
  + Gaussian noise injection    →      same 5-float gate_obs format
        ↓                                       ↓
        policy ─────────────────────────── same policy weights
```

To deploy on the competition sim:
1. Build a CV module that detects the next gate from the onboard camera and outputs `[rel_x, rel_y, rel_z, dist, gate_yaw_err]` in the drone's yaw frame.
2. Replace the `_compute_gate_obs()` call in `_computeObs()` with your CV module's output.
3. The trained policy and `GateObsExtractor` require zero changes.

### Step Cache

`_get_step_state()` populates `_step_cache` lazily so `_computeObs`, `_computeReward`, `_computeTerminated`, and `_computeInfo` share a single PyBullet state query and `gate_manager.update()` fires exactly once per control step.

## CLI Reference

### train.py

```
--timesteps INT   Total environment timesteps          (default: 3_000_000)
--n_envs    INT   Number of parallel environments      (default: 4)
--seed      INT   Random seed                          (default: 42)
--device    STR   auto | cpu | cuda | mps              (default: auto)
--resume    STR   Path to a .zip checkpoint to resume  (default: none)
```

### evaluate.py

```
--model      STR   Path to saved .zip policy           (default: ./best_model/best_model.zip)
--episodes   INT   Number of evaluation episodes       (default: 5)
--render_fps INT   Rendering pace in fps               (default: 48)
--record         Record PyBullet GUI to video frames
```
# drone-sim-RL
