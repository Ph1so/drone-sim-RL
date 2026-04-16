# Drone Racing RL — Development Diary

---

## 2026-04-16 — Fix catapult-and-crash behavior (5 targeted changes)

### Problem
After the gate_obs architecture overhaul, the agent developed a "catapult and crash" policy:
it launched at maximum velocity toward gate 1, cleared it, then immediately crashed. Root
causes identified through analysis:
1. Unbounded linear velocity reward → maximum thrust always optimal
2. Policy had no lookahead past the current gate; gate transition was a hard discontinuity
3. Gate 1 bonus (200) > crash penalty (100) → catapult-and-crash was a profitable strategy
4. Fixed positional noise was too large at high velocity / close range
5. Policy never visited post-gate-1 states → value function garbage beyond gate 1

### Changes

**Fix 1 — Saturate velocity reward (`envs/reward.py`)**
- Replaced `12.0 × progress` with `12.0 × tanh(progress / 2.0)`
- Reward now saturates at ±12 rather than growing without bound
- Marginal reward for going faster than ~2 m/s is near-zero → catapult no longer optimal

**Fix 2 — Add next gate to observation (`envs/drone_racing_env.py`)**
- `gate_obs` extended from 5 → 10 floats: `[current_gate(5), next_gate(5)]`
- Agent can now perceive that gate 2 is around a corner before it reaches gate 1
- Gate transition discontinuity eliminated — next gate obs is already visible pre-transition
- `next_*` slice is zeros when current gate is the last active gate (clean sentinel)

**Fix 3 — Curriculum training (`envs/gate_manager.py`, `envs/drone_racing_env.py`, `train.py`, `evaluate.py`)**
- `GateManager` now accepts `num_gates` (1–5); only activates the first N gates
- `DroneRacingEnv` gains `num_gates` param, propagated to `GateManager`
- `train.py` and `evaluate.py` gain `--num_gates` CLI arg
- Workflow: train on 1 gate first → expand to 3 → full 5-gate course
- Forces post-gate states to be explored from the start

**Fix 4 — Distance-scaled noise (`envs/drone_racing_env.py`)**
- Positional noise now scales with true gate distance: σ = `gate_noise_std × max(dist/3.0, 0.2)`
- Farther gates → more noise (realistic CV behavior); closer gates → cleaner signal
- Prevents noise from being catastrophically large at high speed / close range
- Angular noise (σ=0.05 rad) unchanged

**Fix 5 — Escalating gate bonus (`envs/reward.py`)**
- Replaced flat `GATE_PASS_BONUS = 200` with `GATE_BASE_BONUS × num_gates_cleared`
- Gate 1 = 80, Gate 2 = 160, Gate 3 = 240, Gate 4 = 320, Gate 5 = 400
- Gate 1 bonus (80) < crash penalty (100) → catapult-and-crash is now net-negative
- Multi-gate sequences are substantially more valuable than crashing after gate 1

### Why
Catapult behavior was a locally optimal policy given the old reward structure. All five changes
together redefine what "optimal" looks like: the agent must fly at controlled speed, maintain
stability through gate transitions, and string multiple gates together to maximize return.

### Expected outcome
- Catapult strategy becomes unprofitable (net negative with gate 1 only)
- Curriculum (start with `--num_gates 1`) bootstraps stable single-gate flight before adding complexity
- Lookahead obs prevents the hard transition discontinuity that caused post-gate crashes
- Training from scratch required — architecture and observation space changed again

---

## 2026-04-16 — Architecture overhaul: image obs → noisy gate obs

### Changes
- Replaced `"image": Box(64,64,3)` observation with `"gate_obs": Box(5,) float32`
- `gate_obs` is a 5-float gate-relative vector in the drone's yaw frame:
  `[rel_x, rel_y, rel_z, dist, gate_yaw_err]`
  with Gaussian noise injected at every step (σ=0.3 m position, σ=0.05 rad heading)
- Removed `MultimodalExtractor` (CNN + MLP) from `train.py`; replaced with `GateObsExtractor` (two small MLPs, no CNN)
  - Telemetry branch: Linear(13→128) → LN → ReLU → Linear(128→64)
  - Gate-obs branch:  Linear(5→64)   → LN → ReLU → Linear(64→64)
  - Fusion: concat(128) → Linear(256) → ReLU → policy/value heads
- `_render_ego_camera()` retained in env but not called from `_computeObs()` — preserved for future CV pipeline integration
- New `gate_noise_std=0.3` param on `DroneRacingEnv` — configurable noise level
- Reward function unchanged; reward still uses ground-truth GateManager positions (intentional — reward is the teacher, obs noise trains robustness)
- Starting training from **blank slate** — old model weights are incompatible (different obs space + network architecture)

### Why
The raw image observation created an unsolvable domain gap: PyBullet renders look nothing like the Anduril competition sim. A policy trained on PyBullet pixels would need full retraining on the real sim. The competition description says the drone gets "general telemetry data and visual data" — gate positions will not be given directly, they must come from a CV pipeline. Designing the policy around a gate-relative observation now means the only swap at competition time is replacing the GateManager lookup with CV pipeline estimates. The policy and architecture require zero changes.

Additionally, the obs/reward coupling is now much tighter: every reward shaping component (velocity toward gate, heading alignment, altitude error, proximity) has a direct corresponding signal in the observation. The agent can perceive exactly what the reward is shaping, which should accelerate learning.

### Expected outcome
- Faster convergence than the image-based policy (simpler obs, cleaner signal)
- Policy cannot memorize absolute gate positions — forced to generalize by design
- Re-trains hover + no-crash behavior quickly (reward structure unchanged)

---

## Prior history (pre-diary, reconstructed from git log)

### Phase 1 — Infrastructure & Setup
- **Goal**: Establish a functional training loop for persistent long runs.
- **Method**: Moved from local laptop to Google Colab (L4 GPU). Used `tmux` sessions so training survives closing the browser.
- **Result**: Enabled multi-million step runs.

### Phase 2 — Reward evolution

| Version | Strategy | Result |
| :--- | :--- | :--- |
| **v1 Baseline** | Simple distance shaping (`dist_delta`) + sparse gate bonuses | **Fail — Hovering Trap.** Agent hovered at start to avoid penalties while farming small shaping rewards. |
| **v2 Exploration** | Raised `TILT_THRESHOLD` to 45°, zeroed `TILT_PENALTY_SCALE` | **Fail — Entropy Collapse.** Stopped exploring after ~1.5M steps; couldn't find consistent path to gate. |
| **v3 Velocity Progress** | Switched to velocity projection: `dot(lin_vel, unit_vec_to_gate)` | **Breakthrough.** Broke the hovering cycle. Agent cleared 3/5 gates consistently. |

### Phase 3 — Technical fixes
- **Gate handoff logic**: Reset `_prev_dist` on gate passage to avoid penalizing the agent when the target shifts to a farther gate.
- **Stability penalties**: Added `ANG_VEL_PENALTY_SCALE = -0.10` to stop high-frequency wobbling / spin-outs.

### Last known status (before 2026-04-16 overhaul)
- Agent clears 3/5 gates consistently.
- Falls out of the air shortly after Gate 3 — likely a physical stall from extreme banking, or an OOB termination while maneuvering for Gate 4.
- Suspected cause: agent memorized the specific visual layout of the training course rather than learning generalizable gate-tracking behavior.
