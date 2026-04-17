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

## 2026-04-17 — Fix slow descent after gate 2

### Observation
After resuming training with `--num_gates 4`, the agent learned to pass gates 1 and 2 with
controlled, responsible turns (~500k steps). However, after clearing gate 2, it began a slow,
controlled descent until contacting the floor — not a crash, a deliberate sink.

### Root cause (diagnosed)
1. Velocity reward goes near-zero after gate 2 (drone still moving in gate 2's direction; gate 3 is perpendicular) — time penalty dominates with no forward signal
2. Banking during the gate 2 turn reduces effective lift; no reward signal to compensate throttle afterward
3. Noisy `rel_z` near true zero (gate 3 and drone at same altitude) provides ambiguous altitude signal
4. Gate 3+ states severely underexplored — PPO value function has low, inaccurate estimates there

### Changes

**Fix A — Vertical velocity penalty (`envs/reward.py`)**
- Added `VDOWN_PENALTY_SCALE = -3.0`
- `reward += -3.0 × min(0, vz)` — fires the moment the drone starts sinking
- At 0.3 m/s descent: costs 0.9/step (9× the time penalty); zero cost when holding or climbing
- Catches descent early, before altitude error compounds and credit assignment becomes too distant

**Fix B — Stronger altitude alignment (`envs/reward.py`)**
- Tripled `ALT_ALIGN_SCALE` from `-0.4` → `-1.5`
- Makes even small altitude drift immediately costly; acts as early warning before large error accumulates

**Fix C — Mid-course spawn randomization (`envs/gate_manager.py`, `envs/drone_racing_env.py`, `train.py`)**
- New `GateManager.fast_forward_to(k)`: marks gates 0..k-1 as passed, sets target to gate k
- New `DroneRacingEnv(spawn_mid_course_prob=0.3)`: with 30% probability, teleports drone to
  1.5m past gate k-1's exit, facing gate k, zero velocity; k chosen randomly from 1..num_gates-1
- Uses PyBullet `resetBasePositionAndOrientation` + `resetBaseVelocity` after `super().reset()`
- New `--spawn_mid_course_prob FLOAT` CLI arg in `train.py` (default 0.0; use 0.3 for training)
- Eval env always uses `spawn_mid_course_prob=0.0` for clean benchmark metrics
- Directly populates gate 3+ experience in PPO rollout buffer; no change to obs/action space

### Resume command
```bash
python train.py --resume {DRIVE}/best_model/best_model.zip \
    --num_gates 4 --timesteps 5_000_000 --spawn_mid_course_prob 0.3
```

### Expected outcome
- Vertical velocity penalty prevents gradual descent; agent must maintain thrust through turns
- Stronger altitude alignment catches small drifts before they become large errors
- Mid-course spawns ensure gate 3/4 transitions are trained, not just incidentally visited

---

## 2026-04-17 — Fix altitude overcorrection between Gate 1 → Gate 2

### Observation
After deploying the vdown penalty (`-3.0`) and tripled alt_align (`-1.5`), the drone began
flying noticeably *higher* during the G1→G2 transit before descending to the floor after G2.
The slow descent after gate 2 persisted even with `--spawn_mid_course_prob 0.3`.

### Root cause
Reward conflict between two altitude signals:
- `VDOWN_PENALTY_SCALE = -3.0` was too aggressive — reward gradient strongly favored climbing
  any time vz < 0, causing overcorrection above gate altitude (z > 1.5 m) between G1→G2
- `ALT_ALIGN_SCALE = -1.5` then penalized the drone for being *above* gate altitude, producing
  a descent gradient — the exact behavior we wanted to prevent
- The two penalties fought each other: vdown pushed up, alt_align pushed down; in the
  underexplored post-gate-2 states the policy settled on descending as the net minimum

### Change

**Reduce `VDOWN_PENALTY_SCALE` from `-3.0` to `-1.5` (`envs/reward.py`)**
- Now matches `ALT_ALIGN_SCALE` magnitude — both altitude signals have the same scale
- At 0.3 m/s descent: costs 0.45/step (4.5× time penalty) — still strong, not overcorrecting
- Removes the contradictory gradient that caused climbing above gate altitude

### Expected outcome
- Drone holds gate altitude more tightly between G1→G2 (no overcorrection)
- After gate 2, vdown and alt_align now pull in a consistent direction — both discourage
  both climbing past gate altitude AND descending below it
- Descent-to-floor behavior should disappear once gate 3 states are explored via mid-course spawns

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
