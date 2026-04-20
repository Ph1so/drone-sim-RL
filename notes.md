# Drone Racing RL — Development Diary

---

## 2026-04-20 — Align fully with Swift architecture (Kaufmann et al., Nature 2023)

### Motivation
The codebase had drifted from the Swift paper across several patches: the observation was a
Dict with 5-D scalar gate descriptors, the network was a custom two-branch extractor with
LayerNorm and a fusion layer, the reward used a different perception formula and a gate bonus
that doesn't exist in the paper, and the crash penalty was 10× too large. This session
brought all four systems into exact alignment with Swift.

### Changes

**`envs/reward.py` — exact Swift reward**

- Changed `r_perc` from `λ₂ exp(−δ_cam / σ)` to `λ₂ exp(λ₃ δ_cam⁴)` with `λ₃ = −10.0`
  (this is the paper's equation 8 exactly; the old Gaussian approximation was incorrect)
- Removed `SIGMA_PERC = 0.5`, `GATE_PASS_BONUS = 5.0`, `r_gate_bonus`
- Changed `CRASH_PENALTY` from `−50.0` → `−5.0` (paper value)
- Unified collision + OOB into a single `r_crash` term (paper treats both as p_z < 0 / collision)
- Removed `r_ang_vel` (was not in Swift paper; physical angular velocity penalty was added
  as a workaround and is no longer needed once the reward is correctly formulated)

**`envs/drone_racing_env.py` — flat 31-D observation**

- `_observationSpace()`: changed from `spaces.Dict` → `spaces.Box(shape=(31,), dtype=np.float32)`
- `_computeObs()`: returns flat ndarray `[pos(3) | lin_vel(3) | rot_mat(9) | gate_corners(12) | prev_action(4)]`
  - Attitude: quaternion (x,y,z,w) → flattened 3×3 rotation matrix via `p.getMatrixFromQuaternion()`
    (avoids gimbal-lock discontinuities; see Zhou et al. 2019)
  - Gate: replaced 5-D `[rel_x, rel_y, rel_z, dist, yaw_err]` scalars with 4 corner positions
    in drone body frame (12-D), computed via `R^T @ (corner_world − pos_drone)` for each corner
  - Previous action: `self._last_action` appended as final 4 floats
- Removed `gate_noise_std` parameter and all Gaussian noise injection
- Removed `_single_gate_obs()`, `_compute_gate_obs()` helper methods
- Added `_compute_gate_corners_body_frame()` helper

**`train.py` — Swift MlpPolicy**

- Removed entire `GateObsExtractor` class and all related imports
- Removed `FEATURES_DIM`, `ENT_COEF` constants
- Changed `N_STEPS = 512` → `N_STEPS = 1500` (one rollout = one full episode; matches paper)
- Added `_LeakyReLU02 = functools.partial(nn.LeakyReLU, negative_slope=0.2)` (picklable
  for SubprocVecEnv; `negative_slope=0.2` matches paper's LeakyReLU activation)
- Changed policy from `"MultiInputPolicy"` → `"MlpPolicy"` with `net_arch=[128, 128]`
- Set `ent_coef=0.0` (not used in Swift; was 0.01)

**`evaluate.py` — sync with new policy**

- Replaced `from train import GateObsExtractor, FEATURES_DIM` with `from train import _LeakyReLU02`
- Updated `custom_objects` to use flat `MlpPolicy` with `net_arch=[128, 128], activation_fn=_LeakyReLU02`
- Updated `_BREAKDOWN_KEYS`: removed `r_gate_bonus`, `r_collision`, `r_oob`; added `r_crash`

### Architecture summary after this change

| Dimension | Before | After (Swift) |
|-----------|--------|---------------|
| Obs type | `Dict{telemetry:13, gate_obs:10}` | `Box(31,)` flat |
| Attitude | Quaternion (4-D) | Rotation matrix (9-D) |
| Gate rep | 5-D scalars per gate | 12-D body-frame corners |
| Prev action | Not included | ✓ 4-D |
| Policy net | GateObsExtractor + MultiInputPolicy | 2×128 MLP, LeakyReLU(0.2) |
| r_perc | exp(−δ/0.5) | exp(−10 δ⁴) |
| Gate bonus | +5.0/gate | None |
| Crash penalty | −50.0 | −5.0 |
| N_STEPS | 512 | 1500 |

### Training impact
Old weights are incompatible — observation shape and policy architecture both changed.
Must train from scratch.

---

## 2026-04-19 — Add flat gate passage bonus (GATE_PASS_BONUS = 5.0)

### Problem
After removing all gate bonuses, the drone reverted to catapulting at gate 1 and missing.
Root cause: `r_prog` only sees scalar distance to the gate **centre**. It cannot distinguish:
- Flying cleanly through the gate opening (correct)
- Crashing into the gate frame at speed (incorrect, gets collision penalty)
- Hovering near the gate centre without passing (exploits r_prog without passing)

The old escalating gate bonus (150×n) was doing double duty: incentivising passage AND
acting as the only signal distinguishing "through the opening" from "close but not through."
Pure distance-delta without any gate signal leaves the agent no reason to aim for the opening
specifically over any other path that reduces distance to the centre.

### Change (`envs/reward.py`)
Added `GATE_PASS_BONUS = 5.0` — flat (non-escalating) bonus per gate cleared.

- **Flat, not escalating**: avoids the previous problem where later gates dominated
  the return and the agent optimised gate count over flight quality.
- **Small (5.0)**: ~2× the r_prog earned on a single gate approach. Large enough to be
  a clear signal, small enough never to dominate the episode return.
- **Not a lap bonus**: no additional reward for completing a full lap — lap completion
  is already captured by accumulating 5× gate bonuses.

### Resume vs. scratch
Start from scratch. The previous best_model.zip was trained with escalating gate bonuses
(150×n) as the dominant signal. Its value function expects bonuses that no longer exist,
producing corrupted advantage estimates. The catapult-toward-gate prior is strong and
harmful — unlearning it is slower than relearning from a clean slate.

---

## 2026-04-19 — Remove gate/lap bonuses, fix transition spike, add ang-vel penalty

### Problem
After 3M steps with the Swift reward the agent learned to spin wildly after gate 3 while
still tumbling toward gate 4. Three related root causes:

1. **Gate bonuses dwarfed every other signal.** Gate bonuses (150+300+450 = 900 over 3
   gates) were ~1500× larger than the combined jerk + body-rate penalties (~−0.6 max).
   The agent optimised gate count and treated smoothness penalties as numerical noise.
   This is the same reward-domination problem that broke earlier versions — just with
   different terms overwhelming different penalties.

2. **Actual spinning was never penalised.** Swift's body-rate penalty (λ₅) penalises the
   *commanded* roll/pitch/yaw in the action, not the resulting physical rotation. The drone's
   dynamics amplified commands into ±75° oscillations, but the reward never saw it.
   Spinning was effectively free as long as r_prog stayed positive (which it did — the
   tumbling arc still reduced distance to G4).

3. **Distance-delta produced a large negative spike at gate transitions.** When a gate is
   passed, `gate_manager.update()` switches the target to the next gate *before* `compute()`
   runs, so `curr_dist` jumps from ~0 → large. This made r_prog = λ₁ × (0 − large) = a
   big negative reward on every gate passage. The gate bonus had been inadvertently
   masking this bug.

### Changes (`envs/reward.py`)

**Remove gate/lap bonuses**
- Removed `GATE_BASE_BONUS`, `LAP_COMPLETE_BONUS`, `r_gate_bonus`, `r_lap_bonus`
- Swift does not include these. With distance-delta as the primary signal, the agent is
  rewarded every step it moves toward a gate — no sparse bonus needed.
- Removing them also exposes the jerk/body-rate penalties which were previously invisible.

**Fix gate-transition spike**
- Added `if gate_passed: self._prev_dist = None` before the progress computation
- On the transition step, `_prev_dist` is None → r_prog = 0 (neutral) instead of a large
  negative spike
- Next step, _prev_dist is set to the new (large) distance, giving large positive r_prog
  as the drone approaches the new gate — correct behaviour

**Add angular velocity penalty (λ₆ = −0.02 on ‖ω‖²)**
- Directly penalises *physical* angular velocity, not just commanded rates
- Sizing: at ±75° oscillation ‖ω‖² ≈ 10–20 rad²/s² → costs −0.2 to −0.4/step
  vs r_prog ≈ +0.3/step — spinning is now unprofitable
- Tight controlled banking (‖ω‖² ≈ 1–2) costs only −0.02–0.04/step — affordable

**Raise crash penalty from −5 → −50**
- Without gate bonuses dominating the return, the crash penalty needs to be meaningful
  relative to r_prog. A 3-gate episode earns ~30–50 in total r_prog; −5 was negligible.
  −50 ensures a crash always costs more than the agent can earn in progress alone.

### Updated reward table

| Term | Formula | Weight |
|---|---|---|
| `r_prog` | `λ₁ [d_{t-1} − d_t]` | λ₁ = 1.0 |
| `r_perc` | `λ₂ exp(−δ_cam / σ)` | λ₂ = 0.02 |
| `r_jerk` | `λ₄ ‖a_t − a_{t-1}‖²` | λ₄ = −2×10⁻⁴ |
| `r_body_rate` | `λ₅ ‖a_t^ω‖²` | λ₅ = −1×10⁻⁴ |
| `r_ang_vel` | `λ₆ ‖ω‖²` | λ₆ = −0.02 |
| Crash / OOB | binary | −50 |

### Resume command
```bash
python train.py --resume best_model.zip
```

### Expected outcome
- Spinning is directly penalised; the agent must trade off angular velocity against
  progress. Clean straight-line flight to the gate is the highest-return trajectory.
- Gate transitions no longer produce a negative spike — progress signal is smooth
  across gate passages.
- Crash penalty is meaningful relative to r_prog; the agent cannot ignore it.

---

## 2026-04-19 — Adopt Swift reward function (Kaufmann et al. 2023)

### Motivation
The hand-crafted reward had grown to 14 terms across two years of patches. Many terms
conflicted (vdown vs alt_align), others were proxies for the same objective (heading +
vel_gate_align + proximity all rewarded "be near and face the gate"). The Swift paper
provides a principled, minimal formulation validated at champion level — a clean reset.

### Changes

**`envs/reward.py` — full rewrite of reward terms**

Removed terms:
- `r_vel_progress` — tanh-saturated velocity projection
- `r_proximity` — smooth ramp inside 1.5 m of gate
- `r_heading` — yaw-forward alignment bonus
- `r_vel_gate_align` — velocity vs gate normal alignment
- `r_time` — per-step time penalty
- `r_tilt` — excess roll+pitch penalty
- `r_flip` — one-shot flip terminal penalty
- `r_ang_vel` — quadratic angular velocity penalty
- `r_alt_align` — altitude offset from gate
- `r_vdown` — downward velocity penalty below gate altitude

Added Swift terms:

| Term | Formula | Weight |
|---|---|---|
| `r_prog` | `λ₁ [d_{t-1} − d_t]` | λ₁ = 1.0 |
| `r_perc` | `λ₂ exp(−δ_cam / σ)` | λ₂ = 0.02, σ = 0.5 rad |
| `r_jerk` | `λ₄ ‖a_t − a_{t-1}‖²` | λ₄ = −2×10⁻⁴ |
| `r_body_rate` | `λ₅ ‖a_t^ω‖²` | λ₅ = −1×10⁻⁴ |

- `r_prog` replaces velocity-projection with distance-delta; requires tracking `_prev_dist`
- `r_perc` approximates gate-in-camera-FOV by computing the angle between body-forward
  (from RPY) and gate-centre direction; bounded in [0, 0.02]
- `r_jerk` penalises abrupt action changes; requires tracking `_prev_action`
- `r_body_rate` penalises large roll/pitch/yaw commands (action[1:4])
- Crash / OOB penalty reduced from −500 / −50 → −5.0 (Swift binary)
- Gate escalating bonus (150 × n) and lap bonus (500) retained — not in Swift but provide
  necessary sparse signal for the 5-gate racing task
- Flip termination kept (practical stability; not in Swift paper which uses a flight controller)
- `RewardComputer.reset()` now clears `_prev_dist` and `_prev_action`

**`envs/drone_racing_env.py` — action forwarding**
- Added `self._last_action = np.zeros(4)` in `__init__`
- `step()` captures `self._last_action = np.asarray(action)` before `super().step()`
- `_computeReward()` passes `action=self._last_action` to `compute()`

### Resume command
```bash
python train.py --resume best_model.zip
```
Resume path applies `lr=5e-5` (down from 3e-4) via `custom_objects` — lets the value
function re-calibrate to the new reward scale without shattering stable flight behavior.

### Expected outcome
- Cleaner reward signal: fewer conflicting gradients
- Jerk + body-rate penalties encourage the smooth trajectories needed for fast lap times
- Perception reward keeps the gate in "view" — relevant when swapping in a real CV pipeline
- Crash-avoidance behavior may loosen initially (penalty 100× smaller); watch early collision rate

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

## 2026-04-17 — Rebalance reward proportions

### Problem
At 2.5M training steps the agent reliably clears gates 1–2 but falls after gate 2. Reward
breakdown revealed the underlying cause: velocity progress (+1210/episode, +9.3/step) was
10× larger than every other signal. All constraint penalties were single-digit percentages of
the velocity reward — effectively noise the agent could ignore.

Specific failures in the old proportions:
- Collision penalty (-100) < 11 steps of vel_progress → crashing barely punished
- alt_align (-0.73/step) = 8% of vel_progress → altitude error had near-zero gradient
- Gate bonus (240 for 2 gates) < 2 seconds of vel_progress → gates weren't the primary objective

The priority order for a racing agent should be: **don't crash > pass gates > stay stable > fly fast**.
The old proportions had this inverted: fly fast dominated everything else.

### Changes (`envs/reward.py`)

| Constant | Old | New | Reason |
|---|---|---|---|
| `DIST_SHAPING_SCALE` | 12.0 | 5.0 | Reduce velocity reward so constraints are audible |
| `COLLISION_PENALTY` | -100.0 | -300.0 | Must be catastrophic; was only 11 steps of vel_progress |
| `GATE_BASE_BONUS` | 80.0 | 150.0 | Gates are the primary objective; make each passage dominant |
| `ALT_ALIGN_SCALE` | -1.5 | -3.0 | Was 8% of vel_progress; needs to actually constrain behavior |

### Expected new proportions
- vel_progress: ~500/episode (5/12 × 1210)
- gate_bonus: ~450/episode (150+300 for 2 gates)
- alt_align: ~-190/episode (-3.0/-1.5 × 95)
- collision: -300 (catastrophic, 2× gate-1 bonus)

Gate passage now rivals vel_progress as the dominant positive signal. Crashing costs more
than clearing gate 1 earns, restoring the catapult-is-net-negative property.

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

## 2026-04-17 — Fix course memorization: replace absolute pos with gate-relative pos in telemetry

### Problem
Generalization test (`--gate_offset 3.0 2.0 0.0`) showed 0 gates passed vs 2/2 on the
original course. The `vel_gate_align` breakdown was strongly positive (+217) even with 0 gates
cleared — the drone flew its trained path, just in the wrong direction relative to the shifted
gates. Root cause: `telemetry[0:3]` contained raw world coordinates. The network learned
implicit waypoints ("when pos ≈ [0, 0, 0.3] → execute gate-1 launch") rather than using the
gate_obs relative signal it was designed for.

### Change (`envs/drone_racing_env.py`, `_computeObs()`)

Replaced absolute `pos = state[0:3]` with gate-relative `pos_rel = pos_world - gate_pos`:

```python
# Before
pos     = state[0:3].astype(np.float32)
telemetry = np.concatenate([pos, quat, lin_vel, ang_vel])   # (13,)

# After
pos_world = state[0:3].astype(np.float32)
gate_pos  = self._gate_manager.current_gate.position.astype(np.float32)
pos_rel   = pos_world - gate_pos          # gate-relative position (3,)
telemetry = np.concatenate([pos_rel, quat, lin_vel, ang_vel])   # (13,)
```

Observation space shape and bounds unchanged (still 13 floats, unbounded). Docstring updated.

### Why this approach
- Absolute pos allowed the network to memorize course geometry as implicit waypoints
- Gate-relative pos forces the network to localize itself w.r.t. the current target gate
- The same gate_manager lookup already ran for reward computation — zero new coupling
- At competition time: replace `gate_manager.current_gate.position` with CV pipeline estimate
  of gate centre in world frame — no network or architecture changes required

### Training impact
Must retrain from scratch. Old weights expected absolute coordinates in the first 3 inputs of
the telemetry branch; those expectations are incompatible with gate-relative inputs.

### Training command
```bash
python train.py --num_gates 2 --timesteps 8_000_000 --spawn_mid_course_prob 0.0
```
Start from scratch with 2 gates. Expand to 5 once 2/2 laps complete consistently.

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
