# Drone Racing RL — Development Diary

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
