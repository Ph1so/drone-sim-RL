# RL Development Log: Drone Racing Task

## Phase 1: Infrastructure & Setup
* **Goal**: Establish a functional training loop that allows for persistent execution.
* **Method**: Transitioned from local laptop training to **Google Colab (L4 GPU)**.
* **Result**: Implemented `tmux` sessions to ensure training continues after closing the laptop, enabling multi-million step runs.

## Phase 2: Reward Policy Evolution
The following table tracks the iterative changes made to the `RewardComputer` to address specific behavioral failures encountered during training.

| Version | Primary Reward Strategy | Result / Observation |
| :--- | :--- | :--- |
| **v1: Baseline** | Simple distance shaping (`dist_delta`) and sparse gate bonuses. | **Failure: "Hovering Trap."** Agent learned to hover at the start to avoid penalties while collecting small shaping rewards. |
| **v2: Exploration** | Increased `TILT_THRESHOLD` to 45° and zeroed out `TILT_PENALTY_SCALE`. | **Failure: "Entropy Collapse."** Agent stopped exploring after ~1.5M steps because it couldn't find a consistent path to the gate. |
| **v3: Velocity Progress** | Switched from distance delta to **Velocity Projection** (`dot(lin_vel, unit_vec_to_gate)`). | **Breakthrough.** Successfully broke the hovering cycle. Agent cleared **3 out of 5 gates** consistently. |

## Phase 3: Technical Constraints & Fixes
* **Gate Handoff Logic**: Implemented a `_prev_dist` reset immediately upon gate passage to prevent the agent from being penalized when the target shifts to a further gate.
* **Observation Space Alignment**: Verified that the agent receives **Relative Vectors** (`gate_pos - drone_pos`) rather than world coordinates to ensure the policy generalizes across different gate positions.
* **Stability Penalties**: Implemented `ANG_VEL_PENALTY_SCALE` (-0.10) to mitigate high-frequency wobbling that previously led to "spinning out".

## Current Status
* **Milestone**: The agent successfully navigates 60% of the course.
* **Ongoing Issue**: The drone "falls out of the air" shortly after passing Gate 3.
* **Hypothesis**: This is likely due to a physical stall from extreme banking or an Out-of-Bounds (OOB) termination as the drone maneuvers for Gate 4.