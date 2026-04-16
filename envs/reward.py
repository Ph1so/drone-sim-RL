"""
RewardComputer — modular reward shaping for the drone racing task.

Reward components
-----------------
1.  Velocity-based progress : tanh(dot(lin_vel, unit_vec_to_gate) / PROGRESS_SAT)
                              saturates at DIST_SHAPING_SCALE — no unbounded catapult reward
2.  Proximity bonus         : smooth ramp reward inside 1.5 m of gate centre
3.  Heading alignment       : yaw-forward dot-product vs direction to next gate
4.  Velocity-gate align     : velocity direction dot-product vs gate normal
5.  Gate bonus              : escalating per-gate reward (GATE_BASE_BONUS × gates_cleared)
                              gate 1 = 80, gate 2 = 160, … gate 5 = 400
                              gate 1 bonus (80) < crash penalty (100) → catapult is net-negative
6.  Time penalty            : small negative reward per step (encourages speed)
7.  Tilt penalty            : negative reward for excessive roll / pitch
8.  Angular vel penalty     : quadratic penalty on angular velocity magnitude
9.  Altitude alignment      : penalty proportional to |drone_z − gate_z|
10. Collision penalty       : large negative reward + episode ends
11. Out-of-bounds           : medium negative reward + episode ends
12. Lap completion          : bonus on completing all gates
"""

from __future__ import annotations

import numpy as np

from .gate_manager import GateManager


# ── Boundary box (world coordinates) ─────────────────────────────────────────
# x_min, y_min, z_min, x_max, y_max, z_max
WORLD_BOUNDS = np.array([-3.0, -3.0, 0.05, 12.0, 10.0, 6.0], dtype=np.float64)


class RewardComputer:
    """
    Stateless reward calculator — no previous-distance bookkeeping needed
    because progress is measured via velocity projection, not dist_delta.

    Must call :meth:`reset` at the start of every episode.
    """

    # ── Reward coefficients ────────────────────────────────────────────
    DIST_SHAPING_SCALE:    float = 12.0   # max reward/step at velocity saturation
    PROGRESS_SAT:          float = 2.0    # m/s at which velocity reward hits ~76% of max
    PROXIMITY_SCALE:       float = 0.5    # max bonus/step inside capture radius
    PROXIMITY_RADIUS:      float = 1.5    # metres — gate capture zone
    HEADING_SCALE:         float = 0.2    # max reward/step at perfect yaw alignment
    VEL_GATE_ALIGN_SCALE:  float = 1.0    # max reward/step when velocity || gate normal
    GATE_BASE_BONUS:       float = 80.0   # multiplied by num_gates_cleared (escalating)
    LAP_COMPLETE_BONUS:    float = 500.0
    TIME_PENALTY:          float = -0.1   # per step
    TILT_THRESHOLD:        float = np.deg2rad(45)  # combined roll+pitch limit
    TILT_PENALTY_SCALE:    float = -0.5
    ANG_VEL_PENALTY_SCALE: float = -0.02  # × ||omega||^2 per step (reduced to allow banked turns)
    ALT_ALIGN_SCALE:       float = -0.4   # × |drone_z − gate_z| per step
    COLLISION_PENALTY:     float = -100.0
    OOB_PENALTY:           float = -50.0

    def __init__(self, gate_manager: GateManager) -> None:
        self._gm = gate_manager

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """No-op — velocity-based shaping requires no per-episode state."""
        pass

    # ------------------------------------------------------------------
    def compute(
        self,
        drone_pos:     np.ndarray,
        drone_rpy:     np.ndarray,
        drone_lin_vel: np.ndarray,
        drone_ang_vel: np.ndarray,
        gate_passed:   bool,
        collision:     bool,
    ) -> tuple[float, dict]:
        reward = 0.0
        info: dict = {}

        # 1. Time penalty
        reward += self.TIME_PENALTY

        # 2. Velocity-based progress toward next gate — saturated via tanh.
        #    Raw progress > 0  → flying toward gate  (reward, capped at DIST_SHAPING_SCALE)
        #    Raw progress = 0  → hovering            (zero shaping)
        #    Raw progress < 0  → flying away         (penalty, floored at −DIST_SHAPING_SCALE)
        #    Saturation prevents catapult: marginal reward for going faster than
        #    PROGRESS_SAT m/s is near-zero, so max-thrust is no longer optimal.
        curr_dist = self._gm.dist_to_next(drone_pos)
        info["dist_to_gate"] = round(curr_dist, 4)
        if curr_dist > 1e-6:
            gate_unit = (self._gm.current_gate.position - drone_pos) / curr_dist
            progress  = float(np.dot(drone_lin_vel, gate_unit))
            shaped    = float(np.tanh(progress / self.PROGRESS_SAT))
            reward   += self.DIST_SHAPING_SCALE * shaped
            info["vel_progress"] = round(progress, 4)

        # 2b. Proximity bonus — smooth ramp into the sparse gate bonus
        if curr_dist < self.PROXIMITY_RADIUS:
            prox    = self.PROXIMITY_SCALE * (1.0 - curr_dist / self.PROXIMITY_RADIUS)
            reward += prox
            info["proximity_bonus"] = round(prox, 4)

        # 3. Heading alignment (yaw-forward vs 2-D direction to gate).
        #    Kept small (0.2) so the agent cannot farm reward by just facing the gate.
        yaw        = drone_rpy[2]
        drone_fwd  = np.array([np.cos(yaw), np.sin(yaw)])
        to_gate_2d = self._gm.current_gate.position[:2] - drone_pos[:2]
        dist_2d    = np.linalg.norm(to_gate_2d)
        if dist_2d > 1e-6:
            alignment = float(np.dot(drone_fwd, to_gate_2d / dist_2d))
            reward   += self.HEADING_SCALE * max(0.0, alignment)
            info["heading_alignment"] = round(alignment, 4)

        # 3b. Velocity-gate-normal alignment (rewards moving through the gate
        #     at the correct approach angle, not just facing it)
        vel_norm = float(np.linalg.norm(drone_lin_vel))
        if vel_norm > 0.5:
            vel_unit  = drone_lin_vel / vel_norm
            vel_align = float(np.dot(vel_unit, self._gm.current_gate.normal))
            reward   += self.VEL_GATE_ALIGN_SCALE * max(0.0, vel_align)
            info["vel_gate_align"] = round(vel_align, 4)

        # 4. Gate passage bonus — escalates with number of gates cleared.
        #    gate 1 = 80, gate 2 = 160, … gate N = 80*N.
        #    Gate 1 bonus (80) < crash penalty (100): catapult-and-crash is net-negative.
        if gate_passed:
            gate_bonus = self.GATE_BASE_BONUS * self._gm.num_passed
            reward += gate_bonus
            info["gate_passed"]  = True
            info["gate_bonus"]   = round(gate_bonus, 1)
            if self._gm.lap_complete:
                reward += self.LAP_COMPLETE_BONUS
                info["lap_complete"] = True

        # 5. Tilt penalty (penalise nose-over / excessive banking)
        tilt = abs(drone_rpy[0]) + abs(drone_rpy[1])
        if tilt > self.TILT_THRESHOLD:
            reward += self.TILT_PENALTY_SCALE * (tilt - self.TILT_THRESHOLD)
            info["tilt_penalty"] = True

        # 5b. Angular velocity penalty — quadratic, punishes spinning/wobbling.
        #     Kept small so rapid banking for tight turns remains affordable.
        ang_sq  = float(np.dot(drone_ang_vel, drone_ang_vel))
        reward += self.ANG_VEL_PENALTY_SCALE * ang_sq
        info["ang_vel_sq"] = round(ang_sq, 4)

        # 5c. Altitude alignment — penalise vertical offset from the next gate.
        #     Teaches throttle compensation during banked turns so the drone
        #     holds gate altitude instead of sinking mid-manoeuvre.
        alt_err  = abs(drone_pos[2] - self._gm.current_gate.position[2])
        reward  += self.ALT_ALIGN_SCALE * alt_err
        info["alt_error"] = round(alt_err, 4)

        # 6. Collision
        if collision:
            reward += self.COLLISION_PENALTY
            info["collision"] = True

        # 7. Out of bounds
        if self._is_oob(drone_pos):
            reward += self.OOB_PENALTY
            info["out_of_bounds"] = True

        info["num_gates_passed"] = self._gm.num_passed
        return float(reward), info

    # ------------------------------------------------------------------
    def is_terminated(self, drone_pos: np.ndarray, collision: bool) -> bool:
        """Episode ends on collision, OOB, or lap completion."""
        return (
            collision
            or self._is_oob(drone_pos)
            or self._gm.lap_complete
        )

    @staticmethod
    def _is_oob(pos: np.ndarray) -> bool:
        return bool(
            np.any(pos[:3] < WORLD_BOUNDS[:3])
            or np.any(pos[:3] > WORLD_BOUNDS[3:])
        )
