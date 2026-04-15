"""
RewardComputer — modular reward shaping for the drone racing task.

Reward components
-----------------
1.  Distance shaping      : proportional to reduction in distance to next gate
2.  Proximity bonus       : smooth ramp reward inside 1.5 m of gate centre
3.  Heading alignment     : yaw-forward dot-product vs direction to next gate
4.  Velocity-gate align   : velocity direction dot-product vs gate normal
5.  Gate bonus            : large one-off reward on clean gate passage
6.  Time penalty          : small negative reward per step (encourages speed)
7.  Tilt penalty          : negative reward for excessive roll / pitch
8.  Angular vel penalty   : quadratic penalty on angular velocity magnitude
9.  Collision penalty     : large negative reward + episode ends
10. Out-of-bounds         : medium negative reward + episode ends
11. Lap completion        : bonus on completing all gates
"""

from __future__ import annotations

import numpy as np

from .gate_manager import GateManager


# ── Boundary box (world coordinates) ─────────────────────────────────────────
# x_min, y_min, z_min, x_max, y_max, z_max
WORLD_BOUNDS = np.array([-3.0, -3.0, 0.05, 12.0, 10.0, 6.0], dtype=np.float64)


class RewardComputer:
    """
    Stateful reward calculator (holds previous distance for shaping).

    Must call :meth:`reset` at the start of every episode.
    """

    # ── Reward coefficients ────────────────────────────────────────────
    DIST_SHAPING_SCALE:    float = 6.0    # reward per metre gained toward gate
    PROXIMITY_SCALE:       float = 0.5    # max bonus/step inside capture radius
    PROXIMITY_RADIUS:      float = 1.5    # metres — gate capture zone
    HEADING_SCALE:         float = 1.5    # max reward/step at perfect yaw alignment
    VEL_GATE_ALIGN_SCALE:  float = 0.5    # max reward/step when velocity || gate normal
    GATE_PASS_BONUS:       float = 100.0
    LAP_COMPLETE_BONUS:    float = 500.0
    TIME_PENALTY:          float = -0.05  # per step
    TILT_THRESHOLD:        float = np.deg2rad(45)  # 20° combined roll+pitch
    TILT_PENALTY_SCALE:    float = -2.0
    ANG_VEL_PENALTY_SCALE: float = -0.10  # × ||omega||^2 per step
    COLLISION_PENALTY:     float = -100.0
    OOB_PENALTY:           float = -50.0

    def __init__(self, gate_manager: GateManager) -> None:
        self._gm               = gate_manager
        self._prev_dist: float = 0.0

    # ------------------------------------------------------------------
    def reset(self, drone_pos: np.ndarray) -> None:
        """Initialise distance baseline at episode start."""
        self._prev_dist = self._gm.dist_to_next(drone_pos)

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

        # 2. Distance shaping (progress toward current target gate)
        curr_dist  = self._gm.dist_to_next(drone_pos)
        dist_delta = self._prev_dist - curr_dist
        reward    += self.DIST_SHAPING_SCALE * dist_delta
        self._prev_dist = curr_dist
        info["dist_to_gate"] = round(curr_dist, 4)
        info["dist_delta"]   = round(dist_delta, 4)

        # 2b. Proximity bonus — smooth ramp into the sparse gate bonus
        if curr_dist < self.PROXIMITY_RADIUS:
            prox    = self.PROXIMITY_SCALE * (1.0 - curr_dist / self.PROXIMITY_RADIUS)
            reward += prox
            info["proximity_bonus"] = round(prox, 4)

        # 3. Heading alignment (yaw-forward vs 2-D direction to gate)
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

        # 4. Gate passage bonus
        if gate_passed:
            reward += self.GATE_PASS_BONUS
            self._prev_dist = self._gm.dist_to_next(drone_pos)
            info["gate_passed"] = True
            if self._gm.lap_complete:
                reward += self.LAP_COMPLETE_BONUS
                info["lap_complete"] = True

        # 5. Tilt penalty (penalise nose-over / excessive banking)
        tilt = abs(drone_rpy[0]) + abs(drone_rpy[1])
        if tilt > self.TILT_THRESHOLD:
            reward += self.TILT_PENALTY_SCALE * (tilt - self.TILT_THRESHOLD)
            info["tilt_penalty"] = True

        # 5b. Angular velocity penalty — quadratic, punishes spinning/wobbling
        ang_sq  = float(np.dot(drone_ang_vel, drone_ang_vel))
        reward += self.ANG_VEL_PENALTY_SCALE * ang_sq
        info["ang_vel_sq"] = round(ang_sq, 4)

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
