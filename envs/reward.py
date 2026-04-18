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
                              gate 1 = 150, gate 2 = 300, … gate 5 = 750
                              gate 1 bonus (150) < crash penalty (300) → catapult is net-negative
6.  Time penalty            : small negative reward per step (encourages speed)
7.  Tilt penalty            : negative reward for excessive roll / pitch
8.  Angular vel penalty     : quadratic penalty on angular velocity magnitude
9.  Altitude alignment      : penalty proportional to |drone_z − gate_z| (scale -1.5)
10. Vertical velocity       : penalty on downward velocity ONLY when below gate altitude —
                              prevents sinking to floor without fighting alt_align from above
11. Collision penalty       : large negative reward + episode ends
12. Out-of-bounds           : medium negative reward + episode ends
13. Lap completion          : bonus on completing all gates
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
    DIST_SHAPING_SCALE:    float = 10.0   # max reward/step at velocity saturation
    PROGRESS_SAT:          float = 1.0    # m/s at which velocity reward hits ~76% of max (lowered to reward slow progress)
    PROXIMITY_SCALE:       float = 0.5    # max bonus/step inside capture radius
    PROXIMITY_RADIUS:      float = 1.5    # metres — gate capture zone
    HEADING_SCALE:         float = 0.2    # max reward/step at perfect yaw alignment
    VEL_GATE_ALIGN_SCALE:  float = 1.0    # max reward/step when velocity || gate normal
    GATE_BASE_BONUS:       float = 150.0  # multiplied by num_gates_cleared (escalating; raised from 80)
    LAP_COMPLETE_BONUS:    float = 500.0
    TIME_PENALTY:          float = -0.1   # per step
    TILT_THRESHOLD:        float = np.deg2rad(45)  # combined roll+pitch limit
    TILT_PENALTY_SCALE:    float = -0.5
    ANG_VEL_PENALTY_SCALE: float = -0.02  # × ||omega||^2 per step (reduced to allow banked turns)
    ALT_ALIGN_SCALE:       float = -1.5   # × |drone_z − gate_z| per step
    VDOWN_PENALTY_SCALE:   float = -3.0   # × abs(vz) when below gate_z — stronger anti-descent signal
    COLLISION_PENALTY:     float = -300.0
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
        info: dict = {}

        # Per-step reward contributions (all initialised to 0.0).
        r_time           = 0.0
        r_vel_progress   = 0.0
        r_proximity      = 0.0
        r_heading        = 0.0
        r_vel_gate_align = 0.0
        r_gate_bonus     = 0.0
        r_lap_bonus      = 0.0
        r_tilt           = 0.0
        r_ang_vel        = 0.0
        r_alt_align      = 0.0
        r_vdown          = 0.0
        r_collision      = 0.0
        r_oob            = 0.0

        # 1. Time penalty
        r_time = self.TIME_PENALTY

        # 2. Velocity-based progress toward next gate — saturated via tanh.
        #    Raw progress > 0  → flying toward gate  (reward, capped at DIST_SHAPING_SCALE)
        #    Raw progress = 0  → hovering            (zero shaping)
        #    Raw progress < 0  → flying away         (penalty, floored at −DIST_SHAPING_SCALE)
        #    Saturation prevents catapult: marginal reward for going faster than
        #    PROGRESS_SAT m/s is near-zero, so max-thrust is no longer optimal.
        curr_dist = self._gm.dist_to_next(drone_pos)
        info["dist_to_gate"] = round(curr_dist, 4)
        if curr_dist > 1e-6:
            gate_unit      = (self._gm.current_gate.position - drone_pos) / curr_dist
            progress       = float(np.dot(drone_lin_vel, gate_unit))
            shaped         = float(np.tanh(progress / self.PROGRESS_SAT))
            r_vel_progress = self.DIST_SHAPING_SCALE * shaped
            info["vel_progress"] = round(progress, 4)

        # 2b. Proximity bonus — smooth ramp into the sparse gate bonus
        if curr_dist < self.PROXIMITY_RADIUS:
            r_proximity = self.PROXIMITY_SCALE * (1.0 - curr_dist / self.PROXIMITY_RADIUS)
            info["proximity_bonus"] = round(r_proximity, 4)

        # 3. Heading alignment (yaw-forward vs 2-D direction to gate).
        #    Kept small (0.2) so the agent cannot farm reward by just facing the gate.
        yaw        = drone_rpy[2]
        drone_fwd  = np.array([np.cos(yaw), np.sin(yaw)])
        to_gate_2d = self._gm.current_gate.position[:2] - drone_pos[:2]
        dist_2d    = np.linalg.norm(to_gate_2d)
        if dist_2d > 1e-6:
            alignment = float(np.dot(drone_fwd, to_gate_2d / dist_2d))
            r_heading = self.HEADING_SCALE * max(0.0, alignment)
            info["heading_alignment"] = round(alignment, 4)

        # 3b. Velocity-gate-normal alignment (rewards moving through the gate
        #     at the correct approach angle, not just facing it)
        vel_norm = float(np.linalg.norm(drone_lin_vel))
        if vel_norm > 0.5:
            vel_unit         = drone_lin_vel / vel_norm
            vel_align        = float(np.dot(vel_unit, self._gm.current_gate.normal))
            r_vel_gate_align = self.VEL_GATE_ALIGN_SCALE * max(0.0, vel_align)
            info["vel_gate_align"] = round(vel_align, 4)

        # 4. Gate passage bonus — escalates with number of gates cleared.
        #    gate 1 = 80, gate 2 = 160, … gate N = 80*N.
        #    Gate 1 bonus (80) < crash penalty (100): catapult-and-crash is net-negative.
        if gate_passed:
            r_gate_bonus = self.GATE_BASE_BONUS * self._gm.num_passed
            info["gate_passed"] = True
            info["gate_bonus"]  = round(r_gate_bonus, 1)
            if self._gm.lap_complete:
                r_lap_bonus      = self.LAP_COMPLETE_BONUS
                info["lap_complete"] = True

        # 5. Tilt penalty (penalise nose-over / excessive banking)
        tilt = abs(drone_rpy[0]) + abs(drone_rpy[1])
        if tilt > self.TILT_THRESHOLD:
            r_tilt = self.TILT_PENALTY_SCALE * (tilt - self.TILT_THRESHOLD)
            info["tilt_penalty"] = True

        # 5b. Angular velocity penalty — quadratic, punishes spinning/wobbling.
        #     Kept small so rapid banking for tight turns remains affordable.
        ang_sq    = float(np.dot(drone_ang_vel, drone_ang_vel))
        r_ang_vel = self.ANG_VEL_PENALTY_SCALE * ang_sq
        info["ang_vel_sq"] = round(ang_sq, 4)

        # 5c. Altitude alignment — penalise vertical offset from the next gate.
        #     Teaches throttle compensation during banked turns so the drone
        #     holds gate altitude instead of sinking mid-manoeuvre.
        alt_err     = abs(drone_pos[2] - self._gm.current_gate.position[2])
        r_alt_align = self.ALT_ALIGN_SCALE * alt_err
        info["alt_error"] = round(alt_err, 4)

        # 5d. Vertical velocity penalty — fires only when the drone is sinking AND already
        #     below gate altitude.  If the drone is above gate_z, descending is correct
        #     behavior (alt_align already provides that gradient); penalising it here would
        #     fight alt_align and cause the drone to hover above the gate opening.
        #     Sign: VDOWN_PENALTY_SCALE is negative; abs(vz) is positive → product is negative (penalty).
        vz     = float(drone_lin_vel[2])
        gate_z = self._gm.current_gate.position[2]
        if vz < 0.0 and drone_pos[2] < gate_z:
            r_vdown = self.VDOWN_PENALTY_SCALE * abs(vz)   # negative scale × positive speed = penalty
            info["vdown_penalty"] = round(r_vdown, 4)

        # 6. Collision
        if collision:
            r_collision      = self.COLLISION_PENALTY
            info["collision"] = True

        # 7. Out of bounds
        if self._is_oob(drone_pos):
            r_oob                = self.OOB_PENALTY
            info["out_of_bounds"] = True

        reward = (
            r_time + r_vel_progress + r_proximity + r_heading + r_vel_gate_align
            + r_gate_bonus + r_lap_bonus + r_tilt + r_ang_vel + r_alt_align
            + r_vdown + r_collision + r_oob
        )

        info["num_gates_passed"] = self._gm.num_passed
        info.update({
            "r_time":           r_time,
            "r_vel_progress":   r_vel_progress,
            "r_proximity":      r_proximity,
            "r_heading":        r_heading,
            "r_vel_gate_align": r_vel_gate_align,
            "r_gate_bonus":     r_gate_bonus,
            "r_lap_bonus":      r_lap_bonus,
            "r_tilt":           r_tilt,
            "r_ang_vel":        r_ang_vel,
            "r_alt_align":      r_alt_align,
            "r_vdown":          r_vdown,
            "r_collision":      r_collision,
            "r_oob":            r_oob,
        })
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
