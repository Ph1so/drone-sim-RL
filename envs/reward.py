"""
RewardComputer — Swift-style reward shaping for the drone racing task.

Based on: "Champion-level drone racing using deep reinforcement learning"
          Kaufmann et al., 2023.

Reward components
-----------------
1.  Progress (r_prog)      : λ₁ [d_{t-1} − d_t]
                             Rewards reduction in distance to the next gate centre.
                             _prev_dist is reset to None on gate passage to avoid a
                             large negative spike when the target switches to the next
                             (farther) gate.
2.  Perception (r_perc)    : λ₂ exp(−δ_cam / σ_perc)
                             Rewards keeping the next gate centred in the body-frame
                             forward direction (proxy for camera FOV in simulation).
3.  Jerk penalty (r_jerk)  : λ₄ ‖a_t − a_{t-1}‖²
                             Penalises large action changes between consecutive steps.
4.  Body-rate penalty      : λ₅ ‖a_t^ω‖²
   (r_body_rate)            Penalises large roll/pitch/yaw commands.
5.  Ang-vel penalty        : λ₆ ‖ω‖²
   (r_ang_vel)              Penalises actual physical angular velocity — directly
                             discourages post-gate spinning/tumbling.
6.  Gate passage bonus     : flat GATE_PASS_BONUS per gate cleared.
   (r_gate_bonus)           Distinguishes "flew through the opening" from "crashed into
                             the frame" — r_prog alone cannot make this distinction since
                             it only sees scalar distance to the gate centre.
                             Kept small (5.0) so it never dominates r_prog.
7.  Crash/OOB penalty      : binary, terminates episode
"""

from __future__ import annotations

import numpy as np

from .gate_manager import GateManager


# ── Boundary box (world coordinates) ─────────────────────────────────────────
# x_min, y_min, z_min, x_max, y_max, z_max
WORLD_BOUNDS = np.array([-3.0, -3.0, 0.05, 12.0, 10.0, 6.0], dtype=np.float64)

class RewardComputer:
    """
    Swift-style reward calculator.

    Tracks two pieces of per-episode state:
      - ``_prev_dist``   : distance to gate at the previous step (for progress delta)
      - ``_prev_action`` : action at the previous step (for jerk penalty)

    Must call :meth:`reset` at the start of every episode.
    """

    # ── Termination threshold (kept for practical stability) ──────────────
    FLIP_THRESHOLD: float = np.deg2rad(90)

    # ── Swift reward coefficients ──────────────────────────────────────────
    LAMBDA_1:      float = 1.0    # progress: distance-delta weight
    LAMBDA_2:      float = 0.02   # perception: gate-in-FOV weight
    SIGMA_PERC:    float = 0.5    # perception: angular bandwidth (rad, ≈28°)
    LAMBDA_4:      float = -2e-4   # jerk: ‖a_t − a_{t-1}‖² penalty weight
    LAMBDA_5:      float = -1e-4   # body-rate: ‖a_t^ω‖² penalty weight
    LAMBDA_6:      float = -0.02   # ang-vel: ‖ω‖² penalty weight (physical rotation)

    # ── Gate passage bonus ────────────────────────────────────────────────
    GATE_PASS_BONUS:    float = 5.0     # flat per-gate (not escalating) — just enough
                                        # to distinguish passage from near-miss/collision

    # ── Terminal rewards ───────────────────────────────────────────────────
    CRASH_PENALTY:      float = -50.0   # collision OR out-of-bounds

    def __init__(self, gate_manager: GateManager) -> None:
        self._gm           = gate_manager
        self._prev_dist:   float | None      = None
        self._prev_action: np.ndarray | None = None

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset per-episode state at the start of each episode."""
        self._prev_dist   = None
        self._prev_action = None

    # ------------------------------------------------------------------
    def compute(
        self,
        drone_pos:     np.ndarray,
        drone_rpy:     np.ndarray,
        drone_lin_vel: np.ndarray,   # unused by Swift formulation; kept for API compat
        drone_ang_vel: np.ndarray,   # unused by Swift formulation; kept for API compat
        action:        np.ndarray,
        gate_passed:   bool,
        collision:     bool,
    ) -> tuple[float, dict]:
        info: dict = {}

        # ── 1. Progress reward ────────────────────────────────────────────
        # r_prog = λ₁ [d_{t-1} − d_t]  (positive when closing in on gate)
        #
        # Gate-transition reset: when a gate is passed, gate_manager has already
        # switched the target to the next (farther) gate before compute() runs.
        # Without the reset, curr_dist would jump from ~0 → large, making
        # r_prog = λ₁ × (0 − large) = large negative spike on every gate passage.
        # Resetting _prev_dist to None gives 0 progress on that one step instead.
        curr_dist = self._gm.dist_to_next(drone_pos)
        info["dist_to_gate"] = round(curr_dist, 4)

        if gate_passed:
            self._prev_dist = None   # suppress transition spike

        r_prog = 0.0
        if self._prev_dist is not None:
            r_prog = self.LAMBDA_1 * (self._prev_dist - curr_dist)
        self._prev_dist = curr_dist
        info["r_prog"] = round(r_prog, 6)

        # ── 2. Perception reward ──────────────────────────────────────────
        # r_perc = λ₂ · exp(−δ_cam / σ_perc)
        # δ_cam: angle between body-frame forward axis and gate centre direction.
        # Body-forward in world frame via ZYX rotation matrix first column:
        #   fwd = [ cos(p)cos(y),  cos(p)sin(y),  sin(p) ]
        cp = np.cos(drone_rpy[1]);  sp = np.sin(drone_rpy[1])
        cy = np.cos(drone_rpy[2]);  sy = np.sin(drone_rpy[2])
        fwd = np.array([cp * cy, cp * sy, sp])

        gate_dir = (self._gm.current_gate.position - drone_pos) / max(curr_dist, 1e-6)
        cos_a    = float(np.clip(np.dot(fwd, gate_dir), -1.0, 1.0))
        delta_cam = np.arccos(cos_a)
        r_perc   = self.LAMBDA_2 * np.exp(-delta_cam / self.SIGMA_PERC)
        info["delta_cam"]  = round(float(delta_cam), 4)
        info["r_perc"]     = round(float(r_perc), 6)

        # ── 3. Jerk penalty ───────────────────────────────────────────────
        # r_jerk = λ₄ · ‖a_t − a_{t-1}‖²
        r_jerk = 0.0
        if self._prev_action is not None:
            da     = action - self._prev_action
            r_jerk = self.LAMBDA_4 * float(np.dot(da, da))
        self._prev_action = action.copy()
        info["r_jerk"] = round(r_jerk, 6)

        # ── 4. Body-rate penalty ──────────────────────────────────────────
        # r_body_rate = λ₅ · ‖a_t^ω‖²  (roll, pitch, yaw channels)
        omega_cmd    = action[1:4]
        r_body_rate  = self.LAMBDA_5 * float(np.dot(omega_cmd, omega_cmd))
        info["r_body_rate"] = round(r_body_rate, 6)

        # ── 5. Angular velocity penalty ───────────────────────────────────
        # r_ang_vel = λ₆ · ‖ω‖²  (actual physical rotation, not commanded)
        # Directly penalises spinning/tumbling regardless of what action caused it.
        ang_sq    = float(np.dot(drone_ang_vel, drone_ang_vel))
        r_ang_vel = self.LAMBDA_6 * ang_sq
        info["ang_vel_sq"] = round(ang_sq, 4)
        info["r_ang_vel"]  = round(r_ang_vel, 6)

        # ── 6. Gate passage bonus ─────────────────────────────────────────────
        # Flat (non-escalating) bonus that fires only when the drone actually
        # passes through the gate opening.  r_prog cannot distinguish "close to
        # gate centre" from "through the gate" because it only sees scalar
        # distance.  This small bonus provides that missing binary signal.
        r_gate_bonus = 0.0
        if gate_passed:
            r_gate_bonus         = self.GATE_PASS_BONUS
            info["gate_passed"]  = True
            info["r_gate_bonus"] = r_gate_bonus
            if self._gm.lap_complete:
                info["lap_complete"] = True

        # ── 7. Crash / OOB penalty ────────────────────────────────────────
        r_collision = 0.0
        r_oob       = 0.0
        if collision:
            r_collision      = self.CRASH_PENALTY
            info["collision"] = True
        if self._is_oob(drone_pos):
            r_oob                = self.CRASH_PENALTY
            info["out_of_bounds"] = True

        reward = r_prog + r_perc + r_jerk + r_body_rate + r_ang_vel + r_gate_bonus + r_collision + r_oob

        info["num_gates_passed"] = self._gm.num_passed
        info.update({
            "r_collision": r_collision,
            "r_oob":       r_oob,
        })
        return float(reward), info

    # ------------------------------------------------------------------
    def is_terminated(
        self,
        drone_pos:  np.ndarray,
        collision:  bool,
        drone_rpy:  np.ndarray | None = None,
    ) -> bool:
        """Episode ends on collision, OOB, lap completion, or flip past 90°."""
        flip = (
            drone_rpy is not None
            and (
                abs(drone_rpy[0]) > self.FLIP_THRESHOLD
                or abs(drone_rpy[1]) > self.FLIP_THRESHOLD
            )
        )
        return collision or self._is_oob(drone_pos) or self._gm.lap_complete or flip

    @staticmethod
    def _is_oob(pos: np.ndarray) -> bool:
        return bool(
            np.any(pos[:3] < WORLD_BOUNDS[:3])
            or np.any(pos[:3] > WORLD_BOUNDS[3:])
        )
