"""
RewardComputer — Swift-style reward shaping for the drone racing task.

Based on: "Champion-level drone racing using deep reinforcement learning"
          Kaufmann et al., 2023.

Reward components
-----------------
1.  Progress (r_prog)      : λ₁ [d_{t-1} − d_t]
                             Rewards reduction in distance to the next gate centre.
2.  Perception (r_perc)    : λ₂ exp(−δ_cam / σ_perc)
                             Rewards keeping the next gate centred in the body-frame
                             forward direction (proxy for camera FOV in simulation).
3.  Jerk penalty (r_jerk)  : λ₄ ‖a_t − a_{t-1}‖²
                             Penalises large action changes between consecutive steps.
4.  Body-rate penalty      : λ₅ ‖a_t^ω‖²
   (r_body_rate)            Penalises large roll/pitch/yaw commands.
5.  Gate bonus             : escalating per-gate reward (GATE_BASE_BONUS × gates_cleared)
6.  Lap completion bonus
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
    LAMBDA_4:      float = -2e-4  # jerk: ‖a_t − a_{t-1}‖² penalty weight
    LAMBDA_5:      float = -1e-4  # body-rate: ‖a_t^ω‖² penalty weight

    # ── Sparse / terminal rewards ──────────────────────────────────────────
    GATE_BASE_BONUS:    float = 150.0   # × num_gates_cleared (escalating)
    LAP_COMPLETE_BONUS: float = 500.0
    CRASH_PENALTY:      float = -5.0    # collision OR out-of-bounds (Swift §3)

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
        curr_dist = self._gm.dist_to_next(drone_pos)
        info["dist_to_gate"] = round(curr_dist, 4)

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

        # ── 5. Gate passage bonus (escalating) ───────────────────────────
        r_gate_bonus = 0.0
        r_lap_bonus  = 0.0
        if gate_passed:
            r_gate_bonus = self.GATE_BASE_BONUS * self._gm.num_passed
            info["gate_passed"] = True
            info["gate_bonus"]  = round(r_gate_bonus, 1)
            if self._gm.lap_complete:
                r_lap_bonus          = self.LAP_COMPLETE_BONUS
                info["lap_complete"] = True

        # ── 6. Crash / OOB penalty ────────────────────────────────────────
        r_collision = 0.0
        r_oob       = 0.0
        if collision:
            r_collision      = self.CRASH_PENALTY
            info["collision"] = True
        if self._is_oob(drone_pos):
            r_oob                = self.CRASH_PENALTY
            info["out_of_bounds"] = True

        reward = r_prog + r_perc + r_jerk + r_body_rate + r_gate_bonus + r_lap_bonus + r_collision + r_oob

        info["num_gates_passed"] = self._gm.num_passed
        info.update({
            "r_gate_bonus": r_gate_bonus,
            "r_lap_bonus":  r_lap_bonus,
            "r_collision":  r_collision,
            "r_oob":        r_oob,
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
