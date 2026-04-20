"""
RewardComputer — Swift reward (Kaufmann et al., Nature 2023).

Exact formulation from the paper:

    r_t = r_prog + r_perc + r_cmd − r_crash

    r_prog  = λ₁ [d_{t-1}^Gate − d_t^Gate]           λ₁ = 1.0
    r_perc  = λ₂ exp(λ₃ · δ_cam⁴)                    λ₂ = 0.02, λ₃ = −10.0
    r_cmd   = λ₄ ‖a_t^ω‖² + λ₅ ‖a_t − a_{t-1}‖²    λ₄ = −2e-4, λ₅ = −1e-4
    r_crash = 5.0  if p_z < 0 OR collision; else 0   (terminates episode)

Notes
-----
- No gate passage bonus (not in the paper).
- r_crash is a positive constant that is subtracted from total reward,
  implemented here as CRASH_PENALTY = −5.0 added to the total (equivalent).
- _prev_dist is reset to None on gate passage to suppress the distance-spike
  that occurs when the target switches to the next (farther) gate.
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

    # ── Swift reward coefficients (Extended Data Table 1a) ────────────────
    LAMBDA_1: float = 1.0     # progress: distance-delta weight
    LAMBDA_2: float = 0.02    # perception: gate-in-FOV weight
    LAMBDA_3: float = -10.0   # perception: δ_cam^4 shaping exponent
    LAMBDA_4: float = -4e-4   # body-rate: ‖a_t^ω‖² penalty weight
    LAMBDA_5: float = -1e-3   # jerk: ‖a_t − a_{t-1}‖² penalty weight # TODO: testing a bigger penalty 

    # ── Terminal reward ────────────────────────────────────────────────────
    # Paper: subtract r_crash=5.0; implemented as adding CRASH_PENALTY=−5.0.
    CRASH_PENALTY: float = -5.0

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
        drone_lin_vel: np.ndarray,   # unused; kept for API compat
        drone_ang_vel: np.ndarray,   # unused; kept for API compat
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
        # r_perc = λ₂ · exp(λ₃ · δ_cam⁴)   (Kaufmann et al. eq. 8)
        # δ_cam: angle between body-frame forward axis and gate centre direction.
        # Body-forward in world frame via ZYX rotation matrix first column:
        #   fwd = [ cos(p)cos(y),  cos(p)sin(y),  sin(p) ]
        cp = np.cos(drone_rpy[1]);  sp = np.sin(drone_rpy[1])
        cy = np.cos(drone_rpy[2]);  sy = np.sin(drone_rpy[2])
        fwd = np.array([cp * cy, cp * sy, sp])

        gate_dir  = (self._gm.current_gate.position - drone_pos) / max(curr_dist, 1e-6)
        cos_a     = float(np.clip(np.dot(fwd, gate_dir), -1.0, 1.0))
        delta_cam = float(np.arccos(cos_a))
        r_perc    = self.LAMBDA_2 * np.exp(self.LAMBDA_3 * delta_cam ** 4)
        info["delta_cam"] = round(delta_cam, 4)
        info["r_perc"]    = round(float(r_perc), 6)

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

        # ── 5. Gate passage tracking (info only — no bonus reward) ───────────
        if gate_passed:
            info["gate_passed"] = True
            if self._gm.lap_complete:
                info["lap_complete"] = True

        # ── 6. Crash penalty ──────────────────────────────────────────────
        # r_crash = 5.0 if p_z < 0 OR collision (paper eq. 9), subtracted
        # from total.  Implemented as CRASH_PENALTY = −5.0 added to total.
        r_crash = 0.0
        if collision or self._is_oob(drone_pos):
            r_crash          = self.CRASH_PENALTY
            info["crash"]    = True
            info["collision"] = collision
            info["out_of_bounds"] = self._is_oob(drone_pos)

        reward = r_prog + r_perc + r_jerk + r_body_rate + r_crash

        info["num_gates_passed"] = self._gm.num_passed
        info["r_crash"]          = r_crash
        return float(reward), info

    # ------------------------------------------------------------------
    def is_terminated(
        self,
        drone_pos: np.ndarray,
        collision: bool,
    ) -> bool:
        """Episode ends on crash (collision or OOB) or lap completion."""
        return collision or self._is_oob(drone_pos) or self._gm.lap_complete

    @staticmethod
    def _is_oob(pos: np.ndarray) -> bool:
        return bool(
            np.any(pos[:3] < WORLD_BOUNDS[:3])
            or np.any(pos[:3] > WORLD_BOUNDS[3:])
        )
