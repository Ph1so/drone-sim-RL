"""
ResidualDynamicsModel — GP-based residual forces and torques following
Kaufmann et al. 2023 (Swift, Nature Vol 620).

Paper specification (Methods — Residual identification)
--------------------------------------------------------
"The perception and dynamics residuals are modelled using Gaussian processes
and k-nearest-neighbour regression, respectively."

"We record onboard sensory observations from the robot together with highly
accurate pose estimates from a motion-capture system while the drone is racing
through the track.  During this data-collection phase, the robot is controlled
by a policy trained in simulation that operates on pose estimates provided by
the motion-capture system."

"The residual force and torque (τ_res, τ_res) account for all aerodynamic
effects not captured by the rigid-body + motor model: blade flapping,
induced drag, ground effects, and battery sag."

Implementation
--------------
In the paper the dynamics residual is identified from *real flight data*.
Here we have no real data so we sample from the GP *prior*, the same approach
used by the ResidualObservationModel.  Input features and scaling are chosen
to match the order of magnitude visible in Extended Data Fig. 1a:

  Z-force residual:   peak ≈ ±0.5 N at race speeds (X-axis ≈ 0.3 N)
  Residual torques:   peak ≈ ±0.01–0.02 N·m

Application
-----------
The model is evaluated once per *control* step (48 Hz) and the resulting
force and torque are applied in the body frame via PyBullet's
``applyExternalForce`` and ``applyExternalTorque``.  The forces persist
until the next ``p.stepSimulation()`` call, so they are consumed by the
first of the 5 physics sub-steps.  This is an approximation relative to
applying them every sub-step, but it is sufficient for training a robust
policy.  Force magnitude is chosen conservatively so the drone remains
physically flyable.

Toggle: ``DroneRacingEnv(obs_noise=False)`` disables both ROM and RDM.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class ResidualDynamicsModel:
    """
    GP-based residual dynamics model (Kaufmann et al. 2023, Swift paper).

    Adds state-conditioned stochastic forces and torques to the physics
    simulation to model aerodynamic effects not captured by the rigid-body
    dynamics model (blade flapping, induced drag, motor delay, etc.).

    Uses the same Random Fourier Features (RFF) architecture as the
    ResidualObservationModel for temporal consistency.

    Parameters
    ----------
    ctrl_freq : int
        Control loop frequency (Hz) — unused at runtime, kept for API symmetry.
    enabled : bool
        When False :meth:`sample` returns zeros.  Set False for clean evaluation.
    seed : int | None
        RNG seed.  Fixing it gives reproducible disturbance patterns.
    n_fourier : int
        Number of Random Fourier Feature components D.
    """

    # ── GP input feature length scales ───────────────────────────────────────
    L_SPEED:    float = 5.0   # m/s
    L_ANG_RATE: float = 3.0   # rad/s
    L_TILT:     float = 0.8   # rad

    # ── Residual force amplitude σ_f(state) = base + k_v·speed + k_ω·ang_rate ──
    # Calibrated against Extended Data Fig. 1a (force residual ≈ ±0.3 N at
    # race speeds for the Z axis, smaller for X/Y).
    _FORCE_BASE:  float = 0.03   # N     — hover residual
    _FORCE_KV:    float = 0.02   # N per m/s
    _FORCE_KW:    float = 0.01   # N per rad/s

    # ── Residual torque amplitude ──────────────────────────────────────────────
    _TORQUE_BASE: float = 0.002  # N·m
    _TORQUE_KV:   float = 0.001  # N·m per m/s
    _TORQUE_KW:   float = 0.001  # N·m per rad/s

    # ── Hard clamps ───────────────────────────────────────────────────────────
    _MAX_FORCE_N:  float = 0.50   # N      (~180% of CF2X weight, hard ceiling)
    _MAX_TORQUE_NM: float = 0.025 # N·m

    # ─────────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        ctrl_freq:  int = 48,
        enabled:    bool = True,
        seed:       Optional[int] = None,
        n_fourier:  int = 64,
    ) -> None:
        self.enabled    = enabled
        self._n_fourier = n_fourier
        self._rng       = np.random.default_rng(seed)

        # ── Fixed Random Fourier Features (define the RBF kernel) ─────────
        L = np.array([self.L_SPEED, self.L_ANG_RATE, self.L_TILT], dtype=np.float64)
        self._omega = self._rng.standard_normal((n_fourier, 3)) / L   # (D, 3)
        self._b     = self._rng.uniform(0.0, 2.0 * np.pi, n_fourier)   # (D,)

        # ── Per-episode GP weights: 6 components [fx, fy, fz, tx, ty, tz] ──
        self._w: np.ndarray = np.zeros((6, n_fourier), dtype=np.float64)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Sample a new dynamics realization for the upcoming episode."""
        self._w = self._rng.standard_normal((6, self._n_fourier))

    # ------------------------------------------------------------------
    def sample(
        self,
        lin_vel: np.ndarray,   # (3,) world frame m/s — for ‖v‖
        ang_vel: np.ndarray,   # (3,) world frame rad/s — for ‖ω‖
        rot_mat: np.ndarray,   # (3,3) body→world — for tilt angle
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the GP realisation at the current robot state.

        Returns
        -------
        force_body  : (3,) residual force  in body frame, Newtons
        torque_body : (3,) residual torque in body frame, N·m

        When ``enabled=False`` returns (zeros, zeros).
        """
        if not self.enabled:
            return np.zeros(3), np.zeros(3)

        # ── GP input features ─────────────────────────────────────────────
        speed    = float(np.linalg.norm(lin_vel))
        ang_rate = float(np.linalg.norm(ang_vel))
        cos_tilt = float(np.clip(rot_mat[2, 2], -1.0, 1.0))
        tilt     = float(np.arccos(cos_tilt))

        z = np.array([speed, ang_rate, tilt], dtype=np.float64)

        # ── RFF map ───────────────────────────────────────────────────────
        phi = np.sqrt(2.0 / self._n_fourier) * np.cos(self._omega @ z + self._b)

        # ── GP sample for each of 6 output components ─────────────────────
        f = self._w @ phi   # (6,)

        # ── State-dependent amplitude ─────────────────────────────────────
        amp_force  = self._FORCE_BASE  + self._FORCE_KV  * speed + self._FORCE_KW  * ang_rate
        amp_torque = self._TORQUE_BASE + self._TORQUE_KV * speed + self._TORQUE_KW * ang_rate

        # ── Compute and clip ──────────────────────────────────────────────
        force_body = np.clip(
            amp_force  * f[0:3], -self._MAX_FORCE_N,   self._MAX_FORCE_N
        ).astype(np.float64)
        torque_body = np.clip(
            amp_torque * f[3:6], -self._MAX_TORQUE_NM, self._MAX_TORQUE_NM
        ).astype(np.float64)

        return force_body, torque_body
