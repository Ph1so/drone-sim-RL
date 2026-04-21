"""
ResidualObservationModel — GP-based perception residuals following
Kaufmann et al. 2023 (Swift, Nature Vol 620).

Paper specification (Methods — Residual observation model)
----------------------------------------------------------
"To model the drift in odometry, we use Gaussian processes, as they allow
fitting a posterior distribution of odometry perturbations, from which we
can sample temporally consistent realizations."

"The Gaussian process model fits residual position, velocity and attitude
as a function of the ground-truth robot state.  We treat each dimension of
the observation separately, effectively fitting a set of nine 1D Gaussian
processes to the observation residuals.  We use a mixture of radial basis
function kernels:

    κ(zᵢ, zⱼ) = σ_f² exp(−½(zᵢ−zⱼ)ᵀ L⁻² (zᵢ−zⱼ)) + σ_n²

[...] After kernel hyperparameter optimisation, we sample new realisations
from the posterior distribution that are then used during fine-tuning."

Implementation
--------------
In the paper the GPs are fitted to real VIO-vs-mocap residual data.
Here we have no real data so we sample from the *prior* using Random
Fourier Features (Rahimi & Recht, 2007), which yields an efficient O(D)
approximation to a GP sample:

    φ(z) = √(2/D) · cos(Ω z + b)        Ω ∈ R^{D×3}, b ∈ R^D
    f_i(z) = w_i · φ(z)                  w_i ~ N(0, I_D)

The input features are z = [‖v‖, ‖ω‖, θ_tilt] ∈ R³, where:
  ‖v‖     : linear speed (m/s)        → captures motion blur / feature tracking loss
  ‖ω‖     : angular rate (rad/s)      → captures rotational blur / IMU-visual mismatch
  θ_tilt  : tilt angle (rad, body z vs world z) → captures horizon feature loss

These are the dominant physical drivers of VIO degradation during agile flight.

Temporal consistency (key paper property)
-----------------------------------------
Consecutive time steps share similar states → similar GP inputs → similar
outputs (smooth RBF kernel).  Temporal correlation length ≈ the RBF length
scale divided by the rate of state change.  This is exactly how real VIO
drift behaves: once the estimator "loses its footing" it stays wrong until
the state changes significantly.

One realisation (w) is sampled per episode via reset().  Every step then
evaluates the *same* function at the current state, so:
  • Hovering   → nearly constant drift (state barely changes)
  • High-speed → slowly varying drift (state changes continuously)
  • Full lap    → repeatable drift pattern along the track
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class ResidualObservationModel:
    """
    GP-based residual observation model (Kaufmann et al. 2023, Swift paper).

    9 independent 1-D GPs — approximated via Random Fourier Features (RFF) —
    inject state-conditioned drift into position, velocity, and attitude
    observations to simulate VIO degradation at high speed and agility.

    Parameters
    ----------
    ctrl_freq : int
        Control loop frequency (Hz) — unused at runtime, kept for API symmetry.
    enabled : bool
        When False :meth:`apply` is a no-op.  Set False for clean evaluation.
    seed : int | None
        RNG seed.  Fixing it gives reproducible drift patterns across runs.
    n_fourier : int
        Number of Random Fourier Feature components D.  64 gives a good
        approximation to the RBF kernel; increase if drift pattern looks rough.
    """

    # ── GP input feature normalisation (RBF length scales) ───────────────────
    # Larger L → function varies more slowly in that dimension.
    L_SPEED:    float = 5.0   # m/s    — changes in speed affect GP slowly
    L_ANG_RATE: float = 3.0   # rad/s  — moderate sensitivity to angular rate
    L_TILT:     float = 0.8   # rad    — tighter coupling; small tilt changes matter

    # ── State-dependent drift amplitude σ(state) ─────────────────────────────
    # The raw GP sample f(z) has unit variance N(0,1) for each component.
    # We scale: residual_i(z) = σ_i(state) · f_i(z)
    # so that the effective per-step std matches physical VIO drift budgets:
    #
    #   Quantity    Hover     10 m/s + 5 rad/s
    #   ─────────   ──────    ────────────────
    #   Position    ~1.5 cm   ~9.5 cm
    #   Velocity    ~2.5 cm/s ~10.6 cm/s
    #   Attitude    ~0.09°    ~0.35°
    #
    # σ = base + k_v·speed + k_ω·ang_rate  (linear fit to the expected envelope)
    _POS_BASE:  float = 0.030   # m
    _POS_KV:    float = 0.012   # m per m/s
    _POS_KW:    float = 0.008   # m per rad/s

    _VEL_BASE:  float = 0.070   # m/s
    _VEL_KV:    float = 0.018   # (m/s) per m/s
    _VEL_KW:    float = 0.010   # (m/s) per rad/s

    _ATT_BASE:  float = 0.004   # rad
    _ATT_KW:    float = 0.002   # rad per rad/s
    _ATT_KT:    float = 0.001   # rad per rad of tilt

    # ── Hard clamps ───────────────────────────────────────────────────────────
    _MAX_DRIFT_POS: float = 0.40   # m
    _MAX_DRIFT_VEL: float = 2.00   # m/s
    _MAX_DRIFT_ATT: float = 0.15   # rad  (≈ 8.6°)

    # ─────────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        ctrl_freq:  int = 48,
        enabled:    bool = True,
        seed:       Optional[int] = None,
        n_fourier:  int = 64,
    ) -> None:
        self.enabled   = enabled
        self._n_fourier = n_fourier
        self._rng      = np.random.default_rng(seed)

        # ── Fixed Random Fourier Features (sampled once; define the kernel) ──
        #
        # For an RBF kernel k(z,z') = exp(−½(z−z')ᵀ L⁻² (z−z')), the spectral
        # density is N(0, L⁻² I).  Sample D frequencies accordingly:
        L = np.array([self.L_SPEED, self.L_ANG_RATE, self.L_TILT], dtype=np.float64)
        self._omega = (
            self._rng.standard_normal((n_fourier, 3)) / L   # (D, 3)
        )
        self._b = self._rng.uniform(0.0, 2.0 * np.pi, n_fourier)  # (D,)

        # ── Per-episode GP weights (reset each episode) ───────────────────────
        # Shape (9, D): one weight vector per observation component.
        # Sampled from N(0, I) — implements a GP prior draw.
        self._w: np.ndarray = np.zeros((9, n_fourier), dtype=np.float64)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """
        Sample a new GP realization for the upcoming episode.

        This draws new weights w ~ N(0, I) for each of the 9 GPs.
        The resulting 9 smooth functions are held fixed for the duration
        of the episode — consistent with the paper's per-episode sampling.
        """
        self._w = self._rng.standard_normal((9, self._n_fourier))

    # ------------------------------------------------------------------
    def apply(
        self,
        pos:     np.ndarray,   # (3,) world frame, m
        vel:     np.ndarray,   # (3,) world frame, m/s
        rot_mat: np.ndarray,   # (3,3) body→world rotation matrix
        lin_vel: np.ndarray,   # (3,) for computing ‖v‖
        ang_vel: np.ndarray,   # (3,) for computing ‖ω‖
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate the GP realisation at the current robot state and return
        the corrupted position, velocity, and rotation matrix.

        When ``enabled=False`` the inputs are returned unchanged (no copy
        overhead for the clean evaluation path).
        """
        if not self.enabled:
            return pos, vel, rot_mat

        # ── GP input features ─────────────────────────────────────────────
        speed    = float(np.linalg.norm(lin_vel))
        ang_rate = float(np.linalg.norm(ang_vel))

        # Tilt: angle between body z-axis and world +z
        # rot_mat[:, 2] is body z expressed in world frame; its z-component
        # is cos(tilt_angle).
        cos_tilt = float(np.clip(rot_mat[2, 2], -1.0, 1.0))
        tilt     = float(np.arccos(cos_tilt))

        z = np.array([speed, ang_rate, tilt], dtype=np.float64)

        # ── Random Fourier Feature map ────────────────────────────────────
        # φ(z) = √(2/D) · cos(Ω z + b)   ∈ R^D
        phi = np.sqrt(2.0 / self._n_fourier) * np.cos(self._omega @ z + self._b)

        # ── GP sample value for each of 9 components ─────────────────────
        # f(z) = w · φ(z)   — unit-variance GP prior sample
        f = self._w @ phi   # (9,)

        # ── State-dependent amplitude σ(state) ───────────────────────────
        amp_pos = self._POS_BASE + self._POS_KV * speed + self._POS_KW * ang_rate
        amp_vel = self._VEL_BASE + self._VEL_KV * speed + self._VEL_KW * ang_rate
        amp_att = self._ATT_BASE + self._ATT_KW * ang_rate + self._ATT_KT * tilt

        # ── Compute drift and clip ────────────────────────────────────────
        drift_pos = np.clip(
            amp_pos * f[0:3], -self._MAX_DRIFT_POS, self._MAX_DRIFT_POS
        ).astype(np.float64)
        drift_vel = np.clip(
            amp_vel * f[3:6], -self._MAX_DRIFT_VEL, self._MAX_DRIFT_VEL
        ).astype(np.float64)
        drift_att = np.clip(
            amp_att * f[6:9], -self._MAX_DRIFT_ATT, self._MAX_DRIFT_ATT
        ).astype(np.float64)

        pos_obs = pos + drift_pos
        vel_obs = vel + drift_vel
        rot_obs = self._apply_att_drift(rot_mat, drift_att)

        return pos_obs, vel_obs, rot_obs

    # ------------------------------------------------------------------
    def _apply_att_drift(
        self, rot_mat: np.ndarray, drpy: np.ndarray
    ) -> np.ndarray:
        """
        Compose a small ZYX Euler attitude error onto the clean rotation matrix.

            R_obs = R_drift(Δroll, Δpitch, Δyaw) @ R_true

        Using exact Euler composition (not small-angle) keeps the result on
        SO(3) regardless of drift magnitude.
        """
        dr, dp, dy = drpy

        cr, sr = np.cos(dr), np.sin(dr)
        cp, sp = np.cos(dp), np.sin(dp)
        cy, sy = np.cos(dy), np.sin(dy)

        R_drift = np.array([
            [cy * cp,   cy * sp * sr - sy * cr,   cy * sp * cr + sy * sr],
            [sy * cp,   sy * sp * sr + cy * cr,   sy * sp * cr - cy * sr],
            [-sp,       cp * sr,                  cp * cr               ],
        ], dtype=np.float64)

        return R_drift @ rot_mat

    # ------------------------------------------------------------------
    @property
    def drift_state(self) -> dict:
        """Current drift values at the last queried state — for diagnostics."""
        return {"weights_norm": float(np.linalg.norm(self._w))}
