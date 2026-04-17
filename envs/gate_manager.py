"""
GateManager — tracks the sequential gate waypoints on the racecourse.

Gate coordinate conventions
----------------------------
Each gate URDF is loaded at world position `pos` with a pure-yaw rotation θ
(euler = [0, 0, θ]).  Under that rotation:

  gate_normal (exit direction) = R_z(θ) · [0, 1, 0] = [−sin θ,  cos θ,  0]
  gate_right  (horizontal)     = R_z(θ) · [1, 0, 0] = [ cos θ,  sin θ,  0]
  gate_up                      =                       [  0,      0,      1]

A drone "passes through" gate i when its signed distance w.r.t. the gate plane
transitions from ≤ 0 (approach side) to > 0 (exit side) while its projected
position is inside the gate opening (±HALF_OPEN_W in right, ±HALF_OPEN_H in up).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ── Gate geometry ────────────────────────────────────────────────────────────
GATE_OUTER   = 1.40   # m  outer frame size (both axes)
GATE_INNER   = 1.20   # m  inner opening size (both axes)
HALF_OPEN_W  = GATE_INNER / 2.0   # ± 0.60 m  horizontal in gate plane
HALF_OPEN_H  = GATE_INNER / 2.0   # ± 0.60 m  vertical   in gate plane
PASS_MARGIN  = 1.10   # scale factor: slightly larger acceptance window


# ── Racecourse definition ─────────────────────────────────────────────────────
#
#  Oval-ish lap (CCW when viewed from above, z-up world):
#
#   Start: (0, 0, 0.3)    ←── drone spawn, facing +Y
#
#   G1 ── G2
#   |       \
#   G5      G3
#    \      /
#     G4───
#
#  yaw_deg: angle such that gate_normal = [−sin(yaw), cos(yaw), 0]
#    yaw=  0° → normal = [0,  1, 0]  (gate faces north / +Y)
#    yaw=-45° → normal = [0.71, 0.71, 0]  (NE diagonal)
#    yaw=-90° → normal = [1,  0, 0]  (gate faces east / +X)
#    yaw=180° → normal = [0, -1, 0]  (gate faces south / −Y)
#
_GATE_DEFS = [
    #  pos (x, y, z-center)     yaw_deg   label
    ([ 0.0,  2.5,  1.50],    0.0,  "G1"),
    ([ 2.5,  5.0,  1.50],  -40.0,  "G2"),
    ([ 5.5,  5.5,  1.50],  -90.0,  "G3"),
    ([ 7.5,  2.8,  1.50], -135.0,  "G4"),
    ([ 5.0,  0.5,  1.50],  180.0,  "G5"),
]


# ── Gate dataclass ─────────────────────────────────────────────────────────────

@dataclass
class Gate:
    position: np.ndarray     # world-space centre of gate opening
    yaw_deg:  float          # rotation about Z (degrees)
    label:    str = ""
    body_id:  int = -1       # PyBullet body ID (set by load_gates)
    passed:   bool = False   # has this gate been cleared this episode?

    # Derived axes (set in __post_init__)
    normal:   np.ndarray = field(init=False, repr=False)
    right:    np.ndarray = field(init=False, repr=False)
    yaw_rad:  float      = field(init=False, repr=False)

    # Internal crossing tracker
    _prev_signed_dist: Optional[float] = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64)
        self.yaw_rad  = np.deg2rad(self.yaw_deg)
        self.normal = np.array([
            -np.sin(self.yaw_rad),
             np.cos(self.yaw_rad),
             0.0,
        ])
        self.right = np.array([
            np.cos(self.yaw_rad),
            np.sin(self.yaw_rad),
            0.0,
        ])

    # ------------------------------------------------------------------
    def signed_distance(self, point: np.ndarray) -> float:
        """Signed distance from *point* to this gate's plane."""
        return float(np.dot(point - self.position, self.normal))

    def is_within_opening(self, point: np.ndarray) -> bool:
        """True if *point* lies within (an expanded) gate opening."""
        delta   = point - self.position
        local_h = float(np.dot(delta, self.right))   # horizontal offset
        local_v = float(delta[2])                    # vertical  offset
        return (
            abs(local_h) <= HALF_OPEN_W * PASS_MARGIN
            and abs(local_v) <= HALF_OPEN_H * PASS_MARGIN
        )

    def check_passing(self, drone_pos: np.ndarray) -> bool:
        """
        Call once per env step.  Returns True on the step the drone
        crosses from the approach side to the exit side through the opening.
        """
        sd = self.signed_distance(drone_pos)
        crossed = False
        if self._prev_signed_dist is not None:
            if self._prev_signed_dist <= 0.0 < sd:
                if self.is_within_opening(drone_pos):
                    crossed = True
        self._prev_signed_dist = sd
        return crossed

    def reset(self) -> None:
        self._prev_signed_dist = None
        self.passed            = False
        self.body_id           = -1


# ── Pre-built gate list (deep-copied per episode) ─────────────────────────────

RACE_GATES: List[Gate] = [
    Gate(position=d[0], yaw_deg=d[1], label=d[2])
    for d in _GATE_DEFS
]


# ── GateManager ───────────────────────────────────────────────────────────────

class GateManager:
    """
    Manages sequential gate passing for a single episode.

    Parameters
    ----------
    num_gates : int
        Number of gates to include from the start of RACE_GATES (1–5).
        Use values < 5 for curriculum training (e.g. ``--num_gates 1``).

    Usage::

        gm = GateManager(num_gates=2)   # curriculum: first 2 gates only
        gm.load_gates(client, urdf)     # load URDFs into PyBullet
        ...
        passed = gm.update(pos)         # call every step
        dist   = gm.dist_to_next(pos)
    """

    def __init__(self, num_gates: int = 5) -> None:
        self._n_gates = min(max(num_gates, 1), len(RACE_GATES))
        self.gates: List[Gate] = [copy.deepcopy(g) for g in RACE_GATES[:self._n_gates]]
        self._idx:  int  = 0
        self.num_passed:   int  = 0
        self.lap_complete: bool = False

    # ------------------------------------------------------------------
    @property
    def num_gates(self) -> int:
        return len(self.gates)

    @property
    def current_gate(self) -> Gate:
        idx = min(self._idx, len(self.gates) - 1)
        return self.gates[idx]

    @property
    def current_gate_idx(self) -> int:
        return self._idx

    @property
    def next_gate(self) -> Optional[Gate]:
        """Gate after the current target, or None if current is the last."""
        if self._idx + 1 >= len(self.gates):
            return None
        return self.gates[self._idx + 1]

    # ------------------------------------------------------------------
    def dist_to_next(self, drone_pos: np.ndarray) -> float:
        """Euclidean distance to the next (current target) gate centre."""
        return float(np.linalg.norm(drone_pos - self.current_gate.position))

    # ------------------------------------------------------------------
    def update(self, drone_pos: np.ndarray) -> bool:
        """
        Check whether the drone just passed through the current gate.
        Advances internal index when a gate is cleared.

        Returns True on the step a gate crossing is detected.
        """
        if self.lap_complete:
            return False

        gate = self.current_gate
        if gate.check_passing(drone_pos):
            gate.passed  = True
            self.num_passed += 1
            self._idx   += 1
            if self._idx >= len(self.gates):
                self.lap_complete = True
            return True
        return False

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset all gates and counters for a new episode."""
        self.gates        = [copy.deepcopy(g) for g in RACE_GATES[:self._n_gates]]
        self._idx         = 0
        self.num_passed   = 0
        self.lap_complete = False

    # ------------------------------------------------------------------
    def fast_forward_to(self, k: int) -> None:
        """
        Mark gates 0 … k-1 as already passed and set the current target to gate k.

        Used by mid-course spawn randomisation so the reward and obs targets the
        correct gate when the drone is teleported to a mid-course position.

        Parameters
        ----------
        k : int
            Index of the gate to target next (0-based).  Clamped to [1, n_gates-1].
        """
        k = max(1, min(k, len(self.gates) - 1))
        for i in range(k):
            self.gates[i].passed            = True
            self.gates[i]._prev_signed_dist = None   # clear crossing tracker
        self.num_passed   = k
        self._idx         = k
        self.lap_complete = False

    # ------------------------------------------------------------------
    def load_gates(self, client: int, urdf_path: str) -> None:
        """
        Load gate URDFs into a PyBullet world.  Must be called after
        p.resetSimulation so previous body IDs are invalidated.
        """
        import pybullet as p  # lazy import — not needed for map visualisation
        for gate in self.gates:
            quat = p.getQuaternionFromEuler(
                [0.0, 0.0, gate.yaw_rad], physicsClientId=client
            )
            body_id = p.loadURDF(
                urdf_path,
                basePosition=gate.position.tolist(),
                baseOrientation=quat,
                useFixedBase=True,
                physicsClientId=client,
            )
            gate.body_id = body_id

    # ------------------------------------------------------------------
    def gate_body_ids(self) -> List[int]:
        return [g.body_id for g in self.gates if g.body_id >= 0]
