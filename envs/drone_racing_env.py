"""
DroneRacingEnv — Gymnasium environment for autonomous drone racing.

Inherits from gym-pybullet-drones' BaseAviary for physics and drone management.

Action space
------------
Box(4,)  — normalised [Throttle, Roll, Pitch, Yaw] each in [−1, +1].
Mapped to per-motor RPMs via a classical quadrotor mixer.

Observation space  (matches Swift paper, Kaufmann et al. 2023)
-----------------
Box(31,)  — flat float32 vector:
  [0:3]   position in world frame (m)
  [3:6]   linear velocity in world frame (m/s)
  [6:15]  attitude as rotation matrix (body→world), flattened row-major (9-D)
           Avoids quaternion discontinuities (Zhou et al., 2019).
  [15:27] next-gate corners in drone body frame (4 corners × 3-D = 12-D)
           Corners are the four vertices of the gate inner opening (±0.6 m).
  [27:31] previous action applied at t−1 (4-D)

Reward / termination
--------------------
See reward.py for the detailed shaping.  Episode ends on:
  • Collision with any object  (crash penalty)
  • Leaving the world-bounds box  (crash penalty)
  • Completing all active gates  (success)
  • Exceeding MAX_EPISODE_STEPS  (truncation)
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from .gate_manager import GateManager
from .reward import RewardComputer


# ── Constants ─────────────────────────────────────────────────────────────────
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
GATE_URDF   = os.path.abspath(os.path.join(ASSETS_DIR, "gate.urdf"))

IMG_H = 64    # camera image height (pixels)
IMG_W = 64    # camera image width  (pixels)
IMG_C = 3     # RGB channels

# Drone starts level, facing +Y (yaw = π/2 in world frame)
_INIT_POS = np.array([[0.0, 0.0, 0.30]])   # shape (1, 3)
_INIT_RPY = np.array([[0.0, 0.0, np.pi / 2]])  # shape (1, 3)

MAX_EPISODE_STEPS = 1500   # ~31 s at 48 Hz ctrl_freq

# Mixer authority scales (fraction of HOVER_RPM)
_THROTTLE_AUTH = 0.50   # T=+1 raises all motors by 50% of hover RPM
_ROLL_AUTH     = 0.20
_PITCH_AUTH    = 0.20
_YAW_AUTH      = 0.10


class DroneRacingEnv(BaseAviary):
    """
    Drone racing environment built on gym-pybullet-drones' BaseAviary.

    Parameters
    ----------
    gui : bool
        Open the PyBullet GUI window (use for evaluation, not training).
    img_size : tuple[int, int]
        Egocentric camera resolution (H, W).  Default: (64, 64).
    ctrl_freq : int
        Control loop frequency (Hz).  pyb_freq is fixed at 5× ctrl_freq.
    record : bool
        Record video frames (only relevant when gui=True).
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        gui:                    bool = False,
        img_size:               tuple[int, int] = (IMG_H, IMG_W),
        ctrl_freq:              int = 48,
        record:                 bool = False,
        num_gates:              int = 5,
        spawn_mid_course_prob:  float = 0.0,
        gate_pos_offset:        Optional[list] = None,
    ) -> None:
        self._img_h, self._img_w = img_size
        self._spawn_mid_course_prob = spawn_mid_course_prob

        # GateManager and RewardComputer are created before super().__init__
        # because BaseAviary.__init__ calls _addObstacles() internally.
        self._gate_manager   = GateManager(num_gates=num_gates, pos_offset=gate_pos_offset)
        self._reward_computer = RewardComputer(self._gate_manager)

        # Per-step cache (populated once, read by reward/term/info methods)
        self._step_cache: dict = {}
        self._episode_steps: int = 0
        self._last_action: np.ndarray = np.zeros(4, dtype=np.float32)

        super().__init__(
            drone_model       = DroneModel.CF2X,
            num_drones        = 1,
            initial_xyzs      = _INIT_POS,
            initial_rpys      = _INIT_RPY,
            physics           = Physics.PYB,
            pyb_freq          = ctrl_freq * 5,   # 5 sub-steps per control step
            ctrl_freq         = ctrl_freq,
            gui               = gui,
            record            = record,
            obstacles         = True,
            user_debug_gui    = False,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Gymnasium API overrides
    # ══════════════════════════════════════════════════════════════════════════

    def _actionSpace(self) -> spaces.Box:
        """4-D normalised action: [Throttle, Roll, Pitch, Yaw] ∈ [−1, +1]."""
        return spaces.Box(
            low  = np.full(4, -1.0, dtype=np.float32),
            high = np.full(4,  1.0, dtype=np.float32),
            dtype = np.float32,
        )

    def _observationSpace(self) -> spaces.Box:
        """31-D flat observation matching Swift paper (Kaufmann et al. 2023)."""
        return spaces.Box(
            low  = -np.inf,
            high =  np.inf,
            shape = (31,),
            dtype = np.float32,
        )

    # ------------------------------------------------------------------
    def reset(
        self,
        seed:    Optional[int] = None,
        options: Optional[dict] = None,
    ):
        self._gate_manager.reset()
        self._episode_steps = 0
        self._step_cache    = {}

        obs, info = super().reset(seed=seed, options=options)
        # super().reset() calls _addObstacles() which loads gate URDFs.

        # Optional mid-course spawn: teleport drone to a random gate approach.
        # Only fires when num_gates > 1 (needs at least one transition to practice).
        if (
            self._spawn_mid_course_prob > 0.0
            and self._gate_manager._n_gates > 1
            and self.np_random.random() < self._spawn_mid_course_prob
        ):
            k = int(self.np_random.integers(1, self._gate_manager._n_gates))
            self._teleport_to_gate_approach(k)

        self._reward_computer.reset()
        # Rebuild obs from current (possibly teleported) drone state.
        obs = self._computeObs()
        return obs, info

    # ------------------------------------------------------------------
    def step(self, action):
        # Clear per-step cache at the start of each step.
        self._step_cache = {}
        self._last_action = np.asarray(action, dtype=np.float32)
        obs, reward, terminated, truncated, info = super().step(action)
        self._episode_steps += 1

        # Merge diagnostics
        info.update({
            "num_gates_passed": self._gate_manager.num_passed,
            "lap_complete":     self._gate_manager.lap_complete,
            "current_gate_idx": self._gate_manager.current_gate_idx,
            "episode_steps":    self._episode_steps,
        })
        return obs, reward, terminated, truncated, info

    # ══════════════════════════════════════════════════════════════════════════
    # BaseAviary abstract method implementations
    # ══════════════════════════════════════════════════════════════════════════

    def _addObstacles(self) -> None:
        """Called by BaseAviary after each p.resetSimulation().  Loads gates."""
        self._gate_manager.load_gates(self.CLIENT, GATE_URDF)

    # ------------------------------------------------------------------
    def _preprocessAction(self, action: np.ndarray) -> np.ndarray:
        """
        Map normalised [T, R, P, Y] ∈ [−1, +1]^4 → per-motor RPMs (1, 4).

        X-configuration motor layout (top-view):
            M0 (front-left,  CCW)    M1 (front-right, CW)
            M2 (rear-left,   CW)     M3 (rear-right,  CCW)

        Mixer (positive = right-roll / nose-up / CCW-yaw convention):
            M0 = hover − roll − pitch + yaw
            M1 = hover + roll − pitch − yaw
            M2 = hover − roll + pitch − yaw
            M3 = hover + roll + pitch + yaw
        """
        action = np.squeeze(np.clip(action, -1.0, 1.0)).astype(np.float64)
        T, R, P, Y = action[0], action[1], action[2], action[3]

        h   = float(self.HOVER_RPM)
        dT  = h * _THROTTLE_AUTH * T
        dR  = h * _ROLL_AUTH     * R
        dP  = h * _PITCH_AUTH    * P
        dY  = h * _YAW_AUTH      * Y

        rpms = np.clip(
            np.array([
                h + dT - dR - dP + dY,   # M0  front-left  CCW
                h + dT + dR - dP - dY,   # M1  front-right CW
                h + dT - dR + dP - dY,   # M2  rear-left   CW
                h + dT + dR + dP + dY,   # M3  rear-right  CCW
            ]),
            a_min = 0.0,
            a_max = float(self.MAX_RPM),
        )
        return rpms.reshape(1, 4)   # (num_drones=1, 4)

    # ------------------------------------------------------------------
    def _computeObs(self) -> np.ndarray:
        """
        Build 31-D flat observation matching Swift paper (Kaufmann et al. 2023).

        Layout:
          [0:3]   position world frame
          [3:6]   linear velocity world frame
          [6:15]  rotation matrix body→world, row-major (9-D)
          [15:27] next-gate corners in drone body frame (4 × 3-D)
          [27:31] previous action
        """
        cache = self._get_step_state()
        state = cache["state"]

        pos_world = state[0:3].astype(np.float32)           # (3,)
        lin_vel   = state[10:13].astype(np.float32)         # (3,)

        # Rotation matrix (body→world) from quaternion (x,y,z,w PyBullet format)
        quat    = state[3:7]
        rot_mat = np.array(
            p.getMatrixFromQuaternion(quat, physicsClientId=self.CLIENT)
        ).reshape(3, 3).astype(np.float32)
        rot_flat = rot_mat.flatten()                         # (9,)

        # Next-gate corners in drone body frame (4 corners × 3-D = 12-D)
        gate_corners = self._compute_gate_corners_body_frame(
            pos_world = state[0:3].astype(np.float64),
            rot_mat   = rot_mat.astype(np.float64),
        )                                                    # (12,)

        # Previous action (set to zeros at episode start, current action thereafter)
        prev_action = self._last_action                      # (4,)

        return np.concatenate([
            pos_world,    # 3
            lin_vel,      # 3
            rot_flat,     # 9
            gate_corners, # 12
            prev_action,  # 4
        ]).astype(np.float32)   # total: 31

    # ------------------------------------------------------------------
    def _computeReward(self) -> float:
        cache       = self._get_step_state()
        gate_passed = self._gate_manager.update(cache["pos"])
        cache["gate_passed"] = gate_passed   # share with terminated/info

        lin_vel = cache["state"][10:13]
        ang_vel = cache["state"][13:16]
        reward, rinfo = self._reward_computer.compute(
            drone_pos     = cache["pos"],
            drone_rpy     = cache["rpy"],
            drone_lin_vel = lin_vel,
            drone_ang_vel = ang_vel,
            action        = self._last_action,
            gate_passed   = gate_passed,
            collision     = cache["collision"],
        )
        cache["reward_info"] = rinfo
        return reward

    # ------------------------------------------------------------------
    def _computeTerminated(self) -> bool:
        cache = self._get_step_state()
        return self._reward_computer.is_terminated(
            drone_pos = cache["pos"],
            collision = cache["collision"],
        )

    # ------------------------------------------------------------------
    def _computeTruncated(self) -> bool:
        return self._episode_steps >= MAX_EPISODE_STEPS

    # ------------------------------------------------------------------
    def _computeInfo(self) -> dict:
        cache = self._get_step_state()
        info  = dict(cache.get("reward_info", {}))
        info["gate_passed"]      = cache.get("gate_passed", False)
        info["collision"]        = cache["collision"]
        info["drone_pos"]        = cache["pos"].copy()   # world-frame (3,) — for trajectory plots
        info["drone_rpy"]        = cache["rpy"].copy()   # roll, pitch, yaw in radians
        info["drone_lin_vel"]    = cache["state"][10:13].copy()   # world-frame linear velocity (m/s)
        info["drone_ang_vel"]    = cache["state"][13:16].copy()   # world-frame angular velocity (rad/s)
        return info

    # ══════════════════════════════════════════════════════════════════════════
    # Internal helpers
    # ══════════════════════════════════════════════════════════════════════════

    # Speed injected at mid-course spawns — approximates realistic post-gate velocity.
    # Matches the evaluation distribution (drone exits a gate with forward momentum)
    # so the policy learns to handle the turn, not just approach from a standstill.
    _SPAWN_OFFSET:    float = 1.5   # m offset from prev gate centre
    _SPAWN_SPEED:     float = 2.0   # m/s for normal gate transitions
    _ARC_SPAWN_SPEED: float = 1.0   # m/s for tight corners (G3→G4)

    def _teleport_to_gate_approach(self, k: int) -> None:
        """
        Teleport the drone to just past gate k-1's exit, aligned with that gate's
        exit direction, with realistic forward velocity.

        Previously the drone spawned at rest already facing gate k.  That mismatch
        caused the policy to learn "approach gate k from a standstill" rather than
        "turn from gate k-1's exit momentum into gate k's approach axis" — the exact
        maneuver that fails in evaluation.

        Normal corners (k ≠ 3):
          - Position : _SPAWN_OFFSET m along gate k-1's exit normal
          - Yaw      : aligned with gate k-1's exit direction
          - Velocity : _SPAWN_SPEED m/s along the exit normal
          - Angular  : zero

        G3→G4 tight corner (k=3):
          G3 exits heading east while G4 is mostly south (~80° turn).  Spawning
          1.5 m east of G3 lands the drone almost at G4's x-coordinate with
          eastward momentum — only ~0.5 m of valid travel before overshoot.
          Instead, spawn _SPAWN_OFFSET m along the G3→G4 chord so the drone
          starts on the arc with enough runway to complete the turn:
          - Position : _SPAWN_OFFSET m from G3 in the G3→G4 direction
          - Yaw      : facing along G3→G4 chord (south-southeast)
          - Velocity : _ARC_SPAWN_SPEED m/s along G3→G4 chord
          - Angular  : zero

        GateManager is fast-forwarded to k in all cases.

        Parameters
        ----------
        k : int
            Index of the gate to target after the teleport (1-based, k ≥ 1).
        """
        prev_gate = self._gate_manager.gates[k - 1]

        # ── Spawn position ────────────────────────────────────────────────────
        # G3→G4 tight corner: spawn along the G3→G4 chord so the drone has
        # enough runway before G4's x-coordinate.  All other transitions spawn
        # just past gate k-1's exit plane.
        if k == 3 and len(self._gate_manager.gates) > k:
            chord = (self._gate_manager.gates[k].position - prev_gate.position).copy()
            chord[2] = 0.0
            chord /= np.linalg.norm(chord)
            spawn_pos   = prev_gate.position.copy() + chord * self._SPAWN_OFFSET
            spawn_speed = self._ARC_SPAWN_SPEED
        else:
            spawn_pos   = prev_gate.position + prev_gate.normal * self._SPAWN_OFFSET
            spawn_speed = self._SPAWN_SPEED

        # ── Face toward the target gate ───────────────────────────────────────
        to_target = self._gate_manager.gates[k].position - spawn_pos
        to_target[2] = 0.0                           # horizontal only
        to_target /= np.linalg.norm(to_target)
        exit_yaw  = float(np.arctan2(to_target[1], to_target[0]))
        spawn_vel = (to_target * spawn_speed).tolist()

        quat = p.getQuaternionFromEuler(
            [0.0, 0.0, exit_yaw], physicsClientId=self.CLIENT
        )
        p.resetBasePositionAndOrientation(
            self.DRONE_IDS[0],
            spawn_pos.tolist(),
            quat,
            physicsClientId=self.CLIENT,
        )
        p.resetBaseVelocity(
            self.DRONE_IDS[0],
            linearVelocity  = spawn_vel,
            angularVelocity = [0.0, 0.0, 0.0],
            physicsClientId = self.CLIENT,
        )
        self._gate_manager.fast_forward_to(k)
        self._step_cache = {}   # invalidate stale state

    # ------------------------------------------------------------------
    def _get_step_state(self) -> dict:
        """
        Lazily populate a per-step cache with shared quantities so that
        _computeObs / _computeReward / _computeTerminated do not duplicate
        expensive PyBullet queries, and gate_manager.update() is called
        exactly once per step.
        """
        if self._step_cache:
            return self._step_cache

        state = self._getDroneStateVector(0)
        pos   = state[0:3]
        rpy   = state[7:10]

        self._step_cache = {
            "state":      state,
            "pos":        pos,
            "rpy":        rpy,
            "collision":  self._check_collision(),
        }
        return self._step_cache

    # ------------------------------------------------------------------
    def _check_collision(self) -> bool:
        """
        Return True if the drone body is in contact with anything other
        than itself (gates, ground plane, etc.).
        """
        drone_id = self.DRONE_IDS[0]
        contacts = p.getContactPoints(
            bodyA = drone_id,
            physicsClientId = self.CLIENT,
        )
        if not contacts:
            return False
        for c in contacts:
            # c[1]=bodyA, c[2]=bodyB — filter self-collisions
            if c[2] != drone_id:
                return True
        return False

    # ------------------------------------------------------------------
    def _compute_gate_corners_body_frame(
        self,
        pos_world: np.ndarray,
        rot_mat:   np.ndarray,
    ) -> np.ndarray:
        """
        Return the 4 corners of the current gate's inner opening expressed in
        the drone body frame.  Output shape: (12,) — 4 corners × 3-D each.

        This matches the Swift paper's gate observation:
          "relative position of the four gate corners with respect to the vehicle"
          — Kaufmann et al. 2023 (Methods, Observations section).

        Gate opening corners (world frame):
          top-right    : pos + right * HALF_OPEN_W + up * HALF_OPEN_H
          top-left     : pos - right * HALF_OPEN_W + up * HALF_OPEN_H
          bottom-left  : pos - right * HALF_OPEN_W - up * HALF_OPEN_H
          bottom-right : pos + right * HALF_OPEN_W - up * HALF_OPEN_H

        Transform to body frame: v_body = R^T @ (v_world - drone_pos)
        where R is the body→world rotation matrix.
        """
        from .gate_manager import HALF_OPEN_W, HALF_OPEN_H

        gate     = self._gate_manager.current_gate
        gpos     = gate.position            # (3,) world frame
        right    = gate.right               # (3,) horizontal axis in gate plane
        up       = np.array([0.0, 0.0, 1.0])

        corners_world = np.array([
            gpos + right * HALF_OPEN_W + up * HALF_OPEN_H,   # top-right
            gpos - right * HALF_OPEN_W + up * HALF_OPEN_H,   # top-left
            gpos - right * HALF_OPEN_W - up * HALF_OPEN_H,   # bottom-left
            gpos + right * HALF_OPEN_W - up * HALF_OPEN_H,   # bottom-right
        ], dtype=np.float64)   # (4, 3)

        # Shift to drone origin then rotate into body frame
        corners_body = (rot_mat.T @ (corners_world - pos_world).T).T  # (4, 3)
        return corners_body.flatten().astype(np.float32)   # (12,)

    # ------------------------------------------------------------------
    def _render_ego_camera(
        self,
        pos:  Optional[np.ndarray] = None,
        quat: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Render a forward-facing egocentric camera image from the drone.

        The camera is positioned at the drone's centre of mass and aimed
        along the body +X axis with a slight downward tilt (−10°).

        Parameters
        ----------
        pos, quat : pre-fetched state vectors (avoids a second query if the
                    caller already holds the drone state).

        Returns
        -------
        rgb : np.ndarray  shape (H, W, 3), dtype uint8
        """
        if pos is None or quat is None:
            state = self._getDroneStateVector(0)
            pos   = state[0:3]
            quat  = state[3:7]

        # Rotation matrix from quaternion (body → world)
        rot = np.array(
            p.getMatrixFromQuaternion(quat, physicsClientId=self.CLIENT)
        ).reshape(3, 3)

        # Body +X is forward, slight downward tilt for better gate visibility
        fwd_body  = np.array([1.0, 0.0, -np.tan(np.deg2rad(10))])
        fwd_world = rot @ fwd_body
        fwd_world = fwd_world / np.linalg.norm(fwd_world)

        up_world  = rot @ np.array([0.0, 0.0, 1.0])

        cam_eye    = pos + rot @ np.array([0.02, 0.0, 0.005])
        cam_target = cam_eye + 0.5 * fwd_world

        view_matrix = p.computeViewMatrix(
            cameraEyePosition    = cam_eye.tolist(),
            cameraTargetPosition = cam_target.tolist(),
            cameraUpVector       = up_world.tolist(),
            physicsClientId      = self.CLIENT,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov          = 90.0,
            aspect       = self._img_w / self._img_h,
            nearVal      = 0.05,
            farVal       = 100.0,
            physicsClientId = self.CLIENT,
        )

        (_, _, rgba, _, _) = p.getCameraImage(
            width            = self._img_w,
            height           = self._img_h,
            viewMatrix       = view_matrix,
            projectionMatrix = proj_matrix,
            renderer         = p.ER_TINY_RENDERER,
            physicsClientId  = self.CLIENT,
        )

        # getCameraImage returns RGBA; drop alpha channel
        rgb = np.array(rgba, dtype=np.uint8).reshape(
            self._img_h, self._img_w, 4
        )[:, :, :3]
        return rgb
