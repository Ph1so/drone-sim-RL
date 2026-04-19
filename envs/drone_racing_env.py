"""
DroneRacingEnv — Gymnasium environment for autonomous drone racing.

Inherits from gym-pybullet-drones' BaseAviary for physics and drone management.

Action space
------------
Box(4,)  — normalised [Throttle, Roll, Pitch, Yaw] each in [−1, +1].
Mapped to per-motor RPMs via a classical quadrotor mixer.

Observation space
-----------------
Dict{
  "telemetry" : Box(13,)  — [pos_rel_to_gate(3), quat(4), lin_vel(3), ang_vel(3)]
  "gate_obs"  : Box(10,)  — noisy gate-relative observation (see _compute_gate_obs)
                            [curr_rel_x, curr_rel_y, curr_rel_z, curr_dist, curr_yaw_err,
                             next_rel_x, next_rel_y, next_rel_z, next_dist, next_yaw_err]
                            expressed in the drone's yaw frame with distance-scaled noise.
                            next_* is zeroed when current gate is the last on the course.
}

Reward / termination
--------------------
See reward.py for the detailed shaping.  Episode ends on:
  • Collision with any object
  • Leaving the world-bounds box
  • Completing all active gates (success)
  • Exceeding MAX_EPISODE_STEPS (truncation)

Modular perception note
-----------------------
_render_ego_camera() is kept but not used in _computeObs().  At competition time,
replace the GateManager-based gate positions with estimates from a CV pipeline that
produces the same 10-float gate_obs format — the policy requires no changes.
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
        gate_noise_std:         float = 0.3,
        num_gates:              int = 5,
        spawn_mid_course_prob:  float = 0.0,
        gate_pos_offset:        Optional[list] = None,
    ) -> None:
        self._img_h, self._img_w = img_size
        self._gate_noise_std        = gate_noise_std
        self._spawn_mid_course_prob = spawn_mid_course_prob

        # GateManager and RewardComputer are created before super().__init__
        # because BaseAviary.__init__ calls _addObstacles() internally.
        self._gate_manager   = GateManager(num_gates=num_gates, pos_offset=gate_pos_offset)
        self._reward_computer = RewardComputer(self._gate_manager)

        # Per-step cache (populated once, read by reward/term/info methods)
        self._step_cache: dict = {}
        self._episode_steps: int = 0

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

    def _observationSpace(self) -> spaces.Dict:
        return spaces.Dict({
            "telemetry": spaces.Box(
                low   = -np.inf,
                high  =  np.inf,
                shape = (13,),
                dtype = np.float32,
            ),
            # 10-D gate-relative obs: [current_gate(5), next_gate(5)] in drone yaw frame.
            # next_gate slice is zeros when current gate is the last active gate.
            # Swap source: replace GateManager lookups with CV pipeline estimates
            # at competition time — policy interface stays identical.
            "gate_obs": spaces.Box(
                low   = -np.inf,
                high  =  np.inf,
                shape = (10,),
                dtype = np.float32,
            ),
        })

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
    def _computeObs(self) -> dict:
        cache = self._get_step_state()
        state = cache["state"]

        pos_world = state[0:3].astype(np.float32)
        gate_pos  = self._gate_manager.current_gate.position.astype(np.float32)
        pos_rel   = pos_world - gate_pos          # gate-relative position (3,)

        quat    = state[3:7].astype(np.float32)
        lin_vel = state[10:13].astype(np.float32)
        ang_vel = state[13:16].astype(np.float32)

        telemetry = np.concatenate([pos_rel, quat, lin_vel, ang_vel])   # (13,)
        gate_obs  = self._compute_gate_obs(
            pos = state[0:3].astype(np.float64),
            rpy = cache["rpy"],
        )

        return {"telemetry": telemetry, "gate_obs": gate_obs}

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
            drone_rpy = cache["rpy"],
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
        return info

    # ══════════════════════════════════════════════════════════════════════════
    # Internal helpers
    # ══════════════════════════════════════════════════════════════════════════

    # Speed injected at mid-course spawns — approximates realistic post-gate velocity.
    # Matches the evaluation distribution (drone exits a gate with forward momentum)
    # so the policy learns to handle the turn, not just approach from a standstill.
    _SPAWN_SPEED: float = 2.0   # m/s along previous gate's exit normal

    def _teleport_to_gate_approach(self, k: int) -> None:
        """
        Teleport the drone to just past gate k-1's exit, aligned with that gate's
        exit direction, with realistic forward velocity.

        Previously the drone spawned at rest already facing gate k.  That mismatch
        caused the policy to learn "approach gate k from a standstill" rather than
        "turn from gate k-1's exit momentum into gate k's approach axis" — the exact
        maneuver that fails in evaluation.

        The drone now spawns:
          - Position : 1.5 m along gate k-1's exit normal (unchanged)
          - Yaw      : aligned with gate k-1's exit direction (was: pointing at gate k)
          - Velocity : _SPAWN_SPEED m/s along the exit normal  (was: zero)
          - Angular  : zero

        GateManager is fast-forwarded to k.

        Parameters
        ----------
        k : int
            Index of the gate to target after the teleport (1-based, k ≥ 1).
        """
        prev_gate = self._gate_manager.gates[k - 1]
        spawn_pos = prev_gate.position + prev_gate.normal * 1.5   # 1.5 m past exit

        # Yaw aligned with the exit direction of the gate just passed.
        # In evaluation the drone exits heading along prev_gate.normal — match that.
        exit_yaw = float(np.arctan2(prev_gate.normal[1], prev_gate.normal[0]))

        quat = p.getQuaternionFromEuler(
            [0.0, 0.0, exit_yaw], physicsClientId=self.CLIENT
        )
        p.resetBasePositionAndOrientation(
            self.DRONE_IDS[0],
            spawn_pos.tolist(),
            quat,
            physicsClientId=self.CLIENT,
        )
        spawn_vel = (prev_gate.normal * self._SPAWN_SPEED).tolist()
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

    # ── Noise scaling constants ────────────────────────────────────────
    _NOISE_DIST_REF:  float = 3.0   # m — distance at which noise == gate_noise_std
    _MIN_NOISE_RATIO: float = 0.2   # floor: noise never drops below 20% of nominal

    # ------------------------------------------------------------------
    def _single_gate_obs(
        self,
        gate,
        pos: np.ndarray,
        rpy: np.ndarray,
    ) -> np.ndarray:
        """
        Compute a noisy 5-D relative observation for one gate in the drone's yaw frame.

        Format: [rel_x, rel_y, rel_z, dist, gate_yaw_err]

        rel_x / rel_y : offset to gate centre along drone forward / left axes
        rel_z         : vertical offset (positive = gate above drone)
        dist          : euclidean distance derived from the noisy offset
        gate_yaw_err  : signed angle (rad) from drone heading to gate normal

        Noise is scaled with true distance: farther gates are noisier, mimicking
        how a real CV detector has more uncertainty at longer range.
        """
        rel_world  = gate.position - pos                        # (3,) world frame
        true_dist  = float(np.linalg.norm(rel_world))

        # ── Distance-scaled position noise ───────────────────────────
        noise_scale  = max(true_dist / self._NOISE_DIST_REF, self._MIN_NOISE_RATIO)
        effective_std = self._gate_noise_std * noise_scale
        noise_xyz    = self.np_random.normal(0.0, effective_std, size=3).astype(np.float64)
        rel_noisy    = rel_world + noise_xyz

        dist = float(np.linalg.norm(rel_noisy))

        # ── Project into drone yaw frame ──────────────────────────────
        yaw = float(rpy[2])
        c, s  = np.cos(yaw), np.sin(yaw)
        rel_x = float( c * rel_noisy[0] + s * rel_noisy[1])
        rel_y = float(-s * rel_noisy[0] + c * rel_noisy[1])
        rel_z = float(rel_noisy[2])

        # ── Gate heading error (fixed small angular noise) ────────────
        drone_fwd    = np.array([c, s])
        gate_n_2d    = gate.normal[:2]
        cross        = float(drone_fwd[0] * gate_n_2d[1] - drone_fwd[1] * gate_n_2d[0])
        dot          = float(np.dot(drone_fwd, gate_n_2d))
        gate_yaw_err = float(np.arctan2(cross, dot))
        gate_yaw_err += float(self.np_random.normal(0.0, 0.05))

        return np.array([rel_x, rel_y, rel_z, dist, gate_yaw_err], dtype=np.float32)

    # ------------------------------------------------------------------
    def _compute_gate_obs(
        self,
        pos: np.ndarray,
        rpy: np.ndarray,
    ) -> np.ndarray:
        """
        Return a noisy 10-D gate observation: [current_gate(5), next_gate(5)].

        The next-gate slice is all zeros when the current gate is the last active
        gate on the curriculum course (sentinel that no further gate exists).

        Swap note
        ---------
        At competition time replace GateManager lookups with CV pipeline estimates
        in the same format.  The policy requires no changes.
        """
        curr_obs = self._single_gate_obs(self._gate_manager.current_gate, pos, rpy)

        next_gate = self._gate_manager.next_gate
        if next_gate is not None:
            next_obs = self._single_gate_obs(next_gate, pos, rpy)
        else:
            next_obs = np.zeros(5, dtype=np.float32)

        return np.concatenate([curr_obs, next_obs])  # (10,)

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
