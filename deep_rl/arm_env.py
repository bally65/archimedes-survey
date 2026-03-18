"""
arm_env.py
==========
Gymnasium environment: 3-DOF probe arm target reaching.
Directly adapted from bally65/Archimedes-Hand-/5. Deep_LR/robot_arm_env.py
but for our 3DOF servo arm (J1 yaw, J2 shoulder, J3 elbow).

Action space  (3,):  [j1_torque, j2_torque, j3_torque]  normalised -1..+1
Observation   (18,):
  rel_pos    (3)   target - ee_pos
  j_angles   (3)   joint positions (rad)
  j_vels     (3)   joint velocities (rad/s)
  prev_ctrl  (3)   previous torques (Nm)
  ee_vel     (3)   end-effector linear velocity
  dist_norm  (1)   distance / max_reach (0..1)
  step_norm  (1)   step / max_steps (0..1)

Joint limits (matching our URDF):
  J1 yaw      : ±60°  = ±1.047 rad
  J2 shoulder : 0-120° = 0..2.094 rad
  J3 elbow    : 0-135° = 0..2.356 rad

Max torques (servo ratings):
  J1, J3: MG996R  → 1.27 Nm
  J2:     DS3225MG → 1.96 Nm
"""

import os
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import mujoco
except ImportError:
    raise ImportError("pip install mujoco")

XML_PATH  = os.path.join(os.path.dirname(__file__), "archimedes_survey.xml")
MAX_REACH = 0.65    # m, estimated max probe reach
SUCCESS_THRESHOLD = 0.015   # 15 mm (probe tip accuracy)
MAX_EPISODE_STEPS = 2000

# Arm actuator indices in ctrl array (ctrl[2..4])
ARM_CTRL_SLICE = slice(2, 5)
ARM_TORQUES    = np.array([1.27, 1.96, 1.27])   # J1, J2, J3


class ArmEnv(gym.Env):
    """
    3-DOF probe arm reaching environment.
    The vehicle chassis is fixed; only the arm is trained.
    Reward structure inspired by bally65's RobotArmEnv (phase rewards,
    direction reward, smooth control penalty).
    """
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()
        self._render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data  = mujoco.MjData(self.model)

        # Lock chassis (set freejoint to fixed position)
        self._lock_chassis()

        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(17,), dtype=np.float32
        )

        # Site ID for end-effector
        self._ee_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

        self._target_pos  = np.zeros(3)
        self._step_count  = 0
        self._prev_dist   = 0.0
        self._min_dist    = 0.0
        self._prev_ctrl   = np.zeros(3)
        self._prev_jvels  = np.zeros(3)
        self._phases_done = set()
        self._viewer      = None

    # ------------------------------------------------------------------
    def _lock_chassis(self):
        """Fix the freejoint so only the arm moves during training."""
        fj_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_free")
        if fj_id >= 0:
            # qpos[0:7] → freejoint; set to identity and freeze via bodyinertia
            pass  # we set qpos in reset() instead

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Fix chassis at origin
        self.data.qpos[0:3] = [0.0, 0.0, 0.12]   # x,y,z
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # w,x,y,z quaternion

        # Random arm start pose (within joint limits)
        j1 = self.np_random.uniform(-1.047, 1.047)
        j2 = self.np_random.uniform(0.0,   2.094)
        j3 = self.np_random.uniform(0.0,   2.356)
        self.data.qpos[7] = j1   # arm_j1 (after freejoint 7DOF)
        self.data.qpos[8] = j2
        self.data.qpos[9] = j3

        mujoco.mj_forward(self.model, self.data)

        # Random reachable target
        self._target_pos = self._sample_target()

        # Reset state
        ee_pos = self.data.site_xpos[self._ee_id].copy()
        self._prev_dist   = float(np.linalg.norm(ee_pos - self._target_pos))
        self._min_dist    = self._prev_dist
        self._prev_ctrl   = np.zeros(3)
        self._prev_jvels  = np.zeros(3)
        self._phases_done = set()
        self._step_count  = 0

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        self._prev_ctrl  = self.data.ctrl[ARM_CTRL_SLICE].copy()
        self._prev_jvels = self.data.qvel[7:10].copy()

        # Apply action
        ctrl = np.clip(action, -1.0, 1.0) * ARM_TORQUES
        self.data.ctrl[ARM_CTRL_SLICE] = ctrl
        self.data.ctrl[0:2] = 0.0   # screws off

        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        obs  = self._get_obs()
        rew  = self._compute_reward()
        done, success = self._is_done()
        trunc = self._step_count >= MAX_EPISODE_STEPS

        return obs, rew, done, trunc, {"success": success}

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        ee_pos  = self.data.site_xpos[self._ee_id].copy()
        rel_pos = self._target_pos - ee_pos

        j_ang  = self.data.qpos[7:10].copy()
        j_vel  = self.data.qvel[7:10].copy()
        p_ctrl = self._prev_ctrl.copy()

        # EE velocity (body velocity of probe_tip body)
        ptip_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "probe_tip")
        ee_vel = self.data.cvel[ptip_id][:3].copy()

        dist = float(np.linalg.norm(rel_pos))
        dist_norm = np.array([dist / MAX_REACH])
        step_norm = np.array([self._step_count / MAX_EPISODE_STEPS])

        return np.concatenate([
            rel_pos, j_ang, j_vel, p_ctrl, ee_vel, dist_norm, step_norm
        ]).astype(np.float32)

    # ------------------------------------------------------------------
    def _compute_reward(self) -> float:
        ee_pos = self.data.site_xpos[self._ee_id].copy()
        dist   = float(np.linalg.norm(ee_pos - self._target_pos))

        # ① Distance improvement (bally65-style)
        rew = 0.0
        if dist < self._min_dist:
            rew += (self._min_dist - dist) * 30.0
            self._min_dist = dist
        elif dist > self._prev_dist:
            rew -= (dist - self._prev_dist) * 20.0
        self._prev_dist = dist

        # ② Smooth exponential proximity
        rew += math.exp(-dist * 5.0) * 1.5

        # ③ Phase bonuses (one-time, from bally65's approach)
        phase_thresholds = [0.5, 0.3, 0.15, 0.08, 0.03, 0.015]
        phase_bonuses    = [10,  20,   50,  100,  300,   600]
        for thresh, bonus in zip(phase_thresholds, phase_bonuses):
            if dist < thresh and thresh not in self._phases_done:
                rew += bonus
                self._phases_done.add(thresh)

        # ④ Direction alignment
        ee_vel = self.data.cvel[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "probe_tip")
        ][:3]
        ee_speed = float(np.linalg.norm(ee_vel))
        if ee_speed > 1e-4 and dist > 1e-3:
            to_target = (self._target_pos - ee_pos) / (dist + 1e-6)
            move_dir  = ee_vel / (ee_speed + 1e-6)
            align     = float(np.dot(to_target, move_dir))
            rew += max(0, align) ** 2 * 2.0

        # ⑤ Control smoothness
        jvel_change = np.abs(self.data.qvel[7:10] - self._prev_jvels)
        rew -= 0.05 * float(np.sum(jvel_change))

        # ⑥ Energy
        rew -= 0.005 * float(np.sum(np.square(self.data.ctrl[ARM_CTRL_SLICE])))

        # ⑦ Step penalty
        rew -= 0.05

        return rew

    # ------------------------------------------------------------------
    def _is_done(self):
        ee_pos = self.data.site_xpos[self._ee_id].copy()
        dist   = float(np.linalg.norm(ee_pos - self._target_pos))

        if dist <= SUCCESS_THRESHOLD:
            return True, True

        if not np.all(np.isfinite(self.data.qpos)):
            return True, False

        return False, False

    # ------------------------------------------------------------------
    def _sample_target(self) -> np.ndarray:
        """Sample a reachable target in world frame near the arm."""
        # Arm base is at approximately (0, 0.15, 0.12+0.162) = (0, 0.15, 0.282)
        arm_base = np.array([0.0, 0.15, 0.282])
        for _ in range(100):
            # Random direction
            theta = self.np_random.uniform(0, math.pi)        # elevation
            phi   = self.np_random.uniform(-math.pi, math.pi) # azimuth
            r     = self.np_random.uniform(0.15, MAX_REACH * 0.85)
            dx = r * math.sin(theta) * math.cos(phi)
            dy = r * math.sin(theta) * math.sin(phi)
            dz = r * math.cos(theta)
            target = arm_base + np.array([dx, dy, dz])
            if target[2] > 0.05:   # above ground
                return target
        return arm_base + np.array([0.3, 0.0, 0.1])

    # ------------------------------------------------------------------
    def render(self):
        if self._render_mode != "human":
            return
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self._viewer.is_running():
            self._viewer.sync()

    def close(self):
        if self._viewer:
            self._viewer.close()
            self._viewer = None
