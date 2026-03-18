"""
loco_env.py
===========
Gymnasium environment: Archimedes dual-screw locomotion.
Trains the vehicle to navigate across variable terrain (sand/mud).

Inspired by bally65/Archimedes-Hand-/5. Deep_LR/whole_body_env.py
but adapted for our 2-screw differential drive (not 4-screw).

Action space  (2,):  [screw_a_torque, screw_b_torque]  normalised -1..+1
Observation   (24,):
  base_pos    (3)  x, y, z
  base_quat   (4)  orientation quaternion
  base_linvel (3)  vx, vy, vz
  base_angvel (3)  wx, wy, wz
  screw_a_vel (1)  rad/s
  screw_b_vel (1)  rad/s
  rel_target  (3)  target - base_pos (local frame)
  terrain_id  (1)  0=dry sand, 1=wet sand, 2=mud
  elevation   (4)  rough local terrain height samples (random noise)

Terrains (matching our Isaac Sim matrix):
  0 = dry sand  (10% moisture) — friction 0.8,  viscosity 0.001
  1 = wet sand  (50% moisture) — friction 1.8,  viscosity 0.05
  2 = mud       (70% moisture) — friction 2.5,  viscosity 0.5
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

XML_PATH = os.path.join(os.path.dirname(__file__), "archimedes_survey.xml")

# ---------------------------------------------------------------------------
# Terrain configs (friction tuple: sliding, torsional, rolling)
# ---------------------------------------------------------------------------
TERRAINS = [
    {"name": "dry_sand",  "floor_friction": (0.8,  0.3, 0.1), "viscosity": 0.001, "density": 1500},
    {"name": "wet_sand",  "floor_friction": (1.8,  0.6, 0.3), "viscosity": 0.05,  "density": 1700},
    {"name": "mud",       "floor_friction": (2.5,  0.8, 0.5), "viscosity": 0.5,   "density": 1900},
]

MAX_EPISODE_STEPS = 3000
MAX_TORQUE        = 30.0    # N·m, matches XML forcerange
ARRIVE_DIST       = 0.5     # m, target reached threshold
MAX_RANGE         = 50.0    # m, random target range

# Screw limits (from simulation: max 0.088 m/s)
MAX_LINEAR_MPS = 0.088
SCREW_GEAR     = 10.0


class LocoEnv(gym.Env):
    """Dual-screw locomotion environment with terrain curriculum."""

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None, terrain_idx: int = -1):
        """
        Args:
            terrain_idx: -1 = random each episode, 0/1/2 = fixed terrain
        """
        super().__init__()
        self._render_mode    = render_mode
        self._terrain_idx    = terrain_idx
        self._current_terrain = 0

        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data  = mujoco.MjData(self.model)

        # Action: screw torques normalised -1..+1
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(23,), dtype=np.float32
        )

        self._target_pos    = np.zeros(3)
        self._step_count    = 0
        self._prev_dist     = 0.0
        self._viewer        = None

        # IDs
        self._base_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "mobile_base")
        self._floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Choose terrain
        if self._terrain_idx < 0:
            self._current_terrain = self.np_random.integers(0, len(TERRAINS))
        else:
            self._current_terrain = self._terrain_idx
        self._apply_terrain(self._current_terrain)

        # Random target within MAX_RANGE
        angle = self.np_random.uniform(0, 2 * math.pi)
        dist  = self.np_random.uniform(5.0, MAX_RANGE)
        self._target_pos = np.array([math.cos(angle) * dist,
                                     math.sin(angle) * dist,
                                     0.0])

        # Set target visual marker
        tid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_body")
        self.model.body_pos[tid] = self._target_pos

        self._step_count = 0
        self._prev_dist  = np.linalg.norm(
            self.data.qpos[:3] - self._target_pos)

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)

        # Map normalised action → torque
        self.data.ctrl[0] = float(action[0]) * MAX_TORQUE  # screw_a
        self.data.ctrl[1] = float(action[1]) * MAX_TORQUE  # screw_b
        # Arm joints: zero torque (arm stowed during locomotion)
        self.data.ctrl[2:5] = 0.0

        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        obs  = self._get_obs()
        rew  = self._compute_reward()
        done = self._is_done()
        trunc = self._step_count >= MAX_EPISODE_STEPS

        return obs, rew, done, trunc, {"terrain": TERRAINS[self._current_terrain]["name"]}

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        base_pos  = self.data.qpos[:3].copy()
        base_quat = self.data.qpos[3:7].copy()
        base_lv   = self.data.qvel[:3].copy()
        base_av   = self.data.qvel[3:6].copy()

        # Screw velocities (joints 6 & 7 after 6-DOF free joint)
        sa_vel = np.array([self.data.qvel[6]])
        sb_vel = np.array([self.data.qvel[7]])

        # Target in local frame (world X forward)
        rel = self._target_pos - base_pos
        rel_local = self._world_to_local(rel, base_quat)

        terrain = np.array([float(self._current_terrain) / 2.0])   # 0..1
        elev    = self.np_random.uniform(-0.01, 0.01, 4)            # terrain roughness

        return np.concatenate([
            base_pos, base_quat, base_lv, base_av,
            sa_vel, sb_vel, rel_local, terrain, elev,
        ]).astype(np.float32)

    # ------------------------------------------------------------------
    def _compute_reward(self) -> float:
        base_pos = self.data.qpos[:3].copy()
        dist = float(np.linalg.norm(base_pos - self._target_pos))

        # ① Distance improvement
        progress   = self._prev_dist - dist
        rew_progress = progress * 15.0
        self._prev_dist = dist

        # ② Proximity bonus (exponential)
        rew_proximity = math.exp(-dist * 0.3) * 0.5

        # ③ Arrival bonus
        rew_arrive = 200.0 if dist < ARRIVE_DIST else 0.0

        # ④ Stability penalty (roll / pitch)
        quat = self.data.qpos[3:7]
        roll, pitch = self._quat_to_rp(quat)
        rew_tilt = -5.0 * (abs(roll) + abs(pitch))

        # ⑤ Energy penalty (smooth control)
        rew_energy = -0.01 * float(np.sum(np.square(self.data.ctrl[:2])))

        # ⑥ Out-of-bounds penalty
        z = float(base_pos[2])
        rew_fall = -50.0 if (z < -0.5 or z > 1.0) else 0.0

        return rew_progress + rew_proximity + rew_arrive + rew_tilt + rew_energy + rew_fall

    # ------------------------------------------------------------------
    def _is_done(self) -> bool:
        base_pos = self.data.qpos[:3].copy()
        dist = float(np.linalg.norm(base_pos - self._target_pos))
        z    = float(base_pos[2])

        if dist < ARRIVE_DIST:
            return True
        if z < -0.5 or z > 1.5:    # fell over / flew up
            return True
        if not np.all(np.isfinite(self.data.qpos)):
            return True
        return False

    # ------------------------------------------------------------------
    def _apply_terrain(self, idx: int):
        t = TERRAINS[idx]
        # Set floor friction
        self.model.geom_friction[self._floor_id] = t["floor_friction"]
        # Set fluid properties
        self.model.opt.viscosity = t["viscosity"]
        self.model.opt.density   = t["density"]

    # ------------------------------------------------------------------
    @staticmethod
    def _world_to_local(vec: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """Rotate world-frame vector into body-local frame using quaternion."""
        w, x, y, z = quat
        # Conjugate rotation
        R = np.array([
            [1-2*(y*y+z*z),  2*(x*y+w*z),   2*(x*z-w*y)],
            [2*(x*y-w*z),    1-2*(x*x+z*z), 2*(y*z+w*x)],
            [2*(x*z+w*y),    2*(y*z-w*x),   1-2*(x*x+y*y)],
        ]).T   # transpose = inverse rotation
        return R @ vec

    @staticmethod
    def _quat_to_rp(quat: np.ndarray):
        """Return roll, pitch from quaternion."""
        w, x, y, z = quat
        roll  = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
        sinp  = 2*(w*y-z*x)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))
        return roll, pitch

    # ------------------------------------------------------------------
    def render(self):
        if self._render_mode != "human":
            return
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    def close(self):
        if self._viewer:
            self._viewer.close()
            self._viewer = None
