"""
survey_env.py
=============
Full survey mission environment: locomotion + arm + burrow probing.
Hierarchical task: navigate to detected burrow location, then probe.

This is the top-level environment used for final combined training.
Inherits terrain variety from LocoEnv and arm precision from ArmEnv.

Action space  (5,):
  [screw_a, screw_b, arm_j1, arm_j2, arm_j3]  all normalised -1..+1

Observation   (42,):
  base_pos     (3)   x, y, z
  base_quat    (4)   orientation
  base_linvel  (3)
  base_angvel  (3)
  screw_vels   (2)   a, b
  j_angles     (3)   arm joints
  j_vels       (3)   arm joints
  ee_pos       (3)   world frame
  rel_burrow   (3)   burrow - base_pos (local frame, navigation target)
  rel_probe    (3)   burrow - ee_pos (arm target)
  terrain_id   (1)
  mission_phase(1)   0=navigate, 1=probe, 2=done
  dist_nav     (1)   base to burrow
  dist_probe   (1)   ee to burrow

Mission phases:
  Phase 0 (NAVIGATE): drive to within NAV_ARRIVE_DIST of burrow
  Phase 1 (PROBE):    lower arm probe into burrow (within PROBE_ARRIVE_DIST)
  Phase 2 (DONE):     episode success
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

from loco_env import TERRAINS, LocoEnv

XML_PATH         = os.path.join(os.path.dirname(__file__), "archimedes_survey.xml")
NAV_ARRIVE_DIST  = 0.35   # m: navigation phase success — must be < arm reach (0.57m)
PROBE_ARRIVE_DIST= 0.025  # m: probe tip inside burrow
MAX_STEPS        = 5000
LOCO_TORQUE      = 30.0
ARM_TORQUES      = np.array([1.27, 1.96, 1.27])

PHASE_NAVIGATE = 0
PHASE_PROBE    = 1
PHASE_DONE     = 2


class SurveyEnv(gym.Env):
    """Full survey environment: navigate + probe."""
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None, terrain_idx: int = -1):
        super().__init__()
        self._render_mode  = render_mode
        self._terrain_idx  = terrain_idx
        self._curr_terrain = 0

        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data  = mujoco.MjData(self.model)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(34,), dtype=np.float32)

        self._ee_id    = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self._floor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._tgt_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target_body")

        self._burrow_pos = np.zeros(3)
        self._phase      = PHASE_NAVIGATE
        self._step_count = 0
        self._prev_nav_dist  = 0.0
        self._prev_probe_dist= 0.0
        self._phases_done    = set()
        self._viewer         = None

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Terrain
        if self._terrain_idx < 0:
            self._curr_terrain = self.np_random.integers(0, len(TERRAINS))
        else:
            self._curr_terrain = self._terrain_idx
        t = TERRAINS[self._curr_terrain]
        self.model.geom_friction[self._floor_id] = t["floor_friction"]
        self.model.opt.viscosity = t["viscosity"]
        self.model.opt.density   = t["density"]

        # Place robot at origin
        self.data.qpos[:3]  = [0.0, 0.0, 0.12]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        # Arm stowed
        self.data.qpos[7:10] = [0.0, 0.0, 0.0]

        # Random burrow location 5–30 m away
        angle = self.np_random.uniform(0, 2 * math.pi)
        dist  = self.np_random.uniform(5.0, 30.0)
        self._burrow_pos = np.array([math.cos(angle) * dist,
                                     math.sin(angle) * dist,
                                     0.0])

        self.model.body_pos[self._tgt_body] = self._burrow_pos

        mujoco.mj_forward(self.model, self.data)

        base_pos = self.data.qpos[:3].copy()
        ee_pos   = self.data.site_xpos[self._ee_id].copy()
        self._prev_nav_dist   = float(np.linalg.norm(base_pos - self._burrow_pos))
        self._prev_probe_dist = float(np.linalg.norm(ee_pos - self._burrow_pos))
        self._phase      = PHASE_NAVIGATE
        self._step_count = 0
        self._phases_done= set()

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)

        if self._phase == PHASE_NAVIGATE:
            # Drive screws, keep arm stowed
            self.data.ctrl[0] = action[0] * LOCO_TORQUE
            self.data.ctrl[1] = action[1] * LOCO_TORQUE
            self.data.ctrl[2:5] = 0.0
        else:
            # Stop screws, deploy arm
            self.data.ctrl[0:2] = 0.0
            self.data.ctrl[2:5] = action[2:5] * ARM_TORQUES

        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        # Phase transition
        base_pos = self.data.qpos[:3].copy()
        ee_pos   = self.data.site_xpos[self._ee_id].copy()
        nav_dist = float(np.linalg.norm(base_pos - self._burrow_pos))
        prb_dist = float(np.linalg.norm(ee_pos - self._burrow_pos))

        if self._phase == PHASE_NAVIGATE and nav_dist < NAV_ARRIVE_DIST:
            self._phase = PHASE_PROBE

        done    = False
        success = False
        if self._phase == PHASE_PROBE and prb_dist < PROBE_ARRIVE_DIST:
            self._phase = PHASE_DONE
            done    = True
            success = True

        obs  = self._get_obs()
        rew  = self._compute_reward(nav_dist, prb_dist, success)
        trunc = self._step_count >= MAX_STEPS

        self._prev_nav_dist   = nav_dist
        self._prev_probe_dist = prb_dist

        info = {
            "terrain":  TERRAINS[self._curr_terrain]["name"],
            "phase":    self._phase,
            "nav_dist": nav_dist,
            "prb_dist": prb_dist,
            "success":  success,
        }
        return obs, rew, done or not np.all(np.isfinite(self.data.qpos)), trunc, info

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        base_pos  = self.data.qpos[:3].copy()
        base_quat = self.data.qpos[3:7].copy()
        base_lv   = self.data.qvel[:3].copy()
        base_av   = self.data.qvel[3:6].copy()
        sa_vel    = np.array([self.data.qvel[6]])
        sb_vel    = np.array([self.data.qvel[7]])
        j_ang     = self.data.qpos[7:10].copy()
        j_vel     = self.data.qvel[7:10].copy()
        ee_pos    = self.data.site_xpos[self._ee_id].copy()

        rel_burrow_world = self._burrow_pos - base_pos
        rel_burrow_local = LocoEnv._world_to_local(rel_burrow_world, base_quat)
        rel_probe        = self._burrow_pos - ee_pos

        terrain   = np.array([float(self._curr_terrain) / 2.0])
        phase     = np.array([float(self._phase) / 2.0])
        nav_dist  = np.array([float(np.linalg.norm(rel_burrow_world)) / 30.0])
        prb_dist  = np.array([float(np.linalg.norm(rel_probe)) / 1.0])

        return np.concatenate([
            base_pos, base_quat, base_lv, base_av,
            sa_vel, sb_vel,
            j_ang, j_vel,
            ee_pos,
            rel_burrow_local, rel_probe,
            terrain, phase, nav_dist, prb_dist,
        ]).astype(np.float32)

    # ------------------------------------------------------------------
    def _compute_reward(self, nav_dist, prb_dist, success) -> float:
        rew = 0.0

        if self._phase == PHASE_NAVIGATE:
            # Progress reward
            rew += (self._prev_nav_dist - nav_dist) * 20.0
            rew += math.exp(-nav_dist * 0.1) * 0.3
            # Phase arrival bonus
            if nav_dist < NAV_ARRIVE_DIST and "nav" not in self._phases_done:
                rew += 50.0
                self._phases_done.add("nav")
            # Stability
            quat = self.data.qpos[3:7]
            r, p = LocoEnv._quat_to_rp(quat)
            rew -= 3.0 * (abs(r) + abs(p))

        elif self._phase == PHASE_PROBE:
            # Arm reach reward
            rew += (self._prev_probe_dist - prb_dist) * 80.0
            rew += math.exp(-prb_dist * 10.0) * 1.0
            # Phase bonuses
            for thresh, bonus in [(0.3, 10), (0.1, 30), (0.05, 80), (0.025, 200)]:
                key = f"prb_{thresh}"
                if prb_dist < thresh and key not in self._phases_done:
                    rew += bonus
                    self._phases_done.add(key)

        if success:
            rew += 500.0 + (MAX_STEPS - self._step_count) * 0.1

        # Energy
        rew -= 0.005 * float(np.sum(np.square(self.data.ctrl)))
        # Step penalty
        rew -= 0.02

        return rew

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
