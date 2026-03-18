"""
deploy_ros2.py
==============
ROS2 node that loads a trained TD3/SAC policy and runs it on the real robot.
Bridges MuJoCo observation space → real sensor readings → robot commands.

Subscribe:
  /gps/fix            (NavSatFix)   → base_pos
  /imu/data           (Imu)         → base_quat, base_angvel
  /auto_command       (String)      → target burrow GPS → rel_burrow
  /arm/joint_states   (JointState)  → j_angles, j_vels

Publish:
  /cmd_vel            (Twist)       ← screw velocities (Stage 1/3 policy)
  /arm_command        (String)      ← arm joint targets (Stage 2/3 policy)

Usage:
  ros2 run archimedes_survey rl_agent \
    --ros-args \
    -p model_path:=/home/pi/archimedes-survey/deep_rl/models/stage3_survey.zip \
    -p norm_path:=/home/pi/archimedes-survey/deep_rl/models/stage3_survey_norm.pkl \
    -p stage:=3
"""

import json
import math
import sys
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import NavSatFix, Imu, JointState
    from std_msgs.msg import String
except ImportError:
    sys.exit("ROS2 not found.")

try:
    from stable_baselines3 import TD3, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    sys.exit("pip install stable-baselines3[extra]")

import os

# ---------------------------------------------------------------------------
# Dummy env for loading VecNormalize (must match training obs/act shapes)
# ---------------------------------------------------------------------------
OBS_SHAPES = {1: 24, 2: 18, 3: 42}
ACT_SHAPES = {1: 2,  2: 3,  3: 5}
MAX_TORQUE_LOCO = 30.0
ARM_TORQUES     = np.array([1.27, 1.96, 1.27])
MAX_LINEAR_MPS  = 0.088
WHEEL_BASE      = 0.30   # m (screw separation)


class _DummyEnv(gym.Env):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(obs_dim,), dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0,
                                            shape=(act_dim,), dtype=np.float32)
    def reset(self, **kw): return np.zeros(self.observation_space.shape), {}
    def step(self, a):     return np.zeros(self.observation_space.shape), 0.0, True, False, {}


class RLAgentNode(Node):
    def __init__(self):
        super().__init__("rl_agent")

        # Params
        self.declare_parameter("model_path", "")
        self.declare_parameter("norm_path",  "")
        self.declare_parameter("stage",      3)
        self.declare_parameter("control_hz", 5.0)

        model_path = self.get_parameter("model_path").value
        norm_path  = self.get_parameter("norm_path").value
        stage      = self.get_parameter("stage").value
        hz         = self.get_parameter("control_hz").value

        if not model_path or not os.path.exists(model_path + ".zip"
                                                 if not model_path.endswith(".zip")
                                                 else model_path):
            self.get_logger().error(
                f"Model not found: {model_path}\n"
                "  Train first: python deep_rl/train.py --stage 3")
            raise SystemExit(1)

        # Load model + normalisation
        obs_dim = OBS_SHAPES[stage]
        act_dim = ACT_SHAPES[stage]
        dummy   = DummyVecEnv([lambda: _DummyEnv(obs_dim, act_dim)])

        if norm_path and os.path.exists(norm_path):
            self._venv = VecNormalize.load(norm_path, dummy)
            self._venv.training = False
        else:
            self._venv = dummy
            self.get_logger().warn("No normalisation file. Results may be degraded.")

        try:
            self._model = TD3.load(model_path, env=self._venv)
            self.get_logger().info("Loaded TD3 model")
        except Exception:
            self._model = SAC.load(model_path, env=self._venv)
            self.get_logger().info("Loaded SAC model")

        self._stage = stage

        # State variables
        self._base_pos   = np.zeros(3)
        self._base_quat  = np.array([1.0, 0, 0, 0])
        self._base_lv    = np.zeros(3)
        self._base_av    = np.zeros(3)
        self._j_angles   = np.zeros(3)
        self._j_vels     = np.zeros(3)
        self._burrow_pos = None    # set when /auto_command received
        self._nav_phase  = 0      # 0=navigate, 1=probe
        self._home_pos   = None

        # Publishers
        self._pub_vel = self.create_publisher(Twist,  "/cmd_vel",     10)
        self._pub_arm = self.create_publisher(String, "/arm_command", 10)

        # Subscribers
        self.create_subscription(NavSatFix, "/gps/fix",        self._cb_gps,     10)
        self.create_subscription(Imu,       "/imu/data",       self._cb_imu,     10)
        self.create_subscription(JointState,"/arm/joint_states",self._cb_joints,  10)
        self.create_subscription(String,    "/auto_command",   self._cb_cmd,     10)

        # Control timer
        self.create_timer(1.0 / hz, self._control_loop)

        self.get_logger().info(
            f"RL Agent ready: stage={stage} obs={obs_dim} act={act_dim} hz={hz}")

    # ------------------------------------------------------------------
    def _cb_gps(self, msg: NavSatFix):
        if msg.status.status < 0:
            return
        # Simple equirectangular approximation (good for <1 km)
        if self._home_pos is None:
            self._home_pos = np.array([msg.latitude, msg.longitude, 0.0])
        lat0, lon0 = self._home_pos[0], self._home_pos[1]
        dx = (msg.longitude - lon0) * math.cos(math.radians(lat0)) * 111_320
        dy = (msg.latitude  - lat0) * 110_574
        self._base_pos = np.array([dx, dy, 0.0])

        if self._burrow_pos is not None:
            nav_dist = float(np.linalg.norm(self._base_pos - self._burrow_pos[:2]))
            if nav_dist < 1.0 and self._nav_phase == 0:
                self._nav_phase = 1
                self.get_logger().info("Arrived at burrow — switching to PROBE phase")

    def _cb_imu(self, msg: Imu):
        q = msg.orientation
        self._base_quat = np.array([q.w, q.x, q.y, q.z])
        self._base_av   = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ])
        self._base_lv = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ]) * 0.002  # crude velocity estimate

    def _cb_joints(self, msg: JointState):
        n2i = {n: i for i, n in enumerate(msg.name)}
        for ji, jname in enumerate(["arm_j1", "arm_j2", "arm_j3"]):
            if jname in n2i:
                idx = n2i[jname]
                self._j_angles[ji] = msg.position[idx] if idx < len(msg.position) else 0.0
                self._j_vels[ji]   = msg.velocity[idx] if idx < len(msg.velocity) else 0.0

    def _cb_cmd(self, msg: String):
        try:
            d = json.loads(msg.data)
            if "target_lat" in d and self._home_pos is not None:
                lat0, lon0 = self._home_pos[0], self._home_pos[1]
                dx = (d["target_lon"] - lon0) * math.cos(math.radians(lat0)) * 111_320
                dy = (d["target_lat"] - lat0) * 110_574
                self._burrow_pos = np.array([dx, dy, 0.0])
                self._nav_phase  = 0
                self.get_logger().info(f"New burrow target: {dx:.1f}, {dy:.1f} m")
        except Exception as e:
            self.get_logger().error(f"cmd parse error: {e}")

    # ------------------------------------------------------------------
    def _build_obs(self) -> np.ndarray:
        if self._stage == 1:
            # Locomotion obs (24,)
            if self._burrow_pos is None:
                rel = np.zeros(3)
            else:
                rel_world = self._burrow_pos - self._base_pos
                rel = self._world_to_local(rel_world, self._base_quat)
            sa_vel = np.array([0.0])
            sb_vel = np.array([0.0])
            terrain = np.array([1.0 / 2.0])   # assume wet sand
            elev = np.zeros(4)
            return np.concatenate([
                self._base_pos, self._base_quat, self._base_lv, self._base_av,
                sa_vel, sb_vel, rel, terrain, elev
            ]).astype(np.float32)

        elif self._stage == 2:
            # Arm obs (18,)
            if self._burrow_pos is None:
                rel = np.zeros(3)
            else:
                rel = self._burrow_pos - self._base_pos  # approximate
            dist_norm = np.array([float(np.linalg.norm(rel)) / 0.65])
            step_norm = np.array([0.5])
            return np.concatenate([
                rel, self._j_angles, self._j_vels,
                np.zeros(3), np.zeros(3), dist_norm, step_norm
            ]).astype(np.float32)

        else:
            # Full survey obs (42,)
            if self._burrow_pos is None:
                rel_b_local = np.zeros(3)
                rel_probe   = np.zeros(3)
                nav_dist = np.array([1.0])
                prb_dist = np.array([1.0])
            else:
                rel_b_world = self._burrow_pos - self._base_pos
                rel_b_local = self._world_to_local(rel_b_world, self._base_quat)
                rel_probe   = self._burrow_pos - self._base_pos  # approx ee
                nav_dist = np.array([float(np.linalg.norm(rel_b_world)) / 30.0])
                prb_dist = np.array([float(np.linalg.norm(rel_probe)) / 1.0])
            terrain = np.array([1.0 / 2.0])
            phase   = np.array([float(self._nav_phase) / 2.0])
            ee_pos  = self._base_pos + np.array([0.0, 0.15, 0.282])  # approx
            sa_vel = np.array([0.0])
            sb_vel = np.array([0.0])
            return np.concatenate([
                self._base_pos, self._base_quat, self._base_lv, self._base_av,
                sa_vel, sb_vel,
                self._j_angles, self._j_vels,
                ee_pos,
                rel_b_local, rel_probe,
                terrain, phase, nav_dist, prb_dist,
            ]).astype(np.float32)

    # ------------------------------------------------------------------
    def _control_loop(self):
        if self._burrow_pos is None:
            return

        obs = self._build_obs()
        obs_norm = self._venv.normalize_obs(obs[None, :])
        action, _ = self._model.predict(obs_norm, deterministic=True)
        action = action[0]

        if self._stage == 1:
            self._publish_loco(action[0], action[1])

        elif self._stage == 2:
            self._publish_arm(action)

        else:  # stage 3
            if self._nav_phase == 0:
                self._publish_loco(action[0], action[1])
            else:
                self._stop_loco()
                self._publish_arm(action[2:5])

    def _publish_loco(self, sa, sb):
        """Convert screw torque actions → Twist for teleop_node."""
        # sa, sb normalised -1..+1 → map to linear/angular
        linear  = (sa + sb) / 2.0 * MAX_LINEAR_MPS
        angular = (sb - sa) / 2.0 * (MAX_LINEAR_MPS / (WHEEL_BASE / 2.0)) * 0.3
        t = Twist()
        t.linear.x  = float(linear)
        t.angular.z = float(angular)
        self._pub_vel.publish(t)

    def _stop_loco(self):
        self._pub_vel.publish(Twist())

    def _publish_arm(self, action_3):
        """Convert normalised arm action → joint angle targets."""
        # Map normalised torque action → joint position command (approximate)
        # In real hardware: servo position is set, not torque
        targets = self._j_angles + action_3 * ARM_TORQUES * 0.02  # crude PID step
        targets[0] = float(np.clip(targets[0], -1.047, 1.047))
        targets[1] = float(np.clip(targets[1], 0.0,   2.094))
        targets[2] = float(np.clip(targets[2], 0.0,   2.356))
        cmd = {"type": "arm",
               "j1": math.degrees(targets[0]),
               "j2": math.degrees(targets[1]),
               "j3": math.degrees(targets[2])}
        self._pub_arm.publish(String(data=json.dumps(cmd)))

    @staticmethod
    def _world_to_local(vec, quat):
        w, x, y, z = quat
        R = np.array([
            [1-2*(y*y+z*z), 2*(x*y+w*z),   2*(x*z-w*y)],
            [2*(x*y-w*z),   1-2*(x*x+z*z), 2*(y*z+w*x)],
            [2*(x*z+w*y),   2*(y*z-w*x),   1-2*(x*x+y*y)],
        ]).T
        return R @ vec


# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = RLAgentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("RL Agent shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
