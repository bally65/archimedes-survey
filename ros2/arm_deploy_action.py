"""
Archimedes Survey Arm -- Deployment Action Script
==================================================
Sends joint trajectory commands to deploy the probe arm:
  Stow   : J1=0,  J2=0,   J3=0,   J4=0  (all retracted)
  Deploy : J1=0,  J2=90°, J3=120°, J4=0  (probe tip -8mm into sand)
  Ultrasound: J1=0, J2=90°, J3=120°, J4=-30° (30° off-normal, IEEE optimal for void detection)
  Custom : specify angles via command line

Geometry note (verified 2026-03-17):
  J2=90°, J3=120° -> probe tip X=-8mm (8mm into sand) [CORRECTED from J3=90°]
  J2=80°, J3=90°  -> probe tip X=+137mm (does NOT reach ground!)
  J3 max limit = 135° (MG996R). J3=120° gives safe margin.

Usage:
  python arm_deploy_action.py stow
  python arm_deploy_action.py deploy
  python arm_deploy_action.py ultrasound   # deploy + activate transducer tilt
  python arm_deploy_action.py custom --j1 0 --j2 90 --j3 120 --j4 0
  python arm_deploy_action.py scan
"""
import math
import time
import argparse
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration


J1_YAW      = "j1_yaw"
J2_SHOULDER = "j2_shoulder"
J3_ELBOW    = "j3_elbow"
J4_TILT     = "j4_tilt"
JOINTS      = [J1_YAW, J2_SHOULDER, J3_ELBOW, J4_TILT]

DEG = math.pi / 180.0


class ArmController(Node):
    def __init__(self):
        super().__init__("arm_deploy_controller")
        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/arm_trajectory_controller/follow_joint_trajectory",
        )

    def send_goal(self, positions_deg: list, duration_s: float = 3.0):
        """
        Send a single-point trajectory goal.
        positions_deg: [j1_deg, j2_deg, j3_deg, j4_deg]
        j4_deg (transducer tilt): default 0.0 if omitted
        duration_s: time to reach goal
        """
        # Pad to 4 values if caller passes 3 (backwards compat)
        p = list(positions_deg) + [0.0] * (4 - len(positions_deg))
        self.get_logger().info(
            f"Sending goal: J1={p[0]:.1f} J2={p[1]:.1f} "
            f"J3={p[2]:.1f} J4={p[3]:.1f} deg"
        )

        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Action server not available!")
            return False

        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = JOINTS

        point = JointTrajectoryPoint()
        point.positions = [v * DEG for v in p]
        point.velocities = [0.0] * 4
        point.time_from_start = Duration(
            sec=int(duration_s),
            nanosec=int((duration_s % 1) * 1e9),
        )
        traj.points = [point]
        goal.trajectory = traj

        future = self._action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected!")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        self.get_logger().info(f"Result: error_code={result.error_code}")
        return result.error_code == 0

    def stow(self):
        """Retract arm to vertical stowed position."""
        self.get_logger().info("=== STOW: arm retracted ===")
        return self.send_goal([0.0, 0.0, 0.0, 0.0], duration_s=4.0)

    def deploy(self):
        """Deploy probe arm so tip reaches sand surface (-8mm).
        Verified: J2=90°, J3=120° -> tip X=-8mm into sand.
        """
        self.get_logger().info("=== DEPLOY: probe to sand surface ===")
        # Step 1: swing J2 forward (keep J3 straight to avoid collision)
        ok = self.send_goal([0.0, 90.0, 0.0, 0.0], duration_s=3.0)
        time.sleep(0.5)
        # Step 2: swing J3 down to 120° -> probe enters sand
        if ok:
            ok = self.send_goal([0.0, 90.0, 120.0, 0.0], duration_s=2.5)
        return ok

    def ultrasound(self):
        """Deploy + activate j4 transducer tilt for acoustic scan.
        J4=-30° tilts transducer 30° forward — IEEE literature optimal for
        void/cavity detection (reduces surface backscatter, enhances internal scatter).
        """
        self.get_logger().info("=== ULTRASOUND: deploy + 30-deg transducer tilt ===")
        ok = self.deploy()
        time.sleep(0.3)
        if ok:
            # Tilt transducer -30° (forward): optimal incident angle per IEEE 7890758
            ok = self.send_goal([0.0, 90.0, 120.0, -30.0], duration_s=1.0)
        return ok

    def scan(self):
        """Sweep J1 yaw ±45° to survey area for burrow openings."""
        self.get_logger().info("=== SCAN: sweeping J1 yaw for area survey ===")
        # Deploy to camera survey pose (J2=30° J3=20° looking forward-down)
        self.send_goal([0.0, 30.0, 20.0, 0.0], duration_s=2.0)
        time.sleep(0.3)
        self.send_goal([-45.0, 30.0, 20.0, 0.0], duration_s=3.0)
        time.sleep(0.3)
        self.send_goal([45.0, 30.0, 20.0, 0.0], duration_s=6.0)
        time.sleep(0.3)
        self.send_goal([0.0, 30.0, 20.0, 0.0], duration_s=3.0)
        self.get_logger().info("Scan complete.")


def main():
    parser = argparse.ArgumentParser(description="Archimedes Arm Controller")
    parser.add_argument("command", choices=["stow", "deploy", "ultrasound", "custom", "scan"],
                        help="Action to perform")
    parser.add_argument("--j1", type=float, default=0.0,
                        help="J1 yaw angle (deg), for custom command")
    parser.add_argument("--j2", type=float, default=0.0,
                        help="J2 shoulder angle (deg), for custom command")
    parser.add_argument("--j3", type=float, default=0.0,
                        help="J3 elbow angle (deg), for custom command")
    parser.add_argument("--j4", type=float, default=0.0,
                        help="J4 transducer tilt angle (deg, -30~+30), for custom command")
    parser.add_argument("--duration", type=float, default=3.0,
                        help="Motion duration (s), for custom command")
    args = parser.parse_args()

    rclpy.init()
    controller = ArmController()

    try:
        if args.command == "stow":
            controller.stow()
        elif args.command == "deploy":
            controller.deploy()
        elif args.command == "ultrasound":
            controller.ultrasound()
        elif args.command == "scan":
            controller.scan()
        elif args.command == "custom":
            controller.send_goal(
                [args.j1, args.j2, args.j3, args.j4],
                duration_s=args.duration,
            )
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
