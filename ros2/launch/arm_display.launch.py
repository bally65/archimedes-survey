"""
Archimedes Survey Arm -- ROS2 Launch File
==========================================
Launches:
  1. robot_state_publisher (URDF → /tf)
  2. joint_state_publisher_gui (manual joint sliders)
  3. RViz2 (visualization)

Usage:
  ros2 launch archimedes_arm arm_display.launch.py
  ros2 launch archimedes_arm arm_display.launch.py use_rviz:=false
"""
import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    pkg_name = "archimedes_arm"

    # ── Paths ──────────────────────────────────────────────────────────────────
    # Allow running directly from the repo without a full ROS package install
    repo_root = Path(__file__).resolve().parents[2]   # archimedes-survey/
    urdf_file = repo_root / "ros2" / "urdf" / "arm_3dof.urdf.xacro"
    rviz_file = repo_root / "ros2" / "config" / "arm_rviz.rviz"

    # ── Arguments ──────────────────────────────────────────────────────────────
    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="true",
        description="Launch RViz2 for visualization",
    )
    use_gui_arg = DeclareLaunchArgument(
        "use_gui",
        default_value="true",
        description="Launch joint_state_publisher_gui for manual joint control",
    )

    use_rviz = LaunchConfiguration("use_rviz")
    use_gui  = LaunchConfiguration("use_gui")

    # ── Robot description (xacro → URDF string) ────────────────────────────────
    robot_description = Command(["xacro ", str(urdf_file)])

    # ── Nodes ──────────────────────────────────────────────────────────────────
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{
            "robot_description": robot_description,
            "publish_frequency": 50.0,
        }],
    )

    joint_state_publisher_gui = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        name="joint_state_publisher_gui",
        output="screen",
        condition=IfCondition(use_gui),
    )

    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", str(rviz_file)] if rviz_file.exists() else [],
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription([
        use_rviz_arg,
        use_gui_arg,
        robot_state_publisher,
        joint_state_publisher_gui,
        rviz2,
    ])
