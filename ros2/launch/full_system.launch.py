"""
full_system.launch.py
=====================
一鍵啟動所有 Archimedes Survey 節點：
  - robot_state_publisher (URDF)
  - teleop_node           (感測器彙整 + 馬達 PWM 輸出)
  - stepper_driver        (GPIO step/dir → TB6600)
  - auto_navigate         (GPS 自主導航)
  - burrow_detector       (YOLOv8-nano 穴口偵測)
  - mission_logger        (GeoJSON 任務記錄)

Run:
    ros2 launch archimedes_survey full_system.launch.py
Optional args:
    model_path:=/home/pi/burrow_best.pt
    camera_height_m:=0.28
    debug:=true
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = get_package_share_directory("archimedes_survey")
    urdf_path = os.path.join(pkg, "urdf", "arm_3dof.urdf.xacro")

    # --------------- Arguments ---------------
    arg_model   = DeclareLaunchArgument("model_path",
                      default_value="yolov8n.pt",
                      description="YOLOv8 weights path")
    arg_height  = DeclareLaunchArgument("camera_height_m",
                      default_value="0.30",
                      description="Camera height above ground (m)")
    arg_conf    = DeclareLaunchArgument("confidence_threshold",
                      default_value="0.50",
                      description="Detection confidence threshold")
    arg_debug   = DeclareLaunchArgument("debug",
                      default_value="false",
                      description="Enable extra logging")

    model_path = LaunchConfiguration("model_path")
    cam_height = LaunchConfiguration("camera_height_m")
    conf       = LaunchConfiguration("confidence_threshold")

    # --------------- Nodes ---------------
    robot_state_pub = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        parameters=[{"robot_description": open(urdf_path).read()
                     if os.path.exists(urdf_path) else ""}],
    )

    teleop = Node(
        package="archimedes_survey",
        executable="teleop_node",
        name="teleop_node",
        output="screen",
    )

    stepper = Node(
        package="archimedes_survey",
        executable="stepper_driver",
        name="stepper_driver",
        output="screen",
    )

    auto_nav = Node(
        package="archimedes_survey",
        executable="auto_navigate",
        name="auto_navigate",
        output="screen",
    )

    burrow_det = Node(
        package="archimedes_survey",
        executable="burrow_detector",
        name="burrow_detector",
        output="screen",
        parameters=[
            {"model_path":           model_path},
            {"camera_height_m":      cam_height},
            {"confidence_threshold": conf},
            {"process_hz":           10.0},
        ],
    )

    logger = Node(
        package="archimedes_survey",
        executable="mission_logger",
        name="mission_logger",
        output="screen",
    )

    # 船底聲納（漲潮期熱區掃描）
    hull_sonar = Node(
        package="archimedes_survey",
        executable="hull_sonar_node",
        name="hull_sonar_node",
        output="screen",
        parameters=[
            {"simulate":           True},   # False 時接 NMEA 魚探串口
            {"nmea_port":          "/dev/ttyUSB1"},
            {"anomaly_threshold":  0.65},
        ],
    )

    # 手臂換能器節點（保留供未來洞內量測使用）
    ultrasound = Node(
        package="archimedes_survey",
        executable="ultrasound_node",
        name="ultrasound_node",
        output="screen",
        parameters=[
            {"simulate":     True},
            {"scan_rate_hz": 2.0},
            {"n_averages":   5},
        ],
    )

    recovery = Node(
        package="archimedes_survey",
        executable="recovery_behavior",
        name="recovery_behavior",
        output="screen",
    )

    acoustic = Node(
        package="archimedes_survey",
        executable="acoustic_processor",
        name="acoustic_processor",
        output="screen",
        parameters=[
            {"min_scan_pts":    12},
            {"recon_threshold": 0.35},
            {"auto_trigger":    True},
            {"temperature_c":   25.0},        # 台灣夏季潮間帶水溫
            {"salinity_ppt":    32.0},        # 台灣西海岸鹽度
            {"tvg_db_per_m":    60.0},        # 彰化粉砂衰減（泥50~砂102之間）
        ],
    )

    return LaunchDescription([
        arg_model,
        arg_height,
        arg_conf,
        arg_debug,
        LogInfo(msg="=== Archimedes Survey Full System Starting ==="),
        robot_state_pub,
        teleop,
        stepper,
        auto_nav,
        burrow_det,
        logger,
        hull_sonar,   # 漲潮期船底掃描
        ultrasound,   # 手臂換能器（保留）
        acoustic,
        recovery,
    ])
