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
  - hull_sonar_node       (漲潮期船底聲納掃描)
  - ultrasound_node       (手臂換能器 A-scan 採集)
  - acoustic_processor    (T-SAFT 精確定位)
  - acoustic_cscan        (A-core-2000 C-scan 三維重建)  ← NEW
  - burrow_cnn            (3D-CNN 洞穴偵測 + 計數)      ← NEW
  - recovery_behavior     (自主恢復)

Run:
    ros2 launch archimedes_survey full_system.launch.py
Optional args:
    model_path:=/home/pi/burrow_best.pt
    camera_height_m:=0.28
    cnn_weights:=/home/pi/burrow_cnn.onnx
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
    arg_cnn     = DeclareLaunchArgument("cnn_weights",
                      default_value="",
                      description="Path to burrow_cnn ONNX weights (empty = rule-based fallback)")
    arg_cnn_thr = DeclareLaunchArgument("cnn_threshold",
                      default_value="0.50",
                      description="3D-CNN burrow presence threshold")

    model_path  = LaunchConfiguration("model_path")
    cam_height  = LaunchConfiguration("camera_height_m")
    conf        = LaunchConfiguration("confidence_threshold")
    cnn_weights = LaunchConfiguration("cnn_weights")
    cnn_thr     = LaunchConfiguration("cnn_threshold")

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

    # A-core-2000 C-scan 三維重建節點
    # 訂閱 /ultrasound/raw_waveform → 發布 /acoustic/cscan_volume
    cscan = Node(
        package="archimedes_survey",
        executable="acoustic_cscan",
        name="acoustic_cscan",
        output="screen",
        parameters=[
            {"min_ascans":       9},      # 3×3 最小掃描點數，觸發重建
            {"attenuation_db_m": 60.0},   # 彰化粉砂 TVG 衰減係數
            {"xy_half_m":        0.25},   # ±25cm 掃描範圍
            {"z_max_m":          0.80},   # 最大穿透深度 80cm（200kHz）
            {"auto_trigger":     True},
        ],
    )

    # 3D-CNN 洞穴偵測 + 計數節點
    # 訂閱 /acoustic/cscan_volume → 發布 /burrow/detection
    burrow_cnn_node = Node(
        package="archimedes_survey",
        executable="burrow_cnn",
        name="burrow_cnn",
        output="screen",
        parameters=[
            {"weights_path": cnn_weights},  # 空字串時自動降級為規則推論
            {"threshold":    cnn_thr},
        ],
    )

    return LaunchDescription([
        arg_model,
        arg_height,
        arg_conf,
        arg_debug,
        arg_cnn,
        arg_cnn_thr,
        LogInfo(msg="=== Archimedes Survey Full System Starting ==="),
        robot_state_pub,
        teleop,
        stepper,
        auto_nav,
        burrow_det,
        logger,
        hull_sonar,        # 漲潮期船底掃描
        ultrasound,        # 手臂換能器 A-scan 採集
        acoustic,          # T-SAFT 精確定位（原有）
        cscan,             # C-scan 三維重建（A-core-2000 風格）
        burrow_cnn_node,   # 3D-CNN 偵測 + 計數
        recovery,
    ])
