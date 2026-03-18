from setuptools import find_packages, setup
import os
from glob import glob

package_name = "archimedes_survey"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        # URDF / xacro
        (f"share/{package_name}/urdf",
         glob("urdf/*.xacro") + glob("urdf/*.urdf")),
        # Config
        (f"share/{package_name}/config",
         glob("config/*.yaml")),
        # Launch files
        (f"share/{package_name}/launch",
         glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="bally65",
    maintainer_email="bally65@example.com",
    description="Archimedes dual-screw survey robot ROS2 package",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # Vision
            "burrow_detector    = archimedes_survey.detect_burrow_node:main",
            # Control
            "teleop_node        = archimedes_survey.teleop_node:main",
            "auto_navigate      = archimedes_survey.auto_navigate:main",
            "stepper_driver     = archimedes_survey.stepper_driver_node:main",
            "camera_calibrate   = archimedes_survey.camera_calibrate:main",
            "mission_logger     = archimedes_survey.mission_logger:main",
            # Deep RL agent
            "rl_agent           = archimedes_survey.rl_agent_node:main",
            # Sonar + Acoustic
            "hull_sonar_node    = archimedes_survey.hull_sonar_node:main",
            "ultrasound_node    = archimedes_survey.ultrasound_node:main",
            "acoustic_processor = archimedes_survey.acoustic_processor:main",
            # Recovery
            "recovery_behavior  = archimedes_survey.recovery_behavior:main",
        ],
    },
)
