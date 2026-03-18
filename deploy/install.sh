#!/usr/bin/env bash
# =============================================================================
# Archimedes Survey -- RPi4 一鍵安裝腳本
# 在 RPi4 執行：bash deploy/install.sh
# =============================================================================
set -e
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ROS_DISTRO="${ROS_DISTRO:-humble}"

echo "===== Archimedes Survey Install ====="
echo "Repo: $REPO_DIR"
echo "ROS2: $ROS_DISTRO"

# ---- 1. 系統套件 ----
sudo apt-get update -qq
sudo apt-get install -y \
  python3-pip python3-venv git \
  ros-${ROS_DISTRO}-rclpy \
  ros-${ROS_DISTRO}-robot-state-publisher \
  ros-${ROS_DISTRO}-joint-state-publisher \
  ros-${ROS_DISTRO}-joint-state-publisher-gui \
  ros-${ROS_DISTRO}-cv-bridge \
  ros-${ROS_DISTRO}-image-transport \
  ros-${ROS_DISTRO}-ros2-control \
  ros-${ROS_DISTRO}-ros2-controllers \
  python3-colcon-common-extensions \
  i2c-tools \
  python3-smbus \
  python3-rpi.gpio

# ---- 2. Python venv（Web dashboard） ----
cd "$REPO_DIR"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r control/requirements.txt
pip install -r vision/requirements.txt
deactivate

# ---- 3. Build ROS2 workspace ----
source /opt/ros/${ROS_DISTRO}/setup.bash
cd "$REPO_DIR/ros2"
colcon build --symlink-install
echo "source $REPO_DIR/ros2/install/setup.bash" >> ~/.bashrc

# ---- 4. Enable I2C / SPI / UART ----
sudo raspi-config nonint do_i2c 0
sudo raspi-config nonint do_spi 0
sudo raspi-config nonint do_serial_hw 0
sudo raspi-config nonint do_serial_cons 1   # disable login shell on serial

# ---- 5. GPIO group ----
sudo usermod -aG gpio,i2c,spi "$USER"

# ---- 6. Install systemd services ----
sudo cp "$REPO_DIR/deploy/archimedes.service"     /etc/systemd/system/
sudo cp "$REPO_DIR/deploy/archimedes-ros.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable archimedes archimedes-ros
echo "Services enabled. Reboot to auto-start."

# ---- 7. Verify I2C devices ----
echo ""
echo "=== I2C Device Check ==="
i2cdetect -y 1 || true
echo "(Expected: 0x10/0x18/0x68=BMX055, 0x40=PCA9685, 0x48=ADS1115)"

echo ""
echo "===== Install Complete ====="
echo "Next steps:"
echo "  1. sudo reboot"
echo "  2. Open browser → http://$(hostname -I | awk '{print $1}'):8080"
echo "  3. Camera calibration: python3 ros2/archimedes_survey/camera_calibrate.py --capture"
