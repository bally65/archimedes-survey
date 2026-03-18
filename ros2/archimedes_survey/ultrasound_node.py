"""
ultrasound_node.py
==================
超音波換能器 ROS2 節點：對準洞穴開口量測深度與傾斜角。

硬體對應（v2.0）：
  換能器型號：PZT-5A Φ20mm 200kHz（CTDCO 台灣陶瓷公司）
  驅動電路  ：HV7361（±100V Active RTZ 脈衝 IC）
  接收鏈    ：AD8331 TGC VGA（-28~+92dB，程式控制）
  DAQ 平台  ：pic0rick（RP2040, 60Msps）或 MCP3208（1Msps 降規）
  觸發介面  ：GPIO 17（脈衝輸出），GPIO 27（回波輸入）
  介面模式  ：USB CDC（pic0rick）或 SPI (CE0)（MCP3208 相容模式）

訊號流程：
  GPIO 17 → 5μs TTL 脈衝 → 換能器驅動電路 → 200kHz 超音波發射
  回波 → 換能器 → MAX9814 放大 → MCP3208 ADC → SPI 讀回 → 飛行時間計算

深度計算：
  d = (TOF × c_sediment) / 2
  c_sediment ≈ 1500 m/s（海水飽和泥沙中）

Publish:
  /ultrasound/raw_tof     (Float32)  — 飛行時間 μs
  /ultrasound/depth_cm    (Float32)  — 計算深度 cm
  /ultrasound/burrow_data (String)   — JSON：深度+信心值+時間戳

Subscribe:
  /arm/joint_states       (JointState) — 用於換算換能器世界座標

Run:
  ros2 run archimedes_survey ultrasound_node
"""

import json
import math
import sys
import time
import numpy as np

try:
    from scipy.signal import hilbert as scipy_hilbert
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32, String
    from sensor_msgs.msg import JointState
except ImportError:
    sys.exit("ROS2 not found.")

# 嘗試匯入 GPIO 函式庫（RPi4 上需要）
try:
    import pigpio
    _HW_AVAILABLE = True
except ImportError:
    _HW_AVAILABLE = False

# 嘗試匯入 SPI ADC
try:
    import spidev
    _SPI_AVAILABLE = True
except ImportError:
    _SPI_AVAILABLE = False

# ---------------------------------------------------------------------------
# 硬體參數
# ---------------------------------------------------------------------------
GPIO_TRIGGER    = 17          # 觸發脈衝輸出腳位
GPIO_ECHO_SPI   = 0           # MCP3208 SPI CH0 作為 ADC 輸入
SPI_BUS         = 0
SPI_DEVICE      = 0
SPI_SPEED_HZ    = 1_000_000
SAMPLE_RATE_PIC0RICK_HZ = 60_000_000  # pic0rick RP2040 60 Msps

PULSE_WIDTH_US  = 5           # 觸發脈衝寬度 (μs)
SOUND_SPEED_MPS = 1500.0      # 泥沙中聲速 (m/s)
SAMPLE_RATE_HZ  = 1_000_000   # ADC 採樣率 (Hz)
MAX_DEPTH_M     = 2.0         # 最大量測深度 (m) → 最大 TOF = 2.67 ms
MIN_DEPTH_M     = 0.02        # 最小量測深度 20mm（防止近場反射誤判）
TOF_TIMEOUT_US  = int(MAX_DEPTH_M * 2 / SOUND_SPEED_MPS * 1e6)  # ≈ 2667 μs

# T-SAFT 基本參數（掃描時使用）
TSAFT_GRID_SPACING_MM = 2.5
TSAFT_DEPTH_CM        = 50

# ---------------------------------------------------------------------------
def envelope_peak_tof(samples: list,
                      sample_rate_hz: float = SAMPLE_RATE_HZ,
                      c_mps: float = SOUND_SPEED_MPS,
                      blank_us: float = 26.7) -> tuple:
    """
    Hilbert 轉換包絡偵測 → TOF 與信心值。
    scipy.signal.hilbert 不可用時退回簡單閾值法。

    samples       : 原始 ADC 數列
    sample_rate_hz: ADC 採樣率 Hz（MCP3208=1Msps，pic0rick=60Msps）
    blank_us      : 消隱時間 μs（忽略近場反射）
    回傳 (tof_us, confidence)，tof_us=-1 表示無有效回波
    """
    arr = np.array(samples, dtype=np.float64)
    n = len(arr)
    if n < 10:
        return -1.0, 0.0

    # 去直流
    baseline = arr[:min(50, n // 10)].mean()
    arr -= baseline

    blank_samp = max(0, int(blank_us * 1e-6 * sample_rate_hz))

    if _SCIPY_AVAILABLE:
        # Hilbert 轉換 → 解析信號包絡
        envelope = np.abs(scipy_hilbert(arr))
        envelope[:blank_samp] = 0.0
        if envelope.max() < 1e-6:
            return -1.0, 0.0
        peak_idx   = int(np.argmax(envelope))
        tof_us     = peak_idx / sample_rate_hz * 1e6
        confidence = float(min(1.0, envelope[peak_idx] / (np.percentile(envelope, 99) + 1e-12)))
    else:
        # 退回閾值法（scipy 未安裝）
        abs_arr = np.abs(arr)
        abs_arr[:blank_samp] = 0.0
        threshold = abs_arr.max() * 0.3
        indices   = np.where(abs_arr > threshold)[0]
        if len(indices) == 0:
            return -1.0, 0.0
        peak_idx   = int(indices[0])
        tof_us     = peak_idx / sample_rate_hz * 1e6
        confidence = float(min(1.0, abs_arr[peak_idx] / (abs_arr.max() + 1e-12)))

    return tof_us, confidence


# ---------------------------------------------------------------------------
class UltrasoundTransducer:
    """硬體抽象層：真實 RPi4 或模擬模式。"""

    def __init__(self, simulate: bool = False):
        self.simulate = simulate or not _HW_AVAILABLE
        self._pi = None
        self._spi = None

        if not self.simulate:
            self._init_hw()
        else:
            print("[Ultrasound] 模擬模式（無硬體）")

    def _init_hw(self):
        self._pi = pigpio.pi()
        if not self._pi.connected:
            print("[Ultrasound] pigpio 未連線，切換模擬模式")
            self.simulate = True
            return
        self._pi.set_mode(GPIO_TRIGGER, pigpio.OUTPUT)
        self._pi.write(GPIO_TRIGGER, 0)

        if _SPI_AVAILABLE:
            self._spi = spidev.SpiDev()
            self._spi.open(SPI_BUS, SPI_DEVICE)
            self._spi.max_speed_hz = SPI_SPEED_HZ
            self._spi.mode = 0
        print("[Ultrasound] 硬體初始化完成")

    def _trigger_pulse(self):
        """發射 5μs TTL 觸發脈衝。"""
        self._pi.gpio_trigger(GPIO_TRIGGER, PULSE_WIDTH_US, 1)

    def _read_adc(self, channel: int = 0) -> list[int]:
        """讀取 MCP3208 ADC，回傳原始計數值列表（1024 點）。"""
        if self._spi is None:
            return [0] * 1024
        cmd = [0x06 | (channel >> 2), (channel & 0x03) << 6, 0]
        results = []
        for _ in range(1024):
            r = self._spi.xfer2(cmd)
            val = ((r[1] & 0x0F) << 8) | r[2]
            results.append(val)
        return results

    def measure_tof(self) -> tuple[float, float]:
        """
        量測飛行時間（TOF）。

        回傳：
          tof_us   : 飛行時間 (μs)，-1 表示無回波
          confidence: 0~1，基於回波峰值強度
        """
        if self.simulate:
            return self._simulate_tof()

        self._trigger_pulse()
        time.sleep(PULSE_WIDTH_US * 1e-6)

        # 讀 ADC 採樣緩衝
        samples = self._read_adc(channel=GPIO_ECHO_SPI)

        # Hilbert 轉換包絡偵測（優先）；scipy 不可用時退回閾值法
        blank_us = MIN_DEPTH_M * 2 / SOUND_SPEED_MPS * 1e6  # 消隱時間
        return envelope_peak_tof(samples, SAMPLE_RATE_HZ, SOUND_SPEED_MPS, blank_us)

    def _simulate_tof(self) -> tuple[float, float]:
        """模擬一個典型蝦猴洞穴回波（深度 0.5~1.5m，加隨機雜訊）。"""
        base_depth_m = 0.8 + 0.4 * math.sin(time.time() * 0.1)
        noise_m      = np.random.normal(0, 0.005)
        depth_m      = max(MIN_DEPTH_M, base_depth_m + noise_m)
        tof_us       = depth_m * 2 / SOUND_SPEED_MPS * 1e6
        confidence   = max(0.3, 0.85 - abs(noise_m) * 10)
        return tof_us, confidence

    def close(self):
        if self._spi:
            self._spi.close()
        if self._pi:
            self._pi.stop()


# ---------------------------------------------------------------------------
def tof_to_depth(tof_us: float, c: float = SOUND_SPEED_MPS) -> float:
    """TOF (μs) → 深度 (cm)"""
    return tof_us * 1e-6 * c / 2.0 * 100.0


def depth_to_burrow_info(depth_cm: float, arm_angles_deg: list,
                         confidence: float) -> dict:
    """
    根據深度量測值與手臂關節角度，推算洞穴 3D 資訊。

    arm_angles_deg: [j1_yaw_deg, j2_shoulder_deg, j3_elbow_deg, j4_tilt_deg]
    j4_tilt：換能器俯仰角（MG90S，±30°），加入末端方向修正。
    """
    angles = [math.radians(a) for a in arm_angles_deg]
    j1 = angles[0]
    j2 = angles[1]
    j3 = angles[2]
    j4 = angles[3] if len(angles) > 3 else 0.0  # 換能器俯仰

    # 換能器在手臂坐標系中的朝向向量（簡化 FK）
    arm_base = np.array([0.0, 0.15, 0.282])     # 手臂底座世界坐標
    L_upper  = 0.168                             # 上臂長度 (m)
    L_fore   = 0.180                             # 前臂長度 (m)

    # 前向運動學（2D 簡化，忽略 j1 yaw 的側向）
    tip_x = arm_base[0] + math.cos(j1) * (L_upper * math.cos(j2) + L_fore * math.cos(j2 + j3))
    tip_y = arm_base[1] + math.sin(j1) * (L_upper * math.cos(j2) + L_fore * math.cos(j2 + j3))
    tip_z = arm_base[2] + L_upper * math.sin(j2) + L_fore * math.sin(j2 + j3)

    # 換能器朝向（末端效應器 Z 軸，含 j4_tilt 俯仰修正）
    dir_elev = j2 + j3 + j4  # 仰角（弧度），j4 加入換能器俯仰
    dir_azim = j1             # 方位角

    # 洞穴底部估算世界坐標
    depth_m = depth_cm / 100.0
    bottom_x = tip_x + depth_m * math.cos(dir_elev) * math.cos(dir_azim)
    bottom_y = tip_y + depth_m * math.cos(dir_elev) * math.sin(dir_azim)
    bottom_z = tip_z - depth_m * math.sin(dir_elev)

    return {
        "depth_cm":      round(depth_cm, 1),
        "confidence":    round(confidence, 3),
        "transducer_pos": [round(tip_x, 3), round(tip_y, 3), round(tip_z, 3)],
        "burrow_bottom":  [round(bottom_x, 3), round(bottom_y, 3), round(bottom_z, 3)],
        "elevation_deg":  round(math.degrees(dir_elev), 1),
        "azimuth_deg":    round(math.degrees(dir_azim), 1),
        "timestamp":      time.time(),
    }


# ---------------------------------------------------------------------------
class UltrasoundNode(Node):
    def __init__(self):
        super().__init__("ultrasound_node")

        self.declare_parameter("simulate",     True)
        self.declare_parameter("scan_rate_hz", 2.0)
        self.declare_parameter("n_averages",   5)

        simulate     = self.get_parameter("simulate").value
        rate_hz      = self.get_parameter("scan_rate_hz").value
        self._n_avg  = self.get_parameter("n_averages").value

        self._transducer = UltrasoundTransducer(simulate=simulate)
        self._arm_angles = [0.0, 45.0, 90.0, 0.0]  # [j1,j2,j3,j4_tilt]

        # Pubs
        self._pub_tof        = self.create_publisher(Float32, "/ultrasound/raw_tof",     10)
        self._pub_depth      = self.create_publisher(Float32, "/ultrasound/depth_cm",    10)
        self._pub_data       = self.create_publisher(String,  "/ultrasound/burrow_data", 10)
        self._pub_scan_point = self.create_publisher(String,  "/ultrasound/scan_point",  20)

        # Sub
        self.create_subscription(JointState, "/arm/joint_states", self._cb_joints, 10)

        # Timer
        self.create_timer(1.0 / rate_hz, self._scan)

        mode = "模擬" if simulate else "硬體"
        self.get_logger().info(f"Ultrasound Node ready ({mode} 模式, {rate_hz} Hz)")

    def _cb_joints(self, msg: JointState):
        n2i = {n: i for i, n in enumerate(msg.name)}
        for ji, jn in enumerate(["arm_j1", "arm_j2", "arm_j3", "j4_tilt"]):
            if jn in n2i:
                idx = n2i[jn]
                if idx < len(msg.position):
                    self._arm_angles[ji] = math.degrees(msg.position[idx])

    def _scan(self):
        """執行量測（多次平均）並發布結果。"""
        tofs = []
        confs = []
        for _ in range(self._n_avg):
            tof, conf = self._transducer.measure_tof()
            if tof > 0:
                tofs.append(tof)
                confs.append(conf)

        if not tofs:
            self.get_logger().debug("無有效回波")
            return

        avg_tof  = sum(tofs) / len(tofs)
        avg_conf = sum(confs) / len(confs)
        depth_cm = tof_to_depth(avg_tof)

        # 深度範圍過濾
        if depth_cm < MIN_DEPTH_M * 100 or depth_cm > MAX_DEPTH_M * 100:
            return

        self._pub_tof.publish(Float32(data=float(avg_tof)))
        self._pub_depth.publish(Float32(data=float(depth_cm)))

        info = depth_to_burrow_info(depth_cm, self._arm_angles, avg_conf)
        self._pub_data.publish(String(data=json.dumps(info)))

        # 發布給 acoustic_processor 的掃描點（帶換能器 3D 世界座標）
        tx, ty, tz = info["transducer_pos"]
        scan_pt = {
            "x_m":        tx,
            "y_m":        ty,
            "z_m":        tz,
            "tof_us":     float(avg_tof),
            "confidence": float(avg_conf),
            "timestamp":  info["timestamp"],
        }
        self._pub_scan_point.publish(String(data=json.dumps(scan_pt)))

        self.get_logger().info(
            f"深度 {depth_cm:.1f} cm  信心 {avg_conf:.2f}  "
            f"方位 {info['azimuth_deg']:.0f}°  仰角 {info['elevation_deg']:.0f}°")


# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = UltrasoundNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Ultrasound node shutting down.")
    finally:
        node._transducer.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
