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

# 嘗試匯入 SPI ADC（MCP3208 相容模式）
try:
    import spidev
    _SPI_AVAILABLE = True
except ImportError:
    _SPI_AVAILABLE = False

# 嘗試匯入 pyserial（pic0rick USB CDC 模式）
try:
    import serial
    import struct
    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False

# ---------------------------------------------------------------------------
# 硬體參數
# ---------------------------------------------------------------------------
GPIO_TRIGGER    = 17          # 觸發脈衝輸出腳位
GPIO_ECHO_SPI   = 0           # MCP3208 SPI CH0 作為 ADC 輸入
SPI_BUS         = 0
SPI_DEVICE      = 0
SPI_SPEED_HZ    = 1_000_000
SAMPLE_RATE_PIC0RICK_HZ = 60_000_000  # pic0rick RP2040 60 Msps
PIC0RICK_PORT   = "/dev/ttyACM0"      # USB CDC 虛擬串口（Linux RPi4）

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
class Pic0rickDriver:
    """
    pic0rick（RP2040 @ 60 Msps）USB CDC 驅動。

    pic0rick 透過 USB CDC（虛擬串口）與 RPi4 通訊：
      - 發送 ASCII 命令 → pic0rick PIO 執行高速 ADC 採集
      - 接收 binary 數列（16-bit little-endian unsigned）

    pic0rick 通訊協議（kelu124/pic0rick GitHub v1.0）：
      Host → pic0rick: "ACQ <n_samples>\\n"
      pic0rick → Host: <n_samples × 2 bytes, little-endian uint16>

      Host → pic0rick: "PING\\n"
      pic0rick → Host: "PONG\\n"

      Host → pic0rick: "RATE\\n"
      pic0rick → Host: "60000000\\n"  （採樣率，Hz）

    注意：
      - RP2040 PIO 在接收到 ACQ 命令後立即觸發內建脈衝輸出（GPIO 0）
        並同步開始高速 ADC 採集，無需另外驅動 GPIO_TRIGGER。
      - 回傳值為 10-bit ADC 計數（0~1023），對應 0~3.3V。
      - 連線失敗時自動降級為模擬模式。
    """

    BAUD     = 115200
    TIMEOUT  = 2.0   # 單次採集最大等待秒數

    def __init__(self, port: str = PIC0RICK_PORT):
        self._port    = port
        self._ser     = None
        self._ok      = False
        self._fs      = SAMPLE_RATE_PIC0RICK_HZ

        if not _SERIAL_AVAILABLE:
            print("[Pic0rick] pyserial 未安裝（pip install pyserial），無法連線。")
            return
        self._connect()

    def _connect(self):
        try:
            self._ser = serial.Serial(
                self._port, self.BAUD,
                timeout=self.TIMEOUT,
                write_timeout=1.0,
            )
            # 握手測試
            self._ser.write(b"PING\n")
            resp = self._ser.readline().decode("ascii", errors="ignore").strip()
            if resp == "PONG":
                self._ok = True
                print(f"[Pic0rick] 連線成功：{self._port} @ 60 Msps")
            else:
                print(f"[Pic0rick] 握手失敗（回應：{repr(resp)}），切換模擬模式。")
        except Exception as e:
            print(f"[Pic0rick] 連線失敗（{e}），切換模擬模式。")

    @property
    def connected(self) -> bool:
        return self._ok and self._ser is not None

    @property
    def sample_rate_hz(self) -> float:
        return float(self._fs)

    def acquire(self, n_samples: int = 2048) -> list:
        """
        觸發一次 A-scan 採集，回傳原始 ADC 整數列表（10-bit, 0~1023）。
        失敗時回傳全零列表。
        """
        if not self.connected:
            return [512] * n_samples  # 中間值佔位

        try:
            cmd = f"ACQ {n_samples}\n".encode("ascii")
            self._ser.write(cmd)
            # 等待 n_samples × 2 bytes（uint16 LE）
            raw = self._ser.read(n_samples * 2)
            if len(raw) != n_samples * 2:
                return [512] * n_samples
            return list(struct.unpack(f"<{n_samples}H", raw))
        except Exception as e:
            print(f"[Pic0rick] 採集失敗：{e}")
            self._ok = False
            return [512] * n_samples

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()


# ---------------------------------------------------------------------------
class UltrasoundTransducer:
    """硬體抽象層：真實 RPi4 或模擬模式。"""

    def __init__(self, simulate: bool = False, use_pic0rick: bool = False,
                 pic0rick_port: str = PIC0RICK_PORT):
        """
        simulate      : True = 純模擬，不連接任何硬體
        use_pic0rick  : True = 使用 pic0rick USB CDC（60 Msps）
                        False = 使用 MCP3208 SPI（1 Msps，舊方案）
        """
        self.simulate     = simulate or not _HW_AVAILABLE
        self._pi          = None
        self._spi         = None
        self._pic0rick    = None
        self._use_pic0rick = use_pic0rick and not self.simulate

        if self.simulate:
            print("[Ultrasound] 模擬模式（無硬體）")
        elif self._use_pic0rick:
            self._init_pic0rick(pic0rick_port)
        else:
            self._init_hw()

    def _init_pic0rick(self, port: str):
        """初始化 pic0rick USB CDC（60 Msps 模式）。"""
        self._pic0rick = Pic0rickDriver(port=port)
        if self._pic0rick.connected:
            print(f"[Ultrasound] pic0rick 模式（{port}, 60 Msps）")
        else:
            print("[Ultrasound] pic0rick 未連線，降級為模擬模式")
            self.simulate = True

    def _init_hw(self):
        """初始化 MCP3208 SPI 模式（舊方案，1 Msps）。"""
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
        print("[Ultrasound] MCP3208 SPI 硬體初始化完成（1 Msps）")

    def _trigger_pulse(self):
        """發射 5μs TTL 觸發脈衝。"""
        self._pi.gpio_trigger(GPIO_TRIGGER, PULSE_WIDTH_US, 1)

    def _read_adc(self, channel: int = 0, n_samples: int = 1024) -> list[int]:
        """讀取 MCP3208 ADC，回傳原始計數值列表（n_samples 點）。"""
        if self._spi is None:
            return [0] * n_samples
        cmd = [0x06 | (channel >> 2), (channel & 0x03) << 6, 0]
        results = []
        for _ in range(n_samples):
            r = self._spi.xfer2(cmd)
            val = ((r[1] & 0x0F) << 8) | r[2]
            results.append(val)
        return results

    def measure_tof(self) -> tuple[float, float]:
        """
        量測飛行時間（TOF）。僅回傳峰值，不含完整波形。

        回傳：
          tof_us   : 飛行時間 (μs)，-1 表示無回波
          confidence: 0~1，基於回波峰值強度
        """
        samples, tof_us, confidence = self.measure_waveform()
        return tof_us, confidence

    @property
    def actual_sample_rate_hz(self) -> float:
        """回傳實際使用的 ADC 採樣率。"""
        if self._use_pic0rick and self._pic0rick:
            return self._pic0rick.sample_rate_hz
        return float(SAMPLE_RATE_HZ)

    def measure_waveform(self, n_samples: int = 2000) -> tuple[list, float, float]:
        """
        完整 A-scan 波形擷取（A-core-2000 模式）。

        自動選擇 ADC 後端：
          - 模擬模式：生成合成波形
          - pic0rick 模式：USB CDC 觸發，60 Msps 採集
          - MCP3208 模式：SPI 觸發，1 Msps 採集

        回傳：
          samples   : 原始 ADC 整數列表（10-bit）
          tof_us    : 偵測到的主要回波飛行時間 (μs)，-1=無回波
          confidence: 0~1 峰值信心值
        """
        if self.simulate:
            return self._simulate_waveform(n_samples)

        # pic0rick 模式（60 Msps，USB CDC 觸發，PIO 內建脈衝）
        if self._use_pic0rick and self._pic0rick and self._pic0rick.connected:
            samples = self._pic0rick.acquire(n_samples)
            fs      = self._pic0rick.sample_rate_hz
            blank_us = MIN_DEPTH_M * 2 / SOUND_SPEED_MPS * 1e6
            tof_us, confidence = envelope_peak_tof(
                samples, fs, SOUND_SPEED_MPS, blank_us)
            return samples, tof_us, confidence

        # MCP3208 SPI 模式（舊方案，1 Msps）
        self._trigger_pulse()
        time.sleep(PULSE_WIDTH_US * 1e-6)
        samples = self._read_adc(channel=GPIO_ECHO_SPI, n_samples=n_samples)
        blank_us = MIN_DEPTH_M * 2 / SOUND_SPEED_MPS * 1e6
        tof_us, confidence = envelope_peak_tof(
            samples, SAMPLE_RATE_HZ, SOUND_SPEED_MPS, blank_us)
        return samples, tof_us, confidence

    def _simulate_waveform(self, n_samples: int = 2000) -> tuple[list, float, float]:
        """
        模擬真實 200kHz 超音波 A-scan（含表面回波、洞穴回波、衰減與雜訊）。
        對應 A-core-2000 的 JPR-300C 脈衝接收器輸出格式。
        """
        t = np.arange(n_samples) / SAMPLE_RATE_HZ  # 時間軸 (s)
        samples = np.random.normal(0, 30, n_samples)  # 背景雜訊（12bit ADC，中心 2048）
        samples += 2048  # ADC 偏置

        # 泥面表面回波（換能器距表面約 5cm）
        standoff_m = 0.05
        t_surface = 2 * standoff_m / SOUND_SPEED_MPS
        idx_s = int(t_surface * SAMPLE_RATE_HZ)
        fc = 200_000.0  # 中心頻率 200kHz
        for i in range(max(0, idx_s - 8), min(n_samples, idx_s + 9)):
            amp = 600 * math.exp(-(i - idx_s) ** 2 / 10.0)
            samples[i] += amp * math.sin(2 * math.pi * fc * i / SAMPLE_RATE_HZ)

        # 蝦猴洞穴壁回波（深度 0.5~1.2m，緩慢變化模擬不同位置）
        base_depth_m = 0.65 + 0.35 * math.sin(time.time() * 0.07)
        noise_m = np.random.normal(0, 0.008)
        depth_m = max(MIN_DEPTH_M, base_depth_m + noise_m)
        t_burrow = 2 * depth_m / SOUND_SPEED_MPS
        idx_b = int(t_burrow * SAMPLE_RATE_HZ)

        # 衰減係數 60 dB/m（台灣彰化粉砂） → 振幅 ∝ 10^(-60*d/20)
        attenuation_factor = 10 ** (-60.0 * depth_m / 20.0)
        burrow_amp = 800 * attenuation_factor  # TVG 補償前的真實衰減幅度
        for i in range(max(0, idx_b - 6), min(n_samples, idx_b + 7)):
            amp = burrow_amp * math.exp(-(i - idx_b) ** 2 / 8.0)
            samples[i] += amp * math.sin(2 * math.pi * fc * i / SAMPLE_RATE_HZ)

        # 洞穴底部二次回波（約 2倍深度處，更弱）
        idx_b2 = min(n_samples - 1, idx_b * 2)
        for i in range(max(0, idx_b2 - 4), min(n_samples, idx_b2 + 5)):
            amp = burrow_amp * 0.15 * math.exp(-(i - idx_b2) ** 2 / 6.0)
            samples[i] += amp * math.sin(2 * math.pi * fc * i / SAMPLE_RATE_HZ)

        # 確保 ADC 範圍（0~4095）
        samples = np.clip(samples, 0, 4095)

        tof_us = t_burrow * 1e6
        confidence = max(0.2, min(0.95, 0.85 - abs(noise_m) * 8))
        return samples.astype(int).tolist(), tof_us, confidence

    def _simulate_tof(self) -> tuple[float, float]:
        """舊接口相容，內部呼叫 _simulate_waveform。"""
        _, tof_us, conf = self._simulate_waveform()
        return tof_us, conf

    def close(self):
        if self._pic0rick:
            self._pic0rick.close()
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

        self.declare_parameter("simulate",       True)
        self.declare_parameter("scan_rate_hz",   2.0)
        self.declare_parameter("n_averages",     5)
        self.declare_parameter("use_pic0rick",   False)
        self.declare_parameter("pic0rick_port",  PIC0RICK_PORT)

        simulate      = self.get_parameter("simulate").value
        rate_hz       = self.get_parameter("scan_rate_hz").value
        self._n_avg   = self.get_parameter("n_averages").value
        use_pic0rick  = self.get_parameter("use_pic0rick").value
        pic0rick_port = self.get_parameter("pic0rick_port").value

        self._transducer = UltrasoundTransducer(
            simulate=simulate,
            use_pic0rick=use_pic0rick,
            pic0rick_port=pic0rick_port,
        )
        self._arm_angles = [0.0, 45.0, 90.0, 0.0]  # [j1,j2,j3,j4_tilt]

        # Pubs
        self._pub_tof        = self.create_publisher(Float32, "/ultrasound/raw_tof",      10)
        self._pub_depth      = self.create_publisher(Float32, "/ultrasound/depth_cm",     10)
        self._pub_data       = self.create_publisher(String,  "/ultrasound/burrow_data",  10)
        self._pub_scan_point = self.create_publisher(String,  "/ultrasound/scan_point",   20)
        # 完整 A-scan 波形（供 acoustic_cscan.py C-scan 重建使用）
        self._pub_waveform   = self.create_publisher(String,  "/ultrasound/raw_waveform", 10)

        # Sub
        self.create_subscription(JointState, "/arm/joint_states", self._cb_joints, 10)

        # Timer
        self.create_timer(1.0 / rate_hz, self._scan)

        if simulate:
            mode = "模擬"
        elif use_pic0rick:
            mode = f"pic0rick USB ({pic0rick_port}, 60Msps)"
        else:
            mode = "MCP3208 SPI (1Msps)"
        self.get_logger().info(f"Ultrasound Node ready ({mode}, {rate_hz} Hz)")

    def _cb_joints(self, msg: JointState):
        n2i = {n: i for i, n in enumerate(msg.name)}
        for ji, jn in enumerate(["arm_j1", "arm_j2", "arm_j3", "j4_tilt"]):
            if jn in n2i:
                idx = n2i[jn]
                if idx < len(msg.position):
                    self._arm_angles[ji] = math.degrees(msg.position[idx])

    def _scan(self):
        """執行量測（多次平均）並發布結果（含完整 A-scan 波形）。"""
        tofs = []
        confs = []
        last_waveform = None  # 保留最後一次的完整波形
        for _ in range(self._n_avg):
            samples, tof, conf = self._transducer.measure_waveform()
            if tof > 0:
                tofs.append(tof)
                confs.append(conf)
                last_waveform = samples  # 只保留有效回波的波形

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

        # 發布完整 A-scan 波形（給 acoustic_cscan.py C-scan 重建使用）
        if last_waveform is not None:
            waveform_msg = {
                "x_m":            tx,
                "y_m":            ty,
                "z_m":            tz,
                "samples":        last_waveform,
                "sample_rate_hz": self._transducer.actual_sample_rate_hz,
                "tof_us":         float(avg_tof),
                "confidence":     float(avg_conf),
                "timestamp":      info["timestamp"],
            }
            self._pub_waveform.publish(String(data=json.dumps(waveform_msg)))

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
