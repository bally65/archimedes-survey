"""
acoustic_processor.py
=====================
T-SAFT 聲學後處理 ROS2 節點。

接收 ultrasound_node 發布的逐點掃描資料，執行 DAS（Delay-and-Sum）
合成孔徑聚焦 + Coherence Factor 加權，輸出 3D 洞穴深度圖。

資料流：
  /ultrasound/scan_point  (String JSON) ← 每次量測發布一點
  /ultrasound/3d_map      (String JSON) → 重建後的 3D 洞穴截面
  /ultrasound/burrow_3d   (String JSON) → 洞穴邊界摘要 + GeoJSON 欄位
  /ultrasound/pointcloud  (String JSON) → PointCloud（x,y,z,intensity）

演算法（DAS Backprojection + CF）：
  對每個候選體素 P(x,y,z)：
    ① 計算各量測位置 S_k 到 P 的預期距離 r_exp = |P - S_k|
    ② 計算高斯權重 w_k = exp(-((r_exp - r_meas_k)²)/(2σ²))
    ③ DAS: I(P) = Σ w_k
    ④ CF:  coherence_factor = (Σw_k)² / (N × Σw_k²)
    ⑤ CF 加權強度：I_cf(P) = I(P) × CF

掃描策略（配合 arm_impedance_controller 或手動 XY 步進）：
  手臂沿 X 方向掃描，步進 5~8mm（滿足空間 Nyquist λ/2 = 3.75mm × 安全係數）
  最少 12 點觸發重建，建議 24~48 點

聲速模型：
  c = 1449 + 4.6*T - 0.055*T² + 0.00029*T³ + (1.39 - 0.012*T)*(S-35) + 0.017*D
  預設：T=20°C, S=35ppt, D=0m → c ≈ 1521 m/s（或使用環境感測修正）

Run:
  ros2 run archimedes_survey acoustic_processor
"""

import json
import math
import sys
import time
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

try:
    from scipy.signal import hilbert as scipy_hilbert
    _SCIPY = True
except ImportError:
    _SCIPY = False

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String, Float32
except ImportError:
    sys.exit("ROS2 not found. Source /opt/ros/<distro>/setup.bash first.")

# ─────────────────────────────────────────────────────────────────────────────
# 物理常數與預設參數
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_SOUND_SPEED_MS = 1531.0   # m/s（台灣彰化近海，T=25°C, S=32ppt, Medwin 1975）
MIN_DEPTH_M  = 0.02               # 最小有效深度 2cm
MAX_DEPTH_M  = 2.00               # 最大量測深度 200cm
SIGMA_M      = 0.005              # 高斯核半寬 5mm（對應分辨率）
MIN_SCAN_PTS = 12                 # 觸發重建最少點數
MAX_BUFFER   = 256                # 掃描緩衝區上限

# TVG 衰減係數（文獻來源：PMC11293796, PMC11531588）
TVG_MUD_DB_PER_M  = 50.0         # 泥質沉積物（鹿港蝦猴棲地）
TVG_SAND_DB_PER_M = 102.0        # 細砂沉積物
DEFAULT_TVG_DB_PER_M = 60.0      # 彰化粉砂（泥砂混合，保守估計）

# 最佳入射角（IEEE 7890758: 30° off-normal 最佳偵測空腔）
OPTIMAL_INCIDENT_DEG = 30.0      # J4_TILT 目標角度（transducer 前傾）

# 重建體素網格（局部座標，相對於掃描中心）
GRID_X_MM  = np.arange(-120, 121, 5, dtype=np.float64)  # ±12cm，5mm 步進
GRID_Y_MM  = np.arange(-120, 121, 5, dtype=np.float64)
GRID_Z_MM  = np.arange(20,  2001, 5, dtype=np.float64)  # 2cm ~ 200cm


# ─────────────────────────────────────────────────────────────────────────────
# 聲速計算（Medwin 簡化公式）
# ─────────────────────────────────────────────────────────────────────────────
def sound_speed(temp_c: float = 20.0,
                salinity_ppt: float = 35.0,
                depth_m: float = 0.0) -> float:
    """回傳海水中聲速 m/s（Medwin, 1975）。"""
    c = (1449.0
         + 4.6 * temp_c
         - 0.055 * temp_c ** 2
         + 0.00029 * temp_c ** 3
         + (1.39 - 0.012 * temp_c) * (salinity_ppt - 35.0)
         + 0.017 * depth_m)
    return max(1400.0, min(1600.0, c))


# ─────────────────────────────────────────────────────────────────────────────
# T-SAFT DAS 核心重建
# ─────────────────────────────────────────────────────────────────────────────
class TSAFTReconstructor:
    """
    Time-domain SAFT（T-SAFT）重建器。
    使用 DAS Backprojection + Coherence Factor。
    """

    def __init__(self,
                 c: float = DEFAULT_SOUND_SPEED_MS,
                 sigma_m: float = SIGMA_M):
        self.c = c
        self.sigma2 = sigma_m ** 2

    def reconstruct(self,
                    scan_pts: List[dict]
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        輸入：
          scan_pts — list of dict:
            {x_m, y_m, z_m, r_meas_m, confidence}
            (x,y,z) = 量測時換能器位置（世界座標, m）
            r_meas_m = tof × c / 2（單程距離, m）

        輸出：
          X_mm, Y_mm, Z_mm, intensity_cf — 3D 體素陣列（mm 單位座標 + CF 強度）
        """
        pts = scan_pts
        N   = len(pts)
        sx  = np.array([p["x_m"] for p in pts])
        sy  = np.array([p["y_m"] for p in pts])
        sz  = np.array([p["z_m"] for p in pts])
        rm  = np.array([p["r_meas_m"] for p in pts])
        cf_weight = np.array([p.get("confidence", 1.0) for p in pts])

        # 建立稀疏重建：僅在量測點 ±5 倍 sigma 範圍內計算
        # 體素格點（世界座標, m）
        gx = GRID_X_MM * 1e-3 + np.mean(sx)
        gy = GRID_Y_MM * 1e-3 + np.mean(sy)
        gz = GRID_Z_MM * 1e-3  # Z = 深度方向

        # 使用廣播計算，避免三重 for 迴圈
        # shape: (Ngx, Ngy, Ngz, N)
        GX = gx[:, None, None, None]
        GY = gy[None, :, None, None]
        GZ = gz[None, None, :, None]

        r_exp = np.sqrt((GX - sx[None, None, None, :]) ** 2
                       + (GY - sy[None, None, None, :]) ** 2
                       + (GZ - sz[None, None, None, :]) ** 2)  # (Nx,Ny,Nz,N)

        # 高斯核權重
        delta_r = r_exp - rm[None, None, None, :]
        w = np.exp(-delta_r ** 2 / (2 * self.sigma2)) * cf_weight[None, None, None, :]

        # DAS 強度
        I_das = np.sum(w, axis=-1)  # (Nx, Ny, Nz)

        # Coherence Factor
        sum_w  = np.sum(w,      axis=-1)
        sum_w2 = np.sum(w ** 2, axis=-1)
        eps    = 1e-12
        CF = sum_w ** 2 / (N * sum_w2 + eps)  # (Nx, Ny, Nz)

        I_cf = I_das * CF

        return gx * 1e3, gy * 1e3, gz * 1e3, I_cf  # mm 單位

    def extract_burrow_boundary(self,
                                gx_mm: np.ndarray,
                                gy_mm: np.ndarray,
                                gz_mm: np.ndarray,
                                intensity: np.ndarray,
                                threshold: float = 0.35
                                ) -> List[dict]:
        """
        從 CF 強度圖提取洞穴邊界點。
        threshold：相對最大強度的閾值比例。
        回傳：[{x_mm, y_mm, z_mm, intensity}...]
        """
        i_max = intensity.max()
        if i_max < 1e-10:
            return []

        mask = (intensity / i_max) >= threshold
        ix, iy, iz = np.where(mask)

        boundary = []
        for i, j, k in zip(ix, iy, iz):
            boundary.append({
                "x_mm": float(gx_mm[i]),
                "y_mm": float(gy_mm[j]),
                "z_mm": float(gz_mm[k]),
                "intensity": float(intensity[i, j, k] / i_max),
            })

        # 按強度排序，取前 500 點（避免 JSON 過大）
        boundary.sort(key=lambda p: -p["intensity"])
        return boundary[:500]

    def estimate_burrow_depth(self,
                              boundary: List[dict]
                              ) -> Tuple[float, float, float]:
        """
        從邊界點估算洞穴深度（最大 Z）、開口直徑（XY 展開寬度）、信心值。
        回傳：(depth_cm, diameter_mm, confidence)
        """
        if not boundary:
            return 0.0, 0.0, 0.0

        z_vals = [p["z_mm"] for p in boundary]
        x_vals = [p["x_mm"] for p in boundary]
        y_vals = [p["y_mm"] for p in boundary]

        max_depth_mm  = max(z_vals)
        span_x_mm     = max(x_vals) - min(x_vals)
        span_y_mm     = max(y_vals) - min(y_vals)
        diameter_mm   = math.sqrt(span_x_mm ** 2 + span_y_mm ** 2)
        mean_intensity = sum(p["intensity"] for p in boundary) / len(boundary)

        return max_depth_mm / 10.0, diameter_mm, float(mean_intensity)


# ─────────────────────────────────────────────────────────────────────────────
# DMOMP 頻散補償器
# ─────────────────────────────────────────────────────────────────────────────
class DMOMPCompensator:
    """
    DMOMP（Dispersion-compensated Multi-Mode Orthogonal Matching Pursuit）
    頻散補償，適配泥質沉積物中 200kHz 脈衝回波超音波。

    背景：
      超音波在非均質水合孔隙沉積物（彰化粉砂泥）中傳播時，
      會發生頻率依賴的相速度偏移（Biot 理論），
      導致時域 A-scan 波形隨距離逐漸展寬，降低 T-SAFT 定位精度。

    方法（對應 MDPI Sensors 23(21):8683, 2023 的頻散補償映射）：
      1. FFT 將時域訊號轉至頻域
      2. 計算每個頻率 f 的實際相速度 c(f)：
           c(f) = c0 + β × (f - f0) / f0
           β = 頻散係數（m/s），泥質沉積物典型值 5~20 m/s
      3. 計算相位修正量 Δφ(f)，補回頻散引起的相位累積差
      4. IFFT → 頻散補償後的 A-scan
      5. 重新 Hilbert 包絡偵測 → 更精準的 TOF

    精度（依論文）：定位正確率 ≥ 98.7%
    適用條件：需要 scipy；ADC 採樣率 ≥ 1 Msps
    """

    def __init__(self,
                 c0_mps: float = DEFAULT_SOUND_SPEED_MS,
                 f0_hz: float = 200_000.0,
                 dispersion_beta: float = 12.0,
                 sample_rate_hz: float = 1_000_000.0):
        """
        c0_mps          : 參考聲速 m/s（Medwin 修正後，預設 1531）
        f0_hz           : 換能器中心頻率 Hz（200kHz）
        dispersion_beta : 頻散係數 m/s（泥質 5~20，彰化粉砂保守取 12）
        sample_rate_hz  : ADC 採樣率 Hz（MCP3208=1M，pic0rick=60M）
        """
        self.c0   = c0_mps
        self.f0   = f0_hz
        self.beta = dispersion_beta
        self.fs   = sample_rate_hz

    def compensate(self,
                   rf_data: np.ndarray,
                   initial_tof_us: float
                   ) -> Tuple[np.ndarray, float]:
        """
        對 A-scan 波形執行頻散補償。

        rf_data        : 原始 ADC 數列（float64，已去直流）
        initial_tof_us : 初始 TOF 估計（μs），由簡單包絡峰值取得，
                         用於計算參考深度（確定補償中心點）

        回傳：
          (compensated_rf : np.ndarray float64,
           corrected_tof_us : float)
        """
        if not _SCIPY:
            return rf_data, initial_tof_us  # scipy 不可用時透通

        N = len(rf_data)
        if N < 64:
            return rf_data, initial_tof_us

        # Step 1: FFT
        F = np.fft.rfft(rf_data)
        freqs_hz = np.fft.rfftfreq(N, d=1.0 / self.fs)

        # Step 2: 線性頻散模型 c(f) = c0 + β*(f-f0)/f0
        c_f = self.c0 + self.beta * (freqs_hz - self.f0) / max(self.f0, 1.0)
        c_f = np.clip(c_f, self.c0 * 0.80, self.c0 * 1.20)

        # Step 3: 計算往返相位修正量
        # 參考深度（初始 TOF → 單程距離）
        d_ref = initial_tof_us * 1e-6 * self.c0 / 2.0  # m
        # 各頻率的相位誤差（實際 vs 理想無頻散）
        omega = 2.0 * np.pi * freqs_hz
        # Δφ = ω × 2d × (1/c0 - 1/c(f))  [往返路徑]
        delta_phi = omega * 2.0 * d_ref * (1.0 / self.c0 - 1.0 / c_f)
        # 避免 DC/0Hz 分量被扭曲
        delta_phi[0] = 0.0

        # Step 4: 乘以補償相位因子 → IFFT
        F_corr    = F * np.exp(1j * delta_phi)
        rf_corr   = np.fft.irfft(F_corr, n=N).astype(np.float64)

        # Step 5: 重新 Hilbert 包絡偵測 → 修正後 TOF
        envelope = np.abs(scipy_hilbert(rf_corr))
        # 保留初始 TOF 的消隱範圍（避免脈衝殘影誤判）
        blank_samp = max(0, int(initial_tof_us * 0.4 * 1e-6 * self.fs))
        envelope[:blank_samp] = 0.0

        if envelope.max() > 1e-8:
            peak_idx         = int(np.argmax(envelope))
            corrected_tof_us = float(peak_idx / self.fs * 1e6)
        else:
            corrected_tof_us = initial_tof_us

        return rf_corr, corrected_tof_us

    def correct_scan_point(self, pt: dict, raw_samples: list) -> dict:
        """
        從完整 A-scan 樣本修正單個掃描點的 TOF 與 r_meas_m。

        pt          : 原始掃描點 dict（含 tof_us, r_meas_m）
        raw_samples : 對應的原始 ADC 數列（int list）

        回傳：修正後的掃描點 dict（in-place 修改並回傳）
        """
        if not raw_samples:
            return pt

        arr = np.array(raw_samples, dtype=np.float64)
        baseline = arr[:min(50, max(1, len(arr) // 10))].mean()
        arr -= baseline

        rf_corr, corrected_tof = self.compensate(arr, pt.get("tof_us", 0.0))
        if corrected_tof > 0:
            pt["tof_us"]    = corrected_tof
            pt["r_meas_m"]  = corrected_tof * 1e-6 * self.c0 / 2.0
            pt["dmomp_corrected"] = True
        return pt


# ─────────────────────────────────────────────────────────────────────────────
# ROS2 節點
# ─────────────────────────────────────────────────────────────────────────────
class AcousticProcessorNode(Node):
    """
    T-SAFT 聲學後處理節點。

    參數（ros2 param）：
      sound_speed        : float  聲速 m/s（預設 1521.0）
      sigma_m            : float  高斯核 sigma m（預設 0.005）
      min_scan_pts       : int    觸發重建最少點數（預設 12）
      recon_threshold    : float  邊界提取閾值 0~1（預設 0.35）
      auto_trigger       : bool   自動觸發（True）或等待 /scan/trigger（False）
      temperature_c      : float  水溫 °C（用於聲速修正，預設 20.0）
      salinity_ppt       : float  鹽度 ppt（預設 35.0）
      dmomp_enabled      : bool   啟用 DMOMP 頻散補償（預設 False）
      dmomp_beta         : float  頻散係數 m/s（預設 12.0，泥質 5~20）
      dmomp_sample_rate  : float  ADC 採樣率（pic0rick=60e6，MCP3208=1e6）
    """

    def __init__(self):
        super().__init__("acoustic_processor")

        # ── 參數 ─────────────────────────────────────────────────────────────
        self.declare_parameter("sound_speed",       DEFAULT_SOUND_SPEED_MS)
        self.declare_parameter("sigma_m",           SIGMA_M)
        self.declare_parameter("min_scan_pts",      MIN_SCAN_PTS)
        self.declare_parameter("recon_threshold",   0.35)
        self.declare_parameter("auto_trigger",      True)
        self.declare_parameter("temperature_c",     25.0)
        self.declare_parameter("salinity_ppt",      32.0)
        self.declare_parameter("tvg_db_per_m",      DEFAULT_TVG_DB_PER_M)
        self.declare_parameter("dmomp_enabled",     False)
        self.declare_parameter("dmomp_beta",        12.0)
        self.declare_parameter("dmomp_sample_rate", 1_000_000.0)

        temp    = self.get_parameter("temperature_c").value
        sal     = self.get_parameter("salinity_ppt").value
        c       = sound_speed(temp, sal)
        sigma   = self.get_parameter("sigma_m").value

        self._min_pts   = int(self.get_parameter("min_scan_pts").value)
        self._thresh    = float(self.get_parameter("recon_threshold").value)
        self._auto      = bool(self.get_parameter("auto_trigger").value)
        self._tvg_alpha = float(self.get_parameter("tvg_db_per_m").value)

        # DMOMP 頻散補償
        self._dmomp_enabled = bool(self.get_parameter("dmomp_enabled").value)
        if self._dmomp_enabled:
            self._dmomp = DMOMPCompensator(
                c0_mps=c,
                f0_hz=200_000.0,
                dispersion_beta=float(self.get_parameter("dmomp_beta").value),
                sample_rate_hz=float(self.get_parameter("dmomp_sample_rate").value),
            )
        else:
            self._dmomp = None

        # 用於 DMOMP：暫存最近一筆 raw_waveform（key=座標 tuple，值=samples list）
        self._waveform_cache: dict = {}

        self._recon = TSAFTReconstructor(c=c, sigma_m=sigma)
        self._buf: deque = deque(maxlen=MAX_BUFFER)
        self._scan_count = 0

        self.get_logger().info(
            f"AcousticProcessor init: c={c:.1f} m/s, sigma={sigma*1000:.1f}mm, "
            f"TVG={self._tvg_alpha:.0f}dB/m, min_pts={self._min_pts}, "
            f"DMOMP={'ON β='+str(self.get_parameter('dmomp_beta').value) if self._dmomp_enabled else 'OFF'}"
        )

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            String, "/ultrasound/scan_point",  self._cb_scan_point,  20)
        self.create_subscription(
            String, "/scan/trigger",           self._cb_trigger,      5)
        # DMOMP：訂閱完整 A-scan 波形，在 scan_point 到達前預存波形
        self.create_subscription(
            String, "/ultrasound/raw_waveform", self._cb_raw_waveform, 20)

        # ── Publishers ────────────────────────────────────────────────────────
        self._pub_map    = self.create_publisher(String, "/ultrasound/3d_map",    5)
        self._pub_burrow = self.create_publisher(String, "/ultrasound/burrow_3d", 5)
        self._pub_pc     = self.create_publisher(String, "/ultrasound/pointcloud", 5)
        self._pub_status = self.create_publisher(String, "/acoustic/status",       10)

        # ── 診斷定時器 ────────────────────────────────────────────────────────
        self.create_timer(5.0, self._pub_heartbeat)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _cb_raw_waveform(self, msg: String):
        """
        暫存 A-scan 原始波形（供 DMOMP 使用）。
        以換能器位置（x,y,z 四捨五入 1mm）為 key，保留最近 64 筆。
        """
        if not self._dmomp_enabled:
            return
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        samples = data.get("samples", [])
        if not samples:
            return
        # 以位置 tuple 為索引鍵（1mm 精度）
        key = (round(data.get("x_m", 0.0), 3),
               round(data.get("y_m", 0.0), 3),
               round(data.get("z_m", 0.0), 3))
        self._waveform_cache[key] = samples
        # 保留最新 64 筆，避免記憶體膨脹
        if len(self._waveform_cache) > 64:
            oldest = next(iter(self._waveform_cache))
            del self._waveform_cache[oldest]

    def _cb_scan_point(self, msg: String):
        """接收單點掃描資料並加入緩衝區。"""
        try:
            pt = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn("Invalid JSON in /ultrasound/scan_point")
            return

        # 必要欄位檢查
        for key in ("x_m", "y_m", "z_m", "tof_us", "confidence"):
            if key not in pt:
                self.get_logger().warn(f"scan_point missing field: {key}")
                return

        # 換算單程距離
        c = self._recon.c
        pt["r_meas_m"] = pt["tof_us"] * 1e-6 * c / 2.0

        # 深度範圍過濾
        if pt["r_meas_m"] < MIN_DEPTH_M or pt["r_meas_m"] > MAX_DEPTH_M:
            return

        # TVG（時間增益補償）：補償沉積物衰減
        # TVG_linear = 10^(alpha_dB/m * r_m / 20)  [振幅補償]
        # 將 TVG 因子乘入 confidence，使深層反射等權參與 DAS
        if self._tvg_alpha > 0.0:
            tvg_factor = 10.0 ** (self._tvg_alpha * pt["r_meas_m"] / 20.0)
            # 限制 TVG 上限，避免雜訊放大過度（最多 40dB = ×100）
            tvg_factor = min(tvg_factor, 100.0)
            pt["confidence"] = pt.get("confidence", 1.0) * tvg_factor

        # DMOMP 頻散補償（若啟用且有對應波形）
        if self._dmomp_enabled and self._dmomp is not None:
            key = (round(pt["x_m"], 3),
                   round(pt["y_m"], 3),
                   round(pt["z_m"], 3))
            raw = self._waveform_cache.get(key, [])
            if raw:
                pt = self._dmomp.correct_scan_point(pt, raw)
                # 修正後重新換算距離（TVG 已在上面套用，但 r_meas_m 需重算）
                pt["r_meas_m"] = pt["tof_us"] * 1e-6 * self._recon.c / 2.0

        self._buf.append(pt)
        self._scan_count += 1

        self.get_logger().debug(
            f"scan_point #{self._scan_count}: r={pt['r_meas_m']*100:.1f}cm "
            f"@ ({pt['x_m']*100:.1f}, {pt['y_m']*100:.1f}, {pt['z_m']*100:.1f}) cm")

        # 自動觸發重建
        if self._auto and len(self._buf) >= self._min_pts:
            self._run_reconstruction()

    def _cb_trigger(self, msg: String):
        """手動觸發重建（發布到 /scan/trigger 任意 JSON）。"""
        self.get_logger().info("Manual reconstruction trigger received.")
        self._run_reconstruction()

    # ── 重建主流程 ────────────────────────────────────────────────────────────
    def _run_reconstruction(self):
        pts = list(self._buf)
        N   = len(pts)
        if N < self._min_pts:
            self.get_logger().warn(
                f"Not enough points ({N}/{self._min_pts}), skipping reconstruction.")
            return

        self.get_logger().info(
            f"Starting T-SAFT reconstruction with {N} scan points...")
        t0 = time.monotonic()

        try:
            gx, gy, gz, intensity = self._recon.reconstruct(pts)
        except Exception as e:
            self.get_logger().error(f"Reconstruction failed: {e}")
            return

        dt = time.monotonic() - t0
        self.get_logger().info(f"Reconstruction done in {dt:.2f}s")

        # 邊界提取
        boundary = self._recon.extract_burrow_boundary(
            gx, gy, gz, intensity, self._thresh)
        depth_cm, diam_mm, conf = self._recon.estimate_burrow_depth(boundary)

        # ── 發布 3D Map（稀疏格點，僅強度 > 0.1 的點）─────────────────────
        i_max = intensity.max()
        if i_max > 1e-10:
            mask_idx = np.argwhere(intensity / i_max > 0.10)
            pc_pts   = []
            for idx in mask_idx[:2000]:  # 最多 2000 點
                ix, iy, iz = idx
                pc_pts.append({
                    "x_mm": float(gx[ix]),
                    "y_mm": float(gy[iy]),
                    "z_mm": float(gz[iz]),
                    "intensity": float(intensity[ix, iy, iz] / i_max),
                })
            self._pub_pc.publish(String(data=json.dumps({
                "header": {"stamp": time.time(), "n_pts": len(pc_pts)},
                "points": pc_pts,
            })))

        # ── 發布洞穴摘要 ────────────────────────────────────────────────────
        burrow_msg = {
            "timestamp": time.time(),
            "n_scan_pts": N,
            "recon_time_s": round(dt, 3),
            "burrow_depth_m": round(depth_cm / 100.0, 4),
            "burrow_depth_cm": round(depth_cm, 2),
            "burrow_diameter_mm": round(diam_mm, 1),
            "ultrasound_confidence": round(conf, 3),
            "boundary_pts": len(boundary),
            # GeoJSON 擴充欄位（與 mission_logger.py 介接）
            "geojson_extra": {
                "burrow_depth_m":   round(depth_cm / 100.0, 4),
                "burrow_angle_deg": self._estimate_angle(boundary),
                "ultrasound_confidence": round(conf, 3),
            },
        }
        self._pub_burrow.publish(String(data=json.dumps(burrow_msg)))

        self.get_logger().info(
            f"Burrow: depth={depth_cm:.1f}cm, "
            f"diam={diam_mm:.0f}mm, conf={conf:.2f}")

        # ── 發布完整 3D Map（前 50 邊界點摘要）──────────────────────────────
        map_msg = {
            "timestamp": time.time(),
            "sound_speed_ms": self._recon.c,
            "sigma_mm": self._recon.sigma2 ** 0.5 * 1000,
            "grid_shape": [len(gx), len(gy), len(gz)],
            "boundary_sample": boundary[:50],
        }
        self._pub_map.publish(String(data=json.dumps(map_msg)))

        # 重建完成後清空緩衝區，準備下一次掃描
        self._buf.clear()
        self._scan_count = 0

    # ── 輔助 ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _estimate_angle(boundary: List[dict]) -> float:
        """
        估算洞穴傾斜角（從入口到最深點的仰角, deg）。
        0° = 垂直向下，90° = 水平。
        """
        if len(boundary) < 2:
            return 0.0
        # 最深點
        deepest = max(boundary, key=lambda p: p["z_mm"])
        # 入口點（最淺）
        shallowest = min(boundary, key=lambda p: p["z_mm"])
        dz = deepest["z_mm"] - shallowest["z_mm"]
        dr = math.sqrt((deepest["x_mm"] - shallowest["x_mm"]) ** 2
                      + (deepest["y_mm"] - shallowest["y_mm"]) ** 2)
        if dz < 1e-3:
            return 90.0
        return round(math.degrees(math.atan2(dr, dz)), 1)

    def _pub_heartbeat(self):
        status = {
            "node": "acoustic_processor",
            "buf_len": len(self._buf),
            "total_pts": self._scan_count,
            "sound_speed_ms": self._recon.c,
            "ready": len(self._buf) >= self._min_pts,
        }
        self._pub_status.publish(String(data=json.dumps(status)))


# ─────────────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = AcousticProcessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Acoustic processor shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
