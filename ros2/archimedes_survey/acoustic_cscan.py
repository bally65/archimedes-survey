"""
acoustic_cscan.py
=================
A-core-2000 風格 C-scan 切片重建與三維體積合成。

根據論文：Mizuno et al. (2022) "Deep-sea infauna with calcified exoskeletons
imaged in situ using a new 3D acoustic coring system (A-core-2000)"

演算法流程：
  ① 接收多個位置的完整 A-scan 波形（來自 /ultrasound/raw_waveform）
  ② TVG 衰減補償（50~102 dB/m，依底質選擇）
  ③ Hilbert 轉換萃取包絡線
  ④ 在各深度 z 切片，從每個掃描點取出對應的包絡振幅
  ⑤ 插值到規則 XY 格點 → C-mode 切片
  ⑥ 沿 Z 方向堆疊所有切片 + Alpha-blending → 三維聲學體積
  ⑦ 發布體積 JSON 給 burrow_cnn.py 做 3D-CNN 偵測

與現有 T-SAFT (acoustic_processor.py) 的差異：
  - T-SAFT：逐點 DAS 反投影，適合單一特征深度精確定位
  - C-scan：逐深度橫截面成像，產生全體積，適合 CNN 特征萃取

訂閱：
  /ultrasound/raw_waveform (String JSON)  ← ultrasound_node 發布

發布：
  /acoustic/cscan_volume   (String JSON)  → burrow_cnn 節點消費
  /acoustic/cscan_slice    (String JSON)  → RViz2 即時顯示單切片
  /acoustic/status         (String JSON)  → 狀態心跳

Run:
  ros2 run archimedes_survey acoustic_cscan
"""

import json
import sys
import time
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.signal import hilbert as scipy_hilbert

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    _ROS2_AVAILABLE = True
except ImportError:
    _ROS2_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# 物理常數（對應 A-core-2000 設計參數）
# ─────────────────────────────────────────────────────────────────────────────
SOUND_SPEED_MPS    = 1500.0    # m/s（飽和泥沙中，論文用值）
SAMPLE_RATE_HZ     = 1_000_000 # 1 Msps（MCP3208，論文為 10 MHz）
BYTES_PER_SAMPLE   = 2         # 10bit ADC

# TVG 衰減係數（論文實測）
TVG_MUD_DB_M    = 50.0         # 泥質 50 dB/m
TVG_SAND_DB_M   = 102.0        # 細砂 102 dB/m
TVG_DEFAULT_DB_M = 60.0        # 彰化粉砂（保守）

# 掃描體積設定（適應 Archimedes 船隻精度）
GRID_XY_SPACING_M   = 0.020   # 2cm 格點間距（A-core-2000 用 1mm）
GRID_XY_HALF_M      = 0.25    # ±25cm 掃描範圍（共 50cm × 50cm）
GRID_Z_MIN_M        = 0.03    # 最小深度 3cm（消隱帶）
GRID_Z_MAX_M        = 0.80    # 最大深度 80cm（200kHz 有效穿透）
GRID_Z_SPACING_M    = SOUND_SPEED_MPS / (2 * SAMPLE_RATE_HZ)  # ≈0.75mm/樣本

# C-scan 切片輸出控制
Z_SLICE_STEP_M      = 0.010   # 每 1cm 輸出一個切片（減少 JSON 大小）
MIN_ASCAN_COUNT     = 9       # 觸發重建最少 A-scan 數（A-core: 自動，我們: 3×3 最小）
MAX_ASCAN_BUFFER    = 400     # 最大緩衝 A-scan 數

# Alpha-blending 平滑係數（對應論文的 alpha-blending processing）
ALPHA_SMOOTH_SIGMA  = 1.2     # Gaussian 平滑 sigma（格點數）


# ─────────────────────────────────────────────────────────────────────────────
# TVG 衰減補償
# ─────────────────────────────────────────────────────────────────────────────
def apply_tvg(raw_samples: np.ndarray,
              sample_rate_hz: float = SAMPLE_RATE_HZ,
              sound_speed_mps: float = SOUND_SPEED_MPS,
              attenuation_db_per_m: float = TVG_DEFAULT_DB_M,
              max_gain_db: float = 40.0) -> np.ndarray:
    """
    時間變化增益（TVG）補償。

    公式：G(t) = 10^(α_dB/m × d(t) / 20)
          其中 d(t) = t × c / 2（單程深度）

    max_gain_db：最大放大量（防止雜訊無限放大，論文建議 40~60dB）
    """
    n = len(raw_samples)
    t = np.arange(n) / sample_rate_hz          # 時間軸 (s)
    depth = t * sound_speed_mps / 2.0          # 單程深度 (m)

    gain_db = attenuation_db_per_m * depth
    gain_db = np.clip(gain_db, 0.0, max_gain_db)
    gain_linear = 10.0 ** (gain_db / 20.0)

    return raw_samples * gain_linear


# ─────────────────────────────────────────────────────────────────────────────
# 包絡線萃取（Hilbert 轉換）
# ─────────────────────────────────────────────────────────────────────────────
def extract_envelope(samples: np.ndarray, dc_window: int = 50) -> np.ndarray:
    """
    去直流 → Hilbert 轉換 → 包絡線（瞬時振幅）。

    dc_window：用前 N 個樣本估計直流偏置。
    """
    baseline = samples[:min(dc_window, len(samples) // 10)].mean()
    centered = samples.astype(np.float64) - baseline
    envelope = np.abs(scipy_hilbert(centered))
    return envelope


# ─────────────────────────────────────────────────────────────────────────────
# C-scan 切片萃取
# ─────────────────────────────────────────────────────────────────────────────
def depth_to_sample_idx(depth_m: float,
                        sample_rate_hz: float = SAMPLE_RATE_HZ,
                        sound_speed_mps: float = SOUND_SPEED_MPS) -> int:
    """深度 (m) → ADC 樣本索引（來回飛行時間對應的樣本數）。"""
    tof_s = 2.0 * depth_m / sound_speed_mps
    return int(tof_s * sample_rate_hz)


def build_cscan_slice(ascan_list: List[dict],
                      depth_m: float,
                      xy_grid: Tuple[np.ndarray, np.ndarray],
                      sample_rate_hz: float = SAMPLE_RATE_HZ,
                      sound_speed_mps: float = SOUND_SPEED_MPS,
                      window_samples: int = 3) -> np.ndarray:
    """
    建構特定深度的 C-mode 切片（對應 A-core-2000 C-scan 成像）。

    ascan_list: [{"x_m":..., "y_m":..., "envelope":np.ndarray}, ...]
    depth_m:    目標深度 (m)
    xy_grid:    (XX, YY) meshgrid（格點位置）
    window_samples: 在目標樣本索引附近取 ±window_samples 個樣本的最大值

    回傳: 2D ndarray，shape = XY 格點大小，值為歸一化振幅 [0, 1]
    """
    target_idx = depth_to_sample_idx(depth_m, sample_rate_hz, sound_speed_mps)

    # 從每個 A-scan 取出目標深度的包絡振幅
    xs, ys, vals = [], [], []
    for ascan in ascan_list:
        env = ascan["envelope"]
        n = len(env)
        i0 = max(0, target_idx - window_samples)
        i1 = min(n, target_idx + window_samples + 1)
        if i1 <= i0:
            continue
        amplitude = float(env[i0:i1].max())
        xs.append(ascan["x_m"])
        ys.append(ascan["y_m"])
        vals.append(amplitude)

    if len(xs) < 3:
        # 掃描點太少，回傳空切片
        return np.zeros(xy_grid[0].shape)

    # 插值到規則格點（線性）
    points = np.stack([xs, ys], axis=1)
    XX, YY = xy_grid
    grid_pts = np.stack([XX.ravel(), YY.ravel()], axis=1)
    slice_vals = griddata(points, vals, grid_pts,
                          method="linear", fill_value=0.0)
    c_slice = slice_vals.reshape(XX.shape)

    # 注意：不對單一切片歸一化，保留 TVG 補償後的絕對振幅。
    # 歸一化只在 reconstruct_3d_volume 的最終步驟做一次（全體積歸一化）。
    # 若各切片獨立歸一化，所有深度的 max 均為 1.0，深度資訊將丟失。

    return np.maximum(c_slice, 0.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 三維體積重建（C-scan 疊加）
# ─────────────────────────────────────────────────────────────────────────────
def reconstruct_3d_volume(ascan_list: List[dict],
                          xy_half_m: float = GRID_XY_HALF_M,
                          xy_spacing_m: float = GRID_XY_SPACING_M,
                          z_min_m: float = GRID_Z_MIN_M,
                          z_max_m: float = GRID_Z_MAX_M,
                          z_step_m: float = Z_SLICE_STEP_M,
                          attenuation_db_per_m: float = TVG_DEFAULT_DB_M,
                          alpha_sigma: float = ALPHA_SMOOTH_SIGMA
                          ) -> Tuple[np.ndarray, dict]:
    """
    從 A-scan 列表重建三維聲學體積。

    對應 A-core-2000 的三個步驟：
      1. TVG 補償（每個 A-scan）
      2. C-scan 切片（各深度）
      3. Alpha-blending 三維平滑

    回傳：
      volume     : shape (Nx, Ny, Nz)，float32，值 [0,1]
      meta       : dict，包含格點座標資訊與統計
    """
    # 建立 XY 格點
    xy_coords = np.arange(-xy_half_m, xy_half_m + xy_spacing_m / 2, xy_spacing_m)
    XX, YY = np.meshgrid(xy_coords, xy_coords, indexing="ij")
    Nx, Ny = XX.shape

    # 深度軸
    z_coords = np.arange(z_min_m, z_max_m + z_step_m / 2, z_step_m)
    Nz = len(z_coords)

    # 預計算每個 A-scan 的 TVG 補償包絡
    # 重要：必須先移除 DC 偏置，再做 TVG 補償。
    # 若先做 TVG，ADC 的 DC 基準（約 2048 count）也會被放大，
    # 深層樣本的 DC 殘留會完全掩蓋真正的回波訊號。
    processed = []
    for ascan in ascan_list:
        raw = np.array(ascan["samples"], dtype=np.float64)
        sr  = ascan.get("sample_rate_hz", SAMPLE_RATE_HZ)

        # Step 1：DC 移除（用前 50 點估計基準）
        dc_window = min(50, max(1, len(raw) // 10))
        baseline  = raw[:dc_window].mean()
        raw_ac    = raw - baseline              # 零均值 AC 訊號

        # Step 2：TVG 補償（對 AC 訊號放大，不再有 DC 問題）
        tvg = apply_tvg(raw_ac, sample_rate_hz=sr,
                        attenuation_db_per_m=attenuation_db_per_m)

        # Step 3：Hilbert 包絡（訊號已為 AC，直接計算）
        env = np.abs(scipy_hilbert(tvg))

        processed.append({
            "x_m":     ascan["x_m"],
            "y_m":     ascan["y_m"],
            "envelope": env,
        })

    # 建立體積矩陣，逐深度切片填充
    volume = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    for k, depth_m in enumerate(z_coords):
        c_slice = build_cscan_slice(
            processed, depth_m, (XX, YY),
            window_samples=2)
        volume[:, :, k] = c_slice

    # Alpha-blending：三維 Gaussian 平滑（論文用 alpha-blending processing）
    volume = gaussian_filter(volume, sigma=alpha_sigma).astype(np.float32)

    # 全局再歸一化
    vmax = volume.max()
    if vmax > 1e-10:
        volume /= vmax

    meta = {
        "Nx": Nx, "Ny": Ny, "Nz": Nz,
        "xy_spacing_m":  xy_spacing_m,
        "z_step_m":      z_step_m,
        "z_min_m":       z_min_m,
        "z_max_m":       z_max_m,
        "n_ascans":      len(ascan_list),
        "x_coords_m":    xy_coords.tolist(),
        "z_coords_m":    z_coords.tolist(),
        "attenuation_db_per_m": attenuation_db_per_m,
    }
    return volume, meta


# ─────────────────────────────────────────────────────────────────────────────
# 體積統計（峰值位置、訊雜比）
# ─────────────────────────────────────────────────────────────────────────────
def cfar_2d_mip(volume: np.ndarray,
                guard: int = 2,
                training: int = 4,
                false_alarm: float = 1e-3) -> np.ndarray:
    """
    2D-CFAR（Cell-Averaging）作用於 Z-MIP 影像。

    Gemini CLI 建議：在最大強度投影（MIP）上做 CFAR，
    比逐層閾值更能適應換能器邊緣的增益衰減。

    guard:       保護區格點數（防目標能量洩漏至背景統計）
    training:    訓練區格點數（局部雜訊估計）
    false_alarm: 目標虛警率（決定倍乘係數 α）

    回傳: binary mask (shape = volume.shape[:2])
    """
    mip = volume.max(axis=2)          # (Nx, Ny)
    nx, ny = mip.shape
    mask = np.zeros((nx, ny), dtype=bool)
    half = guard + training

    # α 係數（CA-CFAR 公式：α = N*(P_fa^(-1/N) - 1)，N = 訓練格點數）
    n_train = (2 * half + 1) ** 2 - (2 * guard + 1) ** 2
    n_train = max(n_train, 1)
    alpha   = n_train * (false_alarm ** (-1.0 / n_train) - 1.0)

    for i in range(nx):
        for j in range(ny):
            r0, r1 = max(0, i - half), min(nx, i + half + 1)
            c0, c1 = max(0, j - half), min(ny, j + half + 1)
            gr0, gr1 = max(0, i - guard), min(nx, i + guard + 1)
            gc0, gc1 = max(0, j - guard), min(ny, j + guard + 1)

            region     = mip[r0:r1, c0:c1].copy()
            guard_mask = np.zeros_like(region, dtype=bool)
            gi0 = gr0 - r0; gi1 = gr1 - r0
            gj0 = gc0 - c0; gj1 = gc1 - c0
            guard_mask[gi0:gi1, gj0:gj1] = True

            train_vals = region[~guard_mask]
            if len(train_vals) == 0:
                continue
            threshold_cell = alpha * train_vals.mean()
            mask[i, j] = mip[i, j] > threshold_cell

    return mask


def analyze_volume(volume: np.ndarray, meta: dict,
                   threshold: float = 0.5,
                   use_cfar: bool = False) -> dict:
    """
    分析三維體積，找出強反射體位置（候選洞穴）。

    threshold: 相對最大值的偵測閾值（0.5 = 最大值的 50%）
    use_cfar:  是否先用 2D-CFAR（MIP）濾除背景雜訊（Gemini 建議）
    回傳: dict，含候選點列表、峰值深度等
    """
    if volume.max() < 1e-6:
        return {"candidates": [], "peak_depth_m": 0.0, "snr_db": 0.0}

    # ── 2D-CFAR 前處理（Gemini 建議）─────────────────────────────────────
    # 在 XY-MIP 上做 CA-CFAR，先鎖定候選 (X,Y)，再沿 Z 提取深度
    if use_cfar:
        cfar_mask_2d = cfar_2d_mip(volume, guard=2, training=4)
        # 將 CFAR 通過的 XY 格點展開為 3D mask
        cfar_mask_3d = np.broadcast_to(
            cfar_mask_2d[:, :, np.newaxis], volume.shape).copy()
        volume_filtered = np.where(cfar_mask_3d, volume, 0.0)
    else:
        volume_filtered = volume

    # 高於閾值的體素
    mask = volume_filtered >= threshold
    idxs = np.argwhere(mask)

    z_coords = np.array(meta["z_coords_m"])
    xy_coords = np.array(meta["x_coords_m"])

    candidates = []
    for ix, iy, iz in idxs[:200]:  # 最多 200 個候選點
        candidates.append({
            "x_m":      float(xy_coords[ix] if ix < len(xy_coords) else 0),
            "y_m":      float(xy_coords[iy] if iy < len(xy_coords) else 0),
            "z_m":      float(z_coords[iz] if iz < len(z_coords) else 0),
            "intensity": float(volume_filtered[ix, iy, iz]),
        })

    # 按強度排序
    candidates.sort(key=lambda c: -c["intensity"])

    # 峰值位置（從 CFAR 過濾後的體積取峰值）
    peak_idx = np.unravel_index(np.argmax(volume_filtered), volume_filtered.shape)
    peak_depth_m = float(z_coords[peak_idx[2]] if peak_idx[2] < len(z_coords) else 0)

    # 訊雜比：基於非零體素計算（稀疏體積中全局 mean 接近零，無法區分）
    # 信號集中（洞穴）→ 非零區域 max >> mean → 高 SNR（≥20dB）
    # 均勻雜訊         → 非零區域 max ≈ mean → 低 SNR（<10dB）
    nonzero_vals = volume[volume > 0.01]
    if len(nonzero_vals) > 0:
        mean_nonzero = float(nonzero_vals.mean())
        snr_db = float(20 * np.log10(volume.max() / (mean_nonzero + 1e-10)))
    else:
        snr_db = 0.0

    return {
        "candidates":   candidates,
        "n_candidates": len(candidates),
        "peak_depth_m": peak_depth_m,
        "snr_db":       round(snr_db, 1),
        "cfar_applied": use_cfar,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ROS2 節點
# ─────────────────────────────────────────────────────────────────────────────
if _ROS2_AVAILABLE:
    class CScanNode(Node):
        """
        C-scan 三維體積重建節點。

        訂閱 /ultrasound/raw_waveform，累積 A-scan，
        當收集足夠掃描點後重建三維體積並發布。

        ROS2 params:
          min_ascans        : int   觸發重建最少 A-scan 數（預設 9）
          attenuation_db_m  : float TVG 衰減係數（預設 60）
          xy_half_m         : float 掃描範圍半徑 m（預設 0.25）
          z_max_m           : float 最大偵測深度 m（預設 0.80）
          auto_trigger      : bool  自動觸發（預設 True）
        """

        def __init__(self):
            super().__init__("acoustic_cscan")

            # 參數
            self.declare_parameter("min_ascans",       MIN_ASCAN_COUNT)
            self.declare_parameter("attenuation_db_m", TVG_DEFAULT_DB_M)
            self.declare_parameter("xy_half_m",        GRID_XY_HALF_M)
            self.declare_parameter("z_max_m",          GRID_Z_MAX_M)
            self.declare_parameter("auto_trigger",     True)

            self._min_n   = int(self.get_parameter("min_ascans").value)
            self._atten   = float(self.get_parameter("attenuation_db_m").value)
            self._xy_half = float(self.get_parameter("xy_half_m").value)
            self._z_max   = float(self.get_parameter("z_max_m").value)
            self._auto    = bool(self.get_parameter("auto_trigger").value)

            self._buf: deque = deque(maxlen=MAX_ASCAN_BUFFER)
            self._scan_count = 0

            self.get_logger().info(
                f"CScanNode init: min_ascans={self._min_n}, "
                f"TVG={self._atten}dB/m, xy=±{self._xy_half*100:.0f}cm, "
                f"z_max={self._z_max*100:.0f}cm")

            # Subscribers
            self.create_subscription(
                String, "/ultrasound/raw_waveform", self._cb_waveform, 20)
            self.create_subscription(
                String, "/scan/trigger", self._cb_trigger, 5)

            # Publishers
            self._pub_volume = self.create_publisher(
                String, "/acoustic/cscan_volume", 5)
            self._pub_slice  = self.create_publisher(
                String, "/acoustic/cscan_slice",  10)
            self._pub_status = self.create_publisher(
                String, "/acoustic/cscan_status", 10)

            self.create_timer(5.0, self._heartbeat)

        def _cb_waveform(self, msg: String):
            """接收 A-scan 波形，加入緩衝區。"""
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                return

            for key in ("x_m", "y_m", "samples", "sample_rate_hz"):
                if key not in data:
                    self.get_logger().warn(f"raw_waveform missing: {key}")
                    return

            self._buf.append(data)
            self._scan_count += 1

            self.get_logger().debug(
                f"A-scan #{self._scan_count} @ "
                f"({data['x_m']*100:.1f}, {data['y_m']*100:.1f}) cm")

            # 即時發布最新切片（用於 RViz2 預覽）
            self._publish_live_slice(data)

            # 自動觸發重建
            if self._auto and len(self._buf) >= self._min_n:
                self._run_reconstruction()

        def _cb_trigger(self, msg: String):
            self.get_logger().info("Manual C-scan trigger.")
            self._run_reconstruction()

        def _publish_live_slice(self, ascan: dict):
            """發布單個 A-scan 的包絡線（供即時監視用）。"""
            raw = np.array(ascan["samples"], dtype=np.float64)
            sr  = ascan.get("sample_rate_hz", SAMPLE_RATE_HZ)
            tvg = apply_tvg(raw, sample_rate_hz=sr,
                            attenuation_db_per_m=self._atten)
            env = extract_envelope(tvg)
            vmax = env.max()
            if vmax > 1e-10:
                env /= vmax

            # 僅輸出前 N 個樣本對應的深度-振幅對（每 5 點取一個）
            z_m = [depth_to_sample_idx(d, sr) for d in
                   np.arange(GRID_Z_MIN_M, self._z_max, 0.005)]
            profile = []
            for idx in z_m:
                if idx < len(env):
                    profile.append(float(env[idx]))

            slice_msg = {
                "x_m":       ascan["x_m"],
                "y_m":       ascan["y_m"],
                "z_min_m":   GRID_Z_MIN_M,
                "z_step_m":  0.005,
                "amplitude": profile,
                "timestamp": ascan.get("timestamp", time.time()),
            }
            self._pub_slice.publish(String(data=json.dumps(slice_msg)))

        def _run_reconstruction(self):
            ascans = list(self._buf)
            n = len(ascans)
            if n < self._min_n:
                self.get_logger().warn(
                    f"Not enough A-scans ({n}/{self._min_n})")
                return

            self.get_logger().info(f"C-scan reconstruction: {n} A-scans...")
            t0 = time.monotonic()

            try:
                volume, meta = reconstruct_3d_volume(
                    ascans,
                    xy_half_m=self._xy_half,
                    z_max_m=self._z_max,
                    attenuation_db_per_m=self._atten,
                )
            except Exception as e:
                self.get_logger().error(f"Reconstruction failed: {e}")
                return

            dt = time.monotonic() - t0
            self.get_logger().info(
                f"Volume ready: {meta['Nx']}×{meta['Ny']}×{meta['Nz']} "
                f"in {dt:.2f}s")

            # 分析體積
            analysis = analyze_volume(volume, meta)
            self.get_logger().info(
                f"Candidates: {analysis['n_candidates']}, "
                f"peak_depth: {analysis['peak_depth_m']*100:.1f}cm, "
                f"SNR: {analysis['snr_db']:.1f}dB")

            # 發布（volume 序列化為稀疏格式，僅輸出 >0.3 的體素）
            threshold = 0.30
            sparse_pts = []
            nz = len(meta["z_coords_m"])
            nx = len(meta["x_coords_m"])
            for ix in range(meta["Nx"]):
                for iy in range(meta["Ny"]):
                    for iz in range(meta["Nz"]):
                        v = float(volume[ix, iy, iz])
                        if v >= threshold:
                            sparse_pts.append([
                                meta["x_coords_m"][min(ix, nx-1)],
                                meta["x_coords_m"][min(iy, nx-1)],
                                meta["z_coords_m"][min(iz, nz-1)],
                                round(v, 3),
                            ])

            volume_msg = {
                "timestamp":    time.time(),
                "meta":         meta,
                "analysis":     analysis,
                "sparse_voxels": sparse_pts[:3000],  # 最多 3000 點
                "recon_time_s": round(dt, 3),
            }
            self._pub_volume.publish(String(data=json.dumps(volume_msg)))

            # 清空緩衝區
            self._buf.clear()
            self._scan_count = 0

        def _heartbeat(self):
            status = {
                "node":       "acoustic_cscan",
                "buf_len":    len(self._buf),
                "total_scans": self._scan_count,
                "ready":      len(self._buf) >= self._min_n,
            }
            self._pub_status.publish(String(data=json.dumps(status)))


    def main(args=None):
        rclpy.init(args=args)
        node = CScanNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            node.get_logger().info("CScan node shutting down.")
        finally:
            node.destroy_node()
            rclpy.shutdown()

else:
    def main(args=None):
        print("ROS2 not found. Run: source /opt/ros/<distro>/setup.bash")


if __name__ == "__main__":
    # 獨立模式：快速自我測試（無需 ROS2）
    print("=== acoustic_cscan 自我測試 ===")
    import math

    np.random.seed(42)
    n_scans = 16
    sample_rate = 1_000_000
    sound_speed = 1500.0

    # 生成模擬 A-scan（4×4 格點，25cm 範圍）
    ascan_list = []
    for xi in range(4):
        for yi in range(4):
            x_m = (xi - 1.5) * 0.08
            y_m = (yi - 1.5) * 0.08
            raw = np.random.normal(2048, 30, 2000).astype(float)
            # 加入洞穴模擬回波在 0.6m 深度
            depth_m = 0.60
            t_echo = 2 * depth_m / sound_speed
            idx_echo = int(t_echo * sample_rate)
            for i in range(max(0, idx_echo - 5), min(2000, idx_echo + 6)):
                amp = 600 * math.exp(-(i - idx_echo) ** 2 / 8)
                raw[i] += amp
            ascan_list.append({
                "x_m": x_m,
                "y_m": y_m,
                "samples": raw.astype(int).tolist(),
                "sample_rate_hz": sample_rate,
            })

    print(f"輸入：{len(ascan_list)} 個 A-scan")
    volume, meta = reconstruct_3d_volume(ascan_list, z_max_m=0.80)
    print(f"體積形狀：{volume.shape}，最大值：{volume.max():.3f}")

    analysis = analyze_volume(volume, meta)
    print(f"候選點：{analysis['n_candidates']}，"
          f"峰值深度：{analysis['peak_depth_m']*100:.1f}cm，"
          f"SNR：{analysis['snr_db']:.1f}dB")

    # 驗證峰值深度接近 0.60m（誤差 < 5cm）
    assert abs(analysis["peak_depth_m"] - 0.60) < 0.05, \
        f"峰值深度誤差過大：{analysis['peak_depth_m']:.3f}m"
    print("✓ 峰值深度正確（誤差 < 5cm）")
    print("=== 測試通過 ===")
