"""
burrow_analyzer.py
==================
螻蛄蝦洞穴超音波算法庫（Level 1–4）
可獨立執行（無需 ROS2）。

算法層次：
  Level 1  - Hilbert 包絡 + TOF 偵測
  Level 2a - 極性分類
  Level 2b - 多回波 CFAR 計數
  Level 2c - Real Cepstrum 壁厚分析（Gemini 建議）
  Level 3  - 水腔諧振頻率分析
  Level 4  - Y 型分叉偵測（雙開口）

前處理工具：
  stack_ascans() — Coherent Stacking（Gemini 建議，32–64 次疊加 +15dB SNR）

聲學物理常數（彰化潮間帶）：
  Z_SAND  = 2.40 MRayl（背景中沙）
  Z_CLAY  = 1.95 MRayl（洞穴壁，黏土壓實）
  Z_WATER = 1.50 MRayl（洞穴水腔）
  R_SAND_CLAY  = -0.103（負極性，阻抗下降）
  R_CLAY_WATER = -0.130（負極性，阻抗下降）

Run standalone:
  python ros2/archimedes_survey/burrow_analyzer.py
"""

import math
import numpy as np
from typing import Optional

try:
    from scipy.signal import hilbert as _hilbert, find_peaks, correlate
    _SCIPY = True
except ImportError:
    _SCIPY = False
    print("[burrow_analyzer] scipy 未安裝，部分功能降級")

# ─────────────────────────────────────────────────────────────────────────────
# 聲學常數
# ─────────────────────────────────────────────────────────────────────────────
C_MPS        = 1531.0   # 台灣彰化近海聲速 m/s（T=25°C, S=32ppt）
F0_HZ        = 200e3    # 換能器中心頻率 Hz
SAMPLE_RATE  = 1_000_000  # 預設 ADC 採樣率 Hz（MCP3208 模式）
TVG_DB_PER_M = 60.0     # 彰化粉砂衰減係數

Z_SAND  = 2.40   # MRayl
Z_CLAY  = 1.95   # MRayl
Z_WATER = 1.50   # MRayl

R_SAND_CLAY  = (Z_CLAY - Z_SAND)  / (Z_CLAY + Z_SAND)    # -0.1034
R_CLAY_WATER = (Z_WATER - Z_CLAY) / (Z_WATER + Z_CLAY)   # -0.1304

OPENING_DIST_M = 0.232  # 鹿港蝦猴兩開口距離均值（文獻：21.8~26.4 cm）


# ─────────────────────────────────────────────────────────────────────────────
# 合成信號生成（測試用）
# ─────────────────────────────────────────────────────────────────────────────
def _att(depth_m: float, alpha_db_per_m: float = TVG_DB_PER_M) -> float:
    """雙程振幅衰減係數。"""
    return 10.0 ** (-alpha_db_per_m * depth_m / 20.0)


def _make_pulse(n: int, tof_s: float, amplitude: float, polarity: float,
                f0: float = F0_HZ, fs: float = SAMPLE_RATE,
                n_cycles: int = 5) -> np.ndarray:
    """Gaussian 調製正弦脈衝（n_cycles 個週期）。"""
    t = np.arange(n) / fs - tof_s
    sigma = n_cycles / f0 / 6.0
    env = np.exp(-t ** 2 / (2 * sigma ** 2))
    return polarity * amplitude * env * np.sin(2 * np.pi * f0 * t)


def simulate_ascan(scenario: str,
                   fs: float = SAMPLE_RATE,
                   c: float = C_MPS,
                   snr_db: float = 20.0,
                   seed: Optional[int] = 42) -> tuple:
    """
    生成合成 A-scan RF 信號。

    scenario:
      'no_burrow'   — 只有噪音（背景沙灘）
      'clay_wall'   — 壓實黏土壁（30cm，負極性 1 個回波）
      'void_entry'  — 黏土壁（30cm）+ 水腔（壁厚 5cm），2 個負極性回波
      'y_junction'  — Y 型洞穴（分叉 40cm + 深井 100cm），3 個回波
      'deep_burrow' — 深洞壁（80cm），高度衰減，低信心

    回傳 (rf: np.ndarray, ground_truth: dict)
    """
    rng = np.random.default_rng(seed)
    n = int(2700 * fs / 1e6)   # 2.7ms 窗口 → 最大 2m
    sig = np.zeros(n)
    gt: dict = {}

    if scenario == 'no_burrow':
        gt = {'type': 'no_burrow', 'n_echoes': 0, 'burrow': False}

    elif scenario == 'clay_wall':
        z = 0.30
        tof = 2 * z / c
        amp = abs(R_SAND_CLAY) * _att(z)
        sig += _make_pulse(n, tof, amp, -1.0, fs=fs)
        gt = {
            'type': 'clay_wall', 'n_echoes': 1, 'burrow': True,
            'echoes': [{'tof_us': tof * 1e6, 'depth_m': z, 'polarity': 'negative'}],
        }

    elif scenario == 'void_entry':
        z_wall = 0.30
        wall_t = 0.05          # 5 cm 壁厚
        tof_w = 2 * z_wall / c
        tof_v = 2 * (z_wall + wall_t) / c
        sig += _make_pulse(n, tof_w, abs(R_SAND_CLAY)  * _att(z_wall),        -1.0, fs=fs)
        sig += _make_pulse(n, tof_v, abs(R_CLAY_WATER) * _att(z_wall + wall_t), -1.0, fs=fs)
        gt = {
            'type': 'void_entry', 'n_echoes': 2, 'burrow': True,
            'wall_depth_m': z_wall, 'wall_thickness_m': wall_t,
            'echoes': [
                {'tof_us': tof_w * 1e6, 'depth_m': z_wall,          'polarity': 'negative'},
                {'tof_us': tof_v * 1e6, 'depth_m': z_wall + wall_t, 'polarity': 'negative'},
            ],
        }

    elif scenario == 'y_junction':
        z_j = 0.40     # 分叉深度
        z_d = 1.00     # 深井底部
        tof_j = 2 * z_j / c
        tof_d = 2 * z_d / c
        tof_x = tof_j + OPENING_DIST_M / c   # 交叉路徑（分叉後繞到 B 開口）
        sig += _make_pulse(n, tof_j, 0.15 * _att(z_j),   -1.0, fs=fs)
        sig += _make_pulse(n, tof_d, 0.20 * _att(z_d),   -1.0, fs=fs)
        sig += _make_pulse(n, tof_x, 0.05 * _att(z_j),   -1.0, fs=fs)
        gt = {
            'type': 'y_junction', 'n_echoes': 3, 'burrow': True,
            'junction_depth_m': z_j, 'deep_well_depth_m': z_d,
            'opening_dist_m': OPENING_DIST_M,
        }

    elif scenario == 'deep_burrow':
        z = 0.80
        tof = 2 * z / c
        amp = abs(R_SAND_CLAY) * _att(z)
        sig += _make_pulse(n, tof, amp, -1.0, fs=fs)
        gt = {
            'type': 'deep_burrow', 'n_echoes': 1, 'burrow': True,
            'echoes': [{'tof_us': tof * 1e6, 'depth_m': z, 'polarity': 'negative'}],
        }
    else:
        raise ValueError(f"未知場景: {scenario}")

    # 加高斯白噪音（依 SNR）
    sig_rms = float(np.sqrt(np.mean(sig ** 2))) if np.any(sig != 0) else 1e-4
    noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
    sig += rng.normal(0.0, max(noise_rms, 1e-6), n)

    return sig.astype(np.float32), gt


# ─────────────────────────────────────────────────────────────────────────────
# 前處理：Coherent Stacking
# ─────────────────────────────────────────────────────────────────────────────
def stack_ascans(scans: list,
                 align: bool = False,
                 max_lag_us: float = 5.0,
                 fs: float = SAMPLE_RATE) -> np.ndarray:
    """
    Coherent Stacking（相干疊加）N 次 A-scan，SNR 改善 +10×log10(N) dB。

    原理：訊號相干疊加（×N），噪音非相干（×√N），平均後 SNR×√N。
    建議疊加次數：
      深洞 (>50cm) : 32–64 次 → +15~18 dB
      標準場景     :  8–16 次 → +9~12 dB

    align    : True → 用包絡互相關修正觸發抖動（預設 False，無抖動模擬時不需要）
    max_lag_us: 最大允許抖動 μs（align=True 時生效）
    """
    if not scans:
        raise ValueError("stack_ascans: 空列表")
    n_ref = len(scans[0])
    arr = [np.asarray(s, dtype=np.float64)[:n_ref] for s in scans]

    if align and len(arr) > 1 and _SCIPY:
        max_lag_n = max(1, int(max_lag_us * 1e-6 * fs))
        ref_env = np.abs(_hilbert(arr[0]))
        aligned = [arr[0]]
        for a in arr[1:]:
            env = np.abs(_hilbert(a))
            corr = correlate(env, ref_env, mode='full')
            center = n_ref - 1
            lo = max(0, center - max_lag_n)
            hi = min(len(corr), center + max_lag_n + 1)
            best_lag = int(np.argmax(corr[lo:hi])) + (lo - center)
            if best_lag > 0:
                shifted = np.concatenate([np.zeros(best_lag), a[:-best_lag]])
            elif best_lag < 0:
                shifted = np.concatenate([a[-best_lag:], np.zeros(-best_lag)])
            else:
                shifted = a
            aligned.append(shifted[:n_ref])
        arr = aligned

    stacked = np.mean(arr, axis=0)
    return stacked.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Level 1：Hilbert 包絡 + TOF
# ─────────────────────────────────────────────────────────────────────────────
def hilbert_envelope(rf: np.ndarray) -> np.ndarray:
    """回傳 Hilbert 包絡（scipy 或 FFT 手算）。"""
    if _SCIPY:
        return np.abs(_hilbert(rf.astype(np.float64)))
    # 退回：FFT 手算解析信號
    N = len(rf)
    F = np.fft.rfft(rf.astype(np.float64))
    h = np.zeros(N // 2 + 1)
    h[0] = 1.0
    if N % 2 == 0:
        h[1:-1] = 2.0
        h[-1] = 1.0
    else:
        h[1:] = 2.0
    return np.abs(np.fft.irfft(F * h, n=N))


def detect_tof(rf: np.ndarray,
               fs: float = SAMPLE_RATE,
               c: float = C_MPS,
               blank_us: float = None) -> dict:
    """
    Level 1：Hilbert 包絡偵測第一回波 TOF。
    blank_us: 消隱時間 μs（預設 = 最小偵測深度 20mm）
    回傳 {'tof_us', 'depth_m', 'confidence', 'envelope'}
    """
    if blank_us is None:
        blank_us = 0.02 * 2 / c * 1e6   # 20mm 消隱

    env = hilbert_envelope(rf)
    blank_n = max(0, int(blank_us * 1e-6 * fs))
    env_masked = env.copy()
    env_masked[:blank_n] = 0.0

    if env_masked.max() < 1e-9:
        return {'tof_us': -1.0, 'depth_m': -1.0, 'confidence': 0.0, 'envelope': env}

    peak_i = int(np.argmax(env_masked))
    tof_us = peak_i / fs * 1e6
    depth_m = tof_us * 1e-6 * c / 2.0
    # SNR 信心值：peak vs 消隱前區域的 Hilbert 包絡均值
    # 擴大基線窗口（3×blank_n）以獲得更穩定的噪音估計
    baseline_n = max(20, min(blank_n * 3, 200))
    noise_floor = float(np.mean(env[:baseline_n])) + 1e-12
    # 乘數 9.0：對 N=2700 的純噪音，max(envelope)/mean(envelope) ≈ 3-4，
    # 所以 conf_noise ≈ 4/9 ≈ 0.44 < 0.5；有信號時 peak >> mean，capped at 1.0
    conf = float(min(1.0, env_masked[peak_i] / (noise_floor * 9.0)))

    return {'tof_us': tof_us, 'depth_m': depth_m, 'confidence': conf, 'envelope': env}


# ─────────────────────────────────────────────────────────────────────────────
# Level 2：極性分類 + 多回波 CFAR 計數
# ─────────────────────────────────────────────────────────────────────────────
def polarity_at(rf: np.ndarray, tof_us: float, fs: float = SAMPLE_RATE) -> str:
    """
    Level 2a：分析首個半週期（envelope 中心之後）的平均極性。

    原理：對稱視窗（±half_win）在低 SNR 時正負相消導致結果隨機；
    改為只看 envelope 中心「之後」的 T/2 時間，此段的積分值符號即為極性。

    回傳 'negative' | 'positive' | 'ambiguous'
    """
    peak_i = int(tof_us * 1e-6 * fs)
    # 向前看 T/2 = fs / (2 × f0) 個樣本
    n_half = max(2, int(fs / F0_HZ / 2))
    lo = max(0, peak_i + 1)
    hi = min(len(rf), lo + n_half + 1)
    segment = rf[lo:hi]
    if len(segment) == 0:
        return 'ambiguous'

    # 噪音標準差：取 peak 前 1/4 段
    baseline_n = max(1, min(peak_i // 4, 50))
    noise_std = float(np.std(rf[:baseline_n])) + 1e-12

    # 第一個半週期的均值決定極性
    mean_val = float(np.mean(segment))
    if mean_val < -0.3 * noise_std:
        return 'negative'
    if mean_val >  0.3 * noise_std:
        return 'positive'
    return 'ambiguous'


def count_echoes(rf: np.ndarray,
                 fs: float = SAMPLE_RATE,
                 c: float = C_MPS,
                 blank_us: float = None,
                 cfar_guard: int = 10,
                 cfar_ref: int = 30,
                 cfar_threshold_factor: float = 4.0) -> list:
    """
    Level 2b：CFAR（Cell-Averaging CFAR）多回波偵測。

    cfar_guard           : 保護單元數（避免峰值洩漏到參考窗）
    cfar_ref             : 參考單元數（每側）
    cfar_threshold_factor: 閾值倍數（4 ≈ 12dB SNR 最小值）

    回傳 echo 列表：[{'tof_us', 'depth_m', 'amp', 'polarity'}, ...]
    """
    if blank_us is None:
        blank_us = 0.02 * 2 / c * 1e6

    env = hilbert_envelope(rf)
    blank_n = max(0, int(blank_us * 1e-6 * fs))
    env[:blank_n] = 0.0
    N = len(env)

    echoes = []

    for i in range(cfar_ref + cfar_guard, N - cfar_ref - cfar_guard):
        cell = env[i]
        ref_lo = env[i - cfar_ref - cfar_guard : i - cfar_guard]
        ref_hi = env[i + cfar_guard + 1       : i + cfar_guard + cfar_ref + 1]
        ref_all = np.concatenate([ref_lo, ref_hi])
        if len(ref_all) == 0:
            continue
        threshold = cfar_threshold_factor * ref_all.mean()
        if cell < threshold:
            continue
        # 局部最大值（峰值）
        lo = max(0, i - cfar_guard)
        hi = min(N, i + cfar_guard + 1)
        if cell < env[lo:hi].max():
            continue   # 非局部最大值

        tof_us = i / fs * 1e6
        pol = polarity_at(rf, tof_us, fs)
        echoes.append({
            'tof_us':  round(tof_us, 2),
            'depth_m': round(tof_us * 1e-6 * c / 2.0, 4),
            'amp':     round(float(cell), 6),
            'polarity': pol,
        })

    # 合併過近的回波（距離 < 2 個脈衝週期）
    min_sep_us = 2.0 / F0_HZ * 1e6
    merged = []
    for e in echoes:
        if merged and (e['tof_us'] - merged[-1]['tof_us']) < min_sep_us:
            if e['amp'] > merged[-1]['amp']:
                merged[-1] = e
        else:
            merged.append(e)

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Level 2c：Real Cepstrum 壁厚分析
# ─────────────────────────────────────────────────────────────────────────────
def cepstrum_wall_thickness(rf: np.ndarray,
                             fs: float = SAMPLE_RATE,
                             c_clay: float = 1465.0,
                             q_lo_us: float = 10.0,
                             q_hi_us: float = 200.0) -> dict:
    """
    Level 2c：Real Cepstrum 壁厚分析。

    原理：
      含兩個回波（壁前後界面，時間差 τ）的信號，其對數功率譜出現
      週期 = 1/τ 的餘弦紋波。Real Cepstrum（對數功率譜的 IFFT）在
      quefrency = τ 處出現峰值。壁厚 = τ × c_clay / 2。

    優勢：即使兩回波時間差 < λ/2（3.6mm@200kHz），紋波仍存在，
    比 CFAR 雙峰差更適合偵測薄壁（壁厚 <5mm）。

    q_lo_us  : quefrency 搜尋下限 μs（預設 10μs，排除載頻 T₀=5μs 諧波干擾）
    q_hi_us  : quefrency 搜尋上限 μs（預設 200μs ≈ 15cm 壁厚上限）
    c_clay   : 黏土聲速 m/s（彰化黏土 1465 m/s）

    回傳 {'wall_thickness_m', 'delay_us', 'cepstrum_confidence'}
    """
    N = len(rf)
    sig = rf.astype(np.float64)

    # Real Cepstrum: irfft( log|rfft(x)|² )
    spectrum  = np.fft.rfft(sig)
    log_power = np.log(np.abs(spectrum) ** 2 + 1e-30)
    cepstrum_full = np.fft.irfft(log_power, n=N)

    # quefrency 搜尋範圍
    q_lo_n = max(1, int(q_lo_us * 1e-6 * fs))
    q_hi_n = min(N // 2, int(q_hi_us * 1e-6 * fs))
    if q_hi_n <= q_lo_n + 2:
        return {'wall_thickness_m': None, 'delay_us': None, 'cepstrum_confidence': 0.0}

    cep   = cepstrum_full[q_lo_n:q_hi_n]
    q_us  = np.arange(q_lo_n, q_hi_n) / fs * 1e6

    # 噪音基準：|cep| 均值（Real Cepstrum ≈ 零均值高斯，mean|x|=σ√(2/π)≈0.8σ）
    # threshold = 6× → P(false_alarm/bin) ≈ Φ(-6.4) ≈ 7×10⁻¹¹ → 誤報幾乎為零
    cep_noise = float(np.mean(np.abs(cep))) + 1e-30
    threshold = cep_noise * 6.0

    if _SCIPY:
        peaks_idx, props = find_peaks(
            cep,
            height=threshold,
            prominence=cep_noise * 3.0,
            distance=max(1, int(5e-6 * fs)),   # 最小間距 5μs
        )
        if len(peaks_idx) == 0:
            return {'wall_thickness_m': None, 'delay_us': None,
                    'cepstrum_confidence': 0.0}
        best_i    = peaks_idx[int(np.argmax(props['peak_heights']))]
        peak_val  = float(cep[best_i])
    else:
        # fallback：純最大值檢查
        best_i   = int(np.argmax(cep))
        peak_val = float(cep[best_i])
        if peak_val < threshold:
            return {'wall_thickness_m': None, 'delay_us': None,
                    'cepstrum_confidence': 0.0}

    delay_us = float(q_us[best_i])
    wall_m   = delay_us * 1e-6 * c_clay / 2.0
    # 信心值：peak 相對於噪音基準 15× 的比值（15× ≈ 符號峰值 ≫ 誤報）
    confidence = float(min(1.0, peak_val / (cep_noise * 15.0)))

    return {
        'wall_thickness_m':    round(wall_m, 4),
        'delay_us':            round(delay_us, 2),
        'cepstrum_confidence': round(confidence, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Level 3：水腔諧振頻率分析
# ─────────────────────────────────────────────────────────────────────────────
def cavity_resonance(rf: np.ndarray,
                     fs: float = SAMPLE_RATE,
                     c_void: float = 1500.0,
                     f_lo: float = 50e3,
                     f_hi: float = 500e3) -> dict:
    """
    Level 3：FFT 功率譜中尋找空腔諧振峰。
    f_n = n × c / (2L)  →  L = n × c / (2 × f_n)

    回傳 {'resonances': [...], 'dominant_khz': float | None,
          'cavity_length_cm': float | None}
    """
    N = len(rf)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    power = np.abs(np.fft.rfft(rf.astype(np.float64))) ** 2

    mask = (freqs >= f_lo) & (freqs <= f_hi)
    f_sub = freqs[mask]
    p_sub = power[mask]

    if len(p_sub) < 5:
        return {'resonances': [], 'dominant_khz': None, 'cavity_length_cm': None}

    peaks = []
    if _SCIPY:
        # 排除載頻帶（發射脈衝本身，非空腔諧振）
        f_exclude = (f_sub >= F0_HZ * 0.7) & (f_sub <= F0_HZ * 1.3)
        p_search = p_sub.copy()
        p_search[f_exclude] = 0.0
        # 閾值：噪音底數均值 × 10（chi²(2) 分布下 P(false_alarm/bin) ≈ exp(-10) ≈ 4.5e-5）
        # 對 1350 頻率格：期望誤報數 ≈ 1350 × 4.5e-5 × 0.5 ≈ 0.03 → 幾乎為零
        # 對 relative-to-max 閾值無效，因為 chi²(2) 的次大值幾乎等於最大值
        p_mean = float(np.mean(p_sub)) + 1e-30
        idx, props = find_peaks(p_search,
                                height=p_mean * 10.0,
                                distance=max(1, len(p_sub) // 30),
                                prominence=p_mean * 5.0)
        for i in idx:
            f_r = float(f_sub[i])
            L_m = c_void / (2 * f_r)   # n=1 基頻假設
            peaks.append({
                'freq_khz':       round(f_r / 1e3, 1),
                'cavity_len_cm':  round(L_m * 100, 1),
                'power_rel':      round(float(p_sub[i] / p_sub.max()), 3),
            })
    else:
        # 退回：只找最大峰
        i_max = int(np.argmax(p_sub))
        f_r = float(f_sub[i_max])
        L_m = c_void / (2 * f_r)
        if p_sub[i_max] > p_sub.mean() * 5:
            peaks.append({
                'freq_khz':      round(f_r / 1e3, 1),
                'cavity_len_cm': round(L_m * 100, 1),
                'power_rel':     1.0,
            })

    dominant = peaks[0] if peaks else None
    return {
        'resonances':      peaks,
        'dominant_khz':    dominant['freq_khz']      if dominant else None,
        'cavity_length_cm': dominant['cavity_len_cm'] if dominant else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Level 4：Y 型分叉偵測
# ─────────────────────────────────────────────────────────────────────────────
def detect_y_junction(rf_a: np.ndarray,
                      rf_b: np.ndarray,
                      fs: float = SAMPLE_RATE,
                      c: float = C_MPS,
                      gps_dist_m: Optional[float] = None,
                      expected_dist_m: float = OPENING_DIST_M,
                      dist_tol_m: float = 0.05) -> dict:
    """
    Level 4：從兩開口 A-scan 聯合分析 Y 型分叉。

    判斷邏輯（不依賴互相關，互相關對相同幾何的兩路信號恆給 lag=0）：
      1. 兩路均有信號（信心 > 0.3）
      2. 兩路均有 ≥2 個回波（分叉點 + 深井）
      3. 兩路第一回波深度吻合（同一分叉點，差距 < 5cm）
      4. GPS 開口間距在文獻範圍（若有提供）

    rf_a, rf_b      : 從 A、B 兩開口分別量測的 RF 信號
    gps_dist_m      : GPS 測量的兩開口間距（m），None 表示未知
    expected_dist_m : 文獻兩開口間距均值（0.232m）
    dist_tol_m      : GPS 間距容許誤差（±5cm）

    回傳 {'is_y_junction', 'junction_depth_m', 'est_opening_dist_m',
          'depth_diff_m', 'conf_a', 'conf_b', 'confidence', 'reason'}
    """
    blank_n = max(0, int(0.02 * 2 / c * 1e6 * 1e-6 * fs))

    def _tof_and_conf(rf: np.ndarray) -> tuple:
        env = hilbert_envelope(rf)
        env_m = env.copy()
        env_m[:blank_n] = 0.0
        pk = int(np.argmax(env_m))
        baseline_n = max(5, min(blank_n, 100))
        noise_floor = float(np.median(env[:baseline_n])) + 1e-12
        cf = float(min(1.0, env_m[pk] / (noise_floor * 7.0)))
        return pk / fs, cf

    tof_a_s, conf_a = _tof_and_conf(rf_a)
    tof_b_s, conf_b = _tof_and_conf(rf_b)

    junction_depth = (tof_a_s + tof_b_s) / 2 * c / 2.0
    depth_diff = abs(tof_a_s - tof_b_s) * c / 2.0

    echoes_a = count_echoes(rf_a, fs, c)
    echoes_b = count_echoes(rf_b, fs, c)

    both_valid  = conf_a > 0.3 and conf_b > 0.3
    both_multi  = len(echoes_a) >= 2 and len(echoes_b) >= 2
    depth_agree = depth_diff < 0.05   # 分叉點深度差 < 5cm

    # GPS 間距驗證（選填）
    if gps_dist_m is not None:
        dist_ok = abs(gps_dist_m - expected_dist_m) <= dist_tol_m
        est_dist = gps_dist_m
    else:
        dist_ok = True   # 未提供 GPS → 不排除
        est_dist = 0.0

    is_y = both_valid and both_multi and depth_agree and dist_ok
    confidence = float(min(1.0, (conf_a + conf_b) / 2.0))
    if not (both_multi and depth_agree):
        confidence *= 0.3

    reason_parts = []
    if not both_valid:
        reason_parts.append(f"weak signal A={conf_a:.2f} B={conf_b:.2f}")
    if not both_multi:
        reason_parts.append(f"echoes too few A={len(echoes_a)} B={len(echoes_b)}")
    if not depth_agree:
        reason_parts.append(f"depth mismatch {depth_diff*100:.1f}cm")
    if gps_dist_m is not None and not dist_ok:
        reason_parts.append(f"GPS dist {gps_dist_m*100:.1f}cm out of range")
    if is_y:
        reason_parts.append("Y-junction confirmed")

    return {
        'is_y_junction':      is_y,
        'junction_depth_m':   round(junction_depth, 3),
        'est_opening_dist_m': round(est_dist, 3),
        'depth_diff_m':       round(depth_diff, 3),
        'conf_a':             round(conf_a, 3),
        'conf_b':             round(conf_b, 3),
        'confidence':         round(confidence, 3),
        'reason':             '; '.join(reason_parts) if reason_parts else 'none',
    }


# ─────────────────────────────────────────────────────────────────────────────
# 整合分析器（單點 A-scan 完整流程）
# ─────────────────────────────────────────────────────────────────────────────
class BurrowAnalyzer:
    """
    單點 A-scan 完整分析流程（Level 1–3）。
    Level 4（Y 型）需要兩路 A-scan，呼叫 detect_y_junction() 函式。
    """

    def __init__(self,
                 fs: float = SAMPLE_RATE,
                 c: float = C_MPS,
                 snr_threshold: float = 0.25):
        self.fs  = fs
        self.c   = c
        self.snr_threshold = snr_threshold

    def analyze(self, rf: np.ndarray) -> dict:
        """
        完整 Level 1–3 分析。

        回傳 {
          'level1': detect_tof 結果,
          'level2': {'echoes': [...], 'n_echoes': int},
          'level3': cavity_resonance 結果,
          'verdict': 'no_burrow' | 'clay_wall' | 'void_entry' | 'y_candidate',
          'confidence': float
        }
        """
        # Level 1
        l1 = detect_tof(rf, self.fs, self.c)

        # Level 2
        echoes = count_echoes(rf, self.fs, self.c)
        n = len(echoes)
        cep = cepstrum_wall_thickness(rf, self.fs)

        # Level 3
        l3 = cavity_resonance(rf, self.fs)

        # 判斷邏輯
        conf = l1['confidence']
        if conf < self.snr_threshold or n == 0:
            verdict = 'no_burrow'
        elif n == 1:
            pol = echoes[0]['polarity']
            verdict = 'clay_wall' if pol == 'negative' else 'hard_layer'
        elif n == 2:
            verdict = 'void_entry'
        else:
            verdict = 'y_candidate'

        return {
            'level1':     l1,
            'level2':     {'echoes': echoes, 'n_echoes': n, 'cepstrum': cep},
            'level3':     l3,
            'verdict':    verdict,
            'confidence': round(conf, 3),
        }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("burrow_analyzer.py — 快速自我測試")
    print(f"scipy 可用: {_SCIPY}")
    print(f"R_SAND_CLAY  = {R_SAND_CLAY:.4f}  ({R_SAND_CLAY*100:.1f}%)")
    print(f"R_CLAY_WATER = {R_CLAY_WATER:.4f}  ({R_CLAY_WATER*100:.1f}%)")
    print()
    az = BurrowAnalyzer()
    for sc in ['no_burrow', 'clay_wall', 'void_entry', 'y_junction', 'deep_burrow']:
        rf, gt = simulate_ascan(sc)
        res = az.analyze(rf)
        ok = "✓" if res['verdict'] in gt['type'] or (
            gt['type'] == 'y_junction' and res['verdict'] == 'y_candidate') else "✗"
        cep = res['level2']['cepstrum']
        wall_str = (f"{cep['wall_thickness_m']*100:.1f}cm" if cep['wall_thickness_m'] else "—")
        print(f"  {sc:15s} → verdict={res['verdict']:12s} "
              f"n_echoes={res['level2']['n_echoes']}  conf={res['confidence']:.2f}  "
              f"cep_wall={wall_str}  {ok}")

    print()
    print("Stacking 測試（16× deep_burrow SNR=10dB）:")
    scans = [simulate_ascan('deep_burrow', snr_db=10.0, seed=200 + i)[0] for i in range(16)]
    stacked = stack_ascans(scans)
    c_single  = detect_tof(scans[0])['confidence']
    c_stacked = detect_tof(stacked)['confidence']
    print(f"  single conf={c_single:.3f}  stacked conf={c_stacked:.3f}  "
          f"{'✓' if c_stacked > c_single else '✗'}")
