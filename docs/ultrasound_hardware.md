# 超音波換能器硬體實作規範
**Archimedes Survey Robot — 200~500kHz 防水超音波模組**
版本：2.0  日期：2026-03-18
更新：採納學術論文建議，升級發射器 IC、TGC 電路、DAQ 平台與 DSP 方法

---

## 1. 換能器選型

### 1.1 目標頻率選擇

根據美食奧螻蛄蝦洞穴探測需求（深達 2m，解析度公分級）：

| 頻率 | 穿透深度 | 空間解析度 | 適用場景 |
|------|---------|----------|---------|
| 500 kHz | ~80cm | ~1.5mm | 淺洞、幼蝦洞（< 50cm）|
| **200 kHz** ★ | **~200cm** | **~3.75mm** | **全深度洞穴探測（推薦）**|
| 130 kHz | ~350cm | ~5.8mm | 超深洞、寬頻脈衝 |
| 50 kHz | >5m | ~15mm | 地層分類（解析度不足）|

**★ 主要工作頻率：200 kHz**（λ ≈ 7.5mm in water，可分辨 3.75mm 特徵）

### 1.2 壓電材料選型（文獻依據）

**推薦：PZT-5A（軟式壓電陶瓷）**

| 參數 | PZT-5A（推薦）| PZT-4（硬式）|
|------|-------------|------------|
| 機電耦合係數 k_t | 0.49 | 0.51 |
| 壓電常數 d33 | 374 pC/N | 289 pC/N |
| 介電損耗 tan δ | 0.02 | 0.004 |
| 居里溫度 | **365°C** | 328°C |
| 適用場景 | 脈衝回波（高靈敏度）| 高功率發射 |
| 台灣採購 | CTDCO（台灣陶瓷公司）| 同廠商 |

> ⚠️ **焊接注意**：PZT 壓電陶瓷居里溫度 365°C，焊接時必須快速（<3秒），
> 避免陶瓷去極化（depolarization）。使用低溫焊錫（Sn-Bi, Tm=138°C）較安全。

### 1.3 聲學匹配層設計（λ/4 理論）

聲學匹配層（Acoustic Matching Layer）放於 PZT 正面，減少聲阻抗跳躍：

```
Z_match = √(Z_pzt × Z_water)
Z_pzt   ≈ 34 MRayl（PZT-5A）
Z_water ≈ 1.5 MRayl

Z_match = √(34 × 1.5) ≈ 7.1 MRayl
材料：  鋁（Z≈17MRayl，偏高）→ 玻璃（Z≈13MRayl）→ 環氧樹脂+鋁粉（可調）
厚度：  λ/4 = c_match / (4 × f0) ≈ 1300/(4×200e3) ≈ 1.625 mm
```

**後置阻尼層（Backing Layer）**：
- 材料：環氧樹脂 + 鎢粉（高密度）
- 功能：吸收向後輻射的聲波，縮短脈衝長度（提升縱向解析度）
- Z_back ≥ Z_pzt 才能有效阻尼

---

## 2. 發射電路（v2.0：HV7360 專用脈衝 IC）

### 2.1 架構對比

| 方案 | v1.0（舊）| **v2.0（新，推薦）**|
|------|---------|-----------------|
| 發射 IC | IR2110（MOSFET 閘驅）+ IRF540N H橋 | **HV7360**（±100V 專用脈衝 IC）|
| 最大電壓 | ±12V（受限 IRF540N Vgs）| **±100V**（HV7360 規格）|
| 阻尼方式 | 無主動阻尼（需外加 Rx 電阻）| **Active RTZ**（Return to Zero，HV7361）|
| 驅動電流 | 峰值 ~500mA | **±2.5A** |
| 脈衝寬度精度 | GPIO 軟體限制（>1μs）| 2ns 精度（內建計時器）|
| 元件數 | 6 顆（IC×1 + MOSFET×4 + 保護二極體）| **1 顆 IC + 少量被動元件**|

### 2.2 HV7360 電路設計

```
RPi/RP2040 GPIO ──→ [HV7360LA-G]
                      ├── OUT_POS (+100V 上拉)
                      └── OUT_NEG (-100V 下拉)
                              │
                           換能器（兩端差動接法）
                              │
                         [AD8331 TGC]──→ ADC（pic0rick）
```

**HV7360 最小電路（台灣元件）：**

| 元件 | 型號 | 規格 | 單價(NT$) | 來源 |
|------|------|------|----------|------|
| 專用脈衝 IC | **HV7360LA-G** | ±100V, ±2.5A, 2ns | ~NT$450 | Digi-Key / 欣晟電子 |
| 電源升壓模組 | MT3608 + 變壓器 | 12V → ±100V | ~NT$200 | 蝦皮 |
| 旁路電容 | 10μF/200V + 100nF | 高壓去耦 | NT$30 | 世代電子 |
| 保護 TVS | P6KE100A | 換能器過電壓保護 | NT$15 | 欣晟電子 |
| **小計** | | | **~NT$700** | |

> HV7361 = HV7360 + 內建 Active RTZ（主動返零阻尼），推薦版本。
> Microchip 官方評估板：HV7361 EV Board，可直接外掛 RPi GPIO。

### 2.3 脈衝時序（更新版）

```
發射時序（HV7360，200kHz，5 個週期脈衝串）：
  T0: IN_POS → HIGH（2.5μs HIGH = 半個 200kHz 週期）
  T1: T0+2.5μs → IN_POS LOW / IN_NEG HIGH（反向半週期）
  T2: 重複 5 次 → 輸出 5 週期 200kHz burst（25μs）

接收消隱時間（Blanking）：
  T_blank: 發射結束後 50μs（對應最小量測距離 37mm）

最小量測深度：50μs × 1500m/s ÷ 2 = 37.5mm ≈ 38mm
最大量測深度：2667μs × 1500m/s ÷ 2 = 2000mm = 2m ✓
```

---

## 3. 接收鏈（v2.0：AD8331 TGC VGA）

### 3.1 架構對比

| 方案 | v1.0（舊）| **v2.0（新，推薦）**|
|------|---------|-----------------|
| 放大 IC | MAX9814（固定 AGC，60dB）| **AD8331**（VGA TGC，-28 to +92dB）|
| 增益控制 | 無（AGC 自動）| **程式控制**（0-1V → -28~+92dB）|
| 頻寬 | 300Hz~20kHz（音頻）| **DC~120MHz**（超音波專用）|
| 雜訊指數 | ~10dB | **~5dB** |
| 輸入阻抗 | 10kΩ | **200Ω**（匹配換能器）|

### 3.2 AD8331 TGC 連接

```python
# TGC 控制信號（DAC 輸出 → AD8331 GAIN 腳）
# 隨 TOF 時間線性增大增益，補償深度衰減
def tvg_gain_voltage(t_us: float, alpha_db_per_m: float = 60.0,
                     c_mps: float = 1531.0) -> float:
    """
    計算 AD8331 GAIN 腳電壓（0~1V → -28dB~+92dB）
    t_us         : 時間 μs（從脈衝發射後算起）
    alpha_db_per_m: 沉積物衰減係數（彰化粉砂 60 dB/m）
    回傳 0~1V（對應 -28~+92dB 總範圍 120dB，1V=120dB）
    """
    depth_m = t_us * 1e-6 * c_mps / 2.0          # 單程距離
    gain_db = alpha_db_per_m * depth_m             # 需補償的 dB
    gain_db = max(0.0, min(120.0, gain_db))        # 限幅
    return gain_db / 120.0                         # 歸一化到 0~1V
```

**AD8331 元件：**

| 元件 | 型號 | 數量 | 單價(NT$) | 來源 |
|------|------|------|----------|------|
| TGC VGA | **AD8331ARUZ** | 1 | ~NT$350 | Digi-Key / 正典UCHI |
| AD8331 評估板 | AD8331-EVALZ | 1 | ~NT$1,200 | Digi-Key（省布線）|
| 輸入匹配電感 | 47nH/0603 | 2 | NT$20 | 世代電子 |
| 旁路電容 | 100nF/0402 × 6 | 1包 | NT$15 | 世代電子 |

---

## 4. 資料擷取平台（v2.0：pic0rick RP2040）

### 4.1 平台對比（echOmods 生態系）

| 平台 | 處理器 | 採樣率 | 價格 | 備注 |
|------|--------|--------|------|------|
| un0rick | FPGA（Altera Cyclone IV）| 64 Msps | $489 | 最高效能，需 FPGA 開發 |
| lit3-32 | FPGA | 40 Msps | $330 | 中階，32 通道 |
| **pic0rick** ★ | **RP2040（雙核 ARM M0+）** | **60 Msps** | **$299** | **最優 CP 值，Python/C API** |

**★ 推薦：pic0rick（RP2040 @ 60Msps）**
- 開源硬體：https://github.com/kelu124/echomods
- 60Msps → 時間解析度 16.7ns → 深度解析度 12.5μm（水中）
- RP2040 PIO 狀態機：精確控制發射時序（2ns 解析度）
- USB CDC：直接接 RPi4 USB，無需 SPI 橋接
- Python API：`us.pulseOn(1)` → `buf = us.acquire(n=2048)`

### 4.2 RPi4 + pic0rick 整合

```python
# hull_sonar_node.py 硬體模式（未來升級）
# 當前：NMEA UART 魚探（NT$800~1,500）
# 升級：pic0rick USB（$299 ≈ NT$9,700）+ AD8331 + HV7360

import serial  # USB CDC
PIC0RICK_PORT = "/dev/ttyACM0"  # RP2040 USB CDC

class Pic0rickDriver:
    def __init__(self, port=PIC0RICK_PORT):
        self._ser = serial.Serial(port, 115200, timeout=1)

    def acquire_ascan(self, n_samples: int = 2048) -> list:
        """觸發一次 A-scan，回傳原始 ADC 數列。"""
        self._ser.write(b"ACQ\n")
        raw = self._ser.read(n_samples * 2)  # 16-bit samples
        import struct
        return list(struct.unpack(f"<{n_samples}H", raw))
```

---

## 5. 訊號處理：Hilbert 轉換包絡偵測（v2.0）

### 5.1 方法對比

| 方法 | v1.0（舊）| **v2.0（新）**|
|------|---------|------------|
| 峰值偵測 | 閾值法（baseline + 30%）| **Hilbert 包絡峰值**|
| 抗雜訊性 | 差（易受尖峰干擾）| **佳（平滑包絡）**|
| 多回波分辨 | 困難 | **容易（包絡谷值分割）**|
| 計算量 | O(n) | O(n log n)（FFT 為基礎）|

### 5.2 Hilbert 轉換實作

```python
# ultrasound_node.py 的 measure_tof() 升級版
from scipy.signal import hilbert

def envelope_peak_tof(samples: list,
                      sample_rate_hz: float = 60e6,
                      c_mps: float = 1531.0,
                      blank_us: float = 50.0) -> tuple:
    """
    Hilbert 轉換包絡偵測 → TOF 與信心值。

    samples      : 原始 ADC 數列（int or float）
    sample_rate_hz: ADC 採樣率 Hz（pic0rick = 60 Msps；MCP3208 = 1 Msps）
    blank_us     : 消隱時間 μs（忽略發射後近場反射）
    回傳 (tof_us, confidence), tof_us=-1 表示無有效回波
    """
    arr = np.array(samples, dtype=np.float64)

    # 去直流（移除基線偏移）
    arr -= arr[:min(50, len(arr)//10)].mean()

    # Hilbert 轉換 → 解析信號 → 取絕對值得包絡
    analytic = hilbert(arr)
    envelope = np.abs(analytic)

    # 消隱遮罩（忽略近場）
    blank_samp = int(blank_us * 1e-6 * sample_rate_hz)
    envelope[:blank_samp] = 0.0

    if envelope.max() < 1e-6:
        return -1.0, 0.0

    # 找包絡最大值 → TOF
    peak_idx  = int(np.argmax(envelope))
    tof_us    = peak_idx / sample_rate_hz * 1e6
    confidence = float(envelope[peak_idx] / (envelope.max() + 1e-12))

    return tof_us, min(1.0, confidence)
```

### 5.3 DMOMP 頻散補償（高精度模式）

針對泥質沉積物的頻散效應（Dispersion）補償：

```python
def dmomp_compensate(rf_data: np.ndarray,
                     freq_hz: float = 200e3,
                     c_water: float = 1531.0,
                     c_sed: float = 1480.0) -> np.ndarray:
    """
    DMOMP（Dispersion Model for Oceanic Mud/Particle）頻散補償。
    文獻精度：98.7% 正確定位（模擬測試）。
    rf_data: 原始 A-scan 時域序列
    回傳: 頻散補償後的序列
    """
    N = len(rf_data)
    F = np.fft.rfft(rf_data)
    freqs = np.fft.rfftfreq(N)          # 歸一化頻率

    # 頻率依賴聲速（泥質沉積物線性頻散近似）
    # c(f) = c0 + dc/df × (f - f0)，典型 dc/df = +0.5 m/s per kHz
    delta_c = 0.5e-3 * (freqs * freq_hz - freq_hz) / 1e3  # m/s
    c_f = c_sed + delta_c

    # 頻域相位修正（在 c_sed 基礎上補回頻散相位）
    phase_corr = np.exp(1j * 2 * np.pi * freqs * N * (1/c_sed - 1/c_f))
    F_corr = F * phase_corr

    return np.fft.irfft(F_corr, n=N)
```

---

## 6. 安裝位置與機械設計

### 6.1 換能器安裝在手臂末端

```
安裝方式：
  - 換能器固定於手臂末端探針管外側（探針管 Φ8mm）
  - J4_TILT = -30°（IEEE 7890758 最佳入射角：30° off-normal）
  - 固定夾：PP 3D 列印，M2.5 不鏽鋼螺絲

聲學耦合：
  - 換能器正面塗抹超音波耦合劑（Aquasonic 100，接觸泥面前）
  - 或在換能器面加裝 1.625mm 聚氨酯（PU）聲學窗（λ/4 匹配層）

電纜走線：
  - RG-174 同軸電纜（外徑 2.8mm）沿手臂連桿溝槽走線
  - 電纜長度：手臂末端 → 主控艙，約 600mm
  - 入艙使用 M12 BNC 防水穿牆頭
```

### 6.2 水下換能器方向

```
量測時最佳手臂姿態（arm_deploy_action.py ultrasound()）：
  J1 = 0°（朝洞口方向）
  J2 = 90°（肩部直立）
  J3 = 120°（肘部前折）
  J4_TILT = -30°（換能器前傾 30°，最佳洞穴空腔偵測角度）
```

---

## 7. 硬體測試步驟

### 7.1 台灣現場條件（彰化粉砂）

```
水溫：25°C（夏季潮間帶）
鹽度：32 ppt
聲速：c = 1531 m/s（Medwin 1975 公式）
TVG 係數：60 dB/m（泥50~砂102 之間估計）
```

### 7.2 測試流程

```
Step 1: 水桶中安裝換能器，距離桶底 30cm
Step 2: 執行 ultrasound_node.py（simulate=False，Hilbert 模式）
Step 3: 預期回波 TOF = 300mm × 2 ÷ 1531m/s ≈ 392μs
Step 4: 包絡峰值 SNR > 20dB → 合格

Step 5: 泥沙穿透測試
  準備：海水飽和泥沙（粗沙 70% + 細沙 20% + 黏土 10%）
  填裝高度：30cm（模擬淺洞）
  換能器接觸泥面，J4_TILT = -30°
  預期：包絡峰值在 392~450μs（依泥沙聲速而定）
```

---

## 8. 採購清單（v2.0 Level 2 DIY 升級方案）

> 此為 **Level 2 研究升級方案**，適用固定著陸器環境。
> 當前 Archimedes 採 Level 1：釣魚魚探（NT$1,150）。
> Level 2 提升至 Raw RF A-scan 資料，啟用 T-SAFT 重建。

| 項目 | 型號 | 數量 | 估計單價(NT$) | 來源 |
|------|------|------|--------------|------|
| **換能器** | PZT-5A Φ20mm 200kHz | 1 | ~1,200 | CTDCO（台灣陶瓷公司）|
| **脈衝 IC** | HV7361LA-G（含 Active RTZ）| 1 | ~500 | Digi-Key / 欣晟電子 |
| **TGC VGA** | AD8331-EVALZ（評估板）| 1 | ~1,200 | Digi-Key |
| **DAQ 板** | pic0rick（RP2040, 60Msps）| 1 | ~9,700 | echomods.github.io |
| 高壓升壓模組 | 12V → ±100V | 1 | ~200 | 蝦皮 |
| RG-174 同軸電纜 + BNC | 1m + 接頭×2 | 1組 | 150 | 露天 |
| M12 BNC 防水穿牆頭 | 面板安裝型 | 1 | 120 | 蝦皮 |
| PU 聲學窗（λ/4 匹配）| Φ20mm × 1.625mm | 1 | ~100 | 創維塑膠（台中）|
| 環氧阻尼膠（後置）| 環氧+鎢粉（自配）| 1組 | ~200 | 蒂沅臻（marine epoxy）|
| **Level 2 合計** | | | **~NT$13,370** | |

> Level 1（釣具行魚探）：NT$1,150 → 適合棲地調查（計數/GPS）
> Level 2（pic0rick + HV7361 + AD8331）：NT$13,370 → 適合學術 A-scan 研究

---

## 9. 探頭自製完整流程（Level 2 DIY 組裝）

> 此章節對應報告〈潮間帶與底棲生態探查之高精度自製超音波感測系統〉的
> Step 1～Step 3，整合聲學物理與機構設計。

### 9.1 Step 1 — 壓電陶瓷焊接

**材料：**
- PZT-5A 圓片 Φ20mm（CTDCO，兩面銀電極）
- RG-174 同軸電纜（外徑 2.8mm）
- 低溫焊錫 Sn58Bi（熔點 138°C）
- 溫控電烙鐵（設定 220°C）
- 含膠熱縮套管 Φ3mm

**流程：**

```
① 確認電極面：PZT 正面（聲波發射面）= 接芯線，背面 = 接屏蔽網
② 電烙鐵預熱至 220°C，錫量最小化（約半顆芝麻大）
③ 芯線焊接：快速接觸正面電極，<3 秒完成，立即移開烙鐵
④ 屏蔽網焊接：背面電極同樣 <3 秒
⑤ 冷卻 30 秒後，套上含膠熱縮管，熱風槍 80°C 收縮固定

⚠️  居禮溫度警告：PZT-5A Tc = 365°C，焊接若超過 5 秒局部過熱
    即可能導致去極化（壓電特性永久喪失），無法修復。
    判斷是否去極化：以 LCR 表量測電容，若比出廠規格低 >30%，
    表示已去極化（需丟棄重購）。
```

**驗證：**
```python
# 用 LCR 表（100kHz）量測電容
# PZT-5A Φ20mm 正常值：約 1200~2000 pF
# < 800 pF → 去極化 → 丟棄
```

---

### 9.2 Step 2 — 聲學匹配層與背襯層製備

#### 匹配層（前方，最大化聲能穿透）

**目標阻抗：Z_m = √(Z_PZT × Z_water) = √(34 × 1.5) ≈ 7.1 MRayl**

| 材料組合 | 密度調配 | 聲阻抗 | 備注 |
|---------|---------|--------|------|
| 環氧 + 空心玻璃微珠 | ρ = ρ_glass×Vf + ρ_epoxy×(1-Vf) | 可調至 7 MRayl | 推薦方案 |
| 聚氨酯（PU）板材 | 不需調配 | ~5 MRayl | 次優，創維塑膠購買 |
| 鋁片 | 固定 | ~17 MRayl | 偏高，可加工打薄 |

**空心玻璃微珠調配步驟（目標 ρ_composite ≈ 1.4 g/cm³）：**
```
環氧樹脂 A+B 混合（1:1）→ 加入空心玻璃微珠（3M S38 型）
體積分率 Vf = 0.35（即 35% 體積比微珠）
→ ρ_composite = 0.38×0.35 + 1.25×0.65 ≈ 0.95 g/cm³（過輕）

Vf = 0.20（20% 微珠）
→ ρ_composite = 0.38×0.20 + 1.25×0.80 ≈ 1.08 g/cm³
→ Z ≈ 1.08 × 2200（環氧聲速）/ 1e6 ≈ 2.4 MRayl（仍偏低）

→ 需加入少量鎢粉提高密度（目標 Z = 7 MRayl）
→ 建議直接採用 PU 板材（1.625mm）最省事
```

**塗佈與厚度控制：**
```
λ/4 厚度 = c_material / (4 × f0)
PU（c≈2000m/s）：λ/4 = 2000/(4×200e3) = 2.5mm
環氧（c≈2500m/s）：λ/4 = 2500/(4×200e3) = 3.1mm

使用游標卡尺量測，打磨至目標厚度 ±0.1mm
```

#### 背襯層（後方，吸收後向聲波）

```
材料：Marine Epoxy（蒂沅臻）+ 鎢粉（W 粉，五金行）
比例：環氧樹脂 100g + 鎢粉 150g（重量比），ρ ≈ 5 g/cm³
目標 Z_back ≥ Z_PZT（34 MRayl）→ 有效阻尼 PZT 餘震
灌注厚度：≥ 10mm

功能驗證：無背襯層 → 脈衝拖尾（脈衝長度 > 20μs）
         有背襯層 → 脈衝縮短至 5μs 以下（解析度提升 4×）
```

---

### 9.3 Step 3 — 防水外殼封裝

**外殼材料：**
- 厚壁 PVC 管（OD 50mm，壁厚 5mm，創維塑膠）
- 3D 列印 PETG 內部支架（固定 PCB + 電池模組）
- M12 BNC 防水穿牆頭（IP68）
- Marine Epoxy（蒂沅臻）

**封裝流程：**

```
① 3D 列印支架裝入 PVC 管，固定 HV7360 板 + AD8331 板 + pic0rick

② 前端（換能器開口）：
   - PZT + 匹配層居中置入管口凹槽
   - 圍邊塗抹 Marine Epoxy，確保無氣泡
   ⚠️  氣泡 = 100% 聲能反射 → 探頭失效
   - 固化 24h（20°C）

③ 電纜出口（後端）：
   - RG-174 穿過 M12 Cable Gland
   - 含膠熱縮套管套住電纜出口處
   - Gland 鎖緊 → IP68

④ 後端封蓋：
   - Silicone 填縫 + Marine Epoxy 封蓋
   - Gore-Tex 通氣閥（防壓差破壞 O-ring，選配）

⑤ 水密測試：
   - 沉入 30cm 水桶，靜置 30 分鐘
   - 取出後拆開，確認內部無水漬
```

**聲學耦合確認：**
```python
# 在水桶中量測，距離桶底 30cm
# 預期 TOF = 0.30m × 2 / 1531m/s × 1e6 ≈ 392μs
# Hilbert 包絡 SNR > 20dB → 匹配層有效
```

---

## 10. DMOMP 頻散補償啟用指南

### 10.1 何時啟用

| 條件 | 建議 |
|------|------|
| Level 1（魚探）| 不適用（不輸出 Raw RF）|
| Level 2 + MCP3208（1 Msps）| 可選，效益有限 |
| **Level 2 + pic0rick（60 Msps）** | **強烈建議啟用** |
| 深度 > 50cm | 效益最顯著（頻散累積明顯）|

### 10.2 ROS2 啟動參數

```bash
# 啟用 DMOMP（pic0rick 模式）
ros2 run archimedes_survey acoustic_processor \
  --ros-args \
  -p dmomp_enabled:=true \
  -p dmomp_beta:=12.0 \
  -p dmomp_sample_rate:=60000000.0

# 同步啟用 pic0rick
ros2 run archimedes_survey ultrasound_node \
  --ros-args \
  -p simulate:=false \
  -p use_pic0rick:=true \
  -p pic0rick_port:=/dev/ttyACM0
```

### 10.3 β 參數調校

`dmomp_beta`（m/s）= 頻散係數，依沉積物類型調整：

| 底質 | β 建議值 | 說明 |
|------|---------|------|
| 純沙（低頻散）| 3~5 | 顆粒均勻，頻散弱 |
| 彰化粉砂（混合）| **15~20** | 泥沙混合，中等頻散（原估值 10~15 偏低）|
| 軟泥（高頻散）| 20~28 | 高含水量黏土，頻散強 |

調校方式：對已知深度目標（如水桶測試）量測殘差，調整 β 使 TOF 誤差最小。

---

> ⚠️ **彰化粉砂穿透深度限制（重要）**
>
> 在鹿港/彰化海岸粉砂（孔隙度 φ ≈ 0.60–0.72）中，200kHz 超音波的**有效穿透深度約為 10–15 cm**（衰減係數 α ≈ 0.5–2.0 dB/cm，SNR 降至 10dB 以下即無效）。
>
> **美食奧螻蛄蝦（Austinogebia edulis）的洞穴深度可達 30–60 cm**（Lee & Chao 2003）。
>
> 因此，聲學探測**只能確認洞口是否存在（頂部 10–15cm 段）**，無法描繪完整洞穴形態。洞穴深度量測需依靠機械臂探針（endoscope probe）搭配人工挖掘驗證。

---

## 11. 台灣供應商聯絡

| 廠商 | 供應品項 | 備注 |
|------|---------|------|
| **CTDCO（台灣陶瓷公司）** | PZT-5A 圓片/方片 | 需電話詢價，最小訂量 10 片 |
| **欣晟電子**（台北士林）| HV7360/HV7361, MOSFET, 電容 | Yahoo 拍賣亦有 |
| **正典UCHI**（台北）| AD8331, 信號處理 IC | 代理 Analog Devices |
| **創維塑膠**（台中）| 聚氨酯（PU）板材 | 可切割 λ/4 厚度 |
| **蒂沅臻**（網路）| 防水環氧樹脂 | Marine-grade，耐鹽水 |
