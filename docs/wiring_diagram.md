# 阿基米德探勘船 GPIO 接線圖

## 主控板：Raspberry Pi 4B (BCM 腳位編號)

---

## 1. 步進馬達驅動器 TB6600 × 4

每條螺旋用 2 顆 TB6600（前馬達 + 後馬達），共 4 顆。

### DIP 撥碼開關設定（NEMA23 57BYGH78）
| SW1 | SW2 | SW3 | 細分 | SW4 | SW5 | SW6 | 電流 |
|:---:|:---:|:---:|:----:|:---:|:---:|:---:|:----:|
|  0  |  0  |  1  | 1/1  |  1  |  1  |  0  | 2.0A |

> 細分選 1/1（全步）= 最大扭力；電流選 2.0A = 57BYGH78 額定電流

### 接線（RPi4 → TB6600）

| 功能 | TB6600 端子 | RPi4 BCM | 說明 |
|------|------------|----------|------|
| 螺旋A前 DIR | ENA+ / DIR+ | GPIO **17** | 方向信號 |
| 螺旋A前 PUL | PUL+ | GPIO **27** | 脈衝信號 |
| 螺旋A後 DIR | DIR+ | GPIO **10** | 方向信號 |
| 螺旋A後 PUL | PUL+ | GPIO **9** | 脈衝信號 |
| 螺旋B前 DIR | DIR+ | GPIO **22** | 方向信號 |
| 螺旋B前 PUL | PUL+ | GPIO **23** | 脈衝信號 |
| 螺旋B後 DIR | DIR+ | GPIO **11** | 方向信號 |
| 螺旋B後 PUL | PUL+ | GPIO **25** | 脈衝信號 |
| 所有 ENA+ | ENA+ | 3.3V | 永久致能 |
| 所有負端 | ENA- / DIR- / PUL- | GND | 共地 |

> 注意：RPi4 GPIO 輸出 3.3V，TB6600 訊號輸入接受 3.3V，**不需要電位轉換器**

### TB6600 電源端子
| 端子 | 連接 |
|------|------|
| VCC+ | 24V 電源正極 |
| GND  | 24V 電源負極 |
| A+/A-| NEMA23 線圈 A（通常紅/白） |
| B+/B-| NEMA23 線圈 B（通常藍/黑） |

---

## 2. PCA9685 伺服驅動板 (I2C)

| 功能 | PCA9685 端子 | RPi4 BCM | 說明 |
|------|-------------|----------|------|
| 資料 | SDA | GPIO **2** (SDA1) | I2C1 |
| 時鐘 | SCL | GPIO **3** (SCL1) | I2C1 |
| 電源 | VCC | 3.3V | 邏輯電源 |
| 地線 | GND | GND | |
| 伺服電源 | V+ | 5V（外部 BEC） | 伺服馬達電源，**不可用 RPi 5V** |

| 通道 | 連接 |
|------|------|
| CH0  | J1 Yaw — MG996R |
| CH1  | J2 Shoulder — DS3225MG |
| CH2  | J3 Elbow — MG996R |
| CH3  | J4 Tilt (換能器俯仰) — MG90S（Phase 3 新增）|

---

## 3. ATGM336H GPS 模組 (UART)

| GPS 端子 | RPi4 BCM | 說明 |
|---------|----------|------|
| TX | GPIO **15** (RXD0) | GPS TX → RPi RX |
| RX | GPIO **14** (TXD0) | GPS RX → RPi TX |
| VCC | 3.3V | |
| GND | GND | |

> `/boot/config.txt` 需加 `enable_uart=1`，並停用 serial console：
> `sudo raspi-config → Interface Options → Serial Port → No (login shell) / Yes (hardware)`

---

## 4. BMX055 IMU (I2C，與 PCA9685 共用匯流排)

| IMU 端子 | RPi4 BCM | 說明 |
|---------|----------|------|
| SDA | GPIO **2** | I2C1（位址 0x18 加速度 / 0x68 陀螺儀 / 0x10 磁力計） |
| SCL | GPIO **3** | I2C1 |
| VCC | 3.3V | |
| GND | GND | |

---

## 5. JSN-SR04T-2.0 潮位感測器 × 2 (UART 模式)

| 感測器 | JSN 端子 | RPi4 BCM | 說明 |
|--------|---------|----------|------|
| 前置感測器 | TX | GPIO **13** | 軟體 UART 或 AMA1 |
| 前置感測器 | RX | GPIO **19** | |
| 後置感測器 | TX | GPIO **6** | 軟體 UART |
| 後置感測器 | RX | GPIO **26** | |
| 兩顆 VCC | | 5V | **需要 5V 供電** |
| 兩顆 GND | | GND | |

> 啟用第二組 UART：`/boot/config.txt` 加 `dtoverlay=uart3,txd3_pin=13,rxd3_pin=19`

---

## 6. NTC 熱敏電阻 10K × 4 馬達測溫 (需 ADC)

RPi4 沒有類比輸入，需要 **ADS1115 ADC 模組**（I2C）：

| ADS1115 端子 | RPi4 | 說明 |
|-------------|------|------|
| VDD | 3.3V | |
| GND | GND | |
| SDA | GPIO 2 | I2C1（位址 0x48） |
| SCL | GPIO 3 | I2C1 |
| A0 | NTC 螺旋A前馬達 | NTC + 10K 上拉分壓 |
| A1 | NTC 螺旋A後馬達 | |
| A2 | NTC 螺旋B前馬達 | |
| A3 | NTC 螺旋B後馬達 | |

NTC 分壓電路（每顆）：
```
3.3V ─── 10kΩ (固定) ─── AIN ─── NTC 10K ─── GND
```
溫度換算公式（Steinhart–Hart 簡化式）：
```python
import math
R_fixed = 10000
Vcc = 3.3
V_adc = adc_raw * (4.096 / 32768)   # ADS1115 gain=1
R_ntc = R_fixed * V_adc / (Vcc - V_adc)
T_K = 1.0 / (1/298.15 + math.log(R_ntc/10000)/3950)
T_C = T_K - 273.15
```

---

## 7. Ra-02 LoRa 433MHz (SPI)

| LoRa 端子 | RPi4 BCM | 說明 |
|---------|----------|------|
| NSS (CS) | GPIO **8** (CE0) | SPI0 片選 |
| SCK | GPIO **11** (SCLK) | SPI0 |
| MOSI | GPIO **10** (MOSI) | SPI0 |
| MISO | GPIO **9** (MISO) | SPI0 |
| DIO0 | GPIO **4** | 中斷（RX Done） |
| RST | GPIO **24** | 硬體重置 |
| VCC | 3.3V | **Ra-02 只能接 3.3V！** |
| GND | GND | |

> 啟用 SPI：`sudo raspi-config → Interface Options → SPI → Enable`

---

## 8. RPi Camera 3

直接插入 CSI 排線（J3 接口），不需要 GPIO。
啟用：`sudo raspi-config → Interface Options → Legacy Camera → Enable`

---

## RPi4 腳位總覽

```
                  3V3  (1) o o (2)  5V
   I2C SDA  GPIO2 (3)  o o (4)  5V
   I2C SCL  GPIO3 (5)  o o (6)  GND
  LoRa DIO0 GPIO4 (7)  o o (8)  GPIO14 TXD0 → GPS RX
              GND (9)  o o (10) GPIO15 RXD0 ← GPS TX
  TB6600 L_DIR GPIO17(11) o o (12) GPIO18
  TB6600 L_PUL GPIO27(13) o o (14) GND
  TB6600 R_DIR GPIO22(15) o o (16) GPIO23 TB6600 R_PUL
             3V3 (17) o o (18) GPIO24 LoRa RST
 SPI MOSI GPIO10(19) o o (20) GND
  SPI MISO GPIO9(21) o o (22) GPIO25 TB6600 B後 PUL
  SPI SCLK GPIO11(23) o o (24) GPIO8 SPI CE0 LoRa
              GND(25) o o (26) GPIO7
              ID SD(27) o o (28) ID SC
  JSN前 TX GPIO5(29) o o (30) GND
  JSN前 RX GPIO6(31) o o (32) GPIO12
  JSN後 TX GPIO13(33) o o (34) GND
  JSN後 RX GPIO19(35) o o (36) GPIO16
 TB6600 B後DIR GPIO26(37) o o (38) GPIO20 TB6600 A後DIR(GPIO10用)
              GND(39) o o (40) GPIO21
```

> **實際使用腳位：** GPIO 2,3,4,6,7,8,9,10,11,13,14,15,17,22,23,24,25,26,27

---

## 電源分配

```
電池 3S LiPo 12.6V
  │
  ├── 24V 升壓模組 → TB6600 × 4 VCC（馬達電源）
  │
  ├── 12V → 5V 降壓 BEC → RPi4（5V/3A）
  │                      └── PCA9685 V+（伺服電源 5V/5A）
  │
  └── 3.3V LDO (RPi 內建) → LoRa、BMX055、ADS1115、ATGM336H
```

> 重要：TB6600 馬達電源（24V）與 RPi 邏輯電源（3.3V/5V）共用 GND，
> 但**不共用 VCC**。確保所有 GND 連接在一起。

---

## 組裝檢查清單

- [ ] TB6600 DIP 撥碼：細分 1/1，電流 2.0A
- [ ] 馬達線圈極性：用三用電錶量通路確認 A+A- 和 B+B-
- [ ] LoRa 天線接好再上電（無天線燒晶片）
- [ ] 伺服電源走 BEC，不走 RPi 5V
- [ ] I2C 掃描確認：`i2cdetect -y 1` 應看到 0x18, 0x40, 0x48, 0x68
- [ ] GPS UART 測試：`cat /dev/ttyS0` 應看到 `$GNRMC,...` 字串
- [ ] NTC 分壓電阻確認 10kΩ（25°C 時 ADS1115 讀值應約在中間值）
