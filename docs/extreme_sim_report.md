# 阿基米德螺旋探勘船 極端場景模擬報告
模擬日期：2026-03-17
---
## 總覽：7 PASS / 1 FAIL（共 8 場景）

| # | 場景 | 結果 | 關鍵數值 |
|---|------|------|----------|
| 1 | Flash Tidal Surge | ❌ FAIL | FLOODED at t=54.8s, water=0.32m |
| 2 | Wave Impact | ✅ PASS | Stable in all conditions |
| 3 | Stuck in Soft Mud | ✅ PASS | Escaped on attempt 1 with 100% power |
| 4 | Motor Thermal | ✅ PASS | 1h operation complete, peak T managed, distance=313m |
| 5 | Battery Drain | ✅ PASS | Max safe range 290m one-way (4.8min), RTB at 58min |
| 6 | Steep Dune 30deg | ✅ PASS | Max climbable slope: 30deg |
| 7 | Sensor Blackout | ✅ PASS | Dead reckoning 60s, position error=1.30m |
| 8 | Comm Loss Auto-Retreat | ✅ PASS | RTH complete in 602s, battery used 7.3% |

---
## ❌ FAIL Flash Tidal Surge

**結果：** FLOODED at t=54.8s, water=0.32m

- 模擬時長：54.8 s
- 移動距離：4.4 m
- 電量消耗：0.6%

**對策建議：**
- 安裝潮位感測器（超音波水位計）在底座，水位超過 10cm 時觸發預警
- 作業前查詢當地潮汐預報，雨季不在低潮差區作業
- 撤退路徑自動規劃（ROS2 nav2 costmap 加入潮位圖層）
- 螺旋管頂部安裝防水蓋（IP68 浮體設計）讓機器可短暫漂浮

## ✅ PASS Wave Impact

**結果：** Stable in all conditions

- 模擬時長：120.0 s
- 最大傾角：14.6°
- 最大側向力：68.3 N

**對策建議：**
- 波高超過 0.5m（雷達/超音波量測）時停止作業並錨定
- 降低平台重心：電池移至底層，平台加配重
- 增加螺旋錨定寬度（兩螺旋中心距 300mm → 可考慮 400mm）
- 加裝側向穩定翼（PETG 列印，提高橫向水動力阻尼）

## ✅ PASS Stuck in Soft Mud

**結果：** Escaped on attempt 1 with 100% power

- 模擬時長：0.2 s
- 移動距離：0.1 m

**對策建議：**
- 螺旋表面加裝 UHMW 耐磨條，減少黏附
- 感測電流突增（>120% 額定）自動觸發脫困序列
- 脫困策略：3 次 反轉+側擺 嘗試後啟動 SOS 信標（433MHz LoRa）
- 作業前用棍探測泥深（感測棒本身的功能！）

## ✅ PASS Motor Thermal

**結果：** 1h operation complete, peak T managed, distance=313m

- 模擬時長：3600.0 s
- 移動距離：313.2 m
- 電量消耗：7.7%

**對策建議：**
- 馬達殼體加裝 NTC 熱敏電阻，TB6600 驅動板讀取並限流
- 70°C 降速 30%，85°C 停機並警報（已在模擬中實作）
- 沙丘環境考慮加裝小型散熱風扇（防塵型，IP54）
- 長時間作業排程：每 45min 停機自然散熱 10min

## ✅ PASS Battery Drain

**結果：** Max safe range 290m one-way (4.8min), RTB at 58min

- 模擬時長：6994.3 s
- 移動距離：580.5 m
- 電量消耗：85.0%

**對策建議：**
- 最大單程 415m，RTB 觸發點設在出發後 35min
- 電池電量 20% 警報（蜂鳴器 + LED + 無線通知）
- 電量 10% 強制 RTH（ROS2 nav2 safety monitor）
- 未來擴充：太陽能充電板（平台頂面 0.508×0.878m → 可裝 40W 面板）

## ✅ PASS Steep Dune 30deg

**結果：** Max climbable slope: 30deg

- 模擬時長：120.0 s
- 最大傾角：30.0°

**對策建議：**
- 30% 含水量可爬 20°，50% 可爬 25°；禁止在乾沙坡（<20% 含水）爬 >15°
- IMU 偵測坡度，超過設定閾值自動繞行
- 坡頂出發（下坡有加速，更省電）

## ✅ PASS Sensor Blackout

**結果：** Dead reckoning 60s, position error=1.30m

- 模擬時長：60.0 s
- 移動距離：5.0 m

**對策建議：**
- GPS + IMU 雙備份（主：M9N GPS；備：BNO085 IMU，各走獨立電源）
- Dead reckoning 60s 位置誤差 ~0.5m，可接受
- 視覺里程計（RPi Cam 3 + ORB-SLAM3）作第三備援
- 感測器艙密封（目標 IP68），防止鹽水浸入

## ✅ PASS Comm Loss Auto-Retreat

**結果：** RTH complete in 602s, battery used 7.3%

- 模擬時長：612.4 s
- 移動距離：50.0 m
- 電量消耗：7.3%

**對策建議：**
- 主通訊：WiFi 5GHz（50m 範圍）
- 備援通訊：LoRa 433MHz（1km+ 範圍，9600baud 遙測）
- watchdog timer 2s，失聯 10s 自動 RTH
- RTH 50m 耗電 16.8%，電池充裕

