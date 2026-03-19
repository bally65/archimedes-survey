# 開發日誌 — 2026-03-19

## 工作摘要

本次 session 與 Gemini CLI 協作，根據優先順序實作三個模組。

---

## Gemini 優先排序結果

| 排名 | 項目 | 理由 |
|------|------|------|
| 1 | 視覺伺服節點 | 閉環控制的關鍵缺口，決定 C-scan 能否精準定位 |
| 2 | validate_density.py | 科學信效度基礎，無此無法發表成果 |
| 3 | fleet_coordinator.py | 5 台艦隊效率提升，後勤優化 |
| 4 | 快拆螺旋 CAD | 軟體穩定後再優化實體 |
| 5 | 公民科學 APP | 影響力擴展，非當前關鍵路徑 |

---

## 已實作模組

### 1. `ros2/archimedes_survey/visual_servo_node.py`

ROS2 視覺伺服節點（IBVS 簡化版）。

**功能：**
- 訂閱 `/camera/burrow_detections`（JSON 陣列，含 `distance_m`, `center_world_m`, `confidence`）
- 比例控制器：`v = Kp * (dist - 0.35m)`，角速度 `w = -Kp_ang * atan2(dx, dy)`
- 距離 < 0.35m → 發布 `/visual_servo/aligned = True`，停止輸出
- 超過 5s 無偵測 → `aligned = False`
- 距離 > 1.5m → 不伺服（避免大幅移動）
- 發布 `/visual_servo/status`（JSON，含 state/dist_m/cmd 等除錯資訊）

**Gemini 修正後新增（第二輪）：**
- `min_confidence` 參數（預設 0.6）：過濾反光/陰影誤判
- 洞穴鎖定機制（`_locked_burrow_id`）：防止多洞穴場景中擺盪
- `lock_hysteresis_m`（預設 0.15m）：切換目標需新目標近 15cm 以上
- 鎖定邏輯：維持當前目標，除非有顯著更近的洞穴

**ROS2 參數：**
```
arm_reach_m:         0.35   # 對準距離閾值
max_approach_m:      1.5    # 最大伺服距離
kp_linear:           0.3    # 線速度增益
kp_angular:          0.8    # 角速度增益
max_v:               0.10   # 最大線速度 m/s
max_w:               0.20   # 最大角速度 rad/s
no_detect_timeout_s: 5.0    # 無偵測超時
min_confidence:      0.6    # 最低信心閾值
lock_hysteresis_m:   0.15   # 目標切換遲滯
```

---

### 2. `scripts/validate_density.py`

機器人密度估計統計驗證工具。

**功能：**
- 載入機器人 GeoJSON（`~/archimedes_missions/*_burrows.geojson`）
- 載入人工樣方 CSV（`lat,lon,manual_count,area_m2`）
- Haversine 配對（搜尋半徑預設 5m）
- 計算四項指標：
  - RMSE / MAE（誤差量）
  - Pearson r / r²（相關性，scipy.stats）
  - Bland-Altman（偏差 + 95% LoA，系統性高估/低估）
  - Moran's I（空間自相關；PySAL 或手動反距離加權）
- 輸出 `validation_report.json`
- 可選 matplotlib 圖表（`--plot`）：Bland-Altman 圖 + 散佈圖

**使用：**
```bash
python scripts/validate_density.py \
  --quadrat data/manual_quadrats.csv \
  --missions ~/archimedes_missions \
  --radius 5.0 \
  --plot \
  --output validation_report.json
```

---

### 3. `control/fleet_coordinator.py`

5 台機器人艦隊任務協調器。

**功能：**
- Zone-aware 貪婪分配（`data/priority_cells.json` → `data/robot_N_waypoints.json`）
- 按地理區域分組，確保同一台機器人主要在同一 zone 作業（減少移動距離）
- 輸出 `data/fleet_summary.json`（分配摘要）
- 任務時間估算：移動 + SETTLING(3min) + 掃描(2min) 每格
- LoRa 心跳模擬（`--simulate-heartbeat`）：30s 廣播位置 + 電量
- 碰撞迴避：兩台 < 10m → 低 ID 機器人暫停 60s
- 動態接管（`check_retask()`）：逾時失聯機器人的剩餘格網自動轉交最近可用機器人

**使用：**
```bash
python control/fleet_coordinator.py \
  --cells data/priority_cells.json \
  --n-robots 5 \
  --output-dir data \
  --simulate-heartbeat
```

---

## Gemini 第二輪建議（已/未處理）

| 建議 | 狀態 | 備注 |
|------|------|------|
| 多洞穴擺盪防護 | ✅ 已實作 | `_locked_burrow_id` + `lock_hysteresis_m` |
| 低信心過濾 | ✅ 已實作 | `min_confidence=0.6` 參數 |
| 動態接管機制 | ✅ 已實作 | `check_retask()` 方法 |
| PI 控制器（線速度） | 未實作 | 記錄於 TODO，目前 P 控制夠用 |
| 潮汐感知調度 | 未實作 | 需整合 `tide_predictor.py`，下一 sprint |
| LoRa 心跳 → 本地反應式避障 | 未實作 | 正確觀察：10m 避障不能依賴 30s 心跳；需 VFH/DWA，下一 sprint |
| LISA（局部空間自相關） | 未實作 | 列入 validate_density v2 |
| 零膨脹處理（r² 計算） | 未實作 | 列入 validate_density v2 |
| 格網連通性檢查 | 未實作 | fleet_coordinator v2 需加圖論連通驗證 |

---

## 新想法（討論中浮現）

1. **潮汐整合優先度計算**：`fleet_coordinator` 應讀取 `tide_predictor.py` 的 `next_flood_min`，
   動態調整格網優先度——即將被淹沒的格網排序前移，確保低潮窗口充分利用。

2. **Visual Servo → SETTLING 銜接**：`auto_navigate.py` 目前監聽 `/auto_command`，
   可新增：當 `/visual_servo/aligned = True` 時，自動切入 `SETTLING` 模式。
   需在 `auto_navigate.py` 增加一個 aligned 訂閱。

3. **Burrow Identity Error 指標**：`validate_density.py` 未來應加入
   Y 型洞穴配對錯誤率（`burrow_identity_error`）——當兩個開口被誤認為兩隻個體時的錯誤比例，
   這是本系統獨有的誤差來源（非標準生態調查指標）。

---

## 待完成（Next Sprint）

- [ ] `auto_navigate.py`：訂閱 `/visual_servo/aligned`，aligned=True 時自動切 SETTLING
- [ ] `validate_density.py`：加入 LISA 局部熱點分析、零膨脹樣本分離
- [ ] `fleet_coordinator.py`：整合 `tide_predictor.py`（任務窗口感知）
- [ ] `fleet_coordinator.py`：格網連通性驗證（避免跨深水區路徑）
- [ ] `visual_servo_node.py`：PI 控制器升級（消除穩態誤差）
- [ ] 野外測試：收集真實洞穴偵測數據，驗證 `min_confidence=0.6` 閾值是否合適
- [ ] 建立 `data/manual_quadrats.csv` 範例（n=30 樣方，用於 validate_density 首次測試）
