# Phase 3 列印清單 — Bambu Studio 參數

更新：2026-03-17
機器：Bambu Lab（熱床 220×220mm）
材料：PETG（主要）/ PLA（非結構件）

---

## 列印優先順序

| 優先 | 零件 | STL 路徑 | 說明 |
|------|------|----------|------|
| ★★★ | arm_transducer_mount | stl/arm/arm_transducer_mount.stl | 換能器座，須先到位才能測試超音波 |
| ★★★ | motor_backplate_nema23 | stl/motor_backplate_nema23.stl | 結構件，固定馬達 |
| ★★  | arm_upper_arm_oct | stl/arm/arm_upper_arm_oct.stl | 八角上臂，替換矩形版 |
| ★★  | arm_forearm_oct | stl/arm/arm_forearm_oct.stl | 八角前臂，替換矩形版 |
| ★   | platform_rib | stl/platform_rib.stl | 加強肋，可暫緩（平台先測試） |

---

## 各零件 Bambu Studio 詳細設定

### 1. arm_upper_arm_oct.stl — 八角空心上臂

| 參數 | 值 | 原因 |
|------|---|------|
| 材料 | PETG | 潮間帶防水、韌性 |
| 列印方向 | **長軸水平（躺平）** | 避免層間剝離（承彎矩方向） |
| Layer Height | 0.2mm | 強度 / 速度平衡 |
| Wall Loops | 4 | 壁厚 3.5mm ≈ 4× 0.4mm 噴嘴寬 |
| Infill | **40% Gyroid** | 各向同性強度最佳 |
| Support | 無 | 八角截面無懸挑 |
| 熱床溫度 | 70°C | PETG 標準 |
| 噴嘴溫度 | 235°C | |
| 估計重量 | ~22g | |
| 尺寸確認 | 200mm × 29mm × 29mm | 長軸 < 220mm 熱床限制 ✓ |

> **重要**：列印前在 Bambu Studio 確認長軸平行 X 或 Y 方向（不要直立）。

---

### 2. arm_forearm_oct.stl — 八角空心前臂

| 參數 | 值 | 原因 |
|------|---|------|
| 材料 | PETG | |
| 列印方向 | **長軸水平（躺平）** | |
| Layer Height | 0.2mm | |
| Wall Loops | 4 | 壁厚 3.5mm |
| Infill | **40% Gyroid** | |
| Support | 無 | |
| 尺寸確認 | 220mm × 25mm × 25mm | 剛好 220mm，貼邊 ⚠️ 留意翹曲 |

> **注意**：220mm 剛好塞滿熱床。建議開啟 Brim（5mm）防翹曲，列印後切除。

---

### 3. arm_transducer_mount.stl — 換能器座

| 參數 | 值 | 原因 |
|------|---|------|
| 材料 | PETG | 防水，耐鹽蝕 |
| 列印方向 | 底座朝下（底面貼熱床） | 底部接合面平整 |
| Layer Height | 0.15mm | 夾環圓弧精度 |
| Wall Loops | 4 | |
| Infill | 30% Cubic | 此件結構複雜，Cubic 較好支撐 |
| Support | **需要**（MG90S 伺服腔體懸挑） | 選 Tree Support，移除易 |
| Support 間距 | 0.2mm | |
| 後處理 | O-Ring 溝槽用 Ø22mm × 2.5mm O-Ring | |
| 估計重量 | ~38g（含支撐廢料移除前） | |

> **後處理步驟**：
> 1. 移除 Tree Support
> 2. 夾環內孔用 M10 鑽頭修整至 Ø20.5mm（換能器滑入）
> 3. O-Ring 溝槽塗矽脂安裝
> 4. M3×10 鎖緊螺絲 × 2

---

### 4. motor_backplate_nema23.stl — 馬達背板

| 參數 | 值 | 原因 |
|------|---|------|
| 材料 | PETG | |
| 列印方向 | 平放（8mm 厚面朝上） | 最大強度方向平行層方向 |
| Layer Height | 0.25mm | 此件無精度需求，快速列印 |
| Wall Loops | 4 | |
| Infill | 50% Rectilinear | 背板需剛性 |
| Support | 無 | 平板件無懸挑 |
| 後處理 | **Ø5.5mm 鑽頭貫穿 8 個導孔（M5 過孔）** | 導孔 Ø5.5mm = M5 clearance |
| 後處理 | 軸孔 Ø22mm 通孔用 Ø22mm 鑽頭或銼刀修整 | 聯軸器空間 |
| 尺寸 | 386mm × 86mm × 8mm | 分兩件列印（前段 + 後段） |

> ⚠️ **分件列印**：386mm 超過熱床。前段（Z 中心 ≈ -136mm 位置）、後段（Z 中心 ≈ 586mm 位置）各列印一片，共 2 片。

---

### 5. platform_rib.stl — 平台橫向加強肋

| 參數 | 值 | 原因 |
|------|---|------|
| 材料 | PETG | |
| 列印方向 | 肋板直立（20mm 高面朝上） | |
| Layer Height | 0.2mm | |
| Wall Loops | 3 | |
| Infill | 30% Gyroid | 輕量化 |
| Support | 無 | |
| 數量 | **6 片**（一次列印 2~3 片，並列擺放） | 各片尺寸 300×20×12mm |
| 組裝 | M4×12 螺絲固定至平台底面 × 2 孔/片 | |

---

## 耗材估算（Phase 3 新增）

| 零件 | 重量 | PETG 用量（1.75mm, 密度 1.27g/cc） |
|------|------|-----------------------------------|
| arm_upper_arm_oct | 22g | ~17m |
| arm_forearm_oct | 18g | ~14m |
| arm_transducer_mount | 35g | ~28m |
| motor_backplate_nema23 × 2 | 85g | ~67m |
| platform_rib × 6 | 54g | ~42m |
| **合計** | **214g** | **~168m** |

> PETG 1kg 線材（蝦皮均價 NT$450）= 約 330m，Phase 3 用量約佔 **0.65 捲**，耗材成本約 **NT$290**。

---

## 組裝順序建議

1. 列印 `motor_backplate_nema23` × 2 → 鑽孔 → 安裝到 NEMA23 馬達
2. 列印 `platform_rib` × 6 → M4 固定到平台底面
3. 列印 `arm_upper_arm_oct` + `arm_forearm_oct` → 替換原矩形連桿
4. 列印 `arm_transducer_mount` → O-Ring 安裝 → MG90S 伺服裝入 → 換能器固定
5. 連接 MG90S → PCA9685 ch3 → 接線圖參考 `docs/wiring_diagram.md`

---

## 採購補充（Phase 3 小物件）

| 物件 | 規格 | 數量 | 建議來源 | 估價 |
|------|------|------|----------|------|
| MG90S micro servo | 9g，1.8kg·cm | 1 | 宇倉電子 / 露天 | NT$75 |
| O-Ring | Ø22mm × 2.5mm（NBR 耐油） | 2 | 橡膠零件行 / 蝦皮 | NT$30 |
| M3×10 不鏽鋼內六角 | | 10 | 五金行 | NT$30 |
| M4×12 不鏽鋼內六角 | | 16 | 五金行 | NT$35 |
| PETG 線材 1kg | 任意廠（Bambu 原廠最穩定）| 1 | 蝦皮 | NT$450 |
| **Phase 3 小物合計** | | | | **NT$620** |

（PETG 線材 NT$450 為一次性耗材投資，後續列印均可使用）
