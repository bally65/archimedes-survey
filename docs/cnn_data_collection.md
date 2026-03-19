# 3D-CNN 訓練資料採集指南

## 資料集結構

訓練資料放在 `data/cscan_dataset/` 下，分為 4 個子目錄：

```
data/cscan_dataset/
├── C/   # Control  ：確認無洞穴的乾淨底質
├── B/   # Burrow   ：確認有望潮蟹/奧螻蛄蝦洞穴
├── O/   # Obscured ：洞口有雜物/牡蠣殼遮蔽
└── M/   # Multiple ：同一掃描範圍內有 2+ 個洞穴
```

每筆資料包含兩個同名檔案：
- `scan_YYYYMMDD_HHMMSS.npy`  — 3D 體積 numpy array，shape `(25, 25, 40)`，dtype float32，值域 [0, 1]
- `scan_YYYYMMDD_HHMMSS.json` — 對應的 metadata

### JSON 格式

```json
{
  "timestamp": "2026-05-01T09:30:00",
  "location_lat": 24.012,
  "location_lon": 120.456,
  "tide_height_m": 0.12,
  "water_temp_c": 26.5,
  "label": "B",
  "burrow_count": 1,
  "peak_depth_m": 0.35,
  "snr_db": 14.2,
  "notes": "清晰單洞，深度約 35cm"
}
```

---

## 採集流程（現場）

### 前置條件

- 退潮後 1~2 小時（底質裸露，洞口可見）
- 攜帶細竹籤（標記洞口位置，用於事後 ground truth 確認）
- GPS 記錄每個掃描點經緯度

### 步驟

1. **目視確認**：先用肉眼找到洞口，插竹籤標記
2. **部署 Archimedes**：讓船以 2Hz 掃描率在 50cm × 50cm 範圍內緩慢移動（或定點旋轉）
3. **等待自動觸發**：`acoustic_cscan` 節點在累積 ≥9 個 A-scan 後自動重建體積
4. **記錄結果**：`/acoustic/cscan_volume` 話題會發布體積 JSON
5. **存檔**：執行下方的儲存腳本

### 存檔指令

```bash
# 在 RPi4 上執行，將 cscan_volume 話題存為 npy+json
python3 scripts/record_cscan.py \
  --label B \
  --count 1 \
  --notes "清晰單洞深35cm" \
  --output data/cscan_dataset/B/
```

---

## 目標採集量

| 類別 | 最低需求 | 目標 | 說明 |
|------|---------|------|------|
| C (無洞) | 50 筆 | 100 筆 | 乾淨底質，不同底質（泥/砂/碎殼） |
| B (有洞) | 50 筆 | 100 筆 | 不同深度（10~80cm）、不同洞徑（1~3cm） |
| O (遮蔽) | 20 筆 | 40 筆 | 常見雜物：牡蠣殼、石礫、水草 |
| M (多洞) | 20 筆 | 40 筆 | 2個洞或3個洞，位置分散 |

**總計目標：≥280 筆**（論文中 A-core-2000 用約 300 筆達到 ROC-AUC 0.84）

---

## 訓練指令

```bash
# 在有 GPU 的機器上訓練（或 Google Colab）
cd archimedes-survey/ros2
python3 -c "
from archimedes_survey.burrow_cnn import train_burrow_cnn, export_onnx
model, history = train_burrow_cnn(
    data_root='../data/cscan_dataset',
    epochs=80,
    batch_size=8,
    lr=1e-3,
    output_dir='../models/'
)
export_onnx(model, '../models/burrow_cnn.onnx')
print('Done. Copy burrow_cnn.onnx to RPi4.')
"
```

### 部署到 RPi4

```bash
scp models/burrow_cnn.onnx pi@archimedes.local:/home/pi/

# 啟動時指定 ONNX 路徑
ros2 launch archimedes_survey full_system.launch.py \
  cnn_weights:=/home/pi/burrow_cnn.onnx
```

---

## 資料品質檢查

訓練前執行品質檢查腳本：

```bash
python3 scripts/check_dataset.py data/cscan_dataset/
```

會輸出：
- 各類別筆數統計
- 體積形狀一致性（所有 .npy 是否都是 `(25,25,40)`）
- SNR 分布（B 類應 > C 類均值）
- 異常值警告（全零體積、NaN）

---

## 規則推論降級模式

在採集到足夠資料、完成訓練之前，`burrow_cnn` 節點會自動使用規則推論：

```
snr_proxy = volume.max() / mean_nonzero(volume)

if snr_proxy > 5.0 → presence_prob = 0.9  (判定有洞)
if snr_proxy < 2.0 → presence_prob = 0.1  (判定無洞)
else               → presence_prob = 線性插值
```

這個規則在模擬資料中達到 SNR 區分度 >3dB，但在真實底質下需要 CNN 才能達到 ROC-AUC ≥0.80。
