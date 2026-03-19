"""
burrow_cnn.py
=============
仿 A-core-2000 雙軌制 3D-CNN：奧螻蛄蝦洞穴偵測與計數。

架構設計根據論文：
  Mizuno et al. (2024) "Combining three-dimensional acoustic coring and a
  convolutional neural network to quantify species contributions to benthic
  ecosystems"

主要差異（A-core-2000 蛤蜊 → 本系統蝦洞）：
  - 目標更大：蛤蜊 1~3cm vs 蝦洞 3~5cm 開口
  - 目標為中空：蛤蜊=高聲阻抗實心殼，蝦洞=沙-空氣界面強反射
  - 深度更深：蛤蜊 <10cm vs 蝦洞可達 80cm

雙軌模型：
  1. 存在分類器（Classifier）：體素內有無洞穴 → 二元 sigmoid
  2. 數量估計器（Counter）：洞穴數量 → 3 類（0/1/2+）softmax

特殊設計（對應論文）：
  - Z 方向強池化：洞穴反射在時間軸有長拖尾
  - 資料增強：XYZ ±20% 平移 + 翻轉
  - Stratified 5-fold CV
  - Grad-CAM 可解釋性

使用：
  # 訓練（需 PyTorch）
  python burrow_cnn.py --mode train --data /path/to/data

  # ROS2 推論節點
  ros2 run archimedes_survey burrow_cnn

  # ONNX 匯出（RPi4 部署）
  python burrow_cnn.py --mode export --weights model.pt
"""

import json
import math
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# PyTorch（選用，推論時需要）
# ─────────────────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, random_split
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# ROS2（選用）
# ─────────────────────────────────────────────────────────────────────────────
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    _ROS2_AVAILABLE = True
except ImportError:
    _ROS2_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# 模型超參數（對應論文設定）
# ─────────────────────────────────────────────────────────────────────────────
# 輸入體素大小（論文：40×40×40，本系統受限解析度縮小）
VOXEL_NX  = 25   # X 方向（50cm ÷ 2cm/格）
VOXEL_NY  = 25   # Y 方向
VOXEL_NZ  = 40   # Z 方向（0~80cm ÷ 2cm/格）

# 訓練超參數
LEARNING_RATE  = 1e-4    # 論文用 1e-4（Adam）
BATCH_SIZE     = 4       # 論文用 2~3（記憶體限制）
MAX_EPOCHS_CLS = 50      # 分類器最大 Epoch
MAX_EPOCHS_CNT = 30      # 計數器最大 Epoch
EARLY_STOP_PAT = 10      # 若 10 Epoch 無改善則停止

# 推論閾值
PRESENCE_THRESHOLD = 0.50  # 洞穴存在機率閾值

# 訓練數據集類型（對應論文 Type C/A/M/AM）
DATASET_TYPES = {
    "C": "純沙（無洞穴）",
    "B": "有奧螻蛄蝦洞穴",
    "O": "其他生物洞穴（螃蟹/彈塗魚等）",
    "M": "混合（多個洞穴）",
}


# ─────────────────────────────────────────────────────────────────────────────
# 3D-CNN 模型
# ─────────────────────────────────────────────────────────────────────────────
if _TORCH_AVAILABLE:

    class BurrowDetector3DCNN(nn.Module):
        """
        雙軌制 3D 卷積神經網絡。

        輸入：(B, 1, Nx, Ny, Nz) 體素張量
        輸出：
          presence : (B, 1)  洞穴存在機率（sigmoid）
          count    : (B, 3)  數量類別 [0, 1, 2+]（softmax）

        關鍵設計（對應 A-core-2000）：
          - Z 方向強池化（MaxPool3d kernel_size=(1,1,4)）
            → 壓縮洞穴回波在時間軸的拖尾延伸
          - 共用特徵提取器，分叉為兩個任務頭
        """

        def __init__(self, nx: int = VOXEL_NX,
                     ny: int = VOXEL_NY,
                     nz: int = VOXEL_NZ):
            super().__init__()

            # 共用骨幹特徵提取器
            self.backbone = nn.Sequential(
                # Block 1：初步特徵
                nn.Conv3d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm3d(16),
                nn.ReLU(inplace=True),
                # Z 方向強池化（壓縮拖尾反射，保留 XY 空間資訊）
                nn.MaxPool3d(kernel_size=(1, 1, 4)),   # Nz: 40→10

                # Block 2：中層特徵
                nn.Conv3d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(2, 2, 2)),   # XY: 25→12, Nz: 10→5

                # Block 3：高層特徵（Grad-CAM 作用層）
                nn.Conv3d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool3d((4, 4, 2)),        # → (4, 4, 2)
            )

            feat_dim = 64 * 4 * 4 * 2  # = 2048

            # 存在分類頭
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feat_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 1),
                # 不加 sigmoid，使用 BCEWithLogitsLoss
            )

            # 數量計數頭（0 / 1 / 2+ 個洞穴）
            self.counter = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feat_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, 3),
                # 不加 softmax，使用 CrossEntropyLoss
            )

        def forward(self, x: "torch.Tensor"):
            feat = self.backbone(x)
            presence_logit = self.classifier(feat)   # (B, 1)
            count_logit    = self.counter(feat)       # (B, 3)
            return presence_logit, count_logit

        def predict(self, x: "torch.Tensor"):
            """推論模式：回傳機率值（非 logit）。"""
            with torch.no_grad():
                pl, cl = self.forward(x)
                presence_prob  = torch.sigmoid(pl).squeeze(-1)  # (B,)
                count_prob     = F.softmax(cl, dim=-1)           # (B, 3)
            return presence_prob, count_prob

    # ─────────────────────────────────────────────────────────────────────────
    # Grad-CAM 可解釋性（對應論文視覺化分析）
    # ─────────────────────────────────────────────────────────────────────────
    class GradCAM3D:
        """
        三維 Gradient-weighted Class Activation Mapping。

        論文發現：
          - True Positive：熱圖聚焦於深層高聲阻抗邊緣（洞穴壁）
          - True Negative：熱圖集中在淺層（泥面初始反射）
          - False Positive：誤焦在無機物（石塊、死殼）
        """

        def __init__(self, model: BurrowDetector3DCNN):
            self.model = model
            self._grads: Optional["torch.Tensor"] = None
            self._acts:  Optional["torch.Tensor"] = None
            self._hook_handles = []

            # 在最後一個 Conv3d 層（Block 3）掛鉤
            last_conv = None
            for m in model.backbone:
                if isinstance(m, nn.Conv3d):
                    last_conv = m
            if last_conv is not None:
                self._hook_handles.append(
                    last_conv.register_forward_hook(self._save_activation))
                self._hook_handles.append(
                    last_conv.register_full_backward_hook(self._save_gradient))

        def _save_activation(self, module, inp, out):
            self._acts = out.detach()

        def _save_gradient(self, module, grad_in, grad_out):
            self._grads = grad_out[0].detach()

        def compute(self, x: "torch.Tensor",
                    target_class: int = 1) -> np.ndarray:
            """
            計算 Grad-CAM 熱圖。

            x             : (1, 1, Nx, Ny, Nz) 輸入體素
            target_class  : 1=有洞穴（presence classifier）

            回傳: (Nx, Ny, Nz) numpy 陣列，值 [0, 1]
            """
            self.model.eval()
            x = x.requires_grad_(True)
            presence_logit, _ = self.model(x)
            self.model.zero_grad()
            presence_logit[0, 0].backward()

            if self._grads is None or self._acts is None:
                return np.zeros((VOXEL_NX, VOXEL_NY, VOXEL_NZ))

            # 全局平均梯度（GAP）
            weights = self._grads.mean(dim=(2, 3, 4), keepdim=True)  # (1, C, 1, 1, 1)
            cam = (weights * self._acts).sum(dim=1)                   # (1, Nx', Ny', Nz')
            cam = F.relu(cam[0])                                       # (Nx', Ny', Nz')

            # 上採樣到原始體素大小
            cam_up = F.interpolate(
                cam.unsqueeze(0).unsqueeze(0),
                size=(VOXEL_NX, VOXEL_NY, VOXEL_NZ),
                mode="trilinear", align_corners=False,
            ).squeeze().cpu().numpy()

            # 歸一化 [0, 1]
            cam_min, cam_max = cam_up.min(), cam_up.max()
            if cam_max - cam_min > 1e-8:
                cam_up = (cam_up - cam_min) / (cam_max - cam_min)

            return cam_up.astype(np.float32)

        def remove_hooks(self):
            for h in self._hook_handles:
                h.remove()

    # ─────────────────────────────────────────────────────────────────────────
    # 訓練數據集
    # ─────────────────────────────────────────────────────────────────────────
    class BurrowVoxelDataset(Dataset):
        """
        三維聲學體素數據集。

        data_dir 結構：
          data_dir/
            C/  ← 純沙（label presence=0, count=0）
            B/  ← 有洞穴（label presence=1, count=1）
            O/  ← 其他生物（label presence=0, count=0）
            M/  ← 混合（label presence=1, count=2+）
            *.npy 或 *.json 格式

        每個檔案：shape (Nx, Ny, Nz) float32 numpy 陣列
        """

        LABEL_MAP = {
            "C": (0, 0),   # (presence, count_class)
            "O": (0, 0),
            "B": (1, 1),   # 1 個洞穴
            "M": (1, 2),   # 2+ 個洞穴
        }

        def __init__(self, data_dir: str, augment: bool = True):
            self.data_dir = Path(data_dir)
            self.augment  = augment
            self.samples  = []  # List of (path, presence, count_class)

            for dtype, (pres, cnt) in self.LABEL_MAP.items():
                subdir = self.data_dir / dtype
                if not subdir.exists():
                    continue
                for f in subdir.glob("*.npy"):
                    self.samples.append((f, pres, cnt))
                for f in subdir.glob("*.json"):
                    self.samples.append((f, pres, cnt))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, pres, cnt = self.samples[idx]

            # 載入體素
            if path.suffix == ".npy":
                voxel = np.load(str(path)).astype(np.float32)
            else:
                with open(path) as f:
                    data = json.load(f)
                voxel = np.array(data["volume"], dtype=np.float32)

            # 確保大小匹配
            if voxel.shape != (VOXEL_NX, VOXEL_NY, VOXEL_NZ):
                # 簡單裁切或補零
                v = np.zeros((VOXEL_NX, VOXEL_NY, VOXEL_NZ), dtype=np.float32)
                sx = min(voxel.shape[0], VOXEL_NX)
                sy = min(voxel.shape[1], VOXEL_NY)
                sz = min(voxel.shape[2], VOXEL_NZ)
                v[:sx, :sy, :sz] = voxel[:sx, :sy, :sz]
                voxel = v

            # 資料增強（A-core-2000 論文：XYZ ±20% 平移 + 翻轉）
            if self.augment:
                voxel = self._augment(voxel)

            # 加入 channel 維度 → (1, Nx, Ny, Nz)
            voxel = torch.from_numpy(voxel).unsqueeze(0)
            return voxel, torch.tensor(pres, dtype=torch.float32), \
                          torch.tensor(cnt,  dtype=torch.long)

        @staticmethod
        def _augment(v: np.ndarray) -> np.ndarray:
            """對應論文：XYZ 三方向 ±20% 平移 + 隨機翻轉。"""
            # 隨機翻轉
            if np.random.rand() > 0.5:
                v = np.flip(v, axis=0).copy()
            if np.random.rand() > 0.5:
                v = np.flip(v, axis=1).copy()

            # ±20% 平移（整數格點移動）
            max_shift_x = max(1, int(VOXEL_NX * 0.20))
            max_shift_y = max(1, int(VOXEL_NY * 0.20))
            max_shift_z = max(1, int(VOXEL_NZ * 0.20))
            dx = np.random.randint(-max_shift_x, max_shift_x + 1)
            dy = np.random.randint(-max_shift_y, max_shift_y + 1)
            dz = np.random.randint(-max_shift_z, max_shift_z + 1)
            v = np.roll(v, (dx, dy, dz), axis=(0, 1, 2))

            return v

    # ─────────────────────────────────────────────────────────────────────────
    # 訓練函數
    # ─────────────────────────────────────────────────────────────────────────
    def train_burrow_cnn(data_dir: str,
                         output_path: str = "burrow_cnn.pt",
                         device_str: str = "cpu") -> BurrowDetector3DCNN:
        """
        訓練雙軌制 3D-CNN（對應論文 stratified 5-fold CV）。

        論文最終效能：
          - 存在/不存在：ROC-AUC = 0.9
          - 數量估算：macro AUC = 0.8，MRE 10~20%

        此處為簡化版（單次分割），完整 5-fold 請使用 sklearn.model_selection。
        """
        device = torch.device(device_str)
        dataset = BurrowVoxelDataset(data_dir, augment=True)
        if len(dataset) == 0:
            raise ValueError(f"data_dir 中無有效數據：{data_dir}")

        # 80/20 分割
        n_train = int(0.8 * len(dataset))
        n_val   = len(dataset) - n_train
        train_set, val_set = random_split(dataset, [n_train, n_val])

        train_dl = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0)
        val_dl   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0)

        model = BurrowDetector3DCNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        cls_loss_fn = nn.BCEWithLogitsLoss()
        cnt_loss_fn = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience_cnt  = 0

        for epoch in range(1, MAX_EPOCHS_CLS + 1):
            # 訓練
            model.train()
            train_loss = 0.0
            for voxels, pres_lbl, cnt_lbl in train_dl:
                voxels  = voxels.to(device)
                pres_lbl = pres_lbl.to(device).unsqueeze(1)
                cnt_lbl  = cnt_lbl.to(device)

                pl, cl = model(voxels)
                loss = cls_loss_fn(pl, pres_lbl) + cnt_loss_fn(cl, cnt_lbl)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # 驗證
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for voxels, pres_lbl, cnt_lbl in val_dl:
                    voxels  = voxels.to(device)
                    pres_lbl = pres_lbl.to(device).unsqueeze(1)
                    cnt_lbl  = cnt_lbl.to(device)
                    pl, cl = model(voxels)
                    val_loss += (cls_loss_fn(pl, pres_lbl)
                                 + cnt_loss_fn(cl, cnt_lbl)).item()

            avg_val = val_loss / max(len(val_dl), 1)
            avg_trn = train_loss / max(len(train_dl), 1)
            print(f"Epoch {epoch:3d}  train={avg_trn:.4f}  val={avg_val:.4f}")

            # Early stopping
            if avg_val < best_val_loss - 1e-4:
                best_val_loss = avg_val
                patience_cnt  = 0
                torch.save(model.state_dict(), output_path)
            else:
                patience_cnt += 1
                if patience_cnt >= EARLY_STOP_PAT:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # 載入最佳權重
        model.load_state_dict(torch.load(output_path, map_location=device))
        print(f"訓練完成，最佳模型儲存至：{output_path}")
        return model

    def export_onnx(model: BurrowDetector3DCNN,
                    output_path: str = "burrow_cnn.onnx"):
        """匯出 ONNX（RPi4 / ONNX Runtime 推論）。"""
        model.eval()
        dummy = torch.zeros(1, 1, VOXEL_NX, VOXEL_NY, VOXEL_NZ)
        torch.onnx.export(
            model, dummy, output_path,
            input_names=["voxel"],
            output_names=["presence_logit", "count_logit"],
            dynamic_axes={"voxel": {0: "batch"}},
            opset_version=12,
        )
        print(f"ONNX 匯出完成：{output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 推論包裝器（支援 PyTorch / ONNX Runtime / 規則退後）
# ─────────────────────────────────────────────────────────────────────────────
class BurrowInference:
    """
    推論包裝器。

    優先順序：PyTorch → ONNX Runtime → 規則退後（閾值法）
    """

    def __init__(self, weights_path: Optional[str] = None):
        self._mode = "rule"
        self._model = None
        self._ort_session = None

        if weights_path and Path(weights_path).exists():
            if weights_path.endswith(".onnx"):
                self._load_onnx(weights_path)
            elif _TORCH_AVAILABLE:
                self._load_torch(weights_path)
        else:
            print("[BurrowInference] 無預訓練模型，使用規則退後模式")

    def _load_torch(self, path: str):
        model = BurrowDetector3DCNN()
        model.load_state_dict(
            torch.load(path, map_location="cpu"))
        model.eval()
        self._model = model
        self._mode  = "torch"
        print(f"[BurrowInference] 載入 PyTorch 模型：{path}")

    def _load_onnx(self, path: str):
        try:
            import onnxruntime as ort
            self._ort_session = ort.InferenceSession(path)
            self._mode = "onnx"
            print(f"[BurrowInference] 載入 ONNX 模型：{path}")
        except ImportError:
            print("[BurrowInference] onnxruntime 未安裝，降級為規則模式")

    def infer(self, volume: np.ndarray) -> dict:
        """
        對三維聲學體積進行推論。

        volume : (Nx, Ny, Nz) float32 numpy 陣列，值 [0, 1]
        回傳   : {"presence_prob": float, "count_class": int,
                   "count_probs": [p0, p1, p2plus],
                   "mode": str}
        """
        if self._mode == "torch":
            return self._infer_torch(volume)
        elif self._mode == "onnx":
            return self._infer_onnx(volume)
        else:
            return self._infer_rule(volume)

    def _infer_torch(self, volume: np.ndarray) -> dict:
        v = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
        pres_prob, cnt_prob = self._model.predict(v)
        pres = float(pres_prob[0])
        cnt_probs = cnt_prob[0].cpu().numpy().tolist()
        cnt_class = int(np.argmax(cnt_probs))
        return {
            "presence_prob": round(pres, 4),
            "count_class":   cnt_class,
            "count_probs":   [round(p, 4) for p in cnt_probs],
            "mode":          "torch",
        }

    def _infer_onnx(self, volume: np.ndarray) -> dict:
        v = volume[np.newaxis, np.newaxis].astype(np.float32)  # (1,1,Nx,Ny,Nz)
        inputs = {self._ort_session.get_inputs()[0].name: v}
        pl, cl = self._ort_session.run(None, inputs)
        pres = float(1 / (1 + math.exp(-float(pl[0, 0]))))  # sigmoid
        cnt_probs_raw = cl[0]
        exp_c = np.exp(cnt_probs_raw - cnt_probs_raw.max())
        cnt_probs = (exp_c / exp_c.sum()).tolist()           # softmax
        cnt_class = int(np.argmax(cnt_probs))
        return {
            "presence_prob": round(pres, 4),
            "count_class":   cnt_class,
            "count_probs":   [round(p, 4) for p in cnt_probs],
            "mode":          "onnx",
        }

    def _infer_rule(self, volume: np.ndarray) -> dict:
        """
        規則退後（無 CNN 時）：
        - 計算非零體素的 SNR 代理分數（max/mean_nonzero）
        - 信號集中（洞穴）→ max >> mean_nonzero → 高 SNR → 高 presence_prob
        - 均勻雜訊         → max ≈ mean_nonzero → 低 SNR → 低 presence_prob

        注意：不能用 max/global_mean，因為稀疏格點導致 global_mean 接近零，
        signal 與 noise 的 global_mean 差異不大。
        """
        vmax = float(volume.max())
        if vmax < 0.01:
            return {"presence_prob": 0.05, "count_class": 0,
                    "count_probs": [0.95, 0.04, 0.01], "mode": "rule"}

        # 使用非零體素計算 SNR 代理分數
        nonzero = volume[volume > 0.01]
        mean_nonzero = float(nonzero.mean()) if len(nonzero) > 0 else vmax
        snr_proxy = float(vmax / (mean_nonzero + 1e-8))

        # 高強度體素計數（可能的洞穴反射）
        n_strong = int((volume > 0.5 * vmax).sum())

        # 規則推論（粗略估計）
        if snr_proxy > 10:
            pres = min(0.90, 0.50 + snr_proxy * 0.04)
        elif snr_proxy > 5:
            pres = 0.40 + snr_proxy * 0.02
        else:
            pres = 0.10

        # 計數估計（依強反射體素的空間分佈）
        if n_strong > 500:
            cnt_class = 2   # 可能多個洞穴
            probs = [1 - pres, pres * 0.3, pres * 0.7]
        elif n_strong > 50:
            cnt_class = 1
            probs = [1 - pres, pres * 0.7, pres * 0.3]
        else:
            cnt_class = 0
            probs = [0.90, 0.08, 0.02]

        return {
            "presence_prob": round(pres, 4),
            "count_class":   cnt_class,
            "count_probs":   [round(p, 4) for p in probs],
            "mode":          "rule",
        }


# ─────────────────────────────────────────────────────────────────────────────
# ROS2 節點
# ─────────────────────────────────────────────────────────────────────────────
if _ROS2_AVAILABLE:
    class BurrowCNNNode(Node):
        """
        3D-CNN 洞穴偵測 ROS2 節點。

        訂閱 /acoustic/cscan_volume，對每個體積進行推論，
        輸出偵測結果到 /acoustic/burrow_detection。

        ROS2 params:
          weights_path : str   模型路徑（.pt 或 .onnx），空白=規則模式
          threshold    : float 存在機率閾值（預設 0.5）
        """

        def __init__(self):
            super().__init__("burrow_cnn")

            self.declare_parameter("weights_path", "")
            self.declare_parameter("threshold",    PRESENCE_THRESHOLD)

            weights = self.get_parameter("weights_path").value
            self._threshold = float(self.get_parameter("threshold").value)

            self._inference = BurrowInference(
                weights_path=weights if weights else None)

            self.get_logger().info(
                f"BurrowCNN ready, mode={self._inference._mode}, "
                f"threshold={self._threshold}")

            self.create_subscription(
                String, "/acoustic/cscan_volume",
                self._cb_volume, 5)

            self._pub_detect = self.create_publisher(
                String, "/acoustic/burrow_detection", 10)
            self._pub_status = self.create_publisher(
                String, "/acoustic/cnn_status",       10)

        def _cb_volume(self, msg: String):
            """接收 C-scan 體積並推論。"""
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                return

            meta = data.get("meta", {})
            sparse = data.get("sparse_voxels", [])
            if not meta or not sparse:
                return

            # 從稀疏格式重建 numpy 體積
            Nx = meta.get("Nx", VOXEL_NX)
            Ny = meta.get("Ny", VOXEL_NY)
            Nz = meta.get("Nz", VOXEL_NZ)
            volume = np.zeros((Nx, Ny, Nz), dtype=np.float32)

            x_coords = np.array(meta.get("x_coords_m", []))
            z_coords = np.array(meta.get("z_coords_m", []))

            for vx in sparse:
                if len(vx) < 4:
                    continue
                # 找最近格點索引
                if len(x_coords) > 0:
                    ix = int(np.argmin(np.abs(x_coords - vx[0])))
                    iy = int(np.argmin(np.abs(x_coords - vx[1])))
                else:
                    ix = iy = 0
                iz = int(np.argmin(np.abs(z_coords - vx[2]))) \
                    if len(z_coords) > 0 else 0
                ix = min(ix, Nx - 1)
                iy = min(iy, Ny - 1)
                iz = min(iz, Nz - 1)
                volume[ix, iy, iz] = float(vx[3])

            # 調整到模型輸入大小
            if volume.shape != (VOXEL_NX, VOXEL_NY, VOXEL_NZ):
                from scipy.ndimage import zoom
                factors = (VOXEL_NX / Nx, VOXEL_NY / Ny, VOXEL_NZ / Nz)
                volume = zoom(volume, factors, order=1).astype(np.float32)

            t0 = time.monotonic()
            result = self._inference.infer(volume)
            dt = time.monotonic() - t0

            # 整合 analysis 資訊
            analysis = data.get("analysis", {})
            result["peak_depth_m"]  = analysis.get("peak_depth_m", 0.0)
            result["n_candidates"]  = analysis.get("n_candidates", 0)
            result["snr_db"]        = analysis.get("snr_db", 0.0)
            result["infer_time_ms"] = round(dt * 1000, 1)
            result["timestamp"]     = time.time()

            # 判斷是否為洞穴
            burrow_present = result["presence_prob"] >= self._threshold
            result["burrow_present"] = burrow_present
            result["count_label"] = ["0", "1", "2+"][result["count_class"]]

            self._pub_detect.publish(String(data=json.dumps(result)))

            self.get_logger().info(
                f"CNN: burrow={'YES' if burrow_present else 'NO'} "
                f"p={result['presence_prob']:.3f} "
                f"count={result['count_label']} "
                f"depth={result['peak_depth_m']*100:.1f}cm "
                f"({dt*1000:.0f}ms, {result['mode']})")

    def main(args=None):
        rclpy.init(args=args)
        node = BurrowCNNNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            node.get_logger().info("BurrowCNN node shutting down.")
        finally:
            node.destroy_node()
            rclpy.shutdown()

else:
    def main(args=None):
        print("ROS2 not found. Source /opt/ros/<distro>/setup.bash first.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口（訓練 / 匯出 / 測試）
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Burrow 3D-CNN")
    parser.add_argument("--mode", choices=["train", "export", "test"],
                        default="test")
    parser.add_argument("--data",    default="data/cscan_dataset")
    parser.add_argument("--weights", default="burrow_cnn.pt")
    parser.add_argument("--onnx",    default="burrow_cnn.onnx")
    parser.add_argument("--device",  default="cpu")
    args = parser.parse_args()

    if args.mode == "train":
        if not _TORCH_AVAILABLE:
            print("需要 PyTorch：pip install torch")
            sys.exit(1)
        model = train_burrow_cnn(args.data, args.weights, args.device)

    elif args.mode == "export":
        if not _TORCH_AVAILABLE:
            print("需要 PyTorch：pip install torch")
            sys.exit(1)
        model = BurrowDetector3DCNN()
        model.load_state_dict(torch.load(args.weights, map_location="cpu"))
        export_onnx(model, args.onnx)

    else:  # test
        print("=== 規則模式推論測試 ===")
        np.random.seed(0)

        # 模擬有洞穴的體積
        v_burrow = np.random.rand(VOXEL_NX, VOXEL_NY, VOXEL_NZ).astype(
            np.float32) * 0.1
        # 在 z=20（≈40cm）放置強反射體
        v_burrow[12:13, 12:13, 18:22] = 0.95

        inf = BurrowInference(weights_path=None)
        result = inf.infer(v_burrow)
        print(f"有洞穴體積 → presence={result['presence_prob']:.3f}, "
              f"count={result['count_label']}, mode={result['mode']}")
        assert result["presence_prob"] >= 0.5, "應偵測到洞穴"

        # 模擬空白體積
        v_empty = np.random.rand(VOXEL_NX, VOXEL_NY, VOXEL_NZ).astype(
            np.float32) * 0.05
        result2 = inf.infer(v_empty)
        print(f"空白體積     → presence={result2['presence_prob']:.3f}, "
              f"count={result2['count_label']}")
        assert result2["presence_prob"] < 0.5, "不應誤判為洞穴"

        print("=== 規則模式測試通過 ===")

        if _TORCH_AVAILABLE:
            print("\n=== PyTorch 模型架構測試 ===")
            model = BurrowDetector3DCNN()
            dummy = torch.zeros(2, 1, VOXEL_NX, VOXEL_NY, VOXEL_NZ)
            pl, cl = model(dummy)
            print(f"presence_logit shape: {pl.shape}")   # (2, 1)
            print(f"count_logit shape:    {cl.shape}")   # (2, 3)
            assert pl.shape == (2, 1)
            assert cl.shape == (2, 3)
            print("=== PyTorch 架構測試通過 ===")
