"""
gen_synthetic_dataset.py
========================
合成 C-scan 訓練數據集生成器（不需現場採集即可預訓練 3D-CNN）。

物理模型：
  每個樣本為 (Nx=25, Ny=25, Nz=40) float32 體素，
  模擬 Hilbert 包絡強度經 TVG 補償後的 C-scan 重建結果。

類別設計（對應 burrow_cnn.py DATASET_TYPES）：
  C - 純沙背景：低振幅均勻散射 + 表面反射
  B - 單一蝦洞：背景 + 深層局部強反射 + Z 方向拖尾
  O - 其他生物：背景 + 緊湊型高振幅實心反射（螃蟹/貝殼）
  M - 多洞穴：背景 + 2~3 個分散的蝦洞反射

每類預設生成 200 個樣本（共 800 個），可調整 --n 參數。

使用：
  python scripts/gen_synthetic_dataset.py --out data/cscan_dataset --n 200
  python scripts/gen_synthetic_dataset.py --out data/cscan_dataset --n 50 --seed 42
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 體素尺寸（與 burrow_cnn.py 一致）
# ─────────────────────────────────────────────────────────────────────────────
NX, NY, NZ = 25, 25, 40   # (X格點, Y格點, Z深度格點)

# Z 對應深度：bin 0 ≈ 0cm，bin 39 ≈ 78cm（每格約 2cm）
Z_BIN_CM = 2.0

# 蝦洞出現的深度範圍（bin）：避開表面盲區(0~2)和最深區(>35)
BURROW_Z_MIN = 4
BURROW_Z_MAX = 32

# XY 邊緣保留，避免 Gaussian 被截斷太嚴重
XY_MARGIN = 3


# ─────────────────────────────────────────────────────────────────────────────
# 輔助函數
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_blob(nx, ny, nz, cx, cy, cz,
                   sx, sy, sz, amplitude=1.0) -> np.ndarray:
    """在體素中放置三維 Gaussian 斑點（模擬聲學反射）。"""
    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    blob = amplitude * np.exp(
        -((X - cx)**2 / (2 * sx**2)
        + (Y - cy)**2 / (2 * sy**2)
        + (Z - cz)**2 / (2 * sz**2))
    )
    return blob.astype(np.float32)


def _surface_reflection(nx, ny, nz, rng: np.random.Generator) -> np.ndarray:
    """
    表面反射層（z=0~2）：所有 XY 位置均有，強度略有起伏。
    模擬底泥-水界面的強回波。
    """
    vol = np.zeros((nx, ny, nz), dtype=np.float32)
    amplitude = rng.uniform(0.4, 0.7, size=(nx, ny))
    for ix in range(nx):
        for iy in range(ny):
            # 表面波形：z=0 強反射 + 指數衰減拖尾
            for iz in range(min(4, nz)):
                vol[ix, iy, iz] = float(amplitude[ix, iy]) * np.exp(-iz * 0.8)
    return vol


def _background_noise(nx, ny, nz, rng: np.random.Generator,
                       level=0.03) -> np.ndarray:
    """低振幅散射雜訊。"""
    return (rng.random((nx, ny, nz)) * level).astype(np.float32)


def _burrow_reflector(nx, ny, nz, rng: np.random.Generator) -> np.ndarray:
    """
    單個蝦洞反射特徵：
    - XY：點狀（sigma 1.5~2.5 格，對應 3~5cm 開口）
    - Z：Z 方向拖尾（sigma_z > sigma_xy，模擬空洞多次反射）
    - 振幅：0.65~0.95（空洞-沙界面強反射）
    """
    cx = rng.integers(XY_MARGIN, nx - XY_MARGIN)
    cy = rng.integers(XY_MARGIN, ny - XY_MARGIN)
    cz = rng.integers(BURROW_Z_MIN, BURROW_Z_MAX)

    sx = rng.uniform(1.2, 2.5)
    sy = rng.uniform(1.2, 2.5)
    sz = rng.uniform(2.5, 5.0)   # Z 拖尾：洞穴空腔多次反射

    amp = rng.uniform(0.65, 0.95)
    return _gaussian_blob(nx, ny, nz, cx, cy, cz, sx, sy, sz, amp)


def _other_reflector(nx, ny, nz, rng: np.random.Generator) -> np.ndarray:
    """
    其他生物（螃蟹/貝殼）反射特徵：
    - XY：稍大（有殼體展開）
    - Z：緊湊（實心，無空洞拖尾）
    - 振幅：0.7~1.0（硬殼高聲阻抗）
    """
    cx = rng.integers(XY_MARGIN, nx - XY_MARGIN)
    cy = rng.integers(XY_MARGIN, ny - XY_MARGIN)
    cz = rng.integers(BURROW_Z_MIN, BURROW_Z_MAX)

    sx = rng.uniform(2.0, 3.5)   # 較大 XY 展開（殼體）
    sy = rng.uniform(2.0, 3.5)
    sz = rng.uniform(0.8, 1.8)   # 緊湊 Z（實心，不拖尾）

    amp = rng.uniform(0.70, 1.00)
    return _gaussian_blob(nx, ny, nz, cx, cy, cz, sx, sy, sz, amp)


# ─────────────────────────────────────────────────────────────────────────────
# 四類樣本生成
# ─────────────────────────────────────────────────────────────────────────────

def gen_C(rng: np.random.Generator) -> np.ndarray:
    """類別 C：純沙背景，無洞穴。"""
    vol = _background_noise(NX, NY, NZ, rng, level=rng.uniform(0.02, 0.05))
    vol += _surface_reflection(NX, NY, NZ, rng)
    return np.clip(vol, 0.0, 1.0)


def gen_B(rng: np.random.Generator) -> np.ndarray:
    """類別 B：單一奧螻蛄蝦洞穴。"""
    vol = _background_noise(NX, NY, NZ, rng, level=rng.uniform(0.02, 0.04))
    vol += _surface_reflection(NX, NY, NZ, rng)
    vol += _burrow_reflector(NX, NY, NZ, rng)
    return np.clip(vol, 0.0, 1.0)


def gen_O(rng: np.random.Generator) -> np.ndarray:
    """類別 O：其他生物（螃蟹/彈塗魚），不算洞穴。"""
    vol = _background_noise(NX, NY, NZ, rng, level=rng.uniform(0.02, 0.05))
    vol += _surface_reflection(NX, NY, NZ, rng)
    vol += _other_reflector(NX, NY, NZ, rng)
    return np.clip(vol, 0.0, 1.0)


def gen_M(rng: np.random.Generator) -> np.ndarray:
    """類別 M：多洞穴（2~3 個）。"""
    vol = _background_noise(NX, NY, NZ, rng, level=rng.uniform(0.02, 0.04))
    vol += _surface_reflection(NX, NY, NZ, rng)
    n_burrows = rng.integers(2, 4)   # 2 或 3 個洞穴
    for _ in range(n_burrows):
        vol += _burrow_reflector(NX, NY, NZ, rng)
    return np.clip(vol, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 主程式：生成並儲存
# ─────────────────────────────────────────────────────────────────────────────

CLASS_GENERATORS = {
    "C": gen_C,
    "B": gen_B,
    "O": gen_O,
    "M": gen_M,
}

CLASS_LABELS = {
    "C": {"presence": 0, "count_class": 0, "description": "pure_sand"},
    "B": {"presence": 1, "count_class": 1, "description": "single_burrow"},
    "O": {"presence": 0, "count_class": 0, "description": "other_organism"},
    "M": {"presence": 1, "count_class": 2, "description": "multiple_burrows"},
}


def generate_dataset(out_dir: str, n_per_class: int = 200,
                     seed: int = 42) -> dict:
    """
    生成並儲存完整 synthetic dataset。

    out_dir/
      C/  synthetic_0000.npy ... synthetic_NNNN.npy
      B/
      O/
      M/
      dataset_meta.json   ← 類別統計 + 生成參數

    回傳: 統計 dict
    """
    out_path = Path(out_dir)
    rng = np.random.default_rng(seed)

    stats = {}
    total = 0

    for cls, gen_fn in CLASS_GENERATORS.items():
        cls_dir = out_path / cls
        cls_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n_per_class):
            vol = gen_fn(rng)
            assert vol.shape == (NX, NY, NZ), \
                f"錯誤形狀 {vol.shape}，應為 ({NX},{NY},{NZ})"
            assert vol.dtype == np.float32

            fname = cls_dir / f"synthetic_{i:04d}.npy"
            np.save(str(fname), vol)

        stats[cls] = {
            "n_samples":   n_per_class,
            "presence":    CLASS_LABELS[cls]["presence"],
            "count_class": CLASS_LABELS[cls]["count_class"],
            "description": CLASS_LABELS[cls]["description"],
        }
        total += n_per_class
        print(f"  [{cls}] {n_per_class} 個樣本 → {cls_dir}")

    # 寫入 metadata
    meta = {
        "generator":    "gen_synthetic_dataset.py",
        "voxel_shape":  [NX, NY, NZ],
        "z_bin_cm":     Z_BIN_CM,
        "n_per_class":  n_per_class,
        "total_samples": total,
        "seed":         seed,
        "classes":      stats,
        "notes": [
            "Synthetic data — physics-based Gaussian blob model",
            "Burrow: wide Z-tail (hollow reverb), narrow XY (3~5cm opening)",
            "Other: compact Z (solid shell), wider XY (crab body)",
            "Surface reflection at z=0~3 in all samples",
        ],
    }
    meta_path = out_path / "dataset_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n完成：共 {total} 個樣本，metadata → {meta_path}")
    return stats


def verify_dataset(out_dir: str) -> bool:
    """快速驗證：每類取 5 個樣本，檢查形狀/範圍/標籤一致性。"""
    out_path = Path(out_dir)
    errors = []

    for cls in CLASS_GENERATORS:
        cls_dir = out_path / cls
        files = sorted(cls_dir.glob("*.npy"))
        if not files:
            errors.append(f"{cls}: 無檔案")
            continue

        # 抽查 5 個
        sample_files = files[:5]
        for f in sample_files:
            vol = np.load(str(f))
            if vol.shape != (NX, NY, NZ):
                errors.append(f"{f}: 形狀錯誤 {vol.shape}")
            if vol.dtype != np.float32:
                errors.append(f"{f}: dtype 錯誤 {vol.dtype}")
            if vol.min() < -0.01 or vol.max() > 1.01:
                errors.append(f"{f}: 值超出 [0,1] 範圍")

        # 驗證有洞穴類別確實有高強度體素
        if cls in ("B", "M"):
            sample = np.load(str(files[0]))
            if sample.max() < 0.5:
                errors.append(f"{cls}: 洞穴樣本最大值過低 ({sample.max():.3f})")

        # 驗證無洞穴類別無明顯深層高強度
        if cls == "C":
            sample = np.load(str(files[0]))
            deep_zone = sample[:, :, BURROW_Z_MIN:]
            if deep_zone.max() > 0.3:
                errors.append(f"{cls}: 純沙樣本深層有異常高值 ({deep_zone.max():.3f})")

        print(f"  [{cls}] {len(files)} 個樣本，抽查 {len(sample_files)} 個 OK")

    if errors:
        print("\n驗證失敗：")
        for e in errors:
            print(f"  ✗ {e}")
        return False

    print("\n驗證通過 ✓")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Windows UTF-8 console fix（只在直接執行時套用，不影響 import）
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="生成 Archimedes C-scan synthetic 訓練數據集")
    parser.add_argument("--out",    default="data/cscan_dataset",
                        help="輸出目錄（預設 data/cscan_dataset）")
    parser.add_argument("--n",      type=int, default=200,
                        help="每類生成樣本數（預設 200，共 800 個）")
    parser.add_argument("--seed",   type=int, default=42,
                        help="隨機種子（預設 42）")
    parser.add_argument("--verify", action="store_true",
                        help="生成後執行驗證")
    args = parser.parse_args()

    print(f"生成 synthetic C-scan dataset → {args.out}")
    print(f"  每類 {args.n} 個樣本，種子 {args.seed}")
    print(f"  體素大小：({NX}, {NY}, {NZ})")
    print()

    generate_dataset(args.out, n_per_class=args.n, seed=args.seed)

    if args.verify:
        print("\n執行驗證...")
        ok = verify_dataset(args.out)
        sys.exit(0 if ok else 1)
