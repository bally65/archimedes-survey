"""
validate_density.py
===================
阿基米德探勘機器人密度估計驗證工具

功能：
  載入機器人調查 GeoJSON 與人工樣方 CSV，計算統計指標：
    - RMSE / MAE：機器人偵測 vs 人工計數誤差
    - R²（Pearson r²）：相關性
    - Bland-Altman 分析：系統性偏差與一致性限
    - Moran's I：空間自相關（聚集/隨機/離散）

使用方式：
  python scripts/validate_density.py \
    --quadrat data/manual_quadrats.csv \
    --missions ~/archimedes_missions \
    --radius 5.0 \
    --plot

CSV 格式（--quadrat）：
  lat,lon,manual_count,area_m2
  24.123,120.456,8,0.25
  ...
  manual_count：樣方內人工計數洞穴數
  area_m2：樣方面積（預設 0.25 m²，即 0.5×0.5m 樣框）

GeoJSON 格式（機器人輸出，mission_logger.py）：
  FeatureCollection，每個 Feature 為 Point，properties 含：
    - is_new_individual (bool)：Y 型洞穴配對後是否為獨立個體
    - confidence (float)：YOLO 信心值

輸出：
  - 終端機：彙整統計
  - validation_report.json：完整數據
  - validation_bland_altman.png（若 --plot）
  - validation_scatter.png（若 --plot）
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path


# --- 依賴庫（優雅降級）---
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    print("[ERROR] numpy 未安裝：pip install numpy")
    sys.exit(1)

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    print("[WARN] scipy 未安裝，跳過 r² 計算：pip install scipy")
    HAS_SCIPY = False

try:
    import matplotlib
    matplotlib.use("Agg")   # 無顯示器環境用 Agg
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# PySAL 為選用（Moran's I）
try:
    import libpysal
    from esda.moran import Moran
    HAS_PYSAL = True
except ImportError:
    HAS_PYSAL = False


# ---------------------------------------------------------------------------
# 地理計算工具
# ---------------------------------------------------------------------------

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine 距離（公尺）"""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# 資料載入
# ---------------------------------------------------------------------------

def load_quadrats(csv_path: str) -> list[dict]:
    """載入人工樣方 CSV"""
    quadrats = []
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 略過 # 開頭的備註行，找 header
    header_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            header_idx = i
            break

    headers = [h.strip().lower() for h in lines[header_idx].split(",")]
    required = {"lat", "lon", "manual_count"}
    if not required.issubset(set(headers)):
        raise ValueError(f"CSV 缺少必要欄位，需要：{required}，實際：{set(headers)}")

    for line in lines[header_idx + 1:]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values = line.split(",")
        row = {}
        for i, h in enumerate(headers):
            if i < len(values):
                try:
                    row[h] = float(values[i].strip())
                except ValueError:
                    row[h] = values[i].strip()
        if "area_m2" not in row:
            row["area_m2"] = 0.25   # 預設 0.5×0.5m 樣框
        quadrats.append(row)

    print(f"[載入] 人工樣方：{len(quadrats)} 筆")
    return quadrats


def load_robot_burrows(missions_dir: str) -> list[dict]:
    """載入機器人任務目錄下所有 GeoJSON 的洞穴偵測點"""
    burrows = []
    missions_path = Path(missions_dir).expanduser()

    if not missions_path.exists():
        print(f"[WARN] 任務目錄不存在：{missions_path}")
        return burrows

    geojson_files = list(missions_path.glob("*_burrows.geojson"))
    if not geojson_files:
        print(f"[WARN] 找不到 *_burrows.geojson 檔案於 {missions_path}")
        return burrows

    for gf in sorted(geojson_files):
        with open(gf, "r", encoding="utf-8") as f:
            gj = json.load(f)
        for feat in gj.get("features", []):
            if feat.get("geometry", {}).get("type") != "Point":
                continue
            lon, lat = feat["geometry"]["coordinates"][:2]
            props = feat.get("properties", {})
            burrows.append({
                "lat": lat,
                "lon": lon,
                "is_individual": props.get("is_new_individual", True),
                "confidence": props.get("confidence", 1.0),
                "source_file": gf.name,
            })

    print(f"[載入] 機器人偵測：{len(burrows)} 個開口，"
          f"{sum(1 for b in burrows if b['is_individual'])} 個個體")
    return burrows


# ---------------------------------------------------------------------------
# 配對：每個樣方計數附近機器人偵測數
# ---------------------------------------------------------------------------

def match_quadrats_to_robot(
    quadrats: list[dict],
    burrows: list[dict],
    radius_m: float = 5.0,
    only_individuals: bool = True,
) -> list[dict]:
    """
    對每個人工樣方，計算 radius_m 範圍內機器人偵測到的洞穴數。
    radius_m 建議設為樣方對角線一半 + GPS 誤差 (~1m) → 預設 5m 寬容
    """
    matched = []
    for q in quadrats:
        robot_count = 0
        for b in burrows:
            if only_individuals and not b["is_individual"]:
                continue
            d = haversine_m(q["lat"], q["lon"], b["lat"], b["lon"])
            if d <= radius_m:
                robot_count += 1

        # 換算為密度（隻/m²）
        area  = q.get("area_m2", 0.25)
        area  = max(area, 0.01)   # 防止除零
        manual_density   = q["manual_count"] / area
        robot_density    = robot_count / area

        matched.append({
            "lat":              q["lat"],
            "lon":              q["lon"],
            "manual_count":     q["manual_count"],
            "robot_count":      robot_count,
            "area_m2":          area,
            "manual_density":   manual_density,
            "robot_density":    robot_density,
        })

    return matched


# ---------------------------------------------------------------------------
# 統計計算
# ---------------------------------------------------------------------------

def compute_rmse_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """RMSE 和 MAE"""
    residuals = y_pred - y_true
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae  = float(np.mean(np.abs(residuals)))
    return {"rmse": rmse, "mae": mae}


def compute_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """Pearson r 和 r²（coefficient of determination）"""
    if not HAS_SCIPY:
        return {"r": None, "r2": None, "p_value": None}

    if len(y_true) < 3:
        return {"r": None, "r2": None, "p_value": None, "note": "樣本數 < 3，無法計算"}

    r, p = scipy_stats.pearsonr(y_true, y_pred)
    return {
        "r":       float(r),
        "r2":      float(r ** 2),
        "p_value": float(p),
    }


def compute_bland_altman(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """
    Bland-Altman 一致性分析
    差值 = robot - manual
    正值 = 機器人過數（高估），負值 = 低估
    """
    diff  = y_pred - y_true
    mean  = (y_pred + y_true) / 2.0

    bias       = float(np.mean(diff))
    std_diff   = float(np.std(diff, ddof=1))
    loa_upper  = bias + 1.96 * std_diff   # 95% 一致性上限
    loa_lower  = bias - 1.96 * std_diff   # 95% 一致性下限

    return {
        "bias":       bias,
        "std_diff":   std_diff,
        "loa_upper":  loa_upper,
        "loa_lower":  loa_lower,
        "mean_vals":  mean.tolist(),
        "diff_vals":  diff.tolist(),
        "interpretation": (
            "機器人系統性高估" if bias > 0.5 else
            "機器人系統性低估" if bias < -0.5 else
            "無顯著偏差"
        ),
    }


def compute_morans_i(
    values: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> dict:
    """
    Moran's I 空間自相關
    使用 PySAL（libpysal + esda），若未安裝則改用手動計算。
    I > 0 → 空間聚集（相似值靠近）
    I ≈ 0 → 空間隨機
    I < 0 → 空間離散
    """
    n = len(values)
    if n < 4:
        return {"I": None, "p_value": None, "note": "樣本數 < 4，跳過"}

    if HAS_PYSAL:
        # 用 PySAL 的反距離加權空間矩陣
        try:
            coords = np.column_stack([lons, lats])
            w = libpysal.weights.DistanceBand.from_array(coords, threshold=0.01)
            moran = Moran(values, w)
            return {
                "I":         float(moran.I),
                "EI":        float(moran.EI),
                "p_value":   float(moran.p_sim),
                "z_score":   float(moran.z_sim),
                "method":    "PySAL",
                "interpretation": _morans_interp(moran.I, moran.p_sim),
            }
        except Exception as e:
            # 降級到手動計算
            pass

    # 手動計算（簡化版，反距離加權）
    return _morans_i_manual(values, lats, lons)


def _morans_i_manual(
    values: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> dict:
    """手動計算 Moran's I（反距離加權，不含 p-value）"""
    n = len(values)
    v_mean = np.mean(values)
    v_dev  = values - v_mean

    # 建立反距離權重矩陣（1/d，d=0 時設 0）
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                d = haversine_m(lats[i], lons[i], lats[j], lons[j])
                W[i, j] = 1.0 / max(d, 1e-6)

    W_sum = np.sum(W)
    if W_sum == 0:
        return {"I": None, "p_value": None, "note": "所有點重疊，無法計算"}

    # Moran's I 公式
    numerator   = np.sum(W * np.outer(v_dev, v_dev))
    denominator = np.sum(v_dev ** 2)

    if denominator == 0:
        return {"I": None, "p_value": None, "note": "所有值相同，無變異"}

    I = (n / W_sum) * (numerator / denominator)

    return {
        "I":         float(I),
        "EI":        float(-1.0 / (n - 1)),
        "p_value":   None,
        "z_score":   None,
        "method":    "手動反距離加權",
        "note":      "安裝 pysal/esda 可獲得 p-value",
        "interpretation": _morans_interp(I, None),
    }


def _morans_interp(I: float, p: float | None) -> str:
    if I is None:
        return "無法計算"
    sig = " (顯著)" if (p is not None and p < 0.05) else " (不顯著)"
    if I > 0.2:
        return f"空間聚集{sig}"
    elif I < -0.2:
        return f"空間離散{sig}"
    else:
        return f"空間隨機{sig}"


# ---------------------------------------------------------------------------
# 繪圖
# ---------------------------------------------------------------------------

def plot_bland_altman(ba: dict, output_path: str):
    """輸出 Bland-Altman 圖"""
    if not HAS_MATPLOTLIB:
        print("[WARN] matplotlib 未安裝，跳過繪圖")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    mean_vals = np.array(ba["mean_vals"])
    diff_vals = np.array(ba["diff_vals"])

    ax.scatter(mean_vals, diff_vals, color="steelblue", alpha=0.7, s=60, label="樣方")
    ax.axhline(ba["bias"],      color="red",    linestyle="-",  lw=2, label=f"偏差 = {ba['bias']:.3f}")
    ax.axhline(ba["loa_upper"], color="orange", linestyle="--", lw=1.5, label=f"95% LoA = [{ba['loa_lower']:.3f}, {ba['loa_upper']:.3f}]")
    ax.axhline(ba["loa_lower"], color="orange", linestyle="--", lw=1.5)
    ax.axhline(0,               color="gray",   linestyle=":",  lw=1, alpha=0.5)

    ax.set_xlabel("平均密度（隻/m²）")
    ax.set_ylabel("差值（機器人 − 人工）（隻/m²）")
    ax.set_title("Bland-Altman 一致性分析\n阿基米德機器人 vs 人工樣方")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[圖表] Bland-Altman 已儲存：{output_path}")


def plot_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    r2: float | None,
    output_path: str
):
    """人工 vs 機器人散佈圖"""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, color="steelblue", alpha=0.7, s=60, label="樣方")

    # 1:1 參考線
    lim_min = min(np.min(y_true), np.min(y_pred)) * 0.9
    lim_max = max(np.max(y_true), np.max(y_pred)) * 1.1
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", lw=1.5, label="1:1 理想線")

    # 線性回歸趨勢線
    if HAS_SCIPY and len(y_true) >= 3:
        slope, intercept, _, _, _ = scipy_stats.linregress(y_true, y_pred)
        x_fit = np.linspace(lim_min, lim_max, 100)
        ax.plot(x_fit, slope * x_fit + intercept, "g-", lw=1.5,
                label=f"回歸線 y={slope:.2f}x+{intercept:.2f}")

    r2_str = f"r²={r2:.3f}" if r2 is not None else "r²=N/A"
    ax.set_xlabel("人工樣方密度（隻/m²）")
    ax.set_ylabel("機器人偵測密度（隻/m²）")
    ax.set_title(f"機器人 vs 人工樣方密度比較\n{r2_str}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[圖表] 散佈圖已儲存：{output_path}")


# ---------------------------------------------------------------------------
# 主函數
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="阿基米德機器人密度估計驗證（RMSE/r²/Bland-Altman/Moran's I）"
    )
    parser.add_argument(
        "--quadrat", "-q",
        required=True,
        help="人工樣方 CSV 路徑（欄位：lat,lon,manual_count,area_m2）"
    )
    parser.add_argument(
        "--missions", "-m",
        default=str(Path.home() / "archimedes_missions"),
        help="機器人任務目錄（含 *_burrows.geojson）"
    )
    parser.add_argument(
        "--radius", "-r",
        type=float,
        default=5.0,
        help="配對搜尋半徑（公尺），建議 = 樣方對角線/2 + GPS誤差，預設 5m"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="輸出 Bland-Altman 圖和散佈圖（需 matplotlib）"
    )
    parser.add_argument(
        "--output", "-o",
        default="validation_report.json",
        help="輸出 JSON 報告路徑（預設 validation_report.json）"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="只計入 confidence >= 此值的偵測（預設 0.5）"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("阿基米德機器人密度驗證工具")
    print("=" * 60)

    # --- 載入資料 ---
    quadrats = load_quadrats(args.quadrat)
    burrows  = load_robot_burrows(args.missions)

    # 過濾低信心偵測
    burrows = [b for b in burrows if b.get("confidence", 1.0) >= args.min_confidence]
    print(f"[過濾] confidence ≥ {args.min_confidence}：保留 {len(burrows)} 個偵測")

    if not quadrats:
        print("[ERROR] 無樣方資料，結束")
        sys.exit(1)

    # --- 配對 ---
    matched = match_quadrats_to_robot(quadrats, burrows, radius_m=args.radius)

    n = len(matched)
    print(f"\n[配對] {n} 個樣方成功配對（搜尋半徑 {args.radius}m）")

    if n < 2:
        print("[ERROR] 配對數 < 2，無法計算統計")
        sys.exit(1)

    y_true = np.array([m["manual_density"] for m in matched])
    y_pred = np.array([m["robot_density"]  for m in matched])
    lats   = np.array([m["lat"] for m in matched])
    lons   = np.array([m["lon"] for m in matched])

    # --- 統計計算 ---
    err_stats = compute_rmse_mae(y_true, y_pred)
    r2_stats  = compute_r2(y_true, y_pred)
    ba_stats  = compute_bland_altman(y_true, y_pred)
    mi_stats  = compute_morans_i(y_true, lats, lons)

    # --- 印出摘要 ---
    print("\n" + "=" * 60)
    print("驗證結果摘要")
    print("=" * 60)
    print(f"  樣本數：            {n}")
    print(f"  人工密度均值：      {np.mean(y_true):.2f} 隻/m²")
    print(f"  機器人密度均值：    {np.mean(y_pred):.2f} 隻/m²")
    print()
    print(f"  RMSE：              {err_stats['rmse']:.3f} 隻/m²")
    print(f"  MAE：               {err_stats['mae']:.3f} 隻/m²")
    r2_val = r2_stats.get("r2")
    r_val  = r2_stats.get("r")
    print(f"  Pearson r：         {r_val:.3f if r_val is not None else 'N/A'}")
    print(f"  r²：                {r2_val:.3f if r2_val is not None else 'N/A'}")
    p_val  = r2_stats.get("p_value")
    print(f"  p-value：           {p_val:.4f if p_val is not None else 'N/A'}")
    print()
    print(f"  Bland-Altman 偏差： {ba_stats['bias']:.3f} 隻/m² [{ba_stats['interpretation']}]")
    print(f"  95% LoA：           [{ba_stats['loa_lower']:.3f}, {ba_stats['loa_upper']:.3f}] 隻/m²")
    print()
    mi_val = mi_stats.get("I")
    print(f"  Moran's I（人工）： {mi_val:.3f if mi_val is not None else 'N/A'} [{mi_stats.get('interpretation', 'N/A')}]")
    print("=" * 60)

    # --- 整合報告 ---
    report = {
        "meta": {
            "tool": "validate_density.py",
            "date": __import__("datetime").datetime.now().isoformat(),
            "quadrat_csv": args.quadrat,
            "missions_dir": args.missions,
            "match_radius_m": args.radius,
            "min_confidence": args.min_confidence,
            "n_quadrats": n,
            "n_robot_burrows": len(burrows),
        },
        "summary": {
            "manual_density_mean":  float(np.mean(y_true)),
            "robot_density_mean":   float(np.mean(y_pred)),
            "manual_density_std":   float(np.std(y_true)),
            "robot_density_std":    float(np.std(y_pred)),
        },
        "rmse_mae":       err_stats,
        "r2":             r2_stats,
        "bland_altman":   ba_stats,
        "morans_i":       mi_stats,
        "matched_data":   matched,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[報告] 已儲存：{args.output}")

    # --- 繪圖 ---
    if args.plot:
        if not HAS_MATPLOTLIB:
            print("[WARN] matplotlib 未安裝，無法繪圖：pip install matplotlib")
        else:
            out_dir = Path(args.output).parent
            plot_bland_altman(ba_stats, str(out_dir / "validation_bland_altman.png"))
            plot_scatter(y_true, y_pred, r2_val, str(out_dir / "validation_scatter.png"))

    # --- 快速品質評估 ---
    print("\n[品質評估]")
    if r2_val is not None:
        if r2_val >= 0.8:
            print(f"  ✓ r²={r2_val:.3f} ≥ 0.8：機器人密度估計具高度相關性")
        elif r2_val >= 0.5:
            print(f"  ~ r²={r2_val:.3f}：中等相關性，建議增加訓練資料或校正 YOLO")
        else:
            print(f"  ✗ r²={r2_val:.3f} < 0.5：相關性不足，需檢查系統偏差")

    if abs(ba_stats["bias"]) > 2.0:
        print(f"  ✗ 偏差 {ba_stats['bias']:.2f} 隻/m² 過大，需校正洞穴計數邏輯")
    else:
        print(f"  ✓ 偏差 {ba_stats['bias']:.2f} 隻/m² 在可接受範圍")

    loa_width = ba_stats["loa_upper"] - ba_stats["loa_lower"]
    print(f"  LoA 寬度：{loa_width:.2f} 隻/m²"
          f"（{'偏寬，一致性待改善' if loa_width > 5 else '可接受'}）")


if __name__ == "__main__":
    main()
