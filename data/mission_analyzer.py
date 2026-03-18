"""
mission_analyzer.py
====================
GeoJSON 任務資料後處理管線。

功能：
  1. 讀取 ~/archimedes_missions/ 下所有任務 GeoJSON 檔案
  2. 計算洞穴密度（隻/m²）與族群估算
  3. 產生互動式熱力圖 HTML（folium）
  4. 時間序列比較（不同日期、不同保育區）
  5. 輸出標準化 CSV 報告
  6. 整合超音波深度資料（若有）

依賴：
  pip install folium pandas shapely geojson

使用：
  python data/mission_analyzer.py --dir ~/archimedes_missions --output report
  python data/mission_analyzer.py --compare 2026-03 2026-04 --station 王功
"""

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import folium
    from folium.plugins import HeatMap
    _FOLIUM = True
except ImportError:
    _FOLIUM = False
    print("[Warning] folium 未安裝，跳過地圖輸出。 pip install folium")

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False


# ---------------------------------------------------------------------------
# 常數
# ---------------------------------------------------------------------------
BURROW_DENSITY_INDIVIDUAL_FACTOR = 1.0   # 每個洞口估算 1 隻（蝦猴獨居）
DEFAULT_MISSIONS_DIR = Path.home() / "archimedes_missions"
MIN_CONFIDENCE       = 0.4   # 低信心度的洞穴偵測過濾閾值


# ---------------------------------------------------------------------------
class BurrowRecord:
    """單一洞穴偵測記錄。"""
    __slots__ = ["lat", "lon", "world_x", "world_y", "confidence",
                 "depth_cm", "burrow_angle_deg", "timestamp", "mission_id"]

    def __init__(self, feature: dict, mission_id: str = ""):
        props = feature.get("properties", {})
        coords = feature.get("geometry", {}).get("coordinates", [0, 0])
        self.lon          = coords[0]
        self.lat          = coords[1]
        self.world_x      = props.get("world_x_m",            0.0)
        self.world_y      = props.get("world_y_m",            0.0)
        self.confidence   = props.get("confidence",           0.0)
        self.depth_cm     = props.get("burrow_depth_cm",      None)
        self.burrow_angle_deg = props.get("burrow_angle_deg", None)
        self.timestamp    = props.get("timestamp",            "")
        self.mission_id   = mission_id


class MissionData:
    """單一任務（一次出航）的完整資料。"""
    def __init__(self, mission_dir: Path):
        self.mission_id    = mission_dir.name if mission_dir.is_dir() else mission_dir.stem
        self.date          = self._parse_date(self.mission_id)
        self.track_points  : list[tuple[float, float]] = []
        self.burrows       : list[BurrowRecord] = []
        self.coverage_m2   : float = 0.0
        self._load(mission_dir)

    def _parse_date(self, name: str) -> Optional[datetime]:
        try:
            return datetime.strptime(name[:10], "%Y-%m-%d")
        except (ValueError, IndexError):
            return None

    def _load(self, path: Path):
        """載入 track + burrows GeoJSON。"""
        if path.is_dir():
            track_files  = list(path.glob("*_track.geojson"))
            burrow_files = list(path.glob("*_burrows.geojson"))
        else:
            # 單一檔案模式（burrow 檔）
            track_files  = []
            burrow_files = [path]

        for f in track_files:
            self._load_track(f)
        for f in burrow_files:
            self._load_burrows(f)

        self.coverage_m2 = self._estimate_coverage()

    def _load_track(self, f: Path):
        data = json.loads(f.read_text())
        for feat in data.get("features", []):
            if feat["geometry"]["type"] == "LineString":
                self.track_points = [
                    (c[1], c[0]) for c in feat["geometry"]["coordinates"]
                ]

    def _load_burrows(self, f: Path):
        data = json.loads(f.read_text())
        for feat in data.get("features", []):
            if feat["geometry"]["type"] == "Point":
                rec = BurrowRecord(feat, mission_id=self.mission_id)
                if rec.confidence >= MIN_CONFIDENCE:
                    self.burrows.append(rec)

    def _estimate_coverage(self) -> float:
        """估算調查覆蓋面積（凸包近似）。"""
        if len(self.track_points) < 3:
            return 0.0
        # 簡單 bounding box × π/4 近似橢圓面積
        lats = [p[0] for p in self.track_points]
        lons = [p[1] for p in self.track_points]
        lat_range_m = (max(lats) - min(lats)) * 110_574
        lon_range_m = (max(lons) - min(lons)) * 111_320 * math.cos(math.radians(sum(lats)/len(lats)))
        return lat_range_m * lon_range_m * math.pi / 4

    @property
    def burrow_count(self) -> int:
        return len(self.burrows)

    @property
    def density_per_m2(self) -> float:
        if self.coverage_m2 <= 0:
            return 0.0
        return self.burrow_count / self.coverage_m2

    @property
    def population_estimate(self) -> int:
        return int(self.burrow_count * BURROW_DENSITY_INDIVIDUAL_FACTOR)

    @property
    def mean_depth_cm(self) -> Optional[float]:
        depths = [b.depth_cm for b in self.burrows if b.depth_cm is not None]
        return sum(depths) / len(depths) if depths else None


# ---------------------------------------------------------------------------
class MissionAnalyzer:
    def __init__(self, missions_dir: Path = DEFAULT_MISSIONS_DIR):
        self.missions_dir = missions_dir
        self.missions: list[MissionData] = []

    def load_all(self):
        """載入資料夾內所有任務。"""
        if not self.missions_dir.exists():
            print(f"[Analyzer] 資料夾不存在：{self.missions_dir}")
            return

        # 嘗試子目錄（每次任務一個資料夾）
        subdirs = sorted([d for d in self.missions_dir.iterdir() if d.is_dir()])
        if subdirs:
            for d in subdirs:
                try:
                    m = MissionData(d)
                    if m.burrows or m.track_points:
                        self.missions.append(m)
                except Exception as e:
                    print(f"[Warning] 載入 {d.name} 失敗：{e}")
        else:
            # 嘗試直接讀 GeoJSON 檔案
            for f in sorted(self.missions_dir.glob("*_burrows.geojson")):
                try:
                    m = MissionData(f)
                    self.missions.append(m)
                except Exception as e:
                    print(f"[Warning] 載入 {f.name} 失敗：{e}")

        print(f"[Analyzer] 已載入 {len(self.missions)} 次任務")

    # ------------------------------------------------------------------
    def print_summary(self):
        """打印各次任務摘要。"""
        print(f"\n{'='*70}")
        print(f"  Archimedes 任務分析報告")
        print(f"{'='*70}")
        print(f"  {'任務 ID':25} {'洞口數':>7} {'密度/m²':>10} {'估計族群':>10} {'平均深度':>10}")
        print(f"{'─'*70}")

        total_burrows = 0
        for m in self.missions:
            depth_str = f"{m.mean_depth_cm:.1f} cm" if m.mean_depth_cm else "N/A"
            print(f"  {m.mission_id:25} {m.burrow_count:>7} "
                  f"{m.density_per_m2:>10.4f} {m.population_estimate:>10} {depth_str:>10}")
            total_burrows += m.burrow_count

        print(f"{'─'*70}")
        print(f"  {'合計':25} {total_burrows:>7}")
        print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    def generate_heatmap(self, output_path: str = "burrow_heatmap.html"):
        """產生互動式洞穴密度熱力圖。"""
        if not _FOLIUM:
            print("[Analyzer] folium 未安裝，無法產生地圖")
            return

        # 計算地圖中心
        all_burrows = [b for m in self.missions for b in m.burrows]
        if not all_burrows:
            print("[Analyzer] 無洞穴資料，無法產生地圖")
            return

        center_lat = sum(b.lat for b in all_burrows) / len(all_burrows)
        center_lon = sum(b.lon for b in all_burrows) / len(all_burrows)

        m_map = folium.Map(location=[center_lat, center_lon], zoom_start=16,
                           tiles="CartoDB positron")

        # 熱力圖圖層
        heat_data = [[b.lat, b.lon, b.confidence] for b in all_burrows]
        HeatMap(heat_data, radius=10, blur=8,
                gradient={"0.2": "blue", "0.5": "lime", "0.8": "orange", "1.0": "red"}
                ).add_to(m_map)

        # 各任務軌跡
        for mission in self.missions:
            if mission.track_points:
                folium.PolyLine(
                    mission.track_points, color="#2196F3", weight=2, opacity=0.7,
                    tooltip=f"{mission.mission_id} 軌跡"
                ).add_to(m_map)

        # 各洞穴標記（點擊顯示詳細資料）
        for b in all_burrows:
            popup_html = (
                f"<b>任務：</b>{b.mission_id}<br>"
                f"<b>信心值：</b>{b.confidence:.2f}<br>"
                f"<b>深度：</b>{b.depth_cm or 'N/A'} cm<br>"
                f"<b>傾角：</b>{b.burrow_angle_deg or 'N/A'} °<br>"
                f"<b>時間：</b>{b.timestamp[:19] if b.timestamp else 'N/A'}"
            )
            color = "red" if b.confidence > 0.7 else "orange" if b.confidence > 0.5 else "gray"
            folium.CircleMarker(
                location=[b.lat, b.lon], radius=4, color=color,
                fill=True, fill_opacity=0.8,
                popup=folium.Popup(popup_html, max_width=200)
            ).add_to(m_map)

        m_map.save(output_path)
        print(f"[Analyzer] 地圖已儲存：{output_path}")

    # ------------------------------------------------------------------
    def export_csv(self, output_path: str = "mission_report.csv"):
        """輸出所有洞穴記錄為 CSV。"""
        rows = []
        for m in self.missions:
            for b in m.burrows:
                rows.append({
                    "mission_id":        b.mission_id,
                    "date":              m.date.strftime("%Y-%m-%d") if m.date else "",
                    "lat":               b.lat,
                    "lon":               b.lon,
                    "world_x_m":         b.world_x,
                    "world_y_m":         b.world_y,
                    "confidence":        b.confidence,
                    "depth_cm":          b.depth_cm or "",
                    "angle_deg":         b.burrow_angle_deg or "",
                    "timestamp":         b.timestamp,
                    "density_per_m2":    round(m.density_per_m2, 6),
                    "coverage_m2":       round(m.coverage_m2, 1),
                })
        if not rows:
            print("[Analyzer] 無資料可輸出")
            return

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[Analyzer] CSV 已儲存：{output_path}  ({len(rows)} 筆記錄)")

    # ------------------------------------------------------------------
    def timeseries_plot(self, output_path: str = "timeseries.html"):
        """產生族群數量時間序列圖表（簡易 HTML）。"""
        dated = [(m.date, m.burrow_count, m.density_per_m2)
                 for m in self.missions if m.date]
        dated.sort(key=lambda x: x[0])
        if not dated:
            print("[Analyzer] 無時間資料可產生時序圖")
            return

        rows_html = ""
        for dt, count, density in dated:
            rows_html += f"<tr><td>{dt.strftime('%Y-%m-%d')}</td><td>{count}</td><td>{density:.4f}</td></tr>\n"

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Archimedes 族群時序報告</title>
<style>body{{font-family:sans-serif;max-width:800px;margin:2em auto}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccc;padding:8px;text-align:center}}
th{{background:#2196F3;color:white}}</style>
</head><body>
<h2>🦐 美食奧螻蛄蝦族群監測時序報告</h2>
<p>資料來源：Archimedes Survey Robot  共 {len(dated)} 次任務</p>
<table>
<tr><th>日期</th><th>偵測洞口數</th><th>密度（隻/m²）</th></tr>
{rows_html}
</table>
<p style="color:#888;font-size:0.85em">
  信心值閾值：{MIN_CONFIDENCE}  |  每洞估算 {BURROW_DENSITY_INDIVIDUAL_FACTOR} 隻（蝦猴獨居特性）
</p>
</body></html>"""
        Path(output_path).write_text(html, encoding="utf-8")
        print(f"[Analyzer] 時序報告已儲存：{output_path}")

    # ------------------------------------------------------------------
    def density_summary(self) -> dict:
        """回傳整體統計摘要字典（可供 ROS2 節點使用）。"""
        if not self.missions:
            return {}
        all_densities = [m.density_per_m2 for m in self.missions if m.coverage_m2 > 0]
        return {
            "total_missions":     len(self.missions),
            "total_burrows":      sum(m.burrow_count for m in self.missions),
            "mean_density_per_m2": sum(all_densities) / len(all_densities) if all_densities else 0,
            "max_density_per_m2": max(all_densities) if all_densities else 0,
            "min_density_per_m2": min(all_densities) if all_densities else 0,
            "total_coverage_m2":  sum(m.coverage_m2 for m in self.missions),
        }


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Archimedes 任務資料分析器")
    parser.add_argument("--dir",     default=str(DEFAULT_MISSIONS_DIR),
                        help="任務資料夾路徑")
    parser.add_argument("--output",  default="report",
                        help="輸出檔案前綴（預設 report）")
    parser.add_argument("--no-map",  action="store_true", help="跳過地圖產生")
    args = parser.parse_args()

    analyzer = MissionAnalyzer(Path(args.dir))
    analyzer.load_all()

    if not analyzer.missions:
        print("無任務資料。請先進行實地調查，或手動放置 GeoJSON 測試資料。")
        print(f"資料夾：{args.dir}")
        return

    analyzer.print_summary()

    summary = analyzer.density_summary()
    print("整體統計：")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    analyzer.export_csv(f"{args.output}.csv")
    analyzer.timeseries_plot(f"{args.output}_timeseries.html")
    if not args.no_map:
        analyzer.generate_heatmap(f"{args.output}_heatmap.html")


if __name__ == "__main__":
    main()
