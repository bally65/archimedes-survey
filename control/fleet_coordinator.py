"""
fleet_coordinator.py
====================
阿基米德艦隊協調器（5 台機器人任務分配）

功能：
  1. 讀取 data/priority_cells.json（151 格網格，含優先分數）
  2. 按地理區域分組，貪婪分配格網給 N 台機器人（預設 5 台）
  3. 輸出 data/robot_{1..5}_waypoints.json（每台機器人的任務清單）
  4. 模擬 LoRa 心跳協議（每 30s 廣播位置）
  5. 碰撞迴避：兩台機器人距離 < 10m 時，低優先機器人暫停 60s

演算法（Zone-aware Greedy Allocation）：
  - 先按 zone（地理區域）分組，確保同一台機器人盡量在同一區域作業
  - 在區域內按 priority_score 降序排列
  - Round-robin 輪流分配給各機器人（保持負載平衡）
  - 輸出為 GeoJSON-friendly waypoint JSON

使用方式：
  python control/fleet_coordinator.py \
    --cells data/priority_cells.json \
    --n-robots 5 \
    --output-dir data \
    --simulate-heartbeat

輸出格式 (robot_N_waypoints.json)：
  {
    "robot_id": 1,
    "zone_primary": "王功",
    "total_cells": 31,
    "estimated_duration_min": 240,
    "waypoints": [
      {
        "seq": 1,
        "cell": "G3",
        "lat": 23.962,
        "lon": 120.314,
        "priority_score": 95.0,
        "zone": "王功",
        "area_m2": 20000.0,
        "action": "SURVEY"
      },
      ...
    ]
  }

LoRa 心跳協議（簡化模擬）：
  每台機器人每 30s 廣播：
    {"robot_id": N, "lat": ..., "lon": ..., "battery_pct": ..., "status": "SURVEY"}
  協調器接收後檢查：
    - 兩台機器人距離 < 10m → 低優先（ID 較大）暫停 60s
    - 任一機器人電量 < 20% → 標記為 RTH
"""

import argparse
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# 常數
# ---------------------------------------------------------------------------
DEFAULT_N_ROBOTS          = 5       # 艦隊規模
COLLISION_RADIUS_M        = 10.0    # 碰撞迴避距離（公尺）
COLLISION_PAUSE_S         = 60      # 低優先機器人暫停時間（秒）
HEARTBEAT_INTERVAL_S      = 30      # LoRa 心跳間隔（秒）
SURVEY_SPEED_M_MIN        = 5.0     # 調查速度（公尺/分鐘），模擬用
LOW_BATTERY_PCT           = 20.0    # 低電量閾值
CELL_SIZE_M               = 20.0    # 格網調查距離估算（公尺，對角線）


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

def load_priority_cells(path: str) -> list[dict]:
    """載入格網優先分數 JSON"""
    with open(path, "r", encoding="utf-8") as f:
        cells = json.load(f)
    print(f"[載入] {len(cells)} 個格網，來自 {path}")
    return cells


# ---------------------------------------------------------------------------
# 分配演算法
# ---------------------------------------------------------------------------

def zone_aware_greedy_allocate(
    cells: list[dict],
    n_robots: int
) -> dict[int, list[dict]]:
    """
    Zone-aware 貪婪分配：
    1. 統計各 zone 的格網數與總優先分數
    2. 依 zone 優先分數排序，高優先區域優先分配（確保重要區域有專屬機器人）
    3. Zone 內格網按 priority_score 降序排列
    4. Round-robin 分配給各機器人
    """
    # 按 zone 分組
    zone_cells: dict[str, list] = defaultdict(list)
    for cell in cells:
        z = cell.get("zone", "未知")
        zone_cells[z].append(cell)

    # 每個 zone 內按 priority 降序排列
    for z in zone_cells:
        zone_cells[z].sort(key=lambda c: c.get("priority_score", 0), reverse=True)

    # Zone 按總優先分數排序（大優先分總和的 zone 排前）
    zone_order = sorted(
        zone_cells.keys(),
        key=lambda z: sum(c.get("priority_score", 0) for c in zone_cells[z]),
        reverse=True,
    )

    print(f"[分配] 發現 {len(zone_order)} 個地理區域：{zone_order}")

    # 初始化機器人容器
    assignments: dict[int, list] = {r: [] for r in range(1, n_robots + 1)}

    # 先嘗試整個 zone 分配給一台機器人（減少移動距離）
    robot_idx  = 1
    zone_robot: dict[str, int] = {}   # zone → 主要負責機器人

    # 計算每台機器人負責的 zone（盡量均等）
    zone_to_robot: dict[str, int] = {}
    for i, z in enumerate(zone_order):
        assigned_robot = (i % n_robots) + 1
        zone_to_robot[z] = assigned_robot
        zone_robot[z]     = assigned_robot

    print(f"[分配] Zone → 機器人：{zone_to_robot}")

    # 按 zone 分配格網
    for z in zone_order:
        r = zone_to_robot[z]
        for cell in zone_cells[z]:
            assignments[r].append(cell)

    # 按 priority 重新排列每台機器人的任務（任務內最優先先做）
    for r in assignments:
        assignments[r].sort(key=lambda c: c.get("priority_score", 0), reverse=True)

    return assignments


def estimate_duration_min(cells: list[dict]) -> float:
    """
    估計任務時間（分鐘）
    = 移動時間 + SETTLING(3min) + 掃描(2min) 每格
    """
    if not cells:
        return 0.0
    n = len(cells)
    # 估算移動距離：相鄰格網之間的距離加總
    travel_m = 0.0
    for i in range(1, n):
        prev = cells[i - 1]
        curr = cells[i]
        travel_m += haversine_m(
            prev.get("lat", 0), prev.get("lon", 0),
            curr.get("lat", 0), curr.get("lon", 0)
        )

    travel_min  = travel_m / SURVEY_SPEED_M_MIN
    settle_min  = n * 3.0     # 每格 SETTLING 3 分鐘
    scan_min    = n * 2.0     # 每格 C-scan 2 分鐘
    return travel_min + settle_min + scan_min


def build_waypoints(robot_id: int, cells: list[dict]) -> dict:
    """建立單台機器人的 waypoint JSON"""
    waypoints = []
    for seq, cell in enumerate(cells, start=1):
        wp = {
            "seq":            seq,
            "cell":           cell.get("cell", f"C{seq:03d}"),
            "lat":            cell.get("lat", 0.0),
            "lon":            cell.get("lon", 0.0),
            "priority_score": cell.get("priority_score", 0.0),
            "zone":           cell.get("zone", "未知"),
            "area_m2":        cell.get("area_m2", 0.0),
            "action":         "SURVEY",   # auto_navigate 可直接用
        }
        waypoints.append(wp)

    # 主要作業區域（出現最多的 zone）
    if cells:
        zone_counts: dict[str, int] = defaultdict(int)
        for c in cells:
            zone_counts[c.get("zone", "未知")] += 1
        primary_zone = max(zone_counts, key=lambda z: zone_counts[z])
    else:
        primary_zone = "無"

    duration = estimate_duration_min(cells)

    return {
        "robot_id":              robot_id,
        "generated_at":          datetime.now().isoformat(),
        "zone_primary":          primary_zone,
        "total_cells":           len(cells),
        "estimated_duration_min": round(duration, 1),
        "lora_heartbeat_s":      HEARTBEAT_INTERVAL_S,
        "waypoints":             waypoints,
    }


# ---------------------------------------------------------------------------
# LoRa 心跳模擬
# ---------------------------------------------------------------------------

class LoraHeartbeatSimulator:
    """
    模擬 LoRa 心跳協議（實際部署時由各機器人 lora_bridge.py 廣播）
    """

    def __init__(self, assignments: dict[int, list]):
        self.robots: dict[int, dict] = {}
        for robot_id, cells in assignments.items():
            if cells:
                start = cells[0]
                self.robots[robot_id] = {
                    "robot_id":    robot_id,
                    "lat":         start.get("lat", 0.0),
                    "lon":         start.get("lon", 0.0),
                    "battery_pct": 100.0,
                    "status":      "SURVEY",
                    "paused_until": None,
                    "current_cell_idx": 0,
                    "cells": cells,
                }
        self.collision_log: list[dict] = []

    def tick(self, elapsed_s: float = 30.0):
        """模擬一個心跳週期"""
        now_time = datetime.now()

        # 更新各機器人位置（簡單線性插值）
        for rid, r in self.robots.items():
            if r["status"] in ("RTH", "PAUSED"):
                continue

            cells = r["cells"]
            idx   = r["current_cell_idx"]
            if idx < len(cells):
                # 移動到當前格網中心
                r["lat"] = cells[idx].get("lat", r["lat"])
                r["lon"] = cells[idx].get("lon", r["lon"])
                # 每 30s 大約完成一小段（5m/min × 0.5min = 2.5m）
                # 簡化：每個 tick 推進一個格網（用於模擬）
                r["current_cell_idx"] = min(idx + 1, len(cells) - 1)

            # 電量消耗模擬（每 tick -0.5%）
            r["battery_pct"] = max(0.0, r["battery_pct"] - 0.5)
            if r["battery_pct"] < LOW_BATTERY_PCT:
                r["status"] = "RTH"
                print(f"  [LoRa] Robot {rid} 電量 {r['battery_pct']:.1f}% < {LOW_BATTERY_PCT}% → RTH")

        # 碰撞迴避檢查（所有機器人兩兩比較）
        robot_ids = list(self.robots.keys())
        for i in range(len(robot_ids)):
            for j in range(i + 1, len(robot_ids)):
                r1 = self.robots[robot_ids[i]]
                r2 = self.robots[robot_ids[j]]

                if r1["status"] in ("RTH",) or r2["status"] in ("RTH",):
                    continue

                d = haversine_m(r1["lat"], r1["lon"], r2["lat"], r2["lon"])
                if d < COLLISION_RADIUS_M:
                    # 低優先（ID 較大）的機器人暫停
                    lower_r = r1 if r1["robot_id"] > r2["robot_id"] else r2
                    if lower_r["status"] != "PAUSED":
                        lower_r["status"] = "PAUSED"
                        lower_r["paused_until"] = now_time + timedelta(seconds=COLLISION_PAUSE_S)
                        collision_event = {
                            "time": now_time.isoformat(),
                            "robot_paused": lower_r["robot_id"],
                            "other_robot": r1["robot_id"] if lower_r is r2 else r2["robot_id"],
                            "distance_m": round(d, 2),
                            "pause_s": COLLISION_PAUSE_S,
                        }
                        self.collision_log.append(collision_event)
                        print(f"  [碰撞迴避] Robot {lower_r['robot_id']} 距 "
                              f"Robot {collision_event['other_robot']} 僅 {d:.2f}m "
                              f"→ 暫停 {COLLISION_PAUSE_S}s")

                # 解除暫停
                for rid in robot_ids:
                    r = self.robots[rid]
                    if r["status"] == "PAUSED" and r["paused_until"]:
                        if now_time >= r["paused_until"]:
                            r["status"] = "SURVEY"
                            r["paused_until"] = None
                            print(f"  [LoRa] Robot {rid} 暫停結束，恢復 SURVEY")

    def status_snapshot(self) -> list[dict]:
        """返回所有機器人當前狀態快照"""
        return [
            {
                "robot_id":    r["robot_id"],
                "lat":         round(r["lat"], 6),
                "lon":         round(r["lon"], 6),
                "battery_pct": round(r["battery_pct"], 1),
                "status":      r["status"],
                "cell_progress": f"{r['current_cell_idx']}/{len(r['cells'])}",
            }
            for r in self.robots.values()
        ]

    def check_retask(
        self,
        stale_threshold_s: float = 300.0,
        last_seen: dict[int, float] | None = None,
    ) -> list[dict]:
        """
        動態接管（Dynamic Re-tasking）：
        若機器人超過 stale_threshold_s 未回報（心跳逾時），
        將其未完成的格網釋放並分配給最近的可用機器人。

        last_seen: {robot_id: timestamp_s}（由外部 LoRa 接收器提供）
        返回：接管事件列表
        """
        if last_seen is None:
            return []

        now = time.monotonic()
        events = []

        for rid, r in self.robots.items():
            if r["status"] == "RTH":
                continue
            ts = last_seen.get(rid, 0)
            if now - ts < stale_threshold_s:
                continue

            # 此機器人逾時，釋放剩餘格網
            remaining_idx = r["current_cell_idx"]
            remaining_cells = r["cells"][remaining_idx:]
            if not remaining_cells:
                continue

            # 尋找最近的可用機器人
            candidates = [
                (other_rid, other_r)
                for other_rid, other_r in self.robots.items()
                if other_rid != rid and other_r["status"] not in ("RTH", "PAUSED")
                and other_r["battery_pct"] > LOW_BATTERY_PCT
            ]
            if not candidates:
                continue

            # 選距離最近且電量最高的候選者
            best_rid, best_r = min(
                candidates,
                key=lambda x: (
                    haversine_m(x[1]["lat"], x[1]["lon"], r["lat"], r["lon"])
                    - x[1]["battery_pct"] * 100  # 電量高者優先（折衷因子）
                )
            )

            # 將剩餘格網加到接管機器人的待辦列表尾端
            best_r["cells"].extend(remaining_cells)
            r["cells"] = r["cells"][:remaining_idx]  # 截斷原機器人
            r["status"] = "LOST"                       # 標記為失聯

            event = {
                "time":           datetime.now().isoformat(),
                "failed_robot":   rid,
                "takeover_robot": best_rid,
                "n_cells_retasked": len(remaining_cells),
            }
            events.append(event)
            print(f"  [接管] Robot {rid} 失聯 → {len(remaining_cells)} 格轉交 Robot {best_rid}")

        return events


# ---------------------------------------------------------------------------
# 主函數
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="阿基米德艦隊協調器：格網任務分配 + LoRa 碰撞迴避"
    )
    parser.add_argument(
        "--cells", "-c",
        default="data/priority_cells.json",
        help="格網優先分數 JSON（預設 data/priority_cells.json）"
    )
    parser.add_argument(
        "--n-robots", "-n",
        type=int,
        default=DEFAULT_N_ROBOTS,
        help=f"機器人數量（預設 {DEFAULT_N_ROBOTS}）"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="data",
        help="輸出目錄（預設 data/）"
    )
    parser.add_argument(
        "--simulate-heartbeat",
        action="store_true",
        help="執行 LoRa 心跳模擬（3 個週期，展示碰撞迴避）"
    )
    parser.add_argument(
        "--min-priority",
        type=float,
        default=0.0,
        help="只分配 priority_score >= 此值的格網（預設 0，全部）"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("阿基米德艦隊協調器")
    print(f"機器人數量：{args.n_robots} 台")
    print("=" * 60)

    # --- 載入格網 ---
    cells = load_priority_cells(args.cells)

    # 過濾低優先格網
    if args.min_priority > 0:
        before = len(cells)
        cells = [c for c in cells if c.get("priority_score", 0) >= args.min_priority]
        print(f"[過濾] priority ≥ {args.min_priority}：{before} → {len(cells)} 格")

    # --- 分配 ---
    assignments = zone_aware_greedy_allocate(cells, args.n_robots)

    # --- 輸出 waypoint 檔案 ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[分配結果]")
    for robot_id in sorted(assignments.keys()):
        robot_cells = assignments[robot_id]
        wp_data = build_waypoints(robot_id, robot_cells)

        out_path = output_dir / f"robot_{robot_id}_waypoints.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(wp_data, f, ensure_ascii=False, indent=2)

        print(f"  Robot {robot_id}: {wp_data['total_cells']:3d} 格 | "
              f"主要區域：{wp_data['zone_primary']:<8} | "
              f"預估 {wp_data['estimated_duration_min']:.0f} 分鐘 | "
              f"→ {out_path}")

    # --- 分配摘要 JSON ---
    summary = {
        "generated_at":    datetime.now().isoformat(),
        "n_robots":        args.n_robots,
        "total_cells":     len(cells),
        "cells_per_robot": {
            str(rid): len(assignments[rid]) for rid in sorted(assignments)
        },
        "lora_protocol": {
            "heartbeat_s":       HEARTBEAT_INTERVAL_S,
            "collision_radius_m": COLLISION_RADIUS_M,
            "pause_on_conflict_s": COLLISION_PAUSE_S,
        },
    }
    summary_path = output_dir / "fleet_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[摘要] 已儲存：{summary_path}")

    # --- LoRa 心跳模擬 ---
    if args.simulate_heartbeat:
        print("\n" + "=" * 60)
        print("LoRa 心跳模擬（3 週期，每週期 30s）")
        print("=" * 60)
        sim = LoraHeartbeatSimulator(assignments)

        for tick_n in range(1, 4):
            print(f"\n--- Tick {tick_n} (t={tick_n * 30}s) ---")
            sim.tick(elapsed_s=30.0)
            snapshot = sim.status_snapshot()
            for s in snapshot:
                print(f"  Robot {s['robot_id']}: [{s['status']:6s}] "
                      f"lat={s['lat']:.4f} lon={s['lon']:.4f} "
                      f"bat={s['battery_pct']:.1f}% "
                      f"進度={s['cell_progress']}")

        # 儲存模擬碰撞記錄
        if sim.collision_log:
            coll_path = output_dir / "fleet_collision_log.json"
            with open(coll_path, "w", encoding="utf-8") as f:
                json.dump(sim.collision_log, f, ensure_ascii=False, indent=2)
            print(f"\n[碰撞記錄] {len(sim.collision_log)} 次 → {coll_path}")
        else:
            print("\n[碰撞記錄] 無碰撞事件（格網分離良好）")

    print("\n[完成] 艦隊任務檔案輸出完畢")
    print(f"使用方式：各機器人載入對應的 robot_N_waypoints.json")
    print(f"搭配 auto_navigate.py 的 GOTO 模式逐點執行")


if __name__ == "__main__":
    main()
