"""
tide_predictor.py
=================
潮汐預測模組：整合中央氣象署開放資料 API，計算彰化海岸安全作業窗口。

功能：
  1. 查詢 CWA API 取得未來 24h 潮位預報
  2. 計算今日低潮時段（退潮開始→低潮→漲潮警戒）
  3. 提供 go/no-go 決策與剩餘安全時間
  4. 緊急漲潮警報（漲潮速度 > 閾值時立即觸發 RTH）

使用方式：
  python control/tide_predictor.py --station 王功 --preview 24

ROS2 整合：
  from control.tide_predictor import TidePredictor
  tp = TidePredictor(station_id="466940")
  window = tp.get_safe_window()
"""

import argparse
import json
import math
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone
from typing import Optional

# ---------------------------------------------------------------------------
# 台灣潮位觀測站 ID（中央氣象署）
# 最近彰化作業區的觀測站
# ---------------------------------------------------------------------------
STATION_IDS = {
    "王功":   "C0R120",   # 王功潮位站
    "台中":   "C0R080",   # 台中港
    "鹿港":   "C0R110",   # 鹿港
    "布袋":   "C0R150",   # 布袋（備用）
}

# 安全參數
SAFE_WATER_LEVEL_CM     =  30    # 最低安全水位 (cm)：低於此值才允許作業
ABORT_RISE_RATE_CM_MIN  =   3.0  # cm/min：超過此漲潮速度立即 RTH（模擬失敗閾值 35cm/60s ≈ 0.583cm/s）
SAFE_MARGIN_MIN         =  20    # 作業結束前預留的撤退緩衝時間（分鐘）
MIN_WINDOW_MIN          =  30    # 至少要有這麼多分鐘才值得出發

# CWA Open Data API（免費，需申請 API Key）
CWA_API_BASE = "https://opendata.cwa.gov.tw/api/v1/rest/datastore"
CWA_TIDE_OBS  = "O-A0075-001"   # 潮位觀測
CWA_TIDE_FORE = "F-A0023-001"   # 潮汐預報

TZ_CST = timezone(timedelta(hours=8))  # 台灣時間 UTC+8


# ---------------------------------------------------------------------------
# 備用：諧波預測（離線模式，基於彰化/王功主要分潮係數）
# 當 API 無法連線時自動切換
# ---------------------------------------------------------------------------
# 分潮係數（H=振幅cm, g=遲角°）- 根據台灣西海岸的實測數據近似值
# 主要分潮：M2（太陰半日潮）+ S2（太陽半日潮）+ K1 + O1（全日潮）
HARMONIC_CONSTITUENTS = {
    "M2": {"H": 180.0, "T": 12.4206, "g": 195.0},   # 主要半日潮
    "S2": {"H":  55.0, "T": 12.0000, "g": 220.0},   # 太陽半日潮
    "N2": {"H":  38.0, "T": 12.6583, "g": 185.0},   # 橢圓半日潮
    "K1": {"H":  35.0, "T": 23.9345, "g": 130.0},   # 全日潮
    "O1": {"H":  28.0, "T": 25.8194, "g": 125.0},   # 全日潮
    "M4": {"H":  18.0, "T":  6.2103, "g":  80.0},   # 四分之一日潮（淺水效應）
}
MLLW_OFFSET_CM = 200  # 平均低低潮面偏移（使水位值為正）


def harmonic_tide_cm(t: datetime) -> float:
    """計算某時刻的理論潮位（公分），離線備用模式。"""
    hours_since_epoch = (t.timestamp() - datetime(2000, 1, 1, 0, 0, 0,
                        tzinfo=timezone.utc).timestamp()) / 3600.0
    level = MLLW_OFFSET_CM
    for name, c in HARMONIC_CONSTITUENTS.items():
        omega = 2 * math.pi / c["T"]           # rad/hour
        phi   = math.radians(c["g"])
        level += c["H"] * math.cos(omega * hours_since_epoch - phi)
    return level


# ---------------------------------------------------------------------------
class TidePredictor:
    def __init__(self, station: str = "王功", api_key: str = "",
                 use_offline: bool = False):
        self.station    = station
        self.station_id = STATION_IDS.get(station, "C0R120")
        self.api_key    = api_key
        self.use_offline = use_offline
        self._cache: list[dict] = []

    # ------------------------------------------------------------------
    def fetch_forecast(self, hours: int = 24) -> list[dict]:
        """取得未來 N 小時的潮位預報，每 10 分鐘一筆。"""
        if self.use_offline or not self.api_key:
            return self._offline_forecast(hours)
        try:
            return self._api_forecast(hours)
        except Exception as e:
            print(f"[TidePredictor] API 連線失敗：{e}，切換離線模式")
            return self._offline_forecast(hours)

    def _offline_forecast(self, hours: int) -> list[dict]:
        """使用諧波模型產生離線預報（10 分鐘解析度）。"""
        now = datetime.now(TZ_CST)
        records = []
        steps = hours * 6  # 每 10 分鐘一筆
        for i in range(steps + 1):
            t = now + timedelta(minutes=10 * i)
            level = harmonic_tide_cm(t)
            records.append({
                "time":  t.isoformat(),
                "level": round(level, 1),
                "source": "harmonic_offline"
            })
        self._cache = records
        return records

    def _api_forecast(self, hours: int) -> list[dict]:
        """呼叫 CWA Open Data API。"""
        url = (f"{CWA_API_BASE}/{CWA_TIDE_FORE}"
               f"?Authorization={self.api_key}"
               f"&StationId={self.station_id}"
               f"&timeFrom={datetime.now(TZ_CST).strftime('%Y-%m-%dT%H:%M:%S')}"
               f"&timeTo={(datetime.now(TZ_CST)+timedelta(hours=hours)).strftime('%Y-%m-%dT%H:%M:%S')}")
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        records = []
        for item in data.get("records", {}).get("Station", [{}])[0].get("TideData", []):
            records.append({
                "time":   item["DateTime"],
                "level":  float(item["WaterLevel"]) * 100,  # m → cm
                "source": "cwa_api"
            })
        self._cache = records
        return records

    # ------------------------------------------------------------------
    def find_low_tides(self, records: list[dict]) -> list[dict]:
        """找出所有低潮時刻（局部最小值）。"""
        lows = []
        levels = [r["level"] for r in records]
        for i in range(1, len(levels) - 1):
            if levels[i] < levels[i-1] and levels[i] < levels[i+1]:
                lows.append({
                    "time":  records[i]["time"],
                    "level": levels[i],
                    "idx":   i
                })
        return lows

    # ------------------------------------------------------------------
    def get_safe_window(self, records: Optional[list[dict]] = None
                        ) -> dict:
        """
        計算距離現在最近的安全作業窗口。

        回傳：
          go         : bool - 是否可以出發
          reason     : str  - 決策理由
          window_start / window_end : datetime
          safe_minutes : int - 可用的安全分鐘數
          current_level: float - 當前潮位 (cm)
          rise_rate   : float - 當前漲潮速度 (cm/min)
          next_low    : dict  - 下一個低潮資訊
        """
        if records is None:
            records = self.fetch_forecast(hours=24)

        now   = datetime.now(TZ_CST)
        now_s = now.isoformat()

        # 找最近時刻
        idx_now = 0
        for i, r in enumerate(records):
            if r["time"] >= now_s:
                idx_now = i
                break

        current_level = records[idx_now]["level"]

        # 當前漲退速度（cm/min，10分鐘差分）
        if idx_now > 0:
            dt_level = records[idx_now]["level"] - records[idx_now-1]["level"]
            rise_rate = dt_level / 10.0   # cm/min（正=漲，負=退）
        else:
            rise_rate = 0.0

        # 找下一個低潮
        lows = self.find_low_tides(records)
        next_low = None
        for low in lows:
            if low["time"] >= now_s:
                next_low = low
                break

        # 緊急狀態：當前已在漲潮且速度超閾值
        if rise_rate > ABORT_RISE_RATE_CM_MIN:
            return {
                "go": False,
                "reason": f"⚠️  漲潮速度 {rise_rate:.2f} cm/min 超過安全閾值 {ABORT_RISE_RATE_CM_MIN} cm/min — 禁止出發／立即 RTH",
                "window_start":   None,
                "window_end":     None,
                "safe_minutes":   0,
                "current_level":  current_level,
                "rise_rate":      rise_rate,
                "next_low":       next_low,
            }

        # 計算作業窗口：找水位低於 SAFE_WATER_LEVEL_CM 的連續時段
        safe_start = None
        safe_end   = None
        for i in range(idx_now, len(records)):
            if records[i]["level"] <= SAFE_WATER_LEVEL_CM:
                if safe_start is None:
                    safe_start = records[i]["time"]
                safe_end = records[i]["time"]
            else:
                if safe_start is not None:
                    break  # 找到第一段連續低潮窗口

        safe_minutes = 0
        if safe_start and safe_end:
            t_start = datetime.fromisoformat(safe_start)
            t_end   = datetime.fromisoformat(safe_end)
            safe_minutes = int((t_end - t_start).total_seconds() / 60)
            # 扣掉撤退緩衝
            usable_minutes = safe_minutes - SAFE_MARGIN_MIN
        else:
            usable_minutes = 0

        go     = usable_minutes >= MIN_WINDOW_MIN
        if go:
            reason = (f"✅ 可出發  作業窗口 {safe_start[11:16]}～{safe_end[11:16]} "
                      f"（{safe_minutes} 分鐘，扣除緩衝 {usable_minutes} 分鐘可用）")
        elif safe_start:
            reason = (f"⏳ 窗口太短（{usable_minutes} 分鐘 < {MIN_WINDOW_MIN} 分鐘）"
                      f"  目前潮位 {current_level:.0f} cm")
        else:
            low_time = next_low["time"][11:16] if next_low else "未知"
            reason = f"⏳ 等待低潮  下一次低潮約 {low_time}  目前潮位 {current_level:.0f} cm"

        return {
            "go":            go,
            "reason":        reason,
            "window_start":  safe_start,
            "window_end":    safe_end,
            "safe_minutes":  safe_minutes,
            "usable_minutes": usable_minutes,
            "current_level": current_level,
            "rise_rate":     rise_rate,
            "next_low":      next_low,
        }

    # ------------------------------------------------------------------
    def realtime_monitor(self, callback_rth=None, interval_s: int = 60):
        """
        持續監控潮位，每 interval_s 秒刷新一次。
        當漲潮速度超過閾值時呼叫 callback_rth()。
        """
        print(f"[TidePredictor] 開始即時潮位監控 (站：{self.station}, 每 {interval_s}s 更新)")
        while True:
            records = self.fetch_forecast(hours=3)
            window  = self.get_safe_window(records)
            now_str = datetime.now(TZ_CST).strftime("%H:%M:%S")
            print(f"[{now_str}] 潮位：{window['current_level']:.0f} cm  "
                  f"漲退：{window['rise_rate']:+.2f} cm/min  {window['reason'][:60]}")
            if not window["go"] and window["rise_rate"] > ABORT_RISE_RATE_CM_MIN:
                print("🚨 緊急漲潮！觸發 RTH！")
                if callback_rth:
                    callback_rth()
            time.sleep(interval_s)

    # ------------------------------------------------------------------
    def print_forecast_table(self, records: list[dict], hours: int = 12):
        """打印未來 N 小時的潮位預報表。"""
        print(f"\n{'─'*50}")
        print(f"  潮位預報  站：{self.station}（來源：{records[0]['source'] if records else 'N/A'}）")
        print(f"{'─'*50}")
        print(f"  {'時間':8}  {'水位(cm)':>9}  {'狀態':10}")
        print(f"{'─'*50}")
        limit = min(len(records), hours * 6)
        for r in records[:limit]:
            t     = r["time"][11:16]
            level = r["level"]
            if level <= SAFE_WATER_LEVEL_CM:
                status = "✅ 可作業"
            elif level <= SAFE_WATER_LEVEL_CM + 30:
                status = "⚠️  警戒"
            else:
                status = "❌ 禁止"
            print(f"  {t:8}  {level:9.1f}  {status}")
        print(f"{'─'*50}\n")


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Archimedes 潮汐預測工具")
    parser.add_argument("--station",  default="王功",
                        choices=list(STATION_IDS.keys()))
    parser.add_argument("--api-key",  default="",
                        help="CWA Open Data API Key（可從 opendata.cwa.gov.tw 免費申請）")
    parser.add_argument("--preview",  type=int, default=12,
                        help="預報顯示時數（預設 12h）")
    parser.add_argument("--monitor",  action="store_true",
                        help="持續監控模式（每 60 秒刷新）")
    parser.add_argument("--offline",  action="store_true",
                        help="使用離線諧波模型（不需 API Key）")
    args = parser.parse_args()

    tp = TidePredictor(station=args.station, api_key=args.api_key,
                       use_offline=args.offline or not args.api_key)

    records = tp.fetch_forecast(hours=args.preview)
    tp.print_forecast_table(records, hours=args.preview)

    window = tp.get_safe_window(records)
    print("作業窗口評估：")
    print(f"  {window['reason']}")
    print(f"  目前潮位：{window['current_level']:.0f} cm")
    print(f"  漲退速度：{window['rise_rate']:+.2f} cm/min")
    if window.get("next_low"):
        print(f"  下一低潮：{window['next_low']['time'][11:16]}  "
              f"水位 {window['next_low']['level']:.0f} cm")

    if args.monitor:
        tp.realtime_monitor()


if __name__ == "__main__":
    main()
