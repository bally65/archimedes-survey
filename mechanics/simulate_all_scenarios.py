# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Archimedes-Survey v2 — 全情境移動方案最佳化模擬
測試項目：
  A. 直行
  B. 左移 / 右移（橫向平移）
  C. 順時針旋轉
  D. 逆時針旋轉
  E. 左轉弧行 (R=1m / 0.5m)
  F. 右轉弧行 (R=1m / 0.5m)

負重條件：0 kg / 5 kg / 10 kg
含水量條件：10% / 30% / 50% / 70% / 90%
"""

import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'DejaVu Sans']
import matplotlib.pyplot as plt
import json, os

OUT_DIR = "C:/Users/aa598/archimedes-survey/docs"

# ══════════════════════════════════════════════════════════════
# 機器參數
# ══════════════════════════════════════════════════════════════
BASE_MASS   = 22.0      # kg 機體本身
PAYLOADS    = [0, 5, 10]  # kg 額外負重
MOTOR_RPM   = 60
PITCH       = 0.1125    # m/圈
N_TURNS     = 4
BLADE_W     = 0.050     # m
BLADE_OD    = 0.268     # m
PIPE_OD     = 0.168     # m
WHEEL_BASE  = 0.300     # m 兩螺旋間距 (DY2=300mm)
GRAVITY     = 9.81

MOISTURE_LIST = [10, 30, 50, 70, 90]

def sand_params(m):
    mu  = np.interp(m, [10,30,50,70,90], [0.65, 0.55, 0.45, 0.35, 0.28])
    c   = np.interp(m, [10,30,50,70,90], [0.5,  3.5,  6.0,  4.0,  1.5])
    eta = np.interp(m, [10,30,50,70,90], [0.40, 0.62, 0.78, 0.70, 0.55])
    return mu, c, eta

# ══════════════════════════════════════════════════════════════
# 移動方案定義
# 雙螺旋差速控制：
#   螺旋A（左）轉速 = RPM * (1 + diff_A)
#   螺旋B（右）轉速 = RPM * (1 + diff_B)
#   反轉 = 負值
# ══════════════════════════════════════════════════════════════
SCENARIOS = [
    {
        "id": "A",
        "name": "直行前進",
        "name_en": "Straight Forward",
        "rpm_A": +1.0,   # 左螺旋
        "rpm_B": +1.0,   # 右螺旋（同向同速）
        "mode": "straight",
    },
    {
        "id": "B1",
        "name": "直行後退",
        "name_en": "Straight Reverse",
        "rpm_A": -1.0,
        "rpm_B": -1.0,
        "mode": "straight",
    },
    {
        "id": "C1",
        "name": "左移（橫向平移）",
        "name_en": "Strafe Left",
        "rpm_A": +1.0,   # 左螺旋前進
        "rpm_B": -1.0,   # 右螺旋後退 → 合力向左
        "mode": "strafe",
        "note": "阿基米德螺旋可差速橫移",
    },
    {
        "id": "C2",
        "name": "右移（橫向平移）",
        "name_en": "Strafe Right",
        "rpm_A": -1.0,
        "rpm_B": +1.0,
        "mode": "strafe",
    },
    {
        "id": "D1",
        "name": "順時針旋轉（原地）",
        "name_en": "Clockwise Spin",
        "rpm_A": +1.0,   # 左螺旋前進
        "rpm_B": -1.0,   # 右螺旋後退（同 strafe，但扭矩主導）
        "mode": "spin",
        "spin_dir": +1,
        "note": "螺旋差速產生轉矩，等同坦克差速轉向",
    },
    {
        "id": "D2",
        "name": "逆時針旋轉（原地）",
        "name_en": "Counter-clockwise Spin",
        "rpm_A": -1.0,
        "rpm_B": +1.0,
        "mode": "spin",
        "spin_dir": -1,
    },
    {
        "id": "E1",
        "name": "左弧轉 R=1m",
        "name_en": "Left Arc R=1m",
        "rpm_A": +0.65,   # 內側螺旋減速
        "rpm_B": +1.00,   # 外側螺旋全速
        "mode": "arc",
        "radius": 1.0,
        "direction": "left",
    },
    {
        "id": "E2",
        "name": "左弧轉 R=0.5m",
        "name_en": "Left Arc R=0.5m",
        "rpm_A": +0.30,
        "rpm_B": +1.00,
        "mode": "arc",
        "radius": 0.5,
        "direction": "left",
    },
    {
        "id": "F1",
        "name": "右弧轉 R=1m",
        "name_en": "Right Arc R=1m",
        "rpm_A": +1.00,
        "rpm_B": +0.65,
        "mode": "arc",
        "radius": 1.0,
        "direction": "right",
    },
    {
        "id": "F2",
        "name": "右弧轉 R=0.5m",
        "name_en": "Right Arc R=0.5m",
        "rpm_A": +1.00,
        "rpm_B": +0.30,
        "mode": "arc",
        "radius": 0.5,
        "direction": "right",
    },
]

# ══════════════════════════════════════════════════════════════
# 核心計算函式
# ══════════════════════════════════════════════════════════════
def calc_scenario(scenario, moisture, payload_kg):
    total_mass = BASE_MASS + payload_kg
    mu, c, eta_base = sand_params(moisture)

    rpm_A = MOTOR_RPM * scenario["rpm_A"]
    rpm_B = MOTOR_RPM * scenario["rpm_B"]
    mode  = scenario["mode"]

    # 負重對效率的影響（增加壓力 → 增加摩擦，但也增加葉片咬合）
    load_factor = 1.0 - (payload_kg / BASE_MASS) * 0.08  # 每 10kg 降低 ~8% 效率
    eta = eta_base * load_factor

    # 各螺旋推進速度
    v_A = PITCH * abs(rpm_A) / 60 * eta  # m/s
    v_B = PITCH * abs(rpm_B) / 60 * eta

    # 阻力
    F_resist = total_mass * GRAVITY * mu

    # 推進力估算
    blade_area = BLADE_W * PITCH * N_TURNS
    F_A = (c * 1000 * blade_area + total_mass * GRAVITY * eta / 2) * np.sign(rpm_A)
    F_B = (c * 1000 * blade_area + total_mass * GRAVITY * eta / 2) * np.sign(rpm_B)

    if mode == "straight":
        # 兩條螺旋合力前進
        F_net = (F_A + F_B) - F_resist * np.sign(rpm_A)
        speed_fwd = (v_A + v_B) / 2 * 60   # m/min
        speed_lat = 0.0
        omega_deg_s = 0.0
        eff = eta * 100

    elif mode == "strafe":
        # 橫向平移：前進分量互消，橫向合力
        # 阿基米德螺旋橫移效率約 55%（葉片橫向咬沙較差）
        strafe_eta = 0.55
        speed_lat = (v_A + v_B) / 2 * strafe_eta * 60
        speed_fwd = 0.0
        omega_deg_s = 0.0
        eff = strafe_eta * eta * 100

    elif mode == "spin":
        # 原地旋轉：兩螺旋差速產生扭矩
        spin_dir = scenario.get("spin_dir", 1)
        torque = (abs(F_A) + abs(F_B)) * WHEEL_BASE / 2
        moment_inertia = total_mass * (PIPE_OD/2)**2 * 2   # 近似
        alpha = torque / moment_inertia   # rad/s²
        # 穩態角速度（阻力矩平衡）
        resist_torque = F_resist * WHEEL_BASE / 2
        omega = (torque - resist_torque) / (total_mass * 0.1)   # 近似穩態
        omega_deg_s = np.degrees(max(omega, 0)) * spin_dir
        speed_fwd = 0.0
        speed_lat = 0.0
        eff = eta * 0.75 * 100   # 旋轉損耗較多

    elif mode == "arc":
        radius = scenario["radius"]
        # 外側速度 / 內側速度 比例 = (R + d/2) / (R - d/2)
        ratio = (radius + WHEEL_BASE/2) / (radius - WHEEL_BASE/2)
        v_outer = max(v_A, v_B)
        v_inner = min(v_A, v_B)
        v_center = (v_outer + v_inner) / 2
        speed_fwd = v_center * 60
        omega_deg_s = np.degrees(v_center / radius)
        speed_lat = 0.0
        eff = eta * (1 - 0.05 / radius) * 100   # 轉彎半徑越小效率越低

    return {
        "speed_fwd_m_min":  round(speed_fwd,  3),
        "speed_lat_m_min":  round(speed_lat,  3),
        "omega_deg_s":      round(omega_deg_s, 2),
        "efficiency_pct":   round(min(eff, 100), 1),
        "net_force_N":      round(float(F_A + F_B - F_resist), 1),
    }

# ══════════════════════════════════════════════════════════════
# 執行全矩陣模擬
# ══════════════════════════════════════════════════════════════
all_results = []
for sc in SCENARIOS:
    for payload in PAYLOADS:
        for moisture in MOISTURE_LIST:
            r = calc_scenario(sc, moisture, payload)
            row = {
                "scenario_id":   sc["id"],
                "scenario_name": sc["name"],
                "mode":          sc["mode"],
                "moisture_pct":  moisture,
                "payload_kg":    payload,
                **r
            }
            all_results.append(row)

# ══════════════════════════════════════════════════════════════
# 印出摘要表
# ══════════════════════════════════════════════════════════════
print("=" * 80)
print("Archimedes-Survey v2  全情境移動最佳化模擬")
print(f"含水量 50%（最佳條件），各方案比較")
print("=" * 80)
print(f"{'方案':<20} {'模式':<8} {'前進(m/min)':<12} {'橫移(m/min)':<12} {'旋轉(°/s)':<10} {'效率%':<8} {'0kg/5kg/10kg'}")
print("-" * 80)
for sc in SCENARIOS:
    row_0  = next(r for r in all_results if r["scenario_id"]==sc["id"] and r["moisture_pct"]==50 and r["payload_kg"]==0)
    row_5  = next(r for r in all_results if r["scenario_id"]==sc["id"] and r["moisture_pct"]==50 and r["payload_kg"]==5)
    row_10 = next(r for r in all_results if r["scenario_id"]==sc["id"] and r["moisture_pct"]==50 and r["payload_kg"]==10)
    v0  = row_0["speed_fwd_m_min"]  or row_0["speed_lat_m_min"]  or abs(row_0["omega_deg_s"])
    v5  = row_5["speed_fwd_m_min"]  or row_5["speed_lat_m_min"]  or abs(row_5["omega_deg_s"])
    v10 = row_10["speed_fwd_m_min"] or row_10["speed_lat_m_min"] or abs(row_10["omega_deg_s"])
    unit = "°/s" if sc["mode"]=="spin" else "m/min"
    print(f"  {sc['name']:<18} {sc['mode']:<8} "
          f"{row_0['speed_fwd_m_min']:<12.2f} {row_0['speed_lat_m_min']:<12.2f} "
          f"{row_0['omega_deg_s']:<10.1f} {row_0['efficiency_pct']:<8.1f} "
          f"{v0:.2f} / {v5:.2f} / {v10:.2f} {unit}")

print()
print("=" * 80)
print("負重影響分析（直行前進，含水量50%）")
print("=" * 80)
for payload in PAYLOADS:
    r = next(x for x in all_results if x["scenario_id"]=="A" and x["moisture_pct"]==50 and x["payload_kg"]==payload)
    print(f"  負重 {payload:2d}kg  速度 {r['speed_fwd_m_min']:.2f} m/min  效率 {r['efficiency_pct']:.1f}%  淨推力 {r['net_force_N']:.0f}N")

# ══════════════════════════════════════════════════════════════
# 儲存 JSON
# ══════════════════════════════════════════════════════════════
out_json = os.path.join(OUT_DIR, "all_scenarios_results.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print(f"\nJSON 已儲存：{out_json}")

# ══════════════════════════════════════════════════════════════
# 圖表輸出
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Archimedes-Survey v2  全情境移動最佳化", fontsize=15, fontweight='bold')

# 圖1：各方案速度比較（含水量50%，無負重）
ax1 = axes[0][0]
labels = [s["name"] for s in SCENARIOS]
vals   = []
for sc in SCENARIOS:
    r = next(x for x in all_results if x["scenario_id"]==sc["id"] and x["moisture_pct"]==50 and x["payload_kg"]==0)
    v = r["speed_fwd_m_min"] or r["speed_lat_m_min"] or abs(r["omega_deg_s"]) * 0.1
    vals.append(v)
colors = ['#2ecc71','#27ae60','#3498db','#2980b9','#e74c3c','#c0392b','#f39c12','#e67e22','#9b59b6','#8e44ad']
bars = ax1.barh(labels, vals, color=colors)
ax1.set_xlabel("速度 (m/min) 或 旋轉×0.1")
ax1.set_title("各移動方案速度（含水量50%，無負重）")
for bar, val in zip(bars, vals):
    ax1.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
             f'{val:.2f}', va='center', fontsize=8)

# 圖2：負重影響（各方案，含水量50%）
ax2 = axes[0][1]
x = np.arange(len(SCENARIOS))
w = 0.25
for i, (payload, color) in enumerate(zip(PAYLOADS, ['#2ecc71','#f39c12','#e74c3c'])):
    vals2 = []
    for sc in SCENARIOS:
        r = next(x2 for x2 in all_results if x2["scenario_id"]==sc["id"] and x2["moisture_pct"]==50 and x2["payload_kg"]==payload)
        v = r["efficiency_pct"]
        vals2.append(v)
    ax2.bar(x + i*w, vals2, w, label=f'{payload}kg 負重', color=color, alpha=0.85)
ax2.set_xticks(x + w)
ax2.set_xticklabels([s["id"] for s in SCENARIOS], fontsize=9)
ax2.set_ylabel("傳動效率 (%)")
ax2.set_title("各方案傳動效率 × 負重（含水量50%）")
ax2.legend()
ax2.set_ylim(0, 100)

# 圖3：直行速度 × 含水量 × 負重
ax3 = axes[1][0]
for payload, ls, color in zip(PAYLOADS, ['-',  '--', ':'], ['#2ecc71','#f39c12','#e74c3c']):
    speeds = [next(x for x in all_results if x["scenario_id"]=="A" and x["moisture_pct"]==m and x["payload_kg"]==payload)["speed_fwd_m_min"]
              for m in MOISTURE_LIST]
    ax3.plot(MOISTURE_LIST, speeds, ls, marker='o', color=color, linewidth=2, label=f'負重 {payload}kg')
ax3.set_xlabel("含水量 (%)")
ax3.set_ylabel("前進速度 (m/min)")
ax3.set_title("直行速度 × 含水量 × 負重")
ax3.legend()
ax3.axvspan(30, 55, alpha=0.1, color='green', label='最佳區間')

# 圖4：旋轉方案 × 含水量 × 負重
ax4 = axes[1][1]
spin_scenarios = [s for s in SCENARIOS if s["mode"]=="spin"]
for sc in spin_scenarios:
    for payload, ls, color in zip(PAYLOADS, ['-','--',':'], ['#e74c3c','#f39c12','#2c3e50']):
        omegas = [abs(next(x for x in all_results if x["scenario_id"]==sc["id"] and x["moisture_pct"]==m and x["payload_kg"]==payload)["omega_deg_s"])
                  for m in MOISTURE_LIST]
        ax4.plot(MOISTURE_LIST, omegas, ls, marker='s', color=color, linewidth=2,
                 label=f'{sc["name"]} {payload}kg')
ax4.set_xlabel("含水量 (%)")
ax4.set_ylabel("角速度 (°/s)")
ax4.set_title("旋轉速度 × 含水量 × 負重")
ax4.legend(fontsize=7)

plt.tight_layout()
out_img = os.path.join(OUT_DIR, "all_scenarios_simulation.png")
plt.savefig(out_img, dpi=150, bbox_inches='tight')
print(f"圖表已儲存：{out_img}")
plt.show()
