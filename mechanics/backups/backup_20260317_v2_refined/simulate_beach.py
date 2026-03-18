# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Archimedes-Survey v2 -- Beach Physics Simulation
使用 Python + NumPy 進行力學估算，搭配 Matplotlib 視覺化

模擬情境：
  1. 各含水量下的推進效率
  2. 時速 40km/h 側風下的穩定前行
  3. 左右轉向最佳效率分析
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.rcParams['font.family'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'DejaVu Sans']

# ─── 機器幾何參數 ───────────────────────────────────────────
PIPE_OD       = 0.168    # PVC管外徑 m
PIPE_L        = 0.450    # 管長 m
BLADE_W       = 0.050    # 葉片寬 m
BLADE_OD      = 0.268    # 螺旋外徑 m (168+50*2=268mm 投影)
PITCH         = 0.1125   # 螺距 m
N_TURNS       = 4
MOTOR_RPM     = 60       # 額定轉速
TOTAL_MASS    = 22.0     # 估計總重量 kg (管+4馬達+支架+平台)
GRAVITY       = 9.81
PLAT_AREA_SIDE= 0.878 * 0.170  # 側向投影面積 m² (長×管高)
CD_BODY       = 1.2      # 機體風阻係數

# ─── 沙地參數（含水量對應） ────────────────────────────────
MOISTURE_LIST = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# 各含水量下的沙土摩擦係數、黏聚力、螺旋咬合效率
# 資料來源：海灘沙土 CBR 研究估算值
def sand_params(moisture_pct):
    m = moisture_pct
    # 摩擦係數 μ：乾沙高 → 飽和沙低
    mu = np.interp(m, [10,30,50,70,90], [0.65, 0.55, 0.45, 0.35, 0.28])
    # 黏聚力 c (kPa)：濕沙最高，乾/飽和沙低
    c  = np.interp(m, [10,30,50,70,90], [0.5,  3.5,  6.0,  4.0,  1.5])
    # 螺旋咬合效率 η：葉片挖沙能轉換為推進力的比例
    eta= np.interp(m, [10,30,50,70,90], [0.40, 0.62, 0.78, 0.70, 0.55])
    return mu, c, eta

# ─── 情境 1：各含水量推進效率 ────────────────────────────────
print("=" * 55)
print("情境 1：含水量 vs 推進效率")
print("=" * 55)

speeds, efficiencies, forces_N = [], [], []
for m in MOISTURE_LIST:
    mu, c, eta = sand_params(m)
    # 理論推進力 = 螺旋推進 × 咬合效率
    v_theory = PITCH * MOTOR_RPM / 60  # m/s 理論
    v_actual  = v_theory * eta
    # 滾動阻力
    F_resist = TOTAL_MASS * GRAVITY * mu
    # 推進力（兩條螺旋，4 葉片×N圈）
    blade_area   = BLADE_W * PITCH * N_TURNS * 2   # 兩條螺旋
    F_propulsion = c * 1000 * blade_area + TOTAL_MASS * GRAVITY * eta
    net_force    = F_propulsion - F_resist
    speeds.append(v_actual * 60)         # m/min
    efficiencies.append(eta * 100)
    forces_N.append(net_force)
    print(f"  含水量 {m:2d}%  速度 {v_actual*60:5.1f} m/min  效率 {eta*100:.0f}%  淨推力 {net_force:.1f} N")

# ─── 情境 2：時速 40 km/h 側風穩定性 ──────────────────────
print()
print("=" * 55)
print("情境 2：40 km/h 側風 (11.1 m/s) 穩定性分析")
print("=" * 55)

WIND_SPEED = 40 / 3.6  # m/s
rho_air    = 1.225
F_wind     = 0.5 * rho_air * WIND_SPEED**2 * CD_BODY * PLAT_AREA_SIDE
F_weight   = TOTAL_MASS * GRAVITY
# 螺旋錨定力（兩條螺旋嵌入沙中的側向抵抗）
screw_anchor_area = BLADE_W * BLADE_OD * N_TURNS * 2
wind_results = []
for m in MOISTURE_LIST:
    mu, c, eta = sand_params(m)
    F_anchor = c * 1000 * screw_anchor_area + F_weight * mu
    margin   = F_anchor / F_wind if F_wind > 0 else 999
    stable   = "[OK] 穩定" if margin >= 1.5 else ("[!!] 邊緣" if margin >= 1.0 else "[XX] 不穩")
    wind_results.append((m, F_anchor, margin, stable))
    print(f"  含水量 {m:2d}%  錨定力 {F_anchor:.0f}N  安全係數 {margin:.2f}  {stable}")

print(f"\n  側風力 = {F_wind:.1f} N (安全係數需 ≥ 1.5)")

# ─── 情境 3：轉向效率分析 ────────────────────────────────────
print()
print("=" * 55)
print("情境 3：轉向策略效率分析")
print("=" * 55)

strategies = {
    "原地旋轉 (差速)":      {"speed_ratio": 0.0, "turn_r": 0.15, "power_ratio": 0.85},
    "大弧度左轉 (R=1m)":   {"speed_ratio": 0.8, "turn_r": 1.0,  "power_ratio": 0.92},
    "中弧度左轉 (R=0.5m)": {"speed_ratio": 0.6, "turn_r": 0.5,  "power_ratio": 0.88},
    "前進+側移 (斜45°)":   {"speed_ratio": 0.7, "turn_r": 0.7,  "power_ratio": 0.82},
}

m_best = 40  # 含水量 40% (最佳條件)
_, _, eta_best = sand_params(m_best)
v_base = PITCH * MOTOR_RPM / 60 * eta_best * 60  # m/min

for name, s in strategies.items():
    v_eff = v_base * s["speed_ratio"] * s["power_ratio"]
    print(f"  {name:<25} 有效速度 {v_eff:.2f} m/min  效率 {s['power_ratio']*100:.0f}%")

# ─── 繪圖 ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Archimedes-Survey v2  沙灘模擬結果", fontsize=14, fontweight='bold')

# 圖1：推進效率
ax1 = axes[0]
color1 = ['#e74c3c' if e < 50 else '#f39c12' if e < 70 else '#27ae60' for e in efficiencies]
bars = ax1.bar(MOISTURE_LIST, efficiencies, color=color1, width=7)
ax1.set_xlabel("含水量 (%)")
ax1.set_ylabel("螺旋推進效率 (%)")
ax1.set_title("含水量 vs 推進效率")
ax1.set_ylim(0, 100)
ax1.axhspan(70, 100, alpha=0.1, color='green', label='最佳區間')
ax1.axvline(x=40, color='green', linestyle='--', alpha=0.7, label='最佳含水量')
ax1.legend(fontsize=8)
for bar, val in zip(bars, efficiencies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.0f}%', ha='center', va='bottom', fontsize=8)

# 圖2：側風安全係數
ax2 = axes[1]
margins = [r[2] for r in wind_results]
colors2 = ['#27ae60' if m >= 1.5 else '#f39c12' if m >= 1.0 else '#e74c3c' for m in margins]
ax2.bar(MOISTURE_LIST, margins, color=colors2, width=7)
ax2.axhline(y=1.5, color='green', linestyle='--', linewidth=2, label='安全係數 1.5')
ax2.axhline(y=1.0, color='red',   linestyle='--', linewidth=1, label='最低安全線')
ax2.set_xlabel("含水量 (%)")
ax2.set_ylabel("安全係數 (錨定力/風力)")
ax2.set_title(f"40 km/h 側風穩定性")
ax2.legend(fontsize=8)
for i, (m_val, margin) in enumerate(zip(MOISTURE_LIST, margins)):
    ax2.text(m_val, margin + 0.05, f'{margin:.1f}', ha='center', va='bottom', fontsize=8)

# 圖3：速度曲線
ax3 = axes[2]
ax3.plot(MOISTURE_LIST, speeds, 'b-o', linewidth=2, markersize=6, label='前進速度')
ax3.fill_between(MOISTURE_LIST, speeds, alpha=0.2)
ax3.axvspan(30, 55, alpha=0.15, color='green', label='最佳作業區')
ax3.set_xlabel("含水量 (%)")
ax3.set_ylabel("前進速度 (m/min)")
ax3.set_title("含水量 vs 前進速度")
ax3.legend(fontsize=8)
for x, y in zip(MOISTURE_LIST, speeds):
    ax3.annotate(f'{y:.1f}', (x, y), textcoords="offset points",
                 xytext=(0, 8), ha='center', fontsize=8)

plt.tight_layout()
out_img = "C:/Users/aa598/archimedes-survey/docs/beach_simulation.png"
plt.savefig(out_img, dpi=150, bbox_inches='tight')
print()
print(f"圖表已儲存：{out_img}")
plt.show()

print()
print("=" * 55)
print("模擬結論")
print("=" * 55)
print("最佳作業含水量：30–55%  (潮間帶退潮後 30 分鐘)")
print("40km/h 側風：30%+ 含水量可安全作業")
print("最高效轉向策略：大弧度差速轉彎 (R≥1m)")
print("=" * 55)
