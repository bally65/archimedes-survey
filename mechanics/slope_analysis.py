# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
"""
Archimedes-Survey v2 -- Slope / Incline Analysis
Tests: slope angles 0-20 deg x 5 moisture conditions
Outputs: docs/slope_analysis.json + docs/slope_analysis.png
"""
import numpy as np
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Machine parameters
BASE_MASS   = 22.0      # kg (no payload)
PAYLOAD_LIST = [0, 5, 10]  # kg
PITCH       = 0.1125    # m
MOTOR_RPM   = 60
G           = 9.81

BLADE_W     = 0.050     # m
BLADE_OD    = 0.268     # m
N_TURNS     = 4
PIPE_R      = 0.084     # m
DY2         = 0.300     # m  (separation)

# Sand parameters by moisture
def sand_params(moisture_pct):
    m = moisture_pct
    mu    = np.interp(m, [10,30,50,70,90], [0.60, 0.50, 0.45, 0.38, 0.32])
    c_kPa = np.interp(m, [10,30,50,70,90], [0.5,  3.5,  6.0,  4.0,  1.5])
    eta   = np.interp(m, [10,30,50,70,90], [0.40, 0.62, 0.78, 0.70, 0.55])
    return mu, c_kPa, eta

ANGLES    = [0, 2, 5, 8, 10, 12, 15, 18, 20]  # degrees
MOISTURES = [10, 30, 50, 70, 90]

DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")

results = []

print("=" * 70)
print("Slope Analysis -- Archimedes-Survey v2")
print("=" * 70)

for moisture in MOISTURES:
    mu, c_kPa, eta = sand_params(moisture)
    blade_area = BLADE_W * BLADE_OD * N_TURNS * 2  # m^2, two screws

    print(f"\nMoisture {moisture}%  mu={mu:.2f}  c={c_kPa} kPa  eta={eta:.2f}")
    print(f"  {'Angle':>6}  {'Mass':>7}  {'F_prop':>8}  {'F_resist':>9}  {'F_gravity':>10}  "
          f"{'Net(N)':>8}  {'Speed':>8}  {'Climbable':>10}")

    for angle_deg in ANGLES:
        theta = np.radians(angle_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        for payload in PAYLOAD_LIST:
            M = BASE_MASS + payload

            # Propulsion force: screw cohesion grip + normal force * eta
            # Normal force is M*g*cos(theta) on slope
            F_normal    = M * G * cos_t
            F_prop      = c_kPa * 1000 * blade_area + F_normal * eta

            # Resistance: rolling friction (normal direction)
            F_resist    = F_normal * mu

            # Gravity component along slope (opposing uphill motion)
            F_gravity   = M * G * sin_t

            # Net force (uphill positive)
            F_net       = F_prop - F_resist - F_gravity

            # Effective speed: if net positive, scale by ratio; else 0
            if F_net > 0:
                # Efficiency reduction from slope (approximate)
                eta_slope = eta * (F_net / (F_prop - F_resist + 1e-9))
                eta_slope = min(eta_slope, eta)
                v_ms = PITCH * MOTOR_RPM / 60 * eta_slope
                climbable = True
            else:
                v_ms = 0.0
                climbable = False

            v_mmin = v_ms * 60

            if payload == 0:  # print only 0kg for readability
                status = "YES" if climbable else "NO"
                print(f"  {angle_deg:>5}deg  {M:>5.0f}kg  {F_prop:>7.1f}N  "
                      f"{F_resist:>8.1f}N  {F_gravity:>9.1f}N  "
                      f"{F_net:>7.1f}N  {v_mmin:>6.2f}m/m  {status:>10}")

            results.append({
                "moisture_pct": moisture,
                "payload_kg":   payload,
                "total_mass_kg": M,
                "angle_deg":    angle_deg,
                "F_propulsion_N": round(F_prop, 1),
                "F_resistance_N": round(F_resist, 1),
                "F_gravity_N":  round(F_gravity, 1),
                "F_net_N":      round(F_net, 1),
                "speed_m_min":  round(v_mmin, 2),
                "climbable":    climbable,
            })

# Max climbable angle per moisture/payload
print("\n" + "=" * 70)
print("Max Climbable Angle Summary")
print("=" * 70)
print(f"  {'Moisture':>10}  {'0kg':>8}  {'+5kg':>8}  {'+10kg':>8}")
print("-" * 50)
for moisture in MOISTURES:
    row = []
    for payload in PAYLOAD_LIST:
        max_ang = max((r["angle_deg"] for r in results
                       if r["moisture_pct"]==moisture and r["payload_kg"]==payload
                       and r["climbable"]), default=0)
        row.append(f"{max_ang}deg")
    print(f"  {moisture:>9}%  {row[0]:>8}  {row[1]:>8}  {row[2]:>8}")

# Save JSON
out_json = os.path.join(DOCS, "slope_analysis.json")
os.makedirs(DOCS, exist_ok=True)
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_json}")

# ── Plot ───────────────────────────────────────────────────────────────────────
MOISTURE_COLORS = {10:'#e74c3c', 30:'#f39c12', 50:'#27ae60', 70:'#2980b9', 90:'#8e44ad'}
PAYLOAD_LS      = {0: '-', 5: '--', 10: ':'}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Archimedes-Survey v2 -- Slope Performance Analysis",
             fontsize=13, fontweight='bold')

# Left: speed vs angle (payload=0kg, all moistures)
ax = axes[0]
for moisture in MOISTURES:
    rows = sorted([r for r in results if r["moisture_pct"]==moisture and r["payload_kg"]==0],
                  key=lambda r: r["angle_deg"])
    xs = [r["angle_deg"] for r in rows]
    ys = [r["speed_m_min"] for r in rows]
    ax.plot(xs, ys, 'o-', color=MOISTURE_COLORS[moisture],
            label=f"moisture {moisture}%", linewidth=2)
ax.set_title("Speed vs Slope Angle (0 kg payload)", fontweight='bold')
ax.set_xlabel("Slope Angle (deg)")
ax.set_ylabel("Speed (m/min)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.axvline(5, color='gray', linestyle='--', alpha=0.5, label='5 deg')
ax.axvline(10, color='gray', linestyle='-.', alpha=0.5, label='10 deg')

# Middle: net force vs angle (moisture=50%, all payloads)
ax = axes[1]
PAYLOAD_COLORS = {0:'#2196F3', 5:'#FF9800', 10:'#F44336'}
for payload in PAYLOAD_LIST:
    rows = sorted([r for r in results if r["moisture_pct"]==50 and r["payload_kg"]==payload],
                  key=lambda r: r["angle_deg"])
    xs = [r["angle_deg"] for r in rows]
    ys = [r["F_net_N"] for r in rows]
    ax.plot(xs, ys, 'o-', color=PAYLOAD_COLORS[payload],
            label=f"{payload} kg (+{payload} kg)", linewidth=2)
ax.axhline(0, color='red', linewidth=1.5, linestyle='--', label='F_net=0 (limit)')
ax.set_title("Net Force vs Slope (moisture=50%)", fontweight='bold')
ax.set_xlabel("Slope Angle (deg)")
ax.set_ylabel("Net Force (N)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Right: climbable angle heatmap-style bar chart
ax = axes[2]
x_pos = np.arange(len(MOISTURES))
width = 0.25
for i, payload in enumerate(PAYLOAD_LIST):
    max_angles = []
    for moisture in MOISTURES:
        ma = max((r["angle_deg"] for r in results
                  if r["moisture_pct"]==moisture and r["payload_kg"]==payload
                  and r["climbable"]), default=0)
        max_angles.append(ma)
    bars = ax.bar(x_pos + i*width, max_angles, width,
                  label=f"+{payload} kg", color=PAYLOAD_COLORS[payload], alpha=0.85)
    for bar, val in zip(bars, max_angles):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                f"{val}deg", ha='center', va='bottom', fontsize=7)
ax.set_title("Max Climbable Angle by Moisture & Payload", fontweight='bold')
ax.set_xlabel("Moisture (%)")
ax.set_ylabel("Max Slope (deg)")
ax.set_xticks(x_pos + width)
ax.set_xticklabels([f"{m}%" for m in MOISTURES])
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
out_png = os.path.join(DOCS, "slope_analysis.png")
plt.savefig(out_png, dpi=150, bbox_inches='tight')
print(f"Saved chart: {out_png}")
print("\nDone.")
