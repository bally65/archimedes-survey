# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
"""
Archimedes-Survey v2 -- Maximum Load & Structural Capacity Analysis

Three limiting factors:
  1. Ground bearing capacity  (sinkage limit)
  2. Motor torque limit       (propulsion stall)
  3. Screw net-force limit    (movement possible?)

Also calculates: safety factors, stability, max operable slope per load.
"""
import numpy as np
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Machine constants ─────────────────────────────────────────────────────────
BASE_MASS   = 22.0      # kg
PITCH       = 0.1125    # m
PIPE_R      = 0.084     # m  (screw outer radius for torque arm)
BLADE_W     = 0.050     # m  (blade width)
BLADE_OD    = 0.268     # m  (screw outer diameter)
N_TURNS     = 4
MOTOR_RPM   = 60
G           = 9.81
N_MOTORS    = 4

# Motor specs: NEMA23 + 10:1 gearbox
MOTOR_RATED_Nm  = 1.26        # N*m at motor shaft
GEAR_RATIO      = 10
ETA_GEARBOX     = 0.85
MOTOR_MAX_Nm    = MOTOR_RATED_Nm * GEAR_RATIO * ETA_GEARBOX  # at screw = 10.71 N*m
MOTOR_TOTAL_Nm  = MOTOR_MAX_Nm * N_MOTORS                    # 4 motors total

blade_area = BLADE_W * BLADE_OD * N_TURNS * 2  # m^2, two screws contact

# Effective contact area of screws on ground
# Each screw: N_TURNS turns * blade_width * projected_chord
# Projected footprint per screw ~ N_TURNS * BLADE_W * (BLADE_OD - PIPE_R*2)/2
blade_chord     = (BLADE_OD - PIPE_R * 2) / 2   # effective blade chord = 0.05 m
contact_area    = N_TURNS * BLADE_W * blade_chord * 2  # both screws = 0.02 m^2
# Add pipe surface contact (pipe settles slightly in sand)
pipe_contact    = PIPE_R * 0.3 * 0.450 * 2     # ~30% of pipe bottom, 2 screws
total_contact   = contact_area + pipe_contact   # ~0.042 m^2

DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")

def sand_params(moisture_pct):
    m = moisture_pct
    mu    = np.interp(m, [10,30,50,70,90], [0.60, 0.50, 0.45, 0.38, 0.32])
    c_kPa = np.interp(m, [10,30,50,70,90], [0.5,  3.5,  6.0,  4.0,  1.5])
    eta   = np.interp(m, [10,30,50,70,90], [0.40, 0.62, 0.78, 0.70, 0.55])
    phi   = np.interp(m, [10,30,50,70,90], [32,   28,   25,   22,   18])   # friction angle deg
    return mu, c_kPa, eta, phi

MOISTURES = [10, 30, 50, 70, 90]
MASS_LIST = np.arange(5, 101, 5)  # 5 to 100 kg total machine mass

print("=" * 75)
print("Max Load Analysis -- Archimedes-Survey v2")
print("=" * 75)
print(f"\nMachine base mass: {BASE_MASS} kg")
print(f"Motor max torque (per motor, after gearbox): {MOTOR_MAX_Nm:.2f} N*m")
print(f"Total torque (4 motors): {MOTOR_TOTAL_Nm:.2f} N*m")
print(f"Screw contact area: {total_contact:.4f} m^2 ({total_contact*1e4:.1f} cm^2)")

# ── Limit 1: Ground Bearing Capacity ─────────────────────────────────────────
print("\n" + "=" * 75)
print("LIMIT 1: Ground Bearing Capacity (Terzaghi shallow foundation)")
print("=" * 75)
print(f"\n{'Moisture':>10}  {'c(kPa)':>8}  {'phi(deg)':>9}  {'q_ult(kPa)':>11}  "
      f"{'Max Weight(N)':>14}  {'Max Mass(kg)':>13}  {'Status':>10}")
print("-" * 80)

bearing_limits = {}
for moisture in MOISTURES:
    mu, c_kPa, eta, phi = sand_params(moisture)
    phi_r = np.radians(phi)

    # Terzaghi bearing capacity factors (strip footing approximation)
    Nq = np.exp(np.pi * np.tan(phi_r)) * (np.tan(np.radians(45 + phi/2)))**2
    Nc = (Nq - 1) / np.tan(phi_r)
    Ng = 2 * (Nq + 1) * np.tan(phi_r)

    # Bearing capacity (ignore surcharge, shallow depth ~5cm)
    # q = c*Nc + 0.5*gamma*B*Ng
    gamma_sand = 1800  # kg/m^3 * g = 17,658 N/m^3 (wet sand density ~1600-2000)
    B = BLADE_OD       # effective footing width
    q_ult = c_kPa * 1000 * Nc + 0.5 * gamma_sand * G * B * Ng  # Pa
    q_ult_kPa = q_ult / 1000

    # Apply safety factor 3 for field conditions
    q_allow = q_ult / 3
    F_max = q_allow * total_contact
    M_max = F_max / G

    bearing_limits[moisture] = {"q_ult_kPa": q_ult_kPa, "q_allow_kPa": q_allow/1000,
                                  "F_max_N": F_max, "M_max_kg": M_max}

    status = "OK" if BASE_MASS <= M_max else "SINKING RISK"
    print(f"  {moisture:>8}%  {c_kPa:>7.1f}  {phi:>8}  {q_ult_kPa:>10.1f}  "
          f"{F_max:>13.0f}  {M_max:>12.1f}  {status:>10}")

# ── Limit 2: Motor Torque / Propulsion ────────────────────────────────────────
print("\n" + "=" * 75)
print("LIMIT 2: Motor Torque & Propulsion Stall Mass")
print("=" * 75)

# Motor max force at screw surface
F_motor_max = MOTOR_TOTAL_Nm / PIPE_R  # N, at blade surface
print(f"\nTotal motor force capacity at screw surface: {F_motor_max:.0f} N")

print(f"\n{'Moisture':>10}  {'c(kPa)':>8}  {'eta':>6}  {'mu':>6}  "
      f"{'Net@22kg(N)':>12}  {'Stall Mass(kg)':>15}  {'Payload Limit(kg)':>18}")
print("-" * 80)

propulsion_limits = {}
for moisture in MOISTURES:
    mu, c_kPa, eta, phi = sand_params(moisture)
    F_cohesion = c_kPa * 1000 * blade_area  # N from cohesion (constant, not mass-dep)

    # Net force equation: F_prop - F_resist > 0
    # (F_cohesion + M*g*eta) - M*g*mu = F_cohesion + M*g*(eta-mu)
    # If eta > mu: net force always positive (grows with mass!) -> no propulsion limit
    # If eta < mu: net force = 0 at M* = F_cohesion / (g*(mu-eta))

    net_at_base = F_cohesion + BASE_MASS * G * (eta - mu)

    if eta >= mu:
        # Propulsion always sufficient; limit is motor torque
        # Motor stall occurs when resistance (friction only, since cohesion helps) > motor force
        # M_stall: M*g*mu = F_motor_max + F_cohesion -> M = (F_motor + F_cohesion)/(g*mu)
        M_stall = (F_motor_max + F_cohesion) / (G * mu)
        payload_limit = M_stall - BASE_MASS
        limit_type = "motor"
    else:
        # Propulsion fails when M*g*(mu-eta) > F_cohesion
        M_stall = F_cohesion / (G * (mu - eta))
        payload_limit = M_stall - BASE_MASS
        limit_type = "friction"

    propulsion_limits[moisture] = {"M_stall_kg": M_stall, "payload_limit_kg": payload_limit,
                                    "net_at_base_N": net_at_base}

    print(f"  {moisture:>8}%  {c_kPa:>7.1f}  {eta:>5.2f}  {mu:>5.2f}  "
          f"{net_at_base:>11.0f}  {M_stall:>14.1f}  {payload_limit:>17.1f}  [{limit_type}]")

# ── Combined: Effective Max Payload ───────────────────────────────────────────
print("\n" + "=" * 75)
print("COMBINED: Effective Maximum Payload (most limiting constraint)")
print("=" * 75)
print(f"\n{'Moisture':>10}  {'Ground Limit(kg)':>17}  {'Motor Limit(kg)':>16}  "
      f"{'Effective Max(kg)':>18}  {'SF @ base 22kg':>15}")
print("-" * 80)

all_limits = {}
for moisture in MOISTURES:
    ground_max = bearing_limits[moisture]["M_max_kg"]
    motor_max  = propulsion_limits[moisture]["M_stall_kg"]
    effective  = min(ground_max, motor_max)
    payload    = effective - BASE_MASS
    sf         = effective / BASE_MASS  # safety factor

    all_limits[moisture] = {"effective_max_kg": effective, "payload_kg": max(0, payload),
                             "safety_factor": sf}

    print(f"  {moisture:>8}%  {ground_max:>16.1f}  {motor_max:>15.1f}  "
          f"{effective:>17.1f}  {sf:>14.2f}x")

# ── Additional Safety Factors ─────────────────────────────────────────────────
print("\n" + "=" * 75)
print("ADDITIONAL COEFFICIENTS (moisture=50%, mass=22kg)")
print("=" * 75)

mu50, c50, eta50, phi50 = sand_params(50)
M = BASE_MASS

F_prop   = c50 * 1000 * blade_area + M * G * eta50
F_resist = M * G * mu50
F_net    = F_prop - F_resist
F_wind   = 0.5 * 1.225 * (40/3.6)**2 * 1.2 * (0.878 * 0.170)

screw_anchor = c50*1000 * (BLADE_W * BLADE_OD * N_TURNS * 2) + M * G * mu50

print(f"\n  Propulsion force       : {F_prop:.1f} N")
print(f"  Rolling resistance     : {F_resist:.1f} N")
print(f"  Net propulsion force   : {F_net:.1f} N")
print(f"  Propulsion SF          : {F_prop/F_resist:.2f}x")
print(f"  Wind force (40km/h)    : {F_wind:.1f} N")
print(f"  Anchor vs wind SF      : {screw_anchor/F_wind:.1f}x")
print(f"  Ground bearing SF      : {bearing_limits[50]['q_ult_kPa']/(M*G/total_contact/1000):.2f}x")
print(f"  Motor torque SF        : {MOTOR_TOTAL_Nm/(F_resist*PIPE_R):.2f}x")

# ── Speed vs Total Mass (moisture=50%) ────────────────────────────────────────
print("\n" + "=" * 75)
print("SPEED VS TOTAL MASS (moisture=50%, flat beach)")
print("=" * 75)
print(f"\n{'Mass(kg)':>10}  {'Payload(kg)':>12}  {'Speed(m/min)':>13}  {'F_net(N)':>10}  {'Note'}")
print("-" * 65)

speed_vs_mass = []
for M_test in [22, 27, 32, 37, 42, 52, 62, 75, 90, 110, 135]:
    mu, c_kPa, eta, phi = sand_params(50)
    F_prop_m   = c_kPa * 1000 * blade_area + M_test * G * eta
    F_resist_m = M_test * G * mu
    F_net_m    = F_prop_m - F_resist_m

    if F_net_m > 0 and M_test <= bearing_limits[50]["M_max_kg"]:
        # Effective eta scales with net force ratio
        eta_eff   = eta * min(1.0, F_net_m / (F_prop_m + 1e-9) * (1 + eta))
        eta_eff   = min(eta_eff, eta)
        v_mmin    = PITCH * MOTOR_RPM / 60 * eta_eff * 60
        note = "OK" if M_test <= bearing_limits[50]["M_max_kg"] else "sinking"
    else:
        v_mmin = 0.0
        note = "stalled/sinking"

    payload = M_test - BASE_MASS
    speed_vs_mass.append({"total_mass_kg": M_test, "payload_kg": payload,
                          "speed_m_min": round(v_mmin, 2), "F_net_N": round(F_net_m, 1)})
    print(f"  {M_test:>8}  {payload:>11}  {v_mmin:>12.2f}  {F_net_m:>9.0f}  {note}")

# ── Save results ──────────────────────────────────────────────────────────────
out_data = {
    "bearing_limits": bearing_limits,
    "propulsion_limits": {k: {kk: round(vv, 2) for kk, vv in v.items()}
                          for k, v in propulsion_limits.items()},
    "effective_limits": {str(k): {kk: round(vv, 2) for kk, vv in v.items()}
                         for k, v in all_limits.items()},
    "speed_vs_mass_m50": speed_vs_mass,
}
out_json = os.path.join(DOCS, "max_load_analysis.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(out_data, f, indent=2)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Archimedes-Survey v2 -- Maximum Load Analysis",
             fontsize=13, fontweight='bold')

MCOLORS = {10:'#e74c3c', 30:'#f39c12', 50:'#27ae60', 70:'#2980b9', 90:'#8e44ad'}

# Left: Max operable mass by moisture
ax = axes[0]
moist_x    = MOISTURES
ground_ys  = [min(bearing_limits[m]["M_max_kg"], 150) for m in MOISTURES]
motor_ys   = [min(propulsion_limits[m]["M_stall_kg"], 150) for m in MOISTURES]
effect_ys  = [min(all_limits[m]["effective_max_kg"], 150) for m in MOISTURES]

ax.plot(moist_x, ground_ys, 's--', color='#e74c3c', label='Ground bearing limit', linewidth=2)
ax.plot(moist_x, motor_ys,  'o--', color='#2196F3', label='Motor torque limit', linewidth=2)
ax.plot(moist_x, effect_ys, 'D-',  color='#27ae60', label='Effective max (governing)', linewidth=2.5, markersize=8)
ax.axhline(BASE_MASS, color='gray', linestyle=':', label=f'Current mass {BASE_MASS}kg')
ax.set_title("Max Operable Mass vs Moisture", fontweight='bold')
ax.set_xlabel("Moisture (%)"); ax.set_ylabel("Max Total Machine Mass (kg)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
for x, y in zip(moist_x, effect_ys):
    ax.annotate(f"{y:.0f}kg", (x, y), textcoords="offset points", xytext=(0,8), ha='center', fontsize=8)

# Middle: Safety factors
ax = axes[1]
sf_prop  = [propulsion_limits[m]["M_stall_kg"] / BASE_MASS for m in MOISTURES]
sf_gnd   = [bearing_limits[m]["M_max_kg"] / BASE_MASS for m in MOISTURES]
x = np.arange(len(MOISTURES))
w = 0.35
b1 = ax.bar(x - w/2, sf_prop, w, label='Motor SF (stall / base mass)', color='#2196F3', alpha=0.85)
b2 = ax.bar(x + w/2, sf_gnd,  w, label='Ground SF (bearing / base mass)', color='#e74c3c', alpha=0.85)
ax.axhline(1.0, color='black', linestyle='--', linewidth=1)
ax.axhline(2.0, color='green', linestyle=':', linewidth=1, label='SF=2 (recommended)')
ax.set_title("Safety Factors by Moisture", fontweight='bold')
ax.set_xlabel("Moisture (%)"); ax.set_ylabel("Safety Factor (x)")
ax.set_xticks(x); ax.set_xticklabels([f"{m}%" for m in MOISTURES])
ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis='y')
for bar in b1:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
            f"{bar.get_height():.1f}x", ha='center', va='bottom', fontsize=7)
for bar in b2:
    h = min(bar.get_height(), 20)
    ax.text(bar.get_x()+bar.get_width()/2, h+0.1,
            f"{bar.get_height():.1f}x", ha='center', va='bottom', fontsize=7)
ax.set_ylim(0, 25)

# Right: Speed vs total mass (moisture=50%)
ax = axes[2]
xs = [r["total_mass_kg"] for r in speed_vs_mass]
ys = [r["speed_m_min"]   for r in speed_vs_mass]
ax.plot(xs, ys, 'o-', color='#27ae60', linewidth=2, markersize=7)
ax.axvline(BASE_MASS, color='blue',  linestyle='--', label=f'Base {BASE_MASS}kg')
ax.axvline(bearing_limits[50]["M_max_kg"], color='red', linestyle='--',
           label=f'Ground limit {bearing_limits[50]["M_max_kg"]:.0f}kg')
ax.set_title("Speed vs Total Mass (moisture=50%)", fontweight='bold')
ax.set_xlabel("Total Machine Mass (kg)"); ax.set_ylabel("Speed (m/min)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
for x_, y_ in zip(xs, ys):
    if y_ > 0:
        ax.annotate(f"{y_:.1f}", (x_, y_), textcoords="offset points", xytext=(0,6), ha='center', fontsize=7)

plt.tight_layout()
out_png = os.path.join(DOCS, "max_load_analysis.png")
plt.savefig(out_png, dpi=150, bbox_inches='tight')
print(f"\nSaved chart: {out_png}")
print(f"Saved JSON:  {out_json}")
print("\nDone.")
