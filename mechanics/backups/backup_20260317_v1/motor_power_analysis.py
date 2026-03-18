# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
"""
Archimedes-Survey v2 -- Motor Power & Torque Analysis
Determines optimal motor specs (RPM, torque, power) for each scenario.
Outputs: docs/motor_analysis.json + docs/motor_analysis.png
"""
import numpy as np
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Machine constants
BASE_MASS   = 22.0      # kg
PITCH       = 0.1125    # m
PIPE_R      = 0.084     # m  (torque arm = pipe radius)
BLADE_W     = 0.050     # m
BLADE_OD    = 0.268     # m
N_TURNS     = 4
G           = 9.81
N_MOTORS    = 4         # total motors driving 2 screws (2 per screw)

DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")

# Sand at moisture 50% (optimal)
MU_K  = 0.45
C_KPA = 6.0
ETA   = 0.78

blade_area   = BLADE_W * BLADE_OD * N_TURNS * 2  # m^2

RPM_LIST = [20, 30, 40, 50, 60, 80, 100, 120]

# ── Scenario force requirements ────────────────────────────────────────────────
# Force needed per motor to drive machine:
#   F_resist = M*g*mu (rolling resistance)
#   F_screw  = F_resist / (eta * N_MOTORS)  + screw_preload
#   Torque   = F_screw * PIPE_R
#   Power    = Torque * omega

print("=" * 70)
print("Motor Power & Torque Analysis -- moisture=50%, mass=22kg")
print("=" * 70)

M = BASE_MASS
F_normal  = M * G
F_resist  = F_normal * MU_K          # N, total rolling resistance
F_prop_cohesion = C_KPA*1000 * blade_area  # N, cohesion contribution

# Force each motor must deliver via screw (two motors per screw)
# We need total propulsion = F_resist, distributed across N_MOTORS
F_per_motor_min = F_resist / N_MOTORS   # N  (minimum, no cohesion overhead)
T_per_motor_min = F_per_motor_min * PIPE_R  # N*m

print(f"\nForce budget (moisture=50%, mass={M}kg):")
print(f"  Rolling resistance (F_resist) = {F_resist:.1f} N")
print(f"  Cohesion thrust (both screws) = {F_prop_cohesion:.1f} N")
print(f"  Net excess force              = {F_prop_cohesion - F_resist:.1f} N")
print(f"  Min torque per motor          = {T_per_motor_min:.2f} N*m (at pipe surface)")

results_rpm = []
print(f"\n{'RPM':>6}  {'Speed(m/m)':>11}  {'omega(rad/s)':>13}  "
      f"{'P_motor(W)':>11}  {'P_total(W)':>11}  {'Torque(N*m)':>12}")
print("-" * 75)

for rpm in RPM_LIST:
    omega  = rpm * 2 * np.pi / 60   # rad/s
    v_ms   = PITCH * rpm / 60 * ETA # m/s actual speed
    v_mmin = v_ms * 60

    # Torque needed at screw: F_resist / (N_MOTORS * eta_mechanical)
    eta_mech = 0.85  # gearbox + coupling efficiency
    F_screw_each = F_resist / (N_MOTORS * eta_mech)
    T_motor  = F_screw_each * PIPE_R    # N*m at motor shaft (after gearbox)
    # Add inertia/acceleration margin (20%)
    T_motor_design = T_motor * 1.20

    P_motor  = T_motor_design * omega   # W per motor
    P_total  = P_motor * N_MOTORS       # W total

    results_rpm.append({
        "rpm": rpm,
        "speed_m_min": round(v_mmin, 2),
        "omega_rad_s": round(omega, 3),
        "torque_Nm":   round(T_motor_design, 3),
        "P_motor_W":   round(P_motor, 1),
        "P_total_W":   round(P_total, 1),
    })
    print(f"  {rpm:>4}  {v_mmin:>10.2f}  {omega:>12.3f}  "
          f"{P_motor:>10.1f}  {P_total:>10.1f}  {T_motor_design:>11.3f}")

# ── Motor selection recommendation ────────────────────────────────────────────
print("\n" + "=" * 70)
print("Motor Selection Recommendation")
print("=" * 70)

design_rpm = 60
r60 = next(r for r in results_rpm if r["rpm"] == design_rpm)
print(f"\nDesign point: RPM={design_rpm}, speed={r60['speed_m_min']} m/min")
print(f"  Required torque per motor: {r60['torque_Nm']:.3f} N*m")
print(f"  Required power per motor:  {r60['P_motor_W']:.1f} W")
print(f"  Total system power:        {r60['P_total_W']:.1f} W")

print("""
Motor candidates:
  NEMA23 (57mm)  -- Rated torque 1.26 N*m, typical 60-100W brushless
    -> Sufficient for flat beach (req. {:.2f} N*m)
    -> Marginal on 10-15deg slope (req. {:.2f} N*m with 2x slope penalty)
    -> RECOMMENDED for cost-optimized build

  NEMA34 (86mm)  -- Rated torque 3.0-4.5 N*m, 150-300W brushless
    -> Overkill for flat, suitable for rocky/steep terrain
    -> 2x cost vs NEMA23

  Gear ratio:  10:1 planetary gearbox
    -> Motor shaft: ~2.5x higher RPM, 10x more torque at screw
    -> Reduces motor spec requirement significantly

  Recommended: NEMA23 + 10:1 planetary gearbox x4 motors
""".format(r60['torque_Nm'], r60['torque_Nm']*2.2))

# ── Battery / Runtime analysis ─────────────────────────────────────────────────
print("=" * 70)
print("Battery Runtime Analysis")
print("=" * 70)

BATTERY_WH = 480       # Wh
SENSOR_W   = 15        # W (camera, MCU, GPS)
COMMS_W    = 5         # W (WiFi/LoRa)

for load_pct in [25, 50, 75, 100]:
    p_motor = r60['P_total_W'] * load_pct / 100
    p_total = p_motor + SENSOR_W + COMMS_W
    runtime_h = BATTERY_WH / p_total
    print(f"  Motor load {load_pct:3d}%:  motor={p_motor:.0f}W  "
          f"total={p_total:.0f}W  runtime={runtime_h:.1f}h  "
          f"({runtime_h*60:.0f}min)")

# ── Save JSON ──────────────────────────────────────────────────────────────────
out_json = os.path.join(DOCS, "motor_analysis.json")
os.makedirs(DOCS, exist_ok=True)
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results_rpm, f, indent=2)
print(f"\nSaved: {out_json}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Archimedes-Survey v2 -- Motor Power Analysis (moisture=50%, 22kg)",
             fontsize=12, fontweight='bold')

rpms   = [r["rpm"] for r in results_rpm]
speeds = [r["speed_m_min"] for r in results_rpm]
torqs  = [r["torque_Nm"] for r in results_rpm]
powers = [r["P_total_W"] for r in results_rpm]

ax = axes[0]
ax.plot(rpms, speeds, 'o-b', linewidth=2)
ax.axvline(60, color='green', linestyle='--', label='Design 60 RPM')
ax.set_xlabel("Motor RPM")
ax.set_ylabel("Speed (m/min)")
ax.set_title("RPM vs Speed", fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
for x, y in zip(rpms, speeds):
    ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0,6), ha='center', fontsize=7)

ax = axes[1]
ax.plot(rpms, torqs, 'o-r', linewidth=2)
ax.axvline(60, color='green', linestyle='--', label='Design 60 RPM')
ax.axhline(1.26, color='orange', linestyle=':', label='NEMA23 rated 1.26 N*m')
ax.axhline(3.0,  color='red',    linestyle=':', label='NEMA34 rated 3.0 N*m')
ax.set_xlabel("Motor RPM")
ax.set_ylabel("Torque per motor (N*m)")
ax.set_title("RPM vs Required Torque", fontweight='bold')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(rpms, powers, 'o-g', linewidth=2)
ax.axvline(60, color='green', linestyle='--', label='Design 60 RPM')
ax.axhline(120, color='blue', linestyle=':', label='System budget 120W')
ax.set_xlabel("Motor RPM")
ax.set_ylabel("Total Motor Power (W)")
ax.set_title("RPM vs Total Power Draw", fontweight='bold')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
for x, y in zip(rpms, powers):
    ax.annotate(f"{y:.0f}W", (x, y), textcoords="offset points", xytext=(0,6), ha='center', fontsize=7)

plt.tight_layout()
out_png = os.path.join(DOCS, "motor_analysis.png")
plt.savefig(out_png, dpi=150, bbox_inches='tight')
print(f"Saved chart: {out_png}")
print("\nDone.")
