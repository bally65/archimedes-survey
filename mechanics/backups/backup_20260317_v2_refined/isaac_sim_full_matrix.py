# -*- coding: utf-8 -*-
"""
Archimedes-Survey v2 -- Isaac Sim Full Test Matrix
Tests: 9 scenarios x 3 payloads x 5 moisture conditions = 135 runs
Output: docs/isaac_sim_matrix_results.json + docs/matrix_summary.png
"""
import os
os.environ['OMNI_KIT_ACCEPT_EULA'] = 'yes'

from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "headless": True,       # headless for batch testing (faster)
    "renderer": "RayTracedLighting",
    "width": 1280,
    "height": 720,
})

import numpy as np
import json

from isaacsim.core.api import World
from isaacsim.core.api.objects import GroundPlane
from pxr import UsdGeom, UsdPhysics, UsdShade, Gf

# ── Constants ────────────────────────────────────────────────
BASE_MASS   = 22.0      # machine mass without payload
PITCH       = 0.1125    # m
MOTOR_RPM   = 60
G           = 9.81
WIND_SPEED  = 40 / 3.6
F_WIND      = 0.5 * 1.225 * WIND_SPEED**2 * 1.2 * (0.878 * 0.170)

# Sand parameters by moisture (from Python physics model)
def sand_params(moisture_pct):
    m = moisture_pct
    mu_k  = np.interp(m, [10, 30, 50, 70, 90], [0.60, 0.50, 0.45, 0.38, 0.32])
    c_kPa = np.interp(m, [10, 30, 50, 70, 90], [0.5,  3.5,  6.0,  4.0,  1.5])
    eta_fwd  = np.interp(m, [10, 30, 50, 70, 90], [0.40, 0.62, 0.78, 0.70, 0.55])
    eta_lat  = eta_fwd * (0.43 / 0.78)
    eta_spin = eta_fwd * (0.47 / 0.78)
    eta_arc  = eta_fwd * (0.74 / 0.78)
    # Screw anchor lateral force
    screw_area = 0.050 * 0.268 * 4 * 2
    return mu_k, c_kPa, eta_fwd, eta_lat, eta_spin, eta_arc, screw_area

MOISTURE_LIST = [10, 30, 50, 70, 90]
PAYLOAD_LIST  = [0, 5, 10]   # kg

# Machine geometry (STL)
PIPE_R      = 84.0   # mm
PIPE_LEN    = 450.0  # mm
DY2         = 300.0  # mm
MACHINE_Z   = PIPE_R / 1000 + 0.006

SIM_DT    = 1 / 60.0
SIM_STEPS = int(15.0 / SIM_DT)

MECHANICS_DIR = os.path.dirname(os.path.abspath(__file__))
USD_PATH  = os.path.join(MECHANICS_DIR, "stl", "machine_v2_ASSEMBLY.usd")
OUT_JSON  = os.path.join(MECHANICS_DIR, "..", "docs", "isaac_sim_matrix_results.json")
OUT_PNG   = os.path.join(MECHANICS_DIR, "..", "docs", "matrix_summary.png")

# ── Trajectory generators ────────────────────────────────────
def make_trajs(eta_fwd, eta_lat, eta_spin, eta_arc, mu_k, c_kPa, total_mass):
    v_fwd  = PITCH * MOTOR_RPM / 60 * eta_fwd
    v_lat  = PITCH * MOTOR_RPM / 60 * eta_lat
    v_spin = (PITCH * MOTOR_RPM / 60 * eta_spin) / (DY2 / 2 / 1000)
    v_arc  = PITCH * MOTOR_RPM / 60 * eta_arc

    screw_area  = 0.050 * 0.268 * 4 * 2
    F_anchor    = c_kPa * 1000 * screw_area + total_mass * G * mu_k
    v_drift     = max(0.0, F_WIND - F_anchor) / (total_mass * 10)

    def lin(vy, vx=0.0):
        return lambda t: (vy*t, vx*t, 0.0)

    def arc(v, sign, R=1.0):
        def f(t):
            w = v / R
            return R*np.sin(w*t), sign*R*(1-np.cos(w*t)), np.degrees(sign*w*t)
        return f

    def spin(sign):
        return lambda t: (0.0, 0.0, np.degrees(sign * v_spin * t))

    return {
        "Straight Forward": lin(+v_fwd),
        "Straight Reverse": lin(-v_fwd),
        "Wind 40 km/h":     lin(+v_fwd, vx=v_drift),
        "Arc Left R=1m":    arc(v_arc, -1, 1.0),
        "Arc Right R=1m":   arc(v_arc, +1, 1.0),
        "Spin CW":          spin(+1),
        "Spin CCW":         spin(-1),
        "Strafe Left":      lin(0.0, -v_lat),
        "Strafe Right":     lin(0.0, +v_lat),
    }, v_spin  # also return spin rate for reference

# ── Build world once (headless) ──────────────────────────────
world = World(stage_units_in_meters=1.0, physics_dt=SIM_DT)
world.scene.add(GroundPlane(prim_path="/World/Ground", name="ground",
                            size=30.0, color=np.array([0.85, 0.75, 0.50])))
stage = world.stage

UsdShade.Material.Define(stage, "/World/SandMaterial")

MACHINE_PATH = "/World/Machine"
ref_prim = stage.DefinePrim(MACHINE_PATH, "Xform")
ref_prim.GetReferences().AddReference(USD_PATH)
xf = UsdGeom.Xform(ref_prim)
xf.ClearXformOpOrder()
_t_op  = xf.AddTranslateOp()
_rz_op = xf.AddRotateZOp()
_or_op = xf.AddOrientOp()
_or_op.Set(Gf.Quatf(0.5, Gf.Vec3f(-0.5, -0.5, -0.5)))
_sc_op = xf.AddScaleOp()
_sc_op.Set(Gf.Vec3f(0.001, 0.001, 0.001))
_ct_op = xf.AddTranslateOp(opSuffix="centering")
_ct_op.Set(Gf.Vec3d(0.0, -DY2/2, -PIPE_LEN/2))

def set_pose(x, y, yaw):
    _t_op.Set(Gf.Vec3d(float(x), float(y), float(MACHINE_Z)))
    _rz_op.Set(float(yaw))

set_pose(0, 0, 0)
world.reset()

# ── Run full matrix ──────────────────────────────────────────
all_results = []
total = len(MOISTURE_LIST) * len(PAYLOAD_LIST) * 9
done  = 0

print(f"Running {total} simulation runs...")

for moisture in MOISTURE_LIST:
    mu_k, c_kPa, eta_fwd, eta_lat, eta_spin, eta_arc, _ = sand_params(moisture)

    for payload in PAYLOAD_LIST:
        total_mass = BASE_MASS + payload
        trajs, v_spin_rad = make_trajs(eta_fwd, eta_lat, eta_spin, eta_arc,
                                       mu_k, c_kPa, total_mass)

        for scenario_name, traj in trajs.items():
            # Settle at origin
            set_pose(0, 0, 0)
            for _ in range(20):
                world.step(render=False)

            for step in range(SIM_STEPS):
                dy, dx, dyaw = traj(step * SIM_DT)
                set_pose(dx, dy, dyaw)
                world.step(render=False)   # headless, no render needed

            # Final metrics
            dy_f, dx_f, dyaw_f = traj((SIM_STEPS-1) * SIM_DT)
            dist_fwd    = abs(dy_f)
            dist_lat    = abs(dx_f)
            speed_m_min = np.sqrt(dx_f**2 + dy_f**2) / 15.0 * 60
            spin_deg_s  = abs(dyaw_f) / 15.0

            all_results.append({
                "scenario":    scenario_name,
                "moisture_pct": moisture,
                "payload_kg":   payload,
                "total_mass_kg": total_mass,
                "forward_m":   round(dist_fwd,    3),
                "lateral_m":   round(dist_lat,    3),
                "speed_m_min": round(speed_m_min,  2),
                "spin_deg_s":  round(spin_deg_s,   2),
            })

            done += 1
            if "Spin" in scenario_name:
                print(f"  [{done:3d}/{total}] moisture={moisture}% payload={payload}kg  "
                      f"{scenario_name:<22}  spin={spin_deg_s:.1f} deg/s")
            else:
                print(f"  [{done:3d}/{total}] moisture={moisture}% payload={payload}kg  "
                      f"{scenario_name:<22}  {speed_m_min:.2f} m/min")

# ── Save JSON ────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved JSON: {os.path.abspath(OUT_JSON)}")

# ── Generate summary chart ───────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle("Archimedes-Survey v2  Full Test Matrix\n"
             "(9 scenarios x 3 payloads x 5 moisture conditions)",
             fontsize=14, fontweight='bold')

SCENARIOS = [
    "Straight Forward", "Straight Reverse", "Wind 40 km/h",
    "Arc Left R=1m",    "Arc Right R=1m",   "Spin CW",
    "Strafe Left",      "Strafe Right",      "Spin CCW",
]
PAYLOAD_COLORS = {0: '#2196F3', 5: '#FF9800', 10: '#F44336'}
PAYLOAD_LABELS = {0: '0 kg', 5: '+5 kg', 10: '+10 kg'}

for ax, scenario in zip(axes.flat, SCENARIOS):
    is_spin = "Spin" in scenario
    for payload in PAYLOAD_LIST:
        rows = [r for r in all_results
                if r["scenario"] == scenario and r["payload_kg"] == payload]
        rows.sort(key=lambda r: r["moisture_pct"])
        xs = [r["moisture_pct"] for r in rows]
        ys = [r["spin_deg_s"] if is_spin else r["speed_m_min"] for r in rows]
        ax.plot(xs, ys, 'o-', color=PAYLOAD_COLORS[payload],
                label=PAYLOAD_LABELS[payload], linewidth=2, markersize=5)

    ax.set_title(scenario, fontsize=10, fontweight='bold')
    ax.set_xlabel("Moisture (%)", fontsize=8)
    ax.set_ylabel("deg/s" if is_spin else "m/min", fontsize=8)
    ax.axvspan(30, 55, alpha=0.12, color='green')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
print(f"Saved chart: {os.path.abspath(OUT_PNG)}")

# ── Print summary table ───────────────────────────────────────
print("\n" + "="*80)
print("SUMMARY: Best conditions (moisture=50%, payload=0kg) vs +10kg payload")
print("="*80)
print(f"{'Scenario':<25} {'0kg (m/min)':>12} {'+10kg (m/min)':>13} {'Delta':>8}")
print("-"*80)
for scenario in SCENARIOS:
    r0  = next(r for r in all_results if r["scenario"]==scenario
               and r["moisture_pct"]==50 and r["payload_kg"]==0)
    r10 = next(r for r in all_results if r["scenario"]==scenario
               and r["moisture_pct"]==50 and r["payload_kg"]==10)
    if "Spin" in scenario:
        v0  = r0["spin_deg_s"]
        v10 = r10["spin_deg_s"]
        unit = "deg/s"
    else:
        v0  = r0["speed_m_min"]
        v10 = r10["speed_m_min"]
        unit = "m/min"
    delta_pct = (v10 - v0) / v0 * 100 if v0 > 0 else 0
    print(f"  {scenario:<23} {v0:>10.2f}  {v10:>11.2f}   {delta_pct:>+6.1f}%  ({unit})")

simulation_app.close()
print("\nDone.")
