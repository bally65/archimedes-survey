# -*- coding: utf-8 -*-
"""
Archimedes-Survey v2 -- Isaac Sim 4.5 Beach Simulation (kinematic)

STL coordinate convention:
  Z = screw axis (forward direction when lying flat)
  Y = lateral separation between the two screws (300 mm)
  X = height above ground (pipe radius 84 mm up/down)

Isaac Sim world convention:
  Y = forward, X = lateral, Z = up

Fix: apply quaternion (w=0.5, x=0.5, y=0.5, z=0.5) = 120 deg around (1,1,1)
     → maps STL-Z to world-Y, STL-Y to world-X, STL-X to world-Z  (machine lies flat)
Pre-center STL:  ty=-150mm (Y centre of pipe pair), tz=-225mm (half pipe length)
"""
import os
os.environ['OMNI_KIT_ACCEPT_EULA'] = 'yes'

from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "headless": False,
    "renderer": "RayTracedLighting",
    "width": 1280,
    "height": 720,
})

import numpy as np
import json

from isaacsim.core.api import World
from isaacsim.core.api.objects import GroundPlane
from pxr import UsdGeom, UsdPhysics, UsdShade, Gf

# ── Machine / physics parameters ────────────────────────────
TOTAL_MASS  = 22.0
PITCH       = 0.1125       # m
MOTOR_RPM   = 60
G           = 9.81
MU_K        = 0.45
ETA_FWD     = 0.78         # 50 % moisture
ETA_LAT     = 0.43
ETA_SPIN    = 0.47
WIND_SPEED  = 40 / 3.6     # m/s
F_WIND      = 0.5 * 1.225 * WIND_SPEED**2 * 1.2 * (0.878 * 0.170)

# STL geometry (mm)
PIPE_R      = 84.0         # pipe outer radius mm
PIPE_LEN    = 450.0        # screw length mm
DY2         = 300.0        # lateral separation between screw centres mm
STL_PRE_TY  = -(DY2 / 2)  # centre Y: -150 mm
STL_PRE_TZ  = -(PIPE_LEN / 2)  # centre Z: -225 mm
# X (height): pipe bottom at -PIPE_R = -84 mm; map to world-Z
# MACHINE_Z keeps pipe bottom just above ground
MACHINE_Z   = PIPE_R / 1000 + 0.006   # 0.084 + 0.006 = 0.090 m

SIM_DT    = 1 / 60.0
SIM_STEPS = int(15.0 / SIM_DT)   # 15 s

MECHANICS_DIR = os.path.dirname(os.path.abspath(__file__))
USD_PATH  = os.path.join(MECHANICS_DIR, "stl", "machine_v2_ASSEMBLY.usd")
OUT_JSON  = os.path.join(MECHANICS_DIR, "..", "docs", "isaac_sim_results.json")

# ── Kinematic velocities ────────────────────────────────────
v_fwd      = PITCH * MOTOR_RPM / 60 * ETA_FWD    # 0.0878 m/s = 5.27 m/min
v_lat      = PITCH * MOTOR_RPM / 60 * ETA_LAT    # 0.0484 m/s = 2.90 m/min
v_arc      = v_fwd * (0.74 / 0.78)               # arc efficiency factor
v_spin_rad = (PITCH * MOTOR_RPM / 60 * ETA_SPIN) / (DY2 / 2 / 1000)

# Wind drift: screw-anchor lateral resistance >> F_WIND → effectively 0 drift
C_KPA           = 6.0    # cohesion at 50% moisture
screw_anchor_area = 0.050 * 0.268 * 4 * 2   # m²
F_anchor_lat    = C_KPA * 1000 * screw_anchor_area + TOTAL_MASS * G * MU_K
v_wind_drift    = max(0.0, F_WIND - F_anchor_lat) / (TOTAL_MASS * 10)

print(f"v_fwd={v_fwd*60:.2f} m/min  v_lat={v_lat*60:.2f} m/min  "
      f"spin={np.degrees(v_spin_rad):.1f} deg/s")
print(f"F_WIND={F_WIND:.1f}N  F_anchor={F_anchor_lat:.0f}N  "
      f"wind_drift={v_wind_drift*60:.2f} m/min (~0)")

# ── Trajectory helpers ──────────────────────────────────────
def traj_linear(t, vy, vx=0.0):
    return vy * t, vx * t, 0.0   # dy, dx, dyaw

def traj_arc(t, v, sign, R=1.0):
    w   = v / R
    dy  = R * np.sin(w * t)
    dx  = sign * R * (1 - np.cos(w * t))
    yaw = np.degrees(sign * w * t)
    return dy, dx, yaw

def traj_spin(t, sign):
    return 0.0, 0.0, np.degrees(sign * v_spin_rad * t)

# ── Scenarios ───────────────────────────────────────────────
SCENARIOS = [
    {"name": "Straight Forward", "traj": lambda t: traj_linear(t, +v_fwd)},
    {"name": "Straight Reverse", "traj": lambda t: traj_linear(t, -v_fwd)},
    {"name": "Wind 40 km/h",     "traj": lambda t: traj_linear(t, +v_fwd,
                                                                vx=v_wind_drift)},
    {"name": "Arc Left R=1m",    "traj": lambda t: traj_arc(t, v_arc, -1, 1.0)},
    {"name": "Arc Right R=1m",   "traj": lambda t: traj_arc(t, v_arc, +1, 1.0)},
    {"name": "Spin CW",          "traj": lambda t: traj_spin(t, +1)},
    {"name": "Spin CCW",         "traj": lambda t: traj_spin(t, -1)},
    {"name": "Strafe Left",      "traj": lambda t: traj_linear(t, 0.0, -v_lat)},
    {"name": "Strafe Right",     "traj": lambda t: traj_linear(t, 0.0, +v_lat)},
]

# ── Build world ─────────────────────────────────────────────
world = World(stage_units_in_meters=1.0, physics_dt=SIM_DT)
world.scene.add(
    GroundPlane(prim_path="/World/Ground", name="ground",
                size=30.0, color=np.array([0.85, 0.75, 0.50]))
)
stage = world.stage

# Sand material (informational)
mat_path = "/World/SandMaterial"
UsdShade.Material.Define(stage, mat_path)
pm = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(mat_path))
pm.CreateStaticFrictionAttr(0.55)
pm.CreateDynamicFrictionAttr(MU_K)
pm.CreateRestitutionAttr(0.02)

# ── Machine USD (kinematic, visual only) ─────────────────────
MACHINE_PATH = "/World/Machine"
ref_prim = stage.DefinePrim(MACHINE_PATH, "Xform")
ref_prim.GetReferences().AddReference(USD_PATH)
xf = UsdGeom.Xform(ref_prim)

# Set up xform ops ONCE; only translate + rotateZ values change per step
# Op chain (left to right, outermost first):
#   translate  → world position (x, y, MACHINE_Z)
#   rotateZ    → yaw (degrees, world Z axis)
#   orient     → fixed: lay machine flat (STL Z→world Y, STL Y→world X, STL X→world Z)
#   scale      → 0.001 (mm → m)
#   translate:centering → pre-centre STL to (0, STL_PRE_TY, STL_PRE_TZ) mm

xf.ClearXformOpOrder()
_t_op    = xf.AddTranslateOp()
_rz_op   = xf.AddRotateZOp()
_or_op   = xf.AddOrientOp()
_or_op.Set(Gf.Quatf(0.5, Gf.Vec3f(-0.5, -0.5, -0.5)))  # conjugate: STL-Z→world-Y(fwd), STL-Y→world-X(lat), STL-X→world-Z(up)
_sc_op   = xf.AddScaleOp()
_sc_op.Set(Gf.Vec3f(0.001, 0.001, 0.001))              # fixed scale
_ct_op   = xf.AddTranslateOp(opSuffix="centering")
_ct_op.Set(Gf.Vec3d(0.0, STL_PRE_TY, STL_PRE_TZ))     # fixed pre-centre

def set_machine_pose(x, y, yaw_deg):
    _t_op.Set(Gf.Vec3d(float(x), float(y), float(MACHINE_Z)))
    _rz_op.Set(float(yaw_deg))

set_machine_pose(0.0, 0.0, 0.0)
world.reset()

results = []

# ── Run scenarios ───────────────────────────────────────────
for scenario in SCENARIOS:
    print(f"\n{'='*50}")
    print(f"Scenario: {scenario['name']}")

    # Settle at origin
    set_machine_pose(0.0, 0.0, 0.0)
    for _ in range(30):
        world.step(render=False)

    positions = []
    traj = scenario["traj"]

    for step in range(SIM_STEPS):
        t = step * SIM_DT
        dy, dx, dyaw = traj(t)
        set_machine_pose(dx, dy, dyaw)    # X=lateral, Y=forward
        world.step(render=True)

        if step % 30 == 0:
            positions.append({
                "t":   round(t, 2),
                "x":   round(float(dx), 4),
                "y":   round(float(dy), 4),
                "z":   round(MACHINE_Z, 4),
                "yaw": round(float(dyaw), 2),
            })

    # Metrics from final state
    dy_f, dx_f, dyaw_f = traj((SIM_STEPS - 1) * SIM_DT)
    dist_fwd    = abs(dy_f)
    dist_lat    = abs(dx_f)
    dist_total  = float(np.sqrt(dx_f**2 + dy_f**2))
    speed_m_min = dist_total / 15.0 * 60
    spin_deg_s  = abs(dyaw_f) / 15.0

    if "Spin" in scenario["name"]:
        print(f"  Rotation {dyaw_f:.1f} deg  Spin {spin_deg_s:.1f} deg/s")
    else:
        print(f"  Fwd {dist_fwd:.3f}m  Lat {dist_lat:.3f}m  "
              f"Speed {speed_m_min:.2f} m/min")

    results.append({
        "scenario":    scenario["name"],
        "forward_m":   round(dist_fwd,    3),
        "lateral_m":   round(dist_lat,    3),
        "speed_m_min": round(speed_m_min,  2),
        "spin_deg_s":  round(spin_deg_s,   2),
        "positions":   positions,
    })

# ── Summary ─────────────────────────────────────────────────
print("\n" + "="*65)
print("RESULTS  (sand mu_k=0.45, moisture 50%, mass=22 kg, no payload)")
print("="*65)
for r in results:
    if "Spin" in r["scenario"]:
        print(f"  {r['scenario']:<25}  spin={r['spin_deg_s']:.1f} deg/s")
    else:
        print(f"  {r['scenario']:<25}  fwd={r['forward_m']:.3f}m  "
              f"lat={r['lateral_m']:.3f}m  {r['speed_m_min']:.2f} m/min")

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {os.path.abspath(OUT_JSON)}")

print("\nPress Enter to close Isaac Sim and exit...")
input()
simulation_app.close()
print("Done.")
