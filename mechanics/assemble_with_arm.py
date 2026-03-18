"""
Archimedes-Survey v2 -- Machine + 3DOF Arm + Phase 3 Combined Assembly

Coordinate system (machine STL convention):
  Z = screw axis (forward), Y = lateral separation, X = height (up)
  Platform top: X = 156 mm
  Platform center (Y): (0 + 300) / 2 = 150 mm
  Platform center (Z): (-214 + 664) / 2 = 225 mm

Arm mounting:
  Arm base bolts to platform top center: X=156, Y=150, Z=225
  Arm local Z-up maps to machine +X (up)

Phase 3 additions (2026-03-17):
  - Octagonal arm links (arm_upper_arm_oct, arm_forearm_oct) -- overlaid on arm assembly
  - Transducer mount (arm_transducer_mount) -- at probe tip, stowed pos
  - Motor backplate x2 (motor_backplate_nema23) -- front/rear motor zones
  - Platform ribs x1 (platform_rib) -- 6 ribs between screws

Output: machine_v2_WITH_ARM_P3.stl
"""
import numpy as np
import struct, os, math, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

MECH_DIR = os.path.dirname(os.path.abspath(__file__))
STL_DIR  = os.path.join(MECH_DIR, "stl")
ARM_DIR  = os.path.join(STL_DIR, "arm")

def read_stl(path):
    tris = []
    with open(path, "rb") as f:
        f.read(80)
        n = struct.unpack("<I", f.read(4))[0]
        for _ in range(n):
            f.read(12)
            v0 = list(struct.unpack("<fff", f.read(12)))
            v1 = list(struct.unpack("<fff", f.read(12)))
            v2 = list(struct.unpack("<fff", f.read(12)))
            f.read(2)
            tris.append((v0, v1, v2))
    return tris

def write_stl(path, tris):
    with open(path, "wb") as f:
        f.write(b"\x00" * 80)
        f.write(struct.pack("<I", len(tris)))
        for v0, v1, v2 in tris:
            a = np.array(v1) - np.array(v0)
            b = np.array(v2) - np.array(v0)
            nm = np.cross(a, b)
            l = np.linalg.norm(nm)
            nm = nm / l if l > 1e-10 else np.array([0, 0, 1])
            f.write(struct.pack("<fff", *nm))
            f.write(struct.pack("<fff", *v0))
            f.write(struct.pack("<fff", *v1))
            f.write(struct.pack("<fff", *v2))
            f.write(struct.pack("<H", 0))

def tv(tris, dx, dy, dz):
    return [([v[0]+dx, v[1]+dy, v[2]+dz] for v in (v0, v1, v2))
            for v0, v1, v2 in tris]

def _tv(tris, dx, dy, dz):
    out = []
    for v0, v1, v2 in tris:
        out.append(([v0[0]+dx, v0[1]+dy, v0[2]+dz],
                    [v1[0]+dx, v1[1]+dy, v1[2]+dz],
                    [v2[0]+dx, v2[1]+dy, v2[2]+dz]))
    return out

def _rot(tris, axis, deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    if axis == 'x':
        def f(v): return [v[0], c*v[1]-s*v[2], s*v[1]+c*v[2]]
    elif axis == 'y':
        def f(v): return [c*v[0]+s*v[2], v[1], -s*v[0]+c*v[2]]
    elif axis == 'z':
        def f(v): return [c*v[0]-s*v[1], s*v[0]+c*v[1], v[2]]
    return [(f(v0), f(v1), f(v2)) for v0, v1, v2 in tris]

# ── Arm coordinate transform: arm-Z -> machine-X, arm-X -> machine-Z ─────────
# Arm local: Z=up, X=fwd, Y=right
# Machine:   X=up, Z=fwd, Y=lateral
# Rotation: swap X and Z -> rotate around Y by -90 deg

def arm_to_machine(tris):
    """Convert arm-local (Z-up) to machine (X-up): rotate -90deg around Y"""
    return _rot(tris, 'y', -90)

# ── Geometry verification ─────────────────────────────────────────────────────
PLAT_X = 156.0   # platform top (machine X)
PLAT_Y = 150.0   # arm mount Y center
PLAT_Z = 225.0   # arm mount Z center (along screw axis)

# Arm heights in arm-local coords (Z=up):
BASE_H   = 96.0   # top of J1 base (v3: 88mm housing + 8mm shaft boss)
J2_Z     = BASE_H + 50.0  # J2 shoulder center = 146mm (turret 20+30mm)
J3_Z_at_0 = J2_Z + 225.0 # J3 elbow at 0deg = 371mm (upper arm J2_off+200mm)
FORE_L   = 220.0           # v3 forearm link length
PROBE_L  = 200.0           # probe 180mm tube + 20mm tip

# Deployed pose: J2=90deg fwd, J3=120deg down -> probe tip X=-8mm (8mm into sand)
# Verified: arm_deploy_action.py deploy() uses J2=90, J3=120
J2_ang = 90.0   # degrees, pitches forward
J3_ang = 120.0  # degrees, pitches down from J2 end

J2_rad = math.radians(J2_ang)
J3_rad = math.radians(J3_ang)

# J3 position in arm local (X=fwd from base, Z=up)
J3_x_local = 200.0 * math.sin(J2_rad)
J3_z_local = J2_Z  + 200.0 * math.cos(J2_rad)

# Probe holder position
cum_ang = J2_rad - (J3_rad - math.pi/2)  # combined angle from vertical
ph_x    = J3_x_local + FORE_L * math.sin(J3_ang - 90 + J2_ang - 90 + 180)
# Simpler: direct calculation
# After J2 rotates fwd by J2_ang, then J3 rotates by J3_ang from that direction
# Direction of forearm (angle from +Z / up):
forearm_angle_from_up = J2_ang + (J3_ang - 90)  # total angle from vertical
fa_rad = math.radians(forearm_angle_from_up)
ph_x_local = J3_x_local + FORE_L * math.sin(fa_rad)
ph_z_local = J3_z_local + FORE_L * math.cos(fa_rad)

# Convert arm local (Z-up, X-fwd) to machine (X-up, Z-fwd)
# arm.x -> machine.z, arm.z -> machine.x
J3_machine_X = PLAT_X + J3_z_local    # machine height
J3_machine_Z = PLAT_Z + J3_x_local    # machine forward position
PH_machine_X = PLAT_X + ph_z_local
PH_machine_Z = PLAT_Z + ph_x_local
probe_tip_X  = PH_machine_X - PROBE_L  # probe hangs down (-X direction)

print("=" * 65)
print("Arm Deployment Geometry Verification")
print("=" * 65)
print(f"\nMounting point: machine X={PLAT_X:.0f}mm, Y={PLAT_Y:.0f}mm, Z={PLAT_Z:.0f}mm")
print(f"\nStowed pose (all joints 0 deg):")
print(f"  J2 shoulder : X={PLAT_X + J2_Z:.0f}mm above ground")
print(f"  J3 elbow    : X={PLAT_X + J3_Z_at_0:.0f}mm above ground")
print(f"  Probe tip   : X={PLAT_X + J3_Z_at_0 + FORE_L + PROBE_L:.0f}mm (pointing up)")

print(f"\nDeployed pose (J2={J2_ang:.0f}deg fwd, J3={J3_ang:.0f}deg down):")
print(f"  J3 elbow    : X={J3_machine_X:.0f}mm,  Z={J3_machine_Z:.0f}mm")
print(f"  Probe holder: X={PH_machine_X:.0f}mm, Z={PH_machine_Z:.0f}mm")
print(f"  Probe tip   : X={probe_tip_X:.0f}mm  (negative = below ground surface)")
if probe_tip_X < 0:
    print(f"  Insertion depth: {-probe_tip_X:.0f}mm below surface")
    print(f"  -> Probe inserts {-probe_tip_X:.0f}mm into sand")
else:
    print(f"  !! Probe does NOT reach ground (still {probe_tip_X:.0f}mm above)")

# Calculate optimal J3 angle to just touch ground
# Need: PLAT_X + J2_Z + 200*cos(J2_ang) + FORE_L * cos(J2_ang + J3_ang - 90) = 0
# Solve for J3_ang:
for j3 in range(80, 170, 5):
    fa = J2_ang + (j3 - 90)
    fa_r = math.radians(fa)
    j3x = J3_x_local + FORE_L * math.sin(fa_r)
    j3z = J3_z_local + FORE_L * math.cos(fa_r)
    tip_X = PLAT_X + j3z - PROBE_L
    if tip_X <= 0:
        print(f"\n  Min J3 angle to reach ground: {j3}deg")
        print(f"  At J3={j3}deg: probe tip at X={tip_X:.0f}mm ({-tip_X:.0f}mm into sand)")
        break

print()

# ── Helper: try-load STL, warn if missing ─────────────────────────────────────
def load_stl_safe(path, label):
    if os.path.exists(path):
        t = read_stl(path)
        print(f"  Loaded  {label:35s} ({len(t):5d} tris)")
        return t
    else:
        print(f"  SKIP    {label:35s} (file not found: {os.path.basename(path)})")
        return []


# ── Build combined assembly ───────────────────────────────────────────────────
print("Building combined machine + arm + Phase 3 assembly...")

machine_tris    = load_stl_safe(os.path.join(STL_DIR, "machine_v2_ASSEMBLY.stl"),  "machine_v2_ASSEMBLY")
arm_asm_tris    = load_stl_safe(os.path.join(ARM_DIR,  "arm_ASSEMBLY.stl"),         "arm_ASSEMBLY")
p3_backplate    = load_stl_safe(os.path.join(STL_DIR,  "motor_backplate_nema23.stl"), "motor_backplate_nema23")
p3_rib          = load_stl_safe(os.path.join(STL_DIR,  "platform_rib.stl"),           "platform_rib")
p3_transducer   = load_stl_safe(os.path.join(ARM_DIR,  "arm_transducer_mount.stl"),   "arm_transducer_mount")

# ── 1. Arm assembly: arm-local (Z-up) → machine (X-up) ───────────────────────
arm_placed = _rot(arm_asm_tris, 'y', 90)
arm_placed = _tv(arm_placed, PLAT_X, PLAT_Y, PLAT_Z)

# ── 2. Transducer mount: place at probe tip (stowed pose, arm all 0 deg) ──────
# In stowed pose: arm extends straight up (+X direction in machine coords)
# Probe tip X = PLAT_X + BASE_H + upper_arm_L + forearm_L + probe_L
UPPER_L = 200.0; FORE_L_MM = 220.0; PROBE_L_MM = 200.0
STOWED_TIP_X = PLAT_X + BASE_H + UPPER_L + FORE_L_MM + PROBE_L_MM  # ~872mm
# Transducer mount is a small add-on at the probe tip
# arm_transducer_mount.stl local origin is at its base (Z=0 in arm coords)
# In arm local: it sits above probe tip at arm-Z direction → rotate and place
trans_local = _rot(p3_transducer, 'y', 90)   # arm Z → machine X
trans_placed = _tv(trans_local,
                   STOWED_TIP_X,  # machine X (height) = above probe tip
                   PLAT_Y,        # machine Y = arm center
                   PLAT_Z)        # machine Z = arm center

# ── 3. Motor backplate: just outside motor outer faces (no overlap) ───────────
# Backplate Z span: -43 ~ +43 (86mm = MOTOR_W)
# Front motors outer face at Z = FRONT_BACK_Z = -214
#   -> plate +Z face = -214 -> plate center = -214 - 43 = -257
#   -> plate Z: -300 ~ -214  (just behind front motors, zero overlap)
# Rear motors outer face at Z = REAR_BACK_Z = 664
#   -> plate -Z face = 664 -> plate center = 664 + 43 = 707
#   -> plate Z: 664 ~ 750  (just behind rear motors, zero overlap)
FRONT_BACK_Z = -214.0
REAR_BACK_Z  =  664.0

backplate_front = _tv(p3_backplate, 0, 43, FRONT_BACK_Z - 43)  # center at -257
backplate_rear  = _tv(p3_backplate, 0, 43, REAR_BACK_Z  + 43)  # center at +707

# ── 4. Platform rib: already in machine coords, just include as-is ────────────
# platform_rib.stl was generated directly in machine coordinate space

# ── 5. Combine everything ─────────────────────────────────────────────────────
combined = (machine_tris
            + arm_placed
            + trans_placed
            + backplate_front
            + backplate_rear
            + p3_rib)

out_path = os.path.join(STL_DIR, "machine_v2_WITH_ARM_P3.stl")
write_stl(out_path, combined)
sz = os.path.getsize(out_path)

print()
print(f"  Machine            : {len(machine_tris):6d} tris")
print(f"  Arm assembly       : {len(arm_asm_tris):6d} tris")
print(f"  Transducer mount   : {len(trans_placed):6d} tris")
print(f"  Backplate x2       : {len(backplate_front)+len(backplate_rear):6d} tris")
print(f"  Platform ribs      : {len(p3_rib):6d} tris")
print(f"  ─────────────────────────────")
print(f"  Total              : {len(combined):6d} tris")
print(f"  Saved: {os.path.basename(out_path)}  ({sz//1024} KB)")
print("\nDone. Open with viewstl.com or Bambu Studio.")
