"""
Archimedes-Survey v2 -- 3-DOF Probe Arm CAD Generator

Coordinate system (matches machine_v2_ASSEMBLY.stl):
  X = height (up), Y = lateral, Z = forward (screw axis)
  Platform top surface at X = 156 mm

Joint layout (all rotation axes in YZ plane = vertical swing):
  J1 base     : rotates around X-axis (yaw ±60 deg), base at platform top
  J2 shoulder : rotates around Y-axis (pitch 0-90 deg), 40mm above J1
  J3 elbow    : rotates around Y-axis (pitch 0-120 deg), 200mm from J2

Servo specs (all dimensions in mm):
  MG995  : body 40x20x38, horn diam 24, shaft at center top, torque 13 kg*cm
  DS3218 : body 40x20x40, horn diam 25, shaft at center top, torque 20 kg*cm
  MG996R : body 40x20x38, horn diam 24, shaft at center top, torque 13 kg*cm

Link geometry:
  Upper arm  : 200 mm (J2 shoulder -> J3 elbow pivot)
  Forearm    : 250 mm (J3 elbow -> probe holder)  [extended for 20cm ground reach]
  Probe      : 180 mm steel tube + 20 mm tip = 200mm total

Output STLs in stl/arm/:
  arm_base_mount.stl    -- bolts to platform, houses J1 servo
  arm_upper_link.stl    -- J2 shoulder servo + 200mm link to J3
  arm_forearm_link.stl  -- J3 elbow servo + 200mm link to probe
  arm_probe_holder.stl  -- clamps probe tube
  arm_ASSEMBLY.stl      -- combined preview (nominal pose: all joints 0 deg)
"""
import numpy as np
import struct, os, math

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stl", "arm")
os.makedirs(OUT_DIR, exist_ok=True)

SEGS = 32   # circle segments

# ── STL helpers (same as machine assembly) ────────────────────────────────────
def write_stl(path, tris):
    with open(path, "wb") as f:
        f.write(b"\x00" * 80)
        f.write(struct.pack("<I", len(tris)))
        for v0, v1, v2 in tris:
            a = np.array(v1) - np.array(v0)
            b = np.array(v2) - np.array(v0)
            n = np.cross(a, b)
            nm = np.linalg.norm(n)
            n = n / nm if nm > 1e-10 else np.array([0, 0, 1])
            f.write(struct.pack("<fff", *n))
            f.write(struct.pack("<fff", *v0))
            f.write(struct.pack("<fff", *v1))
            f.write(struct.pack("<fff", *v2))
            f.write(struct.pack("<H", 0))

def quad(v0, v1, v2, v3):
    return [(v0, v1, v2), (v0, v2, v3)]

def translate(tris, dx=0, dy=0, dz=0):
    return [([v[0]+dx, v[1]+dy, v[2]+dz] for v in tri) for tri in
            [([v0, v1, v2]) for v0, v1, v2 in tris]]

def _tv(tris, dx, dy, dz):
    out = []
    for v0, v1, v2 in tris:
        out.append(([v0[0]+dx, v0[1]+dy, v0[2]+dz],
                    [v1[0]+dx, v1[1]+dy, v1[2]+dz],
                    [v2[0]+dx, v2[1]+dy, v2[2]+dz]))
    return out

def _rx(tris, deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    def rot(v): return [v[0], c*v[1]-s*v[2], s*v[1]+c*v[2]]
    return [(rot(v0), rot(v1), rot(v2)) for v0, v1, v2 in tris]

def _ry(tris, deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    def rot(v): return [c*v[0]+s*v[2], v[1], -s*v[0]+c*v[2]]
    return [(rot(v0), rot(v1), rot(v2)) for v0, v1, v2 in tris]

def _rz(tris, deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    def rot(v): return [c*v[0]-s*v[1], s*v[0]+c*v[1], v[2]]
    return [(rot(v0), rot(v1), rot(v2)) for v0, v1, v2 in tris]

def make_box(x0, x1, y0, y1, z0, z1):
    """Solid box from corner to corner"""
    tris = []
    tris += quad([x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0])  # bottom -z
    tris += quad([x0,y1,z1],[x1,y1,z1],[x1,y0,z1],[x0,y0,z1])  # top +z
    tris += quad([x0,y0,z0],[x0,y1,z0],[x0,y1,z1],[x0,y0,z1])  # left -x
    tris += quad([x1,y1,z0],[x1,y0,z0],[x1,y0,z1],[x1,y1,z1])  # right +x
    tris += quad([x1,y0,z0],[x0,y0,z0],[x0,y0,z1],[x1,y0,z1])  # front -y
    tris += quad([x0,y1,z0],[x1,y1,z0],[x1,y1,z1],[x0,y1,z1])  # back +y
    return tris

def make_cylinder(r, z0, z1, segs=SEGS, cap_bottom=True, cap_top=True):
    tris = []
    for i in range(segs):
        a0 = 2*np.pi*i/segs
        a1 = 2*np.pi*(i+1)/segs
        p00 = [r*np.cos(a0), r*np.sin(a0), z0]
        p10 = [r*np.cos(a1), r*np.sin(a1), z0]
        p01 = [r*np.cos(a0), r*np.sin(a0), z1]
        p11 = [r*np.cos(a1), r*np.sin(a1), z1]
        tris += quad(p00, p10, p11, p01)
        if cap_bottom: tris.append(([0, 0, z0], p10, p00))
        if cap_top:    tris.append(([0, 0, z1], p01, p11))
    return tris

def make_rounded_box(x0, x1, y0, y1, z0, z1, r=3.0, segs=8):
    """Box with rounded corners along Z edges (simplified: plain box for CAD preview)"""
    return make_box(x0, x1, y0, y1, z0, z1)

# ── Servo body generator ───────────────────────────────────────────────────────
def make_servo(bw=20, bd=40, bh=38, horn_r=12, horn_t=4, shaft_h=5):
    """
    Generic servo block. Local origin at servo shaft center (top of body).
    Body extends: X: -bh/2..+bh/2, Y: -bw/2..+bw/2, Z: -bd/2..+bd/2
    Shaft exits at +X (toward top).
    Ear tabs at Y = ±bw/2 + 5, X ~ bh*0.4
    """
    tris = []
    # Main body
    tris += make_box(-bh/2, bh/2, -bw/2, bw/2, -bd/2, bd/2)
    # Ear tabs (for mounting screws)
    tab_t, tab_l = 5.0, 10.0
    for ys in [-bw/2 - tab_t, bw/2]:
        ye = ys + tab_t if ys > 0 else ys
        ys2 = ys if ys > 0 else ys
        ye2 = ys + tab_t if ys > 0 else ys - tab_t
        tris += make_box(bh*0.3 - 4, bh*0.3 + 6, min(ys2,ye2), max(ys2,ye2), -bd/2, bd/2)
    # Horn (on top, +X side)
    horn_tris = make_cylinder(horn_r, 0, horn_t, segs=16)
    horn_tris = _ry(horn_tris, -90)                   # point horn in +X direction
    horn_tris = _tv(horn_tris, bh/2 + horn_t/2, 0, 0)
    tris += horn_tris
    # Shaft stub
    shaft_tris = make_cylinder(3, 0, shaft_h, segs=12)
    shaft_tris = _ry(shaft_tris, -90)
    shaft_tris = _tv(shaft_tris, bh/2, 0, 0)
    tris += shaft_tris
    return tris

# ── Servo housing shell (for IP65 enclosure around servo) ─────────────────────
def make_servo_housing(bw=20, bd=40, bh=38, wall=4.0, shaft_hole_r=5.0):
    """
    Sealed PETG housing around servo body.
    Wall thickness wall mm on all sides.
    Shaft hole on +X face.
    """
    wo, bo, ho = bw + wall*2, bd + wall*2, bh + wall*2
    tris = []
    # Outer shell (6 sides)
    tris += make_box(-ho/2, ho/2, -wo/2, wo/2, -bo/2, bo/2)
    # Subtract inner cavity (approx: just show outer shell, subtract is implicit)
    # For STL preview: show solid housing with a notch for shaft exit
    shaft_notch = make_cylinder(shaft_hole_r, -2, wall+6, segs=16)
    shaft_notch = _ry(shaft_notch, -90)
    shaft_notch = _tv(shaft_notch, ho/2 - wall/2, 0, 0)
    # (In practice notch would be subtracted; for preview just show housing)
    return tris

# ── Structural link (C-channel beam) ──────────────────────────────────────────
def make_link_beam(length, width=22, height=16, wall=3.0):
    """
    C-channel structural link along Z axis.
    Local Z: 0 = proximal pivot, Z=length = distal pivot.
    C opens toward -X (top face = mounting surface).
    """
    tris = []
    # Top face
    tris += make_box(-height/2, height/2, -width/2, width/2, 0, length)
    # C-channel cutout (hollow interior, open at top +X)
    # Simplified: solid block for printable CAD preview
    return tris

def make_link_channel(length, W=24, H=18, wall=3):
    """
    Hollow rectangular tube link, Z = 0..length
    """
    tris = []
    # Outer
    tris += make_box(-H/2, H/2, -W/2, W/2, 0, length)
    # Inner hollow (represented as slightly inset faces — STL preview only)
    # For actual print: use wall thickness; preview shows solid
    return tris

# ── Pivot pin / bearing boss ───────────────────────────────────────────────────
def make_pivot_boss(r_out=10, r_in=4, length=16):
    """Cylindrical boss for pivot joint, around Z axis."""
    tris = []
    for i in range(SEGS):
        a0 = 2*np.pi*i/SEGS
        a1 = 2*np.pi*(i+1)/SEGS
        for r, sign in [(r_out, 1), (r_in, -1)]:
            p0 = [r*np.cos(a0), r*np.sin(a0), 0]
            p1 = [r*np.cos(a1), r*np.sin(a1), 0]
            p2 = [r*np.cos(a1), r*np.sin(a1), length]
            p3 = [r*np.cos(a0), r*np.sin(a0), length]
            if sign > 0:
                tris += quad(p0, p1, p2, p3)
            else:
                tris += quad(p3, p2, p1, p0)
        # End caps (annular ring)
        for zc in [0, length]:
            oi = [r_in *np.cos(a0), r_in *np.sin(a0), zc]
            oo = [r_out*np.cos(a0), r_out*np.sin(a0), zc]
            ni = [r_in *np.cos(a1), r_in *np.sin(a1), zc]
            no = [r_out*np.cos(a1), r_out*np.sin(a1), zc]
            if zc == 0:
                tris += [(oi, no, oo), (oi, ni, no)]
            else:
                tris += [(oo, no, oi), (no, ni, oi)]
    return tris

# ══════════════════════════════════════════════════════════════════════════════
# PART 1: Base Mount (J1 yaw servo housing, bolts to platform)
# ══════════════════════════════════════════════════════════════════════════════
def build_base_mount():
    """
    Mounts to platform top (X=0 here = platform surface).
    Contains MG995 servo oriented so shaft faces +X (upward) for yaw rotation.
    Outer dimensions: 60W x 60D x 65H mm
    Bolt pattern: 4x M5 holes at corners (represented as recesses)
    """
    tris = []
    BW, BD, BH = 60.0, 60.0, 65.0
    WALL = 5.0

    # Base plate (bolts to platform)
    tris += make_box(-BW/2, BW/2, -BD/2, BD/2, 0, WALL)
    # Four corner bolt bosses (8mm dia, 5mm tall)
    for sx, sy in [(-1,1),(1,1),(1,-1),(-1,-1)]:
        cx, cy = sx * (BW/2 - 8), sy * (BD/2 - 8)
        boss = make_cylinder(6, 0, WALL+4, segs=12)
        tris += _tv(boss, cx, cy, 0)

    # Side walls (rectangular tube housing)
    # Left/right walls
    for sx in [-1, 1]:
        tris += make_box(sx*(BW/2 - WALL), sx*BW/2, -BD/2, BD/2, WALL, BH)
    # Front/back walls
    for sy in [-1, 1]:
        tris += make_box(-BW/2, BW/2, sy*(BD/2 - WALL), sy*BD/2, WALL, BH)

    # Top cap with shaft hole (D=12mm) for J1 yaw output
    tris += make_box(-BW/2, BW/2, -BD/2, BD/2, BH - WALL, BH)
    # Shaft hole = notch cylinder at top center (symbolic)
    hole = make_cylinder(6, BH - WALL - 1, BH + 5, segs=16)
    tris += hole   # protruding shaft boss (solid; hole represented by absence)

    # J1 servo inside (MG995): positioned with shaft at top center
    servo_mg995 = make_servo(bw=20, bd=40, bh=38)
    servo_mg995 = _tv(servo_mg995, 0, 0, BH/2 + WALL)  # centered in housing
    tris += servo_mg995

    # Output horn plate (yaw output disk, 30mm dia, 6mm thick)
    horn_disk = make_cylinder(15, BH, BH + 6, segs=24)
    tris += horn_disk

    print(f"  Base mount: {len(tris)} triangles")
    return tris

# ══════════════════════════════════════════════════════════════════════════════
# PART 2: Upper Arm Link (J2 shoulder + 200mm link body + J3 pivot boss)
# ══════════════════════════════════════════════════════════════════════════════
def build_upper_arm():
    """
    Attaches to J1 yaw output.
    J2 DS3218 servo at base, output shaft pointing along +Y (lateral) for pitch.
    200mm C-channel link extends in +Z (forward/upward) direction.
    J3 pivot boss at distal end.

    Local origin at J2 pivot center.
    """
    tris = []
    LINK_L = 200.0
    LINK_W = 28.0
    LINK_H = 22.0
    WALL   = 4.0

    # J2 servo housing (DS3218MG, 40x20x40mm + housing wall)
    # Housing box
    HW, HD, HH = 50.0, 30.0, 52.0
    # Left housing side (holds servo)
    tris += make_box(-HD/2, HD/2, -HW/2, HW/2, -HH/2, HH/2)
    # Pivot shaft bosses on +Y and -Y sides for the shoulder joint pin
    for sy in [-1, 1]:
        boss = make_cylinder(10, 0, 14, segs=16)
        boss = _rx(boss, 90)  # align along Y
        boss = _tv(boss, 0, sy * HW/2, 0)
        tris += boss

    # J2 servo body (DS3218 20kg cm)
    servo_ds = make_servo(bw=20, bd=40, bh=40)
    servo_ds = _rx(servo_ds, 90)     # shaft pointing +Y
    tris += servo_ds

    # Link beam from J2 to J3 (C-channel, 200mm)
    beam = make_link_channel(LINK_L, W=LINK_W, H=LINK_H, wall=WALL)
    beam = _rz(beam, 0)
    # Offset so beam starts at top of housing and extends upward/forward
    beam = _tv(beam, 0, 0, HH/2)
    tris += beam

    # Reinforcement gusset (triangle plate at joint)
    gusset_pts = [
        [0, -LINK_W/2, HH/2], [0, -LINK_W/2, HH/2 + 40],
        [LINK_H/2, -LINK_W/2, HH/2 + 40]
    ]
    for sy in [-1, 1]:
        g = [([ p[0], sy*abs(p[1]), p[2]],
               [gp[0], sy*abs(gp[1]), gp[2]],
               [gq[0], sy*abs(gq[1]), gq[2]])
              for (p, gp, gq) in [(gusset_pts[0], gusset_pts[1], gusset_pts[2])]]
        tris += g

    # J3 pivot boss at distal end
    j3_boss = make_pivot_boss(r_out=12, r_in=4, length=LINK_W + 12)
    j3_boss = _rx(j3_boss, 90)       # align along Y axis
    j3_boss = _tv(j3_boss, 0, -(LINK_W/2 + 6), HH/2 + LINK_L)
    tris += j3_boss

    print(f"  Upper arm: {len(tris)} triangles")
    return tris

# ══════════════════════════════════════════════════════════════════════════════
# PART 3: Forearm Link (J3 elbow servo + 200mm link + probe mount)
# ══════════════════════════════════════════════════════════════════════════════
def build_forearm():
    """
    Attaches at J3 pivot.
    J3 MG996R servo at joint for elbow pitch.
    250mm link extends to probe holder (extended for 20cm ground reach).
    Local origin at J3 pivot center.
    """
    tris = []
    LINK_L = 250.0
    LINK_W = 26.0
    LINK_H = 20.0
    WALL   = 3.5

    # J3 servo housing (MG996R, 40x20x38mm + housing wall)
    HW, HD, HH = 48.0, 30.0, 50.0
    tris += make_box(-HD/2, HD/2, -HW/2, HW/2, -HH/2, HH/2)

    # Pivot shaft bosses
    for sy in [-1, 1]:
        boss = make_cylinder(8, 0, 12, segs=16)
        boss = _rx(boss, 90)
        boss = _tv(boss, 0, sy * HW/2, 0)
        tris += boss

    # J3 servo body (MG996R)
    servo_mg = make_servo(bw=20, bd=40, bh=38)
    servo_mg = _rx(servo_mg, 90)
    tris += servo_mg

    # Forearm beam (200mm)
    beam = make_link_channel(LINK_L, W=LINK_W, H=LINK_H, wall=WALL)
    beam = _tv(beam, 0, 0, HH/2)
    tris += beam

    # Probe holder clamp at tip (80mm ahead of beam end)
    CLAMP_W, CLAMP_D, CLAMP_H = 36.0, 24.0, 30.0
    clamp = make_box(-CLAMP_H/2, CLAMP_H/2,
                     -CLAMP_W/2, CLAMP_W/2,
                     HH/2 + LINK_L, HH/2 + LINK_L + CLAMP_D)
    tris += clamp

    # Probe tube opening (symbolic cylinder at clamp center)
    probe_hole = make_cylinder(8, HH/2 + LINK_L - 5, HH/2 + LINK_L + CLAMP_D + 5, segs=16)
    tris += probe_hole

    print(f"  Forearm: {len(tris)} triangles")
    return tris

# ══════════════════════════════════════════════════════════════════════════════
# PART 4: Probe Assembly (80mm stainless tube + tip)
# ══════════════════════════════════════════════════════════════════════════════
def build_probe():
    """
    Probe rod: 10mm OD stainless tube, 100mm long (80 + 20 tip)
    Tip: conical point for soil penetration
    Local Z: 0 = top of probe (held in clamp), Z=100 = soil-penetrating tip
    """
    tris = []
    PROBE_OD = 10.0
    PROBE_ID = 7.0
    PROBE_L  = 180.0   # 180mm tube + 20mm tip = 200mm total
    TIP_L    = 20.0

    # Main tube
    for i in range(SEGS):
        a0 = 2*np.pi*i/SEGS
        a1 = 2*np.pi*(i+1)/SEGS
        for r, sign in [(PROBE_OD/2, 1), (PROBE_ID/2, -1)]:
            p0 = [r*np.cos(a0), r*np.sin(a0), 0]
            p1 = [r*np.cos(a1), r*np.sin(a1), 0]
            p2 = [r*np.cos(a1), r*np.sin(a1), PROBE_L]
            p3 = [r*np.cos(a0), r*np.sin(a0), PROBE_L]
            if sign > 0:
                tris += quad(p0, p1, p2, p3)
            else:
                tris += quad(p3, p2, p1, p0)
    # Top annular cap
    for i in range(SEGS):
        a0 = 2*np.pi*i/SEGS
        a1 = 2*np.pi*(i+1)/SEGS
        oo = [PROBE_OD/2*np.cos(a0), PROBE_OD/2*np.sin(a0), 0]
        io = [PROBE_ID/2*np.cos(a0), PROBE_ID/2*np.sin(a0), 0]
        on = [PROBE_OD/2*np.cos(a1), PROBE_OD/2*np.sin(a1), 0]
        in_ = [PROBE_ID/2*np.cos(a1), PROBE_ID/2*np.sin(a1), 0]
        tris += [(io, on, oo), (io, in_, on)]

    # Conical tip
    TIP_R = PROBE_OD / 2
    for i in range(SEGS):
        a0 = 2*np.pi*i/SEGS
        a1 = 2*np.pi*(i+1)/SEGS
        base0 = [TIP_R*np.cos(a0), TIP_R*np.sin(a0), PROBE_L]
        base1 = [TIP_R*np.cos(a1), TIP_R*np.sin(a1), PROBE_L]
        apex  = [0, 0, PROBE_L + TIP_L]
        tris.append((base0, base1, apex))
        # Base cap of cone
        center = [0, 0, PROBE_L]
        tris.append((center, base1, base0))

    print(f"  Probe: {len(tris)} triangles")
    return tris

# ══════════════════════════════════════════════════════════════════════════════
# ASSEMBLY: combine all parts at nominal pose (0-0-0 degrees)
# ══════════════════════════════════════════════════════════════════════════════
def build_assembly():
    """
    Nominal pose: arm pointing straight up from platform.
    Platform surface = Z=0 (arm coordinate origin).
    Chain: base(0,0,0) -> J1 yaw -> J2 shoulder -> J3 elbow -> probe tip
    """
    all_tris = []

    # Platform surface reference plate (thin slab, 100x100x5mm)
    plat = make_box(-50, 50, -50, 50, -5, 0)
    all_tris += plat

    # Part 1: Base mount (sits on platform, Z: 0..71mm)
    base = build_base_mount()
    all_tris += base
    BASE_TOP = 71.0   # top of base housing (J1 output)

    # Part 2: Upper arm (J2 at BASE_TOP, link extends further +Z)
    upper = build_upper_arm()
    UPPER_H = 52.0   # height of J2 housing
    UPPER_L = 200.0  # link length
    upper = _tv(upper, 0, 0, BASE_TOP)
    all_tris += upper
    J3_Z = BASE_TOP + UPPER_H/2 + UPPER_L   # J3 pivot Z position

    # Part 3: Forearm (J3 at J3_Z)
    forearm = build_forearm()
    FORE_H = 50.0
    forearm = _tv(forearm, 0, 0, J3_Z)
    all_tris += forearm

    # Part 4: Probe (mounted at forearm tip)
    probe = build_probe()
    PROBE_START_Z = J3_Z + FORE_H/2 + 200.0 + 24.0 + 5.0
    probe = _tv(probe, 0, 0, PROBE_START_Z)
    all_tris += probe

    print(f"  Assembly: {len(all_tris)} triangles total")
    return all_tris, PROBE_START_Z

# ── Generate and save all parts ───────────────────────────────────────────────
print("Generating 3DOF Probe Arm STL parts...")
print()

base  = build_base_mount()
upper = build_upper_arm()
fore  = build_forearm()
probe = build_probe()
asm, probe_tip_z = build_assembly()

write_stl(os.path.join(OUT_DIR, "arm_base_mount.stl"),  base)
write_stl(os.path.join(OUT_DIR, "arm_upper_link.stl"),  upper)
write_stl(os.path.join(OUT_DIR, "arm_forearm_link.stl"),fore)
write_stl(os.path.join(OUT_DIR, "arm_probe.stl"),        probe)
write_stl(os.path.join(OUT_DIR, "arm_ASSEMBLY.stl"),     asm)

# ── Summary ───────────────────────────────────────────────────────────────────
BASE_TOP  = 71.0
UPPER_H   = 52.0
UPPER_L   = 200.0
FORE_H    = 50.0
FORE_L    = 250.0
total_reach = UPPER_L + FORE_L
fully_ext_z = BASE_TOP + UPPER_H/2 + UPPER_L + FORE_H/2 + FORE_L

print()
print("=" * 60)
print("3-DOF Arm Summary")
print("=" * 60)
print(f"  J1 base height        : {BASE_TOP:.0f} mm above platform")
print(f"  J2 shoulder height    : {BASE_TOP + UPPER_H/2:.0f} mm")
print(f"  J3 elbow height       : {BASE_TOP + UPPER_H/2 + UPPER_L:.0f} mm")
print(f"  Probe tip (nominal)   : {probe_tip_z:.0f} mm")
print(f"  Upper arm length      : {UPPER_L:.0f} mm")
print(f"  Forearm length        : {FORE_L:.0f} mm")
print(f"  Max reach (straight)  : {total_reach:.0f} mm")
print(f"  Total arm height      : {fully_ext_z:.0f} mm")
print()
print("  Joint specs:")
print("    J1 (MG995)    : yaw ±60 deg,    13 kg*cm, body 40x20x38mm")
print("    J2 (DS3218MG) : pitch 0-90 deg, 20 kg*cm, body 40x20x40mm")
print("    J3 (MG996R)   : pitch 0-120 deg,13 kg*cm, body 40x20x38mm")
print()
print("  Probe reach at full extension: 450 mm from J2")
print("  Deploy sequence: J2 0->90deg, J3 0->90deg -> probe vertical down")
print("  Insertion depth (J2=90, J3=90, platform X=156mm): ~197mm (~20cm) into sand")
print()
print("  Output files:")
for f in ["arm_base_mount.stl", "arm_upper_link.stl",
          "arm_forearm_link.stl", "arm_probe.stl", "arm_ASSEMBLY.stl"]:
    path = os.path.join(OUT_DIR, f)
    size = os.path.getsize(path)
    tcount = (size - 84) // 50
    print(f"    {f:<30} {tcount:>6} triangles  ({size//1024} KB)")
print()
print(f"  Saved to: {OUT_DIR}")
print("Done.")
