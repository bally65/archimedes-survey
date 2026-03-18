"""
Archimedes-Survey -- 3-DOF Probe Arm CAD Generator v4
======================================================
Visual redesign (SO-ARM100 inspired):
  - Cylindrical joint housings — NO exposed servo placeholder geometry
  - Tapered link profiles with rounded end transitions
  - Consistent inter-part alignment (v3 misalignment fixed)
  - Cleaner bracket geometry

Kinematics identical to v3 (backward compatible with assembly scripts):
  BASE_TOP = 88mm   (J1 shaft output)
  J2_Z     = 146mm  (shoulder pivot above platform)
  J3_Z     = J2_Z + 225mm = 371mm  (elbow pivot)
  PROBE_TIP = J3_Z + 220mm + 200mm = 791mm (stowed, pointing up)

  Upper arm link: 200mm (J2 -> J3)
  Forearm link:   220mm (J3 -> probe holder top)
  Probe:          200mm (180mm tube + 20mm cone tip)

Output (same filenames as v3, overwrites):
  stl/arm/arm_base_mount.stl
  stl/arm/arm_j1_turret.stl
  stl/arm/arm_upper_arm.stl
  stl/arm/arm_forearm.stl
  stl/arm/arm_probe.stl
  stl/arm/arm_pivot_pin.stl
  stl/arm/arm_servo_horn_adp.stl
  stl/arm/arm_cable_clip.stl
  stl/arm/arm_pca9685_tray.stl
  stl/arm/arm_ASSEMBLY.stl
"""
import numpy as np
import struct, os, math, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stl", "arm")
os.makedirs(OUT_DIR, exist_ok=True)

SEGS = 32


# =============================================================================
# STL helpers (identical to v3)
# =============================================================================
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

def _tv(tris, dx, dy, dz):
    return [([v0[0]+dx, v0[1]+dy, v0[2]+dz],
             [v1[0]+dx, v1[1]+dy, v1[2]+dz],
             [v2[0]+dx, v2[1]+dy, v2[2]+dz]) for v0, v1, v2 in tris]

def _rx(tris, deg):
    r = math.radians(deg); c, s = math.cos(r), math.sin(r)
    def rot(v): return [v[0], c*v[1]-s*v[2], s*v[1]+c*v[2]]
    return [(rot(a), rot(b), rot(c2)) for a, b, c2 in tris]

def _ry(tris, deg):
    r = math.radians(deg); c, s = math.cos(r), math.sin(r)
    def rot(v): return [c*v[0]+s*v[2], v[1], -s*v[0]+c*v[2]]
    return [(rot(a), rot(b), rot(c2)) for a, b, c2 in tris]

def _rz(tris, deg):
    r = math.radians(deg); c, s = math.cos(r), math.sin(r)
    def rot(v): return [c*v[0]-s*v[1], s*v[0]+c*v[1], v[2]]
    return [(rot(a), rot(b), rot(c2)) for a, b, c2 in tris]


# =============================================================================
# Primitive shapes
# =============================================================================
def make_box(x0, x1, y0, y1, z0, z1):
    t = []
    t += quad([x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0])
    t += quad([x0,y1,z1],[x1,y1,z1],[x1,y0,z1],[x0,y0,z1])
    t += quad([x0,y0,z0],[x0,y1,z0],[x0,y1,z1],[x0,y0,z1])
    t += quad([x1,y1,z0],[x1,y0,z0],[x1,y0,z1],[x1,y1,z1])
    t += quad([x1,y0,z0],[x0,y0,z0],[x0,y0,z1],[x1,y0,z1])
    t += quad([x0,y1,z0],[x1,y1,z0],[x1,y1,z1],[x0,y1,z1])
    return t

def make_cyl(r, z0, z1, segs=SEGS, cap_bot=True, cap_top=True):
    t = []
    for i in range(segs):
        a0 = 2*math.pi*i/segs; a1 = 2*math.pi*(i+1)/segs
        p00 = [r*math.cos(a0), r*math.sin(a0), z0]
        p10 = [r*math.cos(a1), r*math.sin(a1), z0]
        p01 = [r*math.cos(a0), r*math.sin(a0), z1]
        p11 = [r*math.cos(a1), r*math.sin(a1), z1]
        t += quad(p00, p10, p11, p01)
        if cap_bot: t.append(([0,0,z0], p10, p00))
        if cap_top: t.append(([0,0,z1], p01, p11))
    return t

def make_tube(r_out, r_in, z0, z1, segs=SEGS):
    t = []
    for i in range(segs):
        a0 = 2*math.pi*i/segs; a1 = 2*math.pi*(i+1)/segs
        for r, sign in [(r_out, 1), (r_in, -1)]:
            p0 = [r*math.cos(a0), r*math.sin(a0), z0]
            p1 = [r*math.cos(a1), r*math.sin(a1), z0]
            p2 = [r*math.cos(a1), r*math.sin(a1), z1]
            p3 = [r*math.cos(a0), r*math.sin(a0), z1]
            if sign > 0: t += quad(p0, p1, p2, p3)
            else:        t += quad(p3, p2, p1, p0)
        for zc in [z0, z1]:
            oi = [r_in *math.cos(a0), r_in *math.sin(a0), zc]
            oo = [r_out*math.cos(a0), r_out*math.sin(a0), zc]
            ni = [r_in *math.cos(a1), r_in *math.sin(a1), zc]
            no = [r_out*math.cos(a1), r_out*math.sin(a1), zc]
            if zc == z0: t += [(oi, no, oo), (oi, ni, no)]
            else:        t += [(oo, no, oi), (no, ni, oi)]
    return t


# =============================================================================
# v4 NEW primitives
# =============================================================================

def make_tapered_beam(w0, h0, w1, h1, length, z0=0.0):
    """
    Tapered solid beam along Z axis.
    Cross-section: w0 x h0 at z=z0, tapers to w1 x h1 at z=z0+length.
    X = width direction, Y = height direction.
    """
    z1 = z0 + length
    hw0, hh0 = w0/2, h0/2
    hw1, hh1 = w1/2, h1/2
    # 4 side faces
    t  = quad([-hw0,-hh0,z0],[ hw0,-hh0,z0],[ hw1,-hh1,z1],[-hw1,-hh1,z1])
    t += quad([ hw0, hh0,z0],[-hw0, hh0,z0],[-hw1, hh1,z1],[ hw1, hh1,z1])
    t += quad([-hw0, hh0,z0],[-hw0,-hh0,z0],[-hw1,-hh1,z1],[-hw1, hh1,z1])
    t += quad([ hw0,-hh0,z0],[ hw0, hh0,z0],[ hw1, hh1,z1],[ hw1,-hh1,z1])
    # End caps
    t += quad([-hw0,-hh0,z0],[-hw0, hh0,z0],[ hw0, hh0,z0],[ hw0,-hh0,z0])
    t += quad([-hw1, hh1,z1],[-hw1,-hh1,z1],[ hw1,-hh1,z1],[ hw1, hh1,z1])
    return t

def make_rounded_beam(w, h, length, z0=0.0, end_r=None, segs=16):
    """
    Rectangular beam with cylindrical end caps (along Z axis).
    end_r defaults to h/2 (semi-cylinder caps on the short ends).
    """
    if end_r is None:
        end_r = h / 2
    t = make_box(-w/2, w/2, -h/2, h/2, z0, z0+length)
    # End cap cylinders (axis along X, at z=z0 and z=z0+length)
    cap0 = make_cyl(end_r, -w/2, w/2, segs=segs, cap_bot=False, cap_top=False)
    cap0 = _ry(cap0, 90)
    t += _tv(cap0, 0, 0, z0)
    t += _tv(cap0, 0, 0, z0 + length)
    return t

def make_drum_housing(r_out, y_width, wall=4.0, shaft_r=5.0, segs=SEGS):
    """
    Cylindrical servo housing drum, axis along Y.
    Origin: drum center (Y=0, XZ center of cylinder).
    Outer shell: tube, end caps with shaft holes, shaft boss stubs.
    r_out:    outer radius of drum cylinder
    y_width:  total Y extent of drum body
    wall:     shell wall thickness
    shaft_r:  shaft hole radius
    """
    r_in = r_out - wall
    hw = y_width / 2
    t = []

    # Outer cylinder shell (tube)
    for i in range(segs):
        a0 = 2*math.pi*i/segs; a1 = 2*math.pi*(i+1)/segs
        for r, flip in [(r_out, False), (r_in, True)]:
            p0 = [r*math.cos(a0), -hw, r*math.sin(a0)]
            p1 = [r*math.cos(a1), -hw, r*math.sin(a1)]
            p2 = [r*math.cos(a1),  hw, r*math.sin(a1)]
            p3 = [r*math.cos(a0),  hw, r*math.sin(a0)]
            if not flip: t += quad(p0, p1, p2, p3)
            else:        t += quad(p3, p2, p1, p0)

    # End caps (annular disks with shaft hole)
    for y in [-hw, hw]:
        for i in range(segs):
            a0 = 2*math.pi*i/segs; a1 = 2*math.pi*(i+1)/segs
            oo = [r_out*math.cos(a0), y, r_out*math.sin(a0)]
            io = [r_in *math.cos(a0), y, r_in *math.sin(a0)]
            on = [r_out*math.cos(a1), y, r_out*math.sin(a1)]
            in_ = [r_in *math.cos(a1), y, r_in *math.sin(a1)]
            # Inner shaft hole edge (shaft_r < r_in: open hole)
            sh_o = [shaft_r*math.cos(a0), y, shaft_r*math.sin(a0)]
            sh_n = [shaft_r*math.cos(a1), y, shaft_r*math.sin(a1)]
            if y < 0:
                # End cap face: inner ring (r_in..r_out) + annular between shaft and r_in
                t += [(io, oo, on), (io, on, in_)]        # outer annular
                t += [(sh_o, io, in_), (sh_o, in_, sh_n)] # inner annular
            else:
                t += [(on, oo, io), (in_, on, io)]
                t += [(in_, io, sh_o), (sh_n, in_, sh_o)]

    # Shaft boss stubs (short cylinder at each end)
    for sy in [-1, 1]:
        boss = make_tube(shaft_r + 4, shaft_r, 0, 8, segs=20)
        boss = _rx(boss, 90)
        boss = _tv(boss, 0, sy * hw, 0)
        t += boss

    return t

def make_link_with_drums(drum_r, drum_w, link_w, link_h, link_len,
                         wall=4.0, shaft_r=5.0):
    """
    A combined arm link: one drum housing at z=0 (J-pivot), a tapered beam
    running from z=0 to z=link_len, another drum housing at z=link_len.
    The drums are rotated so their axis is along Y (pivot axis).
    link_len: distance between drum centers.
    """
    t = []

    # Drum at z=0 (start pivot)
    drum_start = make_drum_housing(drum_r, drum_w, wall=wall, shaft_r=shaft_r)
    # Drum axis is Y; drum center at XZ origin.
    # Rotate drum so XZ cylinder axis = arm link Z direction... actually the drum
    # is already oriented with Y as rotation axis. We just translate.
    t += drum_start

    # Tapered beam: starts at z=drum_r (clears start drum radius) to z=link_len-drum_r
    beam_start_z = drum_r + 2  # slight gap / transition
    beam_end_z   = link_len - drum_r - 2
    beam_len = beam_end_z - beam_start_z
    if beam_len > 10:
        beam = make_tapered_beam(link_w, link_h, link_w*0.8, link_h*0.8,
                                 beam_len * 0.3, z0=beam_start_z)
        beam += make_tapered_beam(link_w*0.8, link_h*0.8, link_w, link_h,
                                  beam_len * 0.4, z0=beam_start_z + beam_len*0.3)
        beam += make_tapered_beam(link_w, link_h, link_w, link_h,
                                  beam_len * 0.3, z0=beam_start_z + beam_len*0.7)
        t += beam

    # Drum at z=link_len (end pivot)
    drum_end = _tv(make_drum_housing(drum_r, drum_w, wall=wall, shaft_r=shaft_r),
                   0, 0, link_len)
    t += drum_end

    return t


# =============================================================================
# PART 1 -- Base Mount  (clean cylinder + square flange)
# =============================================================================
def build_base_mount():
    """
    Cylindrical body (Dia 80mm x 82mm) on a square mounting flange (90x90x8mm).
    J1 shaft collar at top (Z=90mm).
    No servo body shown inside -- clean exterior only.

    Z=0: bottom of flange (platform surface)
    Z=90: top of shaft collar (J1 output face)
    """
    t = []
    CYL_R    = 40.0   # outer radius
    CYL_H    = 82.0   # cylinder height above flange
    FLANGE_W = 90.0   # square flange side
    FLANGE_H = 8.0    # flange thickness
    WALL     = 4.5    # shell wall
    COLLAR_R = 12.0   # J1 shaft collar outer radius
    COLLAR_H = 8.0    # collar protrudes above cylinder top

    z_cyl_bot = FLANGE_H
    z_cyl_top = FLANGE_H + CYL_H
    z_collar_top = z_cyl_top + COLLAR_H

    # Square mounting flange
    t += make_box(-FLANGE_W/2, FLANGE_W/2, -FLANGE_W/2, FLANGE_W/2, 0, FLANGE_H)

    # M5 bolt bosses at flange corners
    for sx, sy in [(-1,-1),(-1,1),(1,-1),(1,1)]:
        cx, cy = sx*(FLANGE_W/2 - 11), sy*(FLANGE_W/2 - 11)
        t += _tv(make_cyl(7, 0, FLANGE_H+4, segs=12), cx, cy, 0)

    # Cylindrical body shell (hollow tube)
    t += make_tube(CYL_R, CYL_R - WALL, z_cyl_bot, z_cyl_top, segs=SEGS)

    # Bottom annular cap (flange to cylinder transition)
    t += make_tube(CYL_R, 0, z_cyl_bot - 1, z_cyl_bot + 2, segs=SEGS)

    # Top cap (flat disk closing the top, with collar hole)
    for i in range(SEGS):
        a0 = 2*math.pi*i/SEGS; a1 = 2*math.pi*(i+1)/SEGS
        oo = [CYL_R*math.cos(a0), CYL_R*math.sin(a0), z_cyl_top]
        on = [CYL_R*math.cos(a1), CYL_R*math.sin(a1), z_cyl_top]
        io = [COLLAR_R*math.cos(a0), COLLAR_R*math.sin(a0), z_cyl_top]
        in_ = [COLLAR_R*math.cos(a1), COLLAR_R*math.sin(a1), z_cyl_top]
        t += [(io, oo, on), (io, on, in_)]

    # Shaft collar cylinder
    t += make_cyl(COLLAR_R, z_cyl_top, z_collar_top, segs=SEGS)

    # 3 ventilation slot indicators (flat panels on cylinder side, decorative)
    for ang in [0, 120, 240]:
        panel = make_box(CYL_R - WALL - 1, CYL_R + 0.5, -6, 6,
                         z_cyl_bot + 15, z_cyl_bot + 55)
        panel = _rz(panel, ang)
        t += panel

    print(f"  Base Mount      : {len(t):>6} tris")
    return t


# =============================================================================
# PART 2 -- J1 Turret  (disk + bracket ears for J2 axis)
# =============================================================================
def build_j1_turret():
    """
    Disk (Dia 70mm x 18mm) + two bracket ears for J2 pivot.

    Local Z=0: J1 shaft output face (sits on top of shaft collar).
    J2 pivot at local Z = 50mm (= DISK_H + EAR_H/2 = 18+32/2 = 34 -- adjust below).

    Placed in assembly at Z = BASE_TOP + COLLAR_H = 88+8 = 96mm.
    J2_Z = 96 + 50 = 146mm (matches v3 kinematics).
    EAR_H must satisfy: DISK_H + EAR_H/2 = 50  -> EAR_H = 64mm
    """
    DISK_R   = 35.0
    DISK_H   = 18.0
    EAR_W    = 14.0   # bracket plate width (X direction)
    EAR_SPAN = 50.0   # bracket Y-axis separation
    EAR_H    = 64.0   # bracket height -> J2 at DISK_H + EAR_H/2 = 18+32 = 50
    BOSS_R   = 8.0    # J2 shaft boss outer radius
    WALL     = 3.5

    t = []

    # Disk (filled cylinder -- solid, looks clean)
    t += make_cyl(DISK_R, 0, DISK_H, segs=SEGS)

    # Chamfer ring at disk top edge (visual detail)
    t += make_tube(DISK_R + 2, DISK_R - 1, DISK_H - 3, DISK_H, segs=SEGS)

    # Two bracket ears
    for sy in [-1, 1]:
        ey = sy * EAR_SPAN/2
        # Main bracket plate (solid)
        t += make_box(-EAR_W/2, EAR_W/2, ey - EAR_W/2, ey + EAR_W/2,
                      DISK_H, DISK_H + EAR_H)
        # Triangular gusset (bracket to disk face -- stiffener)
        gx = EAR_W/2; gz = DISK_H; glz = 20.0; glx = 12.0
        for flip_x in [-1, 1]:
            for y in [ey - EAR_W/2 + 1, ey + EAR_W/2 - 1]:
                pts = [[0,y,gz],[glx*flip_x,y,gz],[0,y,gz+glz]]
                t.append((pts[0], pts[1], pts[2]))
                t.append((pts[2], pts[1], pts[0]))  # back face
        # J2 shaft boss (cylinder, axis along Y, centered at mid-ear)
        boss = make_cyl(BOSS_R, 0, EAR_W + 10, segs=20)
        boss = _rx(boss, 90)
        boss = _tv(boss, 0, ey - (EAR_W/2 + 5), DISK_H + EAR_H/2)
        t += boss

    # Central vertical rib between ears
    t += make_box(-3, 3,
                  -(EAR_SPAN/2 - EAR_W/2), (EAR_SPAN/2 - EAR_W/2),
                  DISK_H, DISK_H + EAR_H * 0.6)

    print(f"  J1 Turret       : {len(t):>6} tris")
    return t


# =============================================================================
# PART 3 -- Upper Arm  (J2 drum housing + tapered 200mm link + J3 drum flange)
# =============================================================================
def build_upper_arm():
    """
    Local origin = J2 pivot center.
    J2 drum housing: Dia 76mm x 52mm (Y-axis), centered at Z=0.
    Tapered link: runs from Z=+38mm to Z=+200+38mm (so total J2-J3 = 200mm -- J3 drum
    center at Z=200mm but link+drum together span from drum_r to drum_r+200mm...
    Actually: J2 pivot at Z=0, J3 pivot at Z=200mm. Drum centers at Z=0 and Z=200.
    """
    DRUM_R  = 38.0   # drum outer radius
    DRUM_W  = 52.0   # drum Y-width
    WALL    = 4.0
    SHAFT_R = 5.5    # shaft bore radius
    LINK_W  = 26.0   # link beam width (X)
    LINK_H  = 18.0   # link beam height (Y direction -- Y-axis = arm plane)
    LINK_L  = 200.0  # J2->J3 distance

    t = []

    # J2 drum housing (axis along Y, centered at z=0)
    t += make_drum_housing(DRUM_R, DRUM_W, wall=WALL, shaft_r=SHAFT_R)

    # J3 drum housing (same, centered at z=LINK_L)
    j3_drum = make_drum_housing(DRUM_R - 4, DRUM_W - 6, wall=WALL, shaft_r=SHAFT_R)
    t += _tv(j3_drum, 0, 0, LINK_L)

    # Tapered link beam connecting the two drums (Z = DRUM_R+2 to LINK_L-DRUM_R+2-2)
    beam_z0 = DRUM_R + 4
    beam_z1 = LINK_L - (DRUM_R - 4) - 4
    beam_len = beam_z1 - beam_z0
    if beam_len > 20:
        # Taper: wide at ends, slim in middle (SO-ARM100 silhouette)
        third = beam_len / 3
        t += make_tapered_beam(LINK_W, LINK_H, LINK_W * 0.72, LINK_H * 0.72,
                               third, z0=beam_z0)
        t += make_tapered_beam(LINK_W * 0.72, LINK_H * 0.72, LINK_W * 0.72, LINK_H * 0.72,
                               third, z0=beam_z0 + third)
        t += make_tapered_beam(LINK_W * 0.72, LINK_H * 0.72, LINK_W, LINK_H,
                               third, z0=beam_z0 + 2*third)

    j2_offset_z = DRUM_W / 2
    j3_local_z  = LINK_L

    print(f"  Upper Arm       : {len(t):>6} tris")
    return t, j2_offset_z, j3_local_z


# =============================================================================
# PART 4 -- Forearm  (J3 drum + tapered 220mm link + probe holder)
# =============================================================================
def build_forearm():
    """
    Local origin = J3 pivot center.
    J3 drum: Dia 68mm x 46mm (Y-axis), centered at Z=0.
    Tapered link: Z=0..220mm.
    Probe holder (D-clamp cylinder): at Z=220mm.
    """
    DRUM_R  = 34.0
    DRUM_W  = 46.0
    WALL    = 4.0
    SHAFT_R = 5.0
    LINK_W  = 22.0
    LINK_H  = 15.0
    LINK_L  = 220.0

    # Probe holder dimensions
    PH_R     = 18.0   # outer radius of clamp block
    PH_H     = 34.0   # height of clamp block
    PROBE_R  = 6.0    # probe tube radius (12mm OD)

    t = []

    # J3 drum housing
    t += make_drum_housing(DRUM_R, DRUM_W, wall=WALL, shaft_r=SHAFT_R)

    # Tapered link beam
    beam_z0 = DRUM_R + 4
    beam_z1 = LINK_L - PH_R - 2
    beam_len = beam_z1 - beam_z0
    if beam_len > 20:
        third = beam_len / 3
        t += make_tapered_beam(LINK_W, LINK_H, LINK_W * 0.68, LINK_H * 0.68,
                               third, z0=beam_z0)
        t += make_tapered_beam(LINK_W * 0.68, LINK_H * 0.68, LINK_W * 0.68, LINK_H * 0.68,
                               third, z0=beam_z0 + third)
        t += make_tapered_beam(LINK_W * 0.68, LINK_H * 0.68, LINK_W, LINK_H,
                               third, z0=beam_z0 + 2*third)

    # Probe holder (cylindrical clamp block at tip)
    PH_Z = LINK_L
    # D-clamp body: solid cylinder
    t += _tv(make_cyl(PH_R, 0, PH_H, segs=24), 0, 0, PH_Z)
    # Probe tube channel (shown as inner cavity indicator cylinder, protruding slightly)
    probe_ch = make_cyl(PROBE_R, -4, PH_H + 4, segs=16)
    t += _tv(probe_ch, 0, 0, PH_Z)
    # Two clamping bolt bosses (M3, either side of probe tube)
    for sx in [-1, 1]:
        bolt = make_cyl(3, 0, PH_H, segs=8)
        bolt = _ry(bolt, 90)
        t += _tv(bolt, sx * (PH_R + 4), 0, PH_Z + PH_H*0.3)

    j3_offset = DRUM_W / 2
    tip_z     = LINK_L + PH_H

    print(f"  Forearm         : {len(t):>6} tris")
    return t, j3_offset, tip_z


# =============================================================================
# PART 5 -- Probe  (10mm OD SS tube + cone tip)
# =============================================================================
def build_probe():
    t = []
    OD=10.0; ID=7.0; L=180.0; TIP=20.0
    t += make_tube(OD/2, ID/2, 0, L, segs=SEGS)
    for i in range(SEGS):
        a0=2*math.pi*i/SEGS; a1=2*math.pi*(i+1)/SEGS
        oo=[OD/2*math.cos(a0),OD/2*math.sin(a0),0]
        io=[ID/2*math.cos(a0),ID/2*math.sin(a0),0]
        on=[OD/2*math.cos(a1),OD/2*math.sin(a1),0]
        in_=[ID/2*math.cos(a1),ID/2*math.sin(a1),0]
        t += [(io,on,oo),(io,in_,on)]
    for i in range(SEGS):
        a0=2*math.pi*i/SEGS; a1=2*math.pi*(i+1)/SEGS
        b0=[OD/2*math.cos(a0),OD/2*math.sin(a0),L]
        b1=[OD/2*math.cos(a1),OD/2*math.sin(a1),L]
        apex=[0,0,L+TIP]
        t.append((b0,b1,apex))
        t.append(([0,0,L],b1,b0))
    print(f"  Probe           : {len(t):>6} tris")
    return t


# =============================================================================
# PART 6-9  -- Small parts (pin, horn adapter, cable clip, PCA9685 tray)
#              Identical to v3 -- these parts are small / printed separately
# =============================================================================
def build_pivot_pin():
    t = []
    ROD_R=2.0; FLANGE_R=5.0; L=40.0; FL=8.0
    t += make_cyl(ROD_R, 0, L, segs=16)
    t += make_cyl(FLANGE_R, 0, FL, segs=20)
    t += make_cyl(FLANGE_R, L-FL, L, segs=20)
    print(f"  Pivot Pin       : {len(t):>6} tris")
    return t

def build_servo_horn_adapter():
    t = []
    t += make_cyl(15.0, 0, 8.0, segs=32)
    t += make_cyl(2.5, -1, 9.0, segs=12)
    for ang in [45, 135, 225, 315]:
        rad=math.radians(ang)
        cx, cy = 10*math.cos(rad), 10*math.sin(rad)
        t += _tv(make_cyl(1.5, -1, 9.0, segs=8), cx, cy, 0)
    print(f"  Servo Horn Adp  : {len(t):>6} tris")
    return t

def build_cable_clip():
    t = []
    BW=22.0; BD=10.0; BH=14.0; WALL=2.5; WIRE_R=2.5
    t += make_box(-BW/2, BW/2, -BD/2, BD/2, 0, WALL)
    for sx in [-1,1]:
        t += _tv(make_cyl(3, 0, WALL+3, segs=8), sx*(BW/2-4), 0, 0)
    for sx in [-1,1]:
        t += make_box(sx*(BW/2-WALL), sx*BW/2, -BD/2, BD/2, 0, BH)
    t += make_box(-BW/2, BW/2, -BD/2, BD/2, BH-WALL, BH)
    t += make_cyl(WIRE_R, BH, BH+WALL+1, segs=16)
    print(f"  Cable Clip      : {len(t):>6} tris")
    return t

def build_pca9685_tray():
    t = []
    PCB_W=62.0; PCB_D=25.0; FRAME=2.5; H=10.0
    FW=PCB_W+FRAME*2; FD=PCB_D+FRAME*2
    t += make_box(-FW/2, FW/2, -FD/2, FD/2, 0, FRAME)
    for sx in [-1,1]:
        t += make_box(sx*(FW/2-FRAME), sx*FW/2, -FD/2, FD/2, 0, H)
    for sy in [-1,1]:
        t += make_box(-FW/2, FW/2, sy*(FD/2-FRAME), sy*FD/2, 0, H)
    t += make_box(-FW/2, FW/2, -FD/2, FD/2, H-FRAME, H)
    for sx, sy in [(-1,-1),(-1,1),(1,-1),(1,1)]:
        cx, cy = sx*(FW/2-FRAME-3), sy*(FD/2-FRAME-3)
        t += _tv(make_cyl(3, FRAME, H-FRAME, segs=8), cx, cy, 0)
    print(f"  PCA9685 Tray    : {len(t):>6} tris")
    return t


# =============================================================================
# ASSEMBLY -- All joints at 0 deg (stowed, vertical)
# =============================================================================
def build_assembly():
    all_tris = []

    # Reference platform (thin slab)
    all_tris += make_box(-55, 55, -55, 55, -5, 0)

    # Base mount (Z=0..90, collar top at Z=90)
    base = build_base_mount()
    all_tris += base
    BASE_TOP    = 88.0   # housing top
    COLLAR_H    = 8.0    # collar above housing
    TURRET_BASE = BASE_TOP + COLLAR_H  # = 96mm -- turret sits here

    # J1 Turret (local Z=0 at J1 shaft face, placed at TURRET_BASE)
    # J2 pivot in turret local = DISK_H + EAR_H/2 = 18 + 32 = 50mm
    # J2_Z = 96 + 50 = 146mm (matches v3 kinematics)
    turret = build_j1_turret()
    turret = _tv(turret, 0, 0, TURRET_BASE)
    all_tris += turret
    J2_Z = TURRET_BASE + 18 + 64/2  # 96 + 18 + 32 = 146mm

    # Upper arm (local origin at J2 pivot)
    upper, j2_off, j3_local = build_upper_arm()
    upper = _tv(upper, 0, 0, J2_Z)
    all_tris += upper
    J3_Z = J2_Z + j3_local  # 146 + 200 = 346mm

    # Forearm (local origin at J3 pivot)
    forearm, j3_off, tip_local = build_forearm()
    forearm = _tv(forearm, 0, 0, J3_Z)
    all_tris += forearm
    PROBE_BASE_Z = J3_Z + tip_local  # 346 + 254 = 600mm

    # Probe (attached to probe holder top)
    probe = build_probe()
    probe = _tv(probe, 0, 0, PROBE_BASE_Z)
    all_tris += probe
    PROBE_TIP_Z = PROBE_BASE_Z + 180 + 20

    print(f"  ASSEMBLY total  : {len(all_tris):>6} tris")
    return all_tris, PROBE_TIP_Z, J2_Z, J3_Z


# =============================================================================
# MAIN
# =============================================================================
print("=" * 60)
print("3-DOF Probe Arm v4 -- Generating parts (clean design)...")
print("=" * 60)
print()

base_t   = build_base_mount()
turret_t = build_j1_turret()
upper_t, *_ = build_upper_arm()
fore_t, *_  = build_forearm()
probe_t  = build_probe()
pin_t    = build_pivot_pin()
horn_t   = build_servo_horn_adapter()
clip_t   = build_cable_clip()
tray_t   = build_pca9685_tray()
asm_t, probe_tip_z, J2_Z, J3_Z = build_assembly()

parts = [
    ("arm_base_mount.stl",     base_t),
    ("arm_j1_turret.stl",      turret_t),
    ("arm_upper_arm.stl",      upper_t),
    ("arm_forearm.stl",        fore_t),
    ("arm_probe.stl",          probe_t),
    ("arm_pivot_pin.stl",      pin_t),
    ("arm_servo_horn_adp.stl", horn_t),
    ("arm_cable_clip.stl",     clip_t),
    ("arm_pca9685_tray.stl",   tray_t),
    ("arm_ASSEMBLY.stl",       asm_t),
]

print()
print("Saving STL files...")
for fname, tris in parts:
    path = os.path.join(OUT_DIR, fname)
    write_stl(path, tris)
    sz = os.path.getsize(path)
    print(f"  {fname:<30}  {len(tris):>6} tris  ({sz//1024:>4} KB)")

print()
print("=" * 60)
print("3-DOF Arm v4 Summary")
print("=" * 60)
print(f"  J2 Shoulder pivot  : Z = {J2_Z:.0f} mm above platform")
print(f"  J3 Elbow pivot     : Z = {J3_Z:.0f} mm above platform")
print(f"  Probe tip (stowed) : Z = {probe_tip_z:.0f} mm")
print()
print("  Visual changes vs v3:")
print("    - Cylindrical drum housings (no boxy servo placeholders)")
print("    - Tapered beam links (SO-ARM100 silhouette)")
print("    - No servo body geometry sticking out of housings")
print("    - Consistent J2/J3 pivot alignment")
print()
print(f"  Saved to: {OUT_DIR}")
print("Done.")
