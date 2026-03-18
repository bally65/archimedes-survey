"""
Archimedes-Survey -- 3-DOF Probe Arm CAD Generator v3
======================================================
座標系（與 machine_v2_ASSEMBLY.stl 相同）：
  X = 高度(上), Y = 橫向, Z = 前進(螺旋軸方向)
  平台頂面 X = 156 mm

手臂安裝（手臂自身局部座標：Z=上, X=前, Y=右）
  手臂局部 Z  →  機器 X（上）  ← 組裝時 ry(+90)

關節佈局：
  J1 Base  : 偏搖(Yaw)   ±60 deg，繞 X 軸旋轉，伺服 MG995
  J2 Shoulder: 俯仰(Pitch)  0-120 deg，繞 Y 軸，伺服 DS3218MG
  J3 Elbow : 俯仰(Pitch)  0-135 deg，繞 Y 軸，伺服 MG996R

鏈路長度：
  上臂   200 mm  (J2 → J3 旋轉中心)
  前臂   220 mm  (J3 → 探針夾座)   ← ≤220mm 確保能在220mm熱床印
  探針   180 mm 不鏽鋼管 + 20mm 錐頭 = 200mm

輸出 STL（stl/arm/ 目錄）：
  arm_base_mount.stl     -- 底座（含 J1 MG995 腔體、PCA9685 托盤）
  arm_j1_turret.stl      -- J1 偏搖轉台（含雙耳架）
  arm_upper_arm.stl      -- 上臂連桿（含 J2 DS3218 腔體）
  arm_forearm.stl        -- 前臂連桿（含 J3 MG996R 腔體）
  arm_probe_holder.stl   -- 探針夾座
  arm_pivot_pin.stl      -- 旋轉軸銷（M4×40 對應零件）
  arm_servo_horn_adp.stl -- 伺服 Horn 轉接片（14T 轉 M4 軸）
  arm_cable_clip.stl     -- 走線夾（×4 列印）
  arm_pca9685_tray.stl   -- PCA9685 I2C 控制板托盤
  arm_ASSEMBLY.stl       -- 完整組裝預覽（名義姿態 0-0-0 deg）

Bambu Studio 熱床 220×220mm，各零件均在 220mm 以內。
"""
import numpy as np
import struct, os, math

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stl", "arm")
os.makedirs(OUT_DIR, exist_ok=True)

SEGS = 32

# ═══════════════════════════════════════════════════════════════════════════════
# STL helpers
# ═══════════════════════════════════════════════════════════════════════════════
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

# ─── Primitive shapes ──────────────────────────────────────────────────────────
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
    """Hollow tube (annular cross-section)."""
    t = []
    for i in range(segs):
        a0 = 2*math.pi*i/segs; a1 = 2*math.pi*(i+1)/segs
        for r, sign in [(r_out,1),(r_in,-1)]:
            p0=[r*math.cos(a0),r*math.sin(a0),z0]
            p1=[r*math.cos(a1),r*math.sin(a1),z0]
            p2=[r*math.cos(a1),r*math.sin(a1),z1]
            p3=[r*math.cos(a0),r*math.sin(a0),z1]
            if sign>0: t += quad(p0,p1,p2,p3)
            else:      t += quad(p3,p2,p1,p0)
        # annular end caps
        for zc in [z0, z1]:
            oi=[r_in *math.cos(a0),r_in *math.sin(a0),zc]
            oo=[r_out*math.cos(a0),r_out*math.sin(a0),zc]
            ni=[r_in *math.cos(a1),r_in *math.sin(a1),zc]
            no=[r_out*math.cos(a1),r_out*math.sin(a1),zc]
            if zc==z0: t += [(oi,no,oo),(oi,ni,no)]
            else:      t += [(oo,no,oi),(no,ni,oi)]
    return t

def make_hollow_rect(W, H, L, wall, z0=0.0):
    """Rectangular hollow tube along Z, outer W×H, wall thickness wall.
    4 non-overlapping wall slabs + 2 solid end caps = true hollow section.
    """
    hw, hh = W/2, H/2
    iw, ih = W/2 - wall, H/2 - wall
    z1 = z0 + L
    t = []
    # 4 side walls — corners handled by -X/+X taking full Y width
    t += make_box(-hh, -ih, -hw,  hw, z0, z1)   # -X wall (full Y)
    t += make_box( ih,  hh, -hw,  hw, z0, z1)   # +X wall (full Y)
    t += make_box(-ih,  ih, -hw, -iw, z0, z1)   # -Y wall (inner X span)
    t += make_box(-ih,  ih,  iw,  hw, z0, z1)   # +Y wall (inner X span)
    # 2 solid end caps
    t += make_box(-hh,  hh, -hw,  hw, z0,       z0 + wall)
    t += make_box(-hh,  hh, -hw,  hw, z1 - wall, z1)
    return t

def make_servo_body(bw=20, bd=40, bh=38, segs=16):
    """
    伺服器幾何體（占位模型）
    局部原點：伺服軸中心（機體頂面）
    機體向 -Z 延伸 bd，X±bh/2，Y±bw/2
    伺服軸朝 +Z 伸出
    """
    t = []
    # Main body (shifted so shaft at z=0)
    t += make_box(-bh/2, bh/2, -bw/2, bw/2, -bd, 0)
    # Ear tabs
    for sy in [-1, 1]:
        ty0 = sy*(bw/2); ty1 = ty0 + sy*6
        t += make_box(bh*0.2, bh*0.2+8, min(ty0,ty1), max(ty0,ty1), -bd, 0)
    # Horn disk at shaft (z=0 top)
    horn = make_cyl(12, 0, 5, segs=segs)
    t += horn
    # Shaft stub
    shaft = make_cyl(3, 5, 12, segs=12)
    t += shaft
    return t

def make_bearing_boss(r_out=10, r_in=4, L=18):
    """軸承座（圓環形，繞 Z 軸，Z=0..L）"""
    return make_tube(r_out, r_in, 0, L, segs=24)

def make_gusset(lx, lz, thickness=3, mirror_y=False):
    """
    三角補強板：XZ 平面內直角三角，
    角點在 (0,0,0), (lx,0,0), (0,0,lz)
    thickness 為 Y 方向厚度
    """
    t = []
    y0, y1 = -thickness/2, thickness/2
    for y in [y0, y1]:
        # Triangle face
        t.append(([0,y,0],[lx,y,0],[0,y,lz]))
    # Side faces (3 rectangles)
    pts = [[0,0,0],[lx,0,0],[0,0,lz]]
    for i in range(3):
        a = pts[i]; b = pts[(i+1)%3]
        t += quad([a[0],y0,a[2]],[b[0],y0,b[2]],
                  [b[0],y1,b[2]],[a[0],y1,a[2]])
    if mirror_y:
        t = [tuple([v[0],-v[1],v[2]] for v in tri) for tri in t]
    return t

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 ── Base Mount  (J1 MG995 偏搖 + PCA9685 托盤)
# ═══════════════════════════════════════════════════════════════════════════════
def build_base_mount():
    """
    螺栓固定到平台，容納 MG995 伺服（J1 偏搖）及 PCA9685 托盤。
    外形：80W × 80D × 88H mm  (Z=up)
    底板：8mm，四角 M5 螺栓柱
    頂部：J1 輸出孔 + 轉台接口
    """
    t = []
    BW=80.0; BD=80.0; BH=88.0; WALL=5.0

    # 底板
    t += make_box(-BW/2, BW/2, -BD/2, BD/2, 0, WALL)
    # 四角螺栓柱（M5，6mm 外徑，嵌入底板）
    for sx, sy in [(-1,-1),(-1,1),(1,-1),(1,1)]:
        cx, cy = sx*(BW/2-10), sy*(BD/2-10)
        t += _tv(make_cyl(6, 0, WALL+5, segs=12), cx, cy, 0)

    # 四面牆（空心箱體）
    for sx in [-1,1]:
        t += make_box(sx*(BW/2-WALL), sx*BW/2, -BD/2, BD/2, WALL, BH)
    for sy in [-1,1]:
        t += make_box(-BW/2, BW/2, sy*(BD/2-WALL), sy*BD/2, WALL, BH)

    # 頂蓋（帶 J1 軸孔，以凸出軸座代替孔）
    t += make_box(-BW/2, BW/2, -BD/2, BD/2, BH-WALL, BH)
    # J1 軸座（頂部中心，OD 22mm）
    t += _tv(make_cyl(11, BH-WALL-1, BH+8, segs=24), 0, 0, 0)

    # MG995 伺服機體（軸朝 +Z，居中在腔體內）
    servo = make_servo_body(bw=20, bd=40, bh=38)
    servo = _tv(servo, 0, 0, BH-WALL)   # 軸在頂蓋面
    t += servo

    # PCA9685 托盤（安裝在腔體後壁內側）
    # PCB 尺寸 62×25mm，厚 2mm，固定在 z=15~17mm
    t += make_box(-31, 31, BD/2-WALL-27, BD/2-WALL-2, 12, 14)
    # 四角支柱
    for sx, sy in [(-1,-1),(-1,1),(1,-1),(1,1)]:
        px, py = sx*28, BD/2-WALL - (3 + sy*10 + 10)
        t += _tv(make_cyl(3, WALL, 14, segs=8), px, py, 0)

    # 走線口（前壁下方，8mm 圓形）
    # 以小凸柱示意（實際列印時需後處理鑽孔）
    wire_exit = make_cyl(4, 0, WALL+1, segs=12)
    wire_exit = _tv(wire_exit, 0, -BD/2, WALL+6)
    wire_exit = _rx(wire_exit, 90)
    t += wire_exit

    print(f"  Base Mount      : {len(t):>6} tris")
    return t

# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 ── J1 Turret  (偏搖輸出轉台，帶雙耳架供 J2 安裝)
# ═══════════════════════════════════════════════════════════════════════════════
def build_j1_turret():
    """
    套接 J1 伺服 Horn，轉台上方兩側各有一個耳架（M4 孔）供 J2 肩關節安裝。
    局部 Z=0 = J1 軸端面（即底座頂部 Z=BH+8=96mm 處）
    轉台高度：20mm（Z=0..20）
    耳架高度：再往上 30mm（Z=20..50），耳距 44mm（Y方向）
    """
    t = []
    DISK_R=35.0; DISK_H=20.0
    EAR_W=12.0; EAR_H=30.0; EAR_SPAN=44.0  # Y方向耳距

    # 主盤
    t += make_cyl(DISK_R, 0, DISK_H, segs=36)
    # Horn 中心連接孔（凸起定位銷，實際為母孔；STL 以小凸柱示意）
    t += make_cyl(6, -3, 0, segs=16)

    # 兩側耳架
    for sy in [-1, 1]:
        ey = sy * EAR_SPAN/2
        # 耳板
        t += make_box(-EAR_W/2, EAR_W/2, ey-EAR_W/2, ey+EAR_W/2, DISK_H, DISK_H+EAR_H)
        # M4 軸孔柱（凸柱，實際需鑽孔）
        boss = make_cyl(6, 0, EAR_W+6, segs=12)
        boss = _rx(boss, 90)
        boss = _tv(boss, 0, ey, DISK_H+EAR_H/2)
        t += boss

    # 加強筋（轉台盤面到耳板）
    rib = make_box(-4, 4, -EAR_SPAN/2+EAR_W/2, EAR_SPAN/2-EAR_W/2, DISK_H-2, DISK_H+EAR_H*0.4)
    t += rib

    print(f"  J1 Turret       : {len(t):>6} tris")
    return t

# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 ── Upper Arm  (J2 DS3218 腔體 + 200mm 空心連桿 + J3 軸座)
# ═══════════════════════════════════════════════════════════════════════════════
def build_upper_arm():
    """
    局部原點 = J2 旋轉中心（軸銷中心）。
    J2 DS3218MG 伺服腔體：50W×32D×52H，位於 z=-26..+26（腔體中心在 z=0）。
    連桿（28×20mm 矩形空心管）：z=+26 → z=+226（200mm）。
    J3 軸座：z=+226 末端。
    """
    t = []
    # ── J2 伺服腔體
    HW, HD, HH = 50.0, 32.0, 52.0
    WALL = 4.0
    # 外殼
    t += make_box(-HH/2, HH/2, -HD/2, HD/2, -HW/2, HW/2)
    # 旋轉軸孔柱（兩側 Y 軸承座）
    for sy in [-1,1]:
        boss = make_bearing_boss(r_out=11, r_in=4, L=14)
        boss = _rx(boss, 90)
        boss = _tv(boss, 0, sy*(HW/2), -6)
        t += boss

    # DS3218MG 伺服機體（軸朝 +Y，偏搖俯仰）
    servo = make_servo_body(bw=20, bd=40, bh=40)
    servo = _rx(servo, 90)          # 軸朝 +Y
    servo = _tv(servo, 0, -20, 0)  # 居中在腔體內
    t += servo

    # 伺服輸出 Horn 連接片（厚 5mm 板，接連桿）
    horn_plate = make_box(-5, 5, -HD/2, HD/2, HW/2, HW/2+10)
    t += horn_plate

    # ── 上臂連桿（矩形空心管 28×20，200mm）
    LINK_L = 200.0
    LW, LH = 28.0, 20.0; LWALL = 3.5
    beam = make_hollow_rect(LW, LH, LINK_L, LWALL, z0=0)
    beam = _tv(beam, 0, 0, HW/2)
    t += beam

    # 補強三角板（連桿與腔體交接處）
    for sy in [-1, 1]:
        g = make_box(-3, 3, sy*(HW/2), sy*(HW/2+30), HW/2, HW/2+20)
        t += g

    # ── J3 軸座（連桿末端，Y 方向）
    J3_Z = HW/2 + LINK_L
    for sy in [-1, 1]:
        boss = make_bearing_boss(r_out=10, r_in=3.5, L=14)
        boss = _rx(boss, 90)
        boss = _tv(boss, 0, sy*LW/2, J3_Z)
        t += boss

    # 末端端板
    t += make_box(-LH/2, LH/2, -LW/2, LW/2, J3_Z, J3_Z+LWALL)

    print(f"  Upper Arm       : {len(t):>6} tris")
    return t, HW/2, HW/2 + 200.0  # J2_offset_z, J3_z_local

# ═══════════════════════════════════════════════════════════════════════════════
# PART 4 ── Forearm  (J3 MG996R 腔體 + 220mm 連桿 + 探針座)
# ═══════════════════════════════════════════════════════════════════════════════
def build_forearm():
    """
    局部原點 = J3 旋轉中心。
    J3 MG996R 腔體：48W×30D×50H。
    連桿（25×18 空心管，220mm）。
    探針夾座（末端）。
    """
    t = []
    HW, HD, HH = 48.0, 30.0, 50.0
    WALL = 4.0

    # J3 腔體
    t += make_box(-HH/2, HH/2, -HD/2, HD/2, -HW/2, HW/2)
    # 旋轉軸孔柱
    for sy in [-1,1]:
        boss = make_bearing_boss(r_out=10, r_in=3.5, L=13)
        boss = _rx(boss, 90)
        boss = _tv(boss, 0, sy*(HW/2), -5)
        t += boss

    # MG996R 伺服機體
    servo = make_servo_body(bw=20, bd=40, bh=38)
    servo = _rx(servo, 90)
    servo = _tv(servo, 0, -18, 0)
    t += servo

    # Horn 連接片
    horn_plate = make_box(-5, 5, -HD/2, HD/2, HW/2, HW/2+10)
    t += horn_plate

    # ── 前臂連桿（220mm，25×18mm 空心管）
    LINK_L = 220.0
    LW, LH = 25.0, 18.0; LWALL = 3.0
    beam = make_hollow_rect(LW, LH, LINK_L, LWALL, z0=0)
    beam = _tv(beam, 0, 0, HW/2)
    t += beam

    # ── 探針夾座
    PROBE_Z = HW/2 + LINK_L
    CW, CD, CH = 38.0, 28.0, 32.0
    clamp_base = make_box(-CH/2, CH/2, -CW/2, CW/2, PROBE_Z, PROBE_Z+CD)
    t += clamp_base
    # 探針管孔（OD 12mm，以外凸圓柱示意）
    probe_cyl = make_cyl(6, PROBE_Z-8, PROBE_Z+CD+8, segs=16)
    t += probe_cyl
    # 鎖緊螺栓凸柱（M3×2）
    for sx in [-1,1]:
        bolt = make_cyl(3, PROBE_Z+4, PROBE_Z+CD-4, segs=8)
        bolt = _ry(bolt, 90)
        bolt = _tv(bolt, sx*(CH/2), 0, PROBE_Z+CD/2)
        t += bolt

    print(f"  Forearm         : {len(t):>6} tris")
    return t, HW/2, HW/2 + LINK_L  # J3_offset, tip_z

# ═══════════════════════════════════════════════════════════════════════════════
# PART 5 ── Probe Assembly  (10mm 不鏽鋼管 + 錐頭)
# ═══════════════════════════════════════════════════════════════════════════════
def build_probe():
    t = []
    OD=10.0; ID=7.0; L=180.0; TIP=20.0

    # 管體（空心）
    t += make_tube(OD/2, ID/2, 0, L, segs=SEGS)
    # 頂部環形蓋
    for i in range(SEGS):
        a0=2*math.pi*i/SEGS; a1=2*math.pi*(i+1)/SEGS
        oo=[OD/2*math.cos(a0),OD/2*math.sin(a0),0]
        io=[ID/2*math.cos(a0),ID/2*math.sin(a0),0]
        on=[OD/2*math.cos(a1),OD/2*math.sin(a1),0]
        in_=[ID/2*math.cos(a1),ID/2*math.sin(a1),0]
        t += [(io,on,oo),(io,in_,on)]
    # 錐頭
    for i in range(SEGS):
        a0=2*math.pi*i/SEGS; a1=2*math.pi*(i+1)/SEGS
        b0=[OD/2*math.cos(a0),OD/2*math.sin(a0),L]
        b1=[OD/2*math.cos(a1),OD/2*math.sin(a1),L]
        apex=[0,0,L+TIP]
        t.append((b0,b1,apex))
        t.append(([0,0,L],b1,b0))

    print(f"  Probe           : {len(t):>6} tris")
    return t

# ═══════════════════════════════════════════════════════════════════════════════
# PART 6 ── Pivot Pin  (M4×40 軸銷，帶法蘭)
# ═══════════════════════════════════════════════════════════════════════════════
def build_pivot_pin():
    """
    M4 軸銷替代件（PETG 列印預覽；實際用 M4 不鏽鋼螺栓）
    OD 4mm 桿體，40mm 長，兩端各有 8mm 法蘭（OD 10mm）
    """
    t = []
    ROD_R=2.0; FLANGE_R=5.0; L=40.0; FL=8.0
    t += make_cyl(ROD_R, 0, L, segs=16)
    # 頭法蘭
    t += make_cyl(FLANGE_R, 0, FL, segs=20)
    # 尾法蘭
    t += make_cyl(FLANGE_R, L-FL, L, segs=20)
    print(f"  Pivot Pin       : {len(t):>6} tris")
    return t

# ═══════════════════════════════════════════════════════════════════════════════
# PART 7 ── Servo Horn Adapter  (14T → M4 軸轉接片)
# ═══════════════════════════════════════════════════════════════════════════════
def build_servo_horn_adapter():
    """
    將伺服花鍵 Horn 轉接到 M4 軸銷的轉接片。
    圓盤 OD 30mm，厚 8mm，中心 M4 貫穿孔，兩旁 M3 固定孔（4支）
    """
    t = []
    DISK_R=15.0; THICK=8.0
    t += make_cyl(DISK_R, 0, THICK, segs=32)
    # 中心 M4 孔（以小凸出圓柱示意）
    t += make_cyl(2.5, -1, THICK+1, segs=12)
    # M3 固定孔×4（以凸柱示意）
    for ang in [45, 135, 225, 315]:
        rad=math.radians(ang)
        cx, cy = 10*math.cos(rad), 10*math.sin(rad)
        t += _tv(make_cyl(1.5, -1, THICK+1, segs=8), cx, cy, 0)
    print(f"  Servo Horn Adp  : {len(t):>6} tris")
    return t

# ═══════════════════════════════════════════════════════════════════════════════
# PART 8 ── Cable Clip  (走線夾，固定在連桿表面)
# ═══════════════════════════════════════════════════════════════════════════════
def build_cable_clip():
    """
    U 形走線夾，夾住 3mm 電線束，螺栓固定到連桿外壁。
    建議：列印 4 個（兩條連桿各 2 個）
    外形：22W × 10D × 14H mm
    """
    t = []
    BW=22.0; BD=10.0; BH=14.0; WALL=2.5; WIRE_R=2.5

    # 底板
    t += make_box(-BW/2, BW/2, -BD/2, BD/2, 0, WALL)
    # 固定孔柱（M2.5）
    for sx in [-1,1]:
        t += _tv(make_cyl(3, 0, WALL+3, segs=8), sx*(BW/2-4), 0, 0)
    # 兩側壁
    for sx in [-1,1]:
        t += make_box(sx*(BW/2-WALL), sx*BW/2, -BD/2, BD/2, 0, BH)
    # 頂部橋（帶半圓線槽）
    # 橋體
    t += make_box(-BW/2, BW/2, -BD/2, BD/2, BH-WALL, BH)
    # 線槽（半圓凸弧示意）
    slot = make_cyl(WIRE_R, BH, BH+WALL+1, segs=16)
    t += slot

    print(f"  Cable Clip      : {len(t):>6} tris")
    return t

# ═══════════════════════════════════════════════════════════════════════════════
# PART 9 ── PCA9685 Tray  (I2C 舵機控制板托盤)
# ═══════════════════════════════════════════════════════════════════════════════
def build_pca9685_tray():
    """
    PCA9685 模組托盤（獨立件，可螺栓固定在底座腔體內或平台上）
    PCB 尺寸約 62×25mm；托盤留 2mm 邊框，高 10mm
    四角支柱 M3×3，中心端子列區域鏤空（以框架示意）
    """
    t = []
    PCB_W=62.0; PCB_D=25.0; FRAME=2.5; H=10.0
    FW=PCB_W+FRAME*2; FD=PCB_D+FRAME*2

    # 底板
    t += make_box(-FW/2, FW/2, -FD/2, FD/2, 0, FRAME)
    # 四邊框
    for sx in [-1,1]:
        t += make_box(sx*(FW/2-FRAME), sx*FW/2, -FD/2, FD/2, 0, H)
    for sy in [-1,1]:
        t += make_box(-FW/2, FW/2, sy*(FD/2-FRAME), sy*FD/2, 0, H)
    # 頂框
    t += make_box(-FW/2, FW/2, -FD/2, FD/2, H-FRAME, H)
    # 四角 M3 支柱
    for sx, sy in [(-1,-1),(-1,1),(1,-1),(1,1)]:
        cx, cy = sx*(FW/2-FRAME-3), sy*(FD/2-FRAME-3)
        t += _tv(make_cyl(3, FRAME, H-FRAME, segs=8), cx, cy, 0)
    # 板子定位凸點（4角，防翹起）
    for sx, sy in [(-1,-1),(-1,1),(1,-1),(1,1)]:
        cx, cy = sx*(PCB_W/2-4), sy*(PCB_D/2-4)
        t += _tv(make_cyl(1.5, H-FRAME, H+1, segs=6), cx, cy, 0)

    print(f"  PCA9685 Tray    : {len(t):>6} tris")
    return t

# ═══════════════════════════════════════════════════════════════════════════════
# ASSEMBLY ── 組裝所有零件（名義姿態：全關節 0 度，直立）
# ═══════════════════════════════════════════════════════════════════════════════
def build_assembly():
    """
    手臂局部座標：Z=上
    鏈：平台面(Z=0) → 底座 → J1 轉台 → 上臂(J2) → 前臂(J3) → 探針
    返回：(tris, probe_tip_z)
    """
    all_tris = []

    # 參考平台薄板（100×100×5mm）
    all_tris += make_box(-50, 50, -50, 50, -5, 0)

    # 底座（Z=0..88）
    base = build_base_mount()
    all_tris += base
    BASE_TOP = 88.0

    # J1 轉台（Z=88..88+8+50=146 含軸座突出）
    turret = build_j1_turret()
    turret = _tv(turret, 0, 0, BASE_TOP + 8)  # 軸座高 8mm
    all_tris += turret
    TURRET_TOP = BASE_TOP + 8 + 20 + 30  # 轉台盤 + 耳架
    J2_Z = TURRET_TOP   # J2 旋轉中心高度（手臂局部Z）

    # 上臂（J2 原點在 J2_Z）
    upper, j2_off, j3_local = build_upper_arm()
    upper = _tv(upper, 0, 0, J2_Z)
    all_tris += upper
    J3_Z = J2_Z + j3_local  # J3 旋轉中心高度

    # 前臂（J3 原點在 J3_Z）
    forearm, j3_off, tip_local = build_forearm()
    forearm = _tv(forearm, 0, 0, J3_Z)
    all_tris += forearm
    PROBE_BASE_Z = J3_Z + tip_local

    # 探針（夾座末端往 +Z 伸出）
    probe = build_probe()
    probe = _tv(probe, 0, 0, PROBE_BASE_Z + 28 + 5)
    all_tris += probe
    PROBE_TIP_Z = PROBE_BASE_Z + 28 + 5 + 180 + 20

    print(f"  ASSEMBLY total  : {len(all_tris):>6} tris")
    return all_tris, PROBE_TIP_Z, J2_Z, J3_Z

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("3-DOF Probe Arm v3 -- Generating parts...")
print("=" * 60)
print()

base_t   = build_base_mount()
turret_t = build_j1_turret()
upper_t, *_  = build_upper_arm()
fore_t,  *_  = build_forearm()
probe_t  = build_probe()
pin_t    = build_pivot_pin()
horn_t   = build_servo_horn_adapter()
clip_t   = build_cable_clip()
tray_t   = build_pca9685_tray()
asm_t, probe_tip_z, J2_Z, J3_Z = build_assembly()

parts = [
    ("arm_base_mount.stl",    base_t),
    ("arm_j1_turret.stl",     turret_t),
    ("arm_upper_arm.stl",     upper_t),
    ("arm_forearm.stl",       fore_t),
    ("arm_probe.stl",         probe_t),
    ("arm_pivot_pin.stl",     pin_t),
    ("arm_servo_horn_adp.stl",horn_t),
    ("arm_cable_clip.stl",    clip_t),
    ("arm_pca9685_tray.stl",  tray_t),
    ("arm_ASSEMBLY.stl",      asm_t),
]

print()
print("Saving STL files...")
for fname, tris in parts:
    path = os.path.join(OUT_DIR, fname)
    write_stl(path, tris)
    sz = os.path.getsize(path)
    print(f"  {fname:<30}  {len(tris):>6} tris  ({sz//1024:>4} KB)")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("3-DOF Arm v3 Summary")
print("=" * 60)
print(f"  J1 Base top (yaw output)  : Z = {88+8:.0f} mm")
print(f"  J2 Shoulder pivot         : Z = {J2_Z:.0f} mm above platform")
print(f"  J3 Elbow pivot            : Z = {J3_Z:.0f} mm above platform")
print(f"  Probe tip (stowed, up)    : Z = {probe_tip_z:.0f} mm")
print()
print("  Joint specs:")
print("    J1 MG995    : Yaw  +-60 deg   13 kg.cm")
print("    J2 DS3218MG : Pitch 0-120 deg 20 kg.cm")
print("    J3 MG996R   : Pitch 0-135 deg 13 kg.cm")
print()
print("  Deploy sequence (probe to ground):")
print("    Step 1: J2 +90 deg (arm swings forward)")
print("    Step 2: J3 +90 deg (forearm swings down)")
print("    Step 3: Probe inserts ~20cm into sand")
print()
print("  IP65 requirements:")
print("    - All cavities sealed with 4mm PETG wall, O-ring on shaft holes")
print("    - Wiring bundled via base cable port, covered with heatshrink")
print("    - PCA9685 tray inside base housing (waterproof zone)")
print()
print("  Print notes (Bambu Studio, 220x220mm bed):")
print("    - Upper arm 200mm: print diagonally on bed")
print("    - Forearm 220mm: print diagonally (diagonal ~311mm OK)")
print("    - Other parts <100mm, can batch-print together")
print()
print(f"  Saved to: {OUT_DIR}")
print("Done.")
