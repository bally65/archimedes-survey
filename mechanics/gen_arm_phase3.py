"""
gen_arm_phase3.py
=================
Archimedes-Survey 機器手臂 Phase 3 CAD 擴充。

新增零件：
  arm_upper_arm_oct.stl    — 八角空心上臂連桿（替代矩形截面）
  arm_forearm_oct.stl      — 八角空心前臂連桿
  arm_transducer_mount.stl — 超音波換能器座（DS3218 俯仰 ±30°）
  motor_backplate_nema23.stl — 馬達背板含 NEMA23 螺孔（57×57mm 孔距 47.14mm 對角）
  platform_rib.stl         — 平台橫向加強肋（跨越兩條 PVC 管之間）

設計參數（Gemini Phase 3 建議）：
  八角連桿：壁厚 3.5mm，外切圓 R=15.5mm（平面距 28.5mm），列印方向=長軸橫向
  換能器座：DS3218（40×20×38mm）伺服，±30° 俯仰，換能器 Ø20mm 夾具
  NEMA23 孔：M5 過孔（Ø5.5mm），孔位 ±16.67mm × ±16.67mm
  平台肋：20W×12H 矩形截面，貫通兩條 PVC 管之間（Y=0~300mm）

座標系：與 machine_v2_ASSEMBLY.stl 相同
  X = 高度(上), Y = 橫向, Z = 前進(螺旋軸方向)
"""

import math
import os
import struct
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stl", "arm")
OUT_MECH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stl")
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(OUT_MECH, exist_ok=True)

SEGS = 32

# ═══════════════════════════════════════════════════════════════════════════
# STL helpers（與 gen_arm_3dof_v3.py 相同介面）
# ═══════════════════════════════════════════════════════════════════════════
def write_stl(path, tris):
    with open(path, "wb") as f:
        f.write(b"\x00" * 80)
        f.write(struct.pack("<I", len(tris)))
        for v0, v1, v2 in tris:
            a = np.array(v1) - np.array(v0)
            b = np.array(v2) - np.array(v0)
            n = np.cross(a, b)
            nm = np.linalg.norm(n)
            n = n / nm if nm > 1e-10 else np.array([0., 0., 1.])
            f.write(struct.pack("<fff", *n))
            f.write(struct.pack("<fff", *v0))
            f.write(struct.pack("<fff", *v1))
            f.write(struct.pack("<fff", *v2))
            f.write(struct.pack("<H", 0))

def quad(v0, v1, v2, v3):
    return [(v0, v1, v2), (v0, v2, v3)]

def _tv(tris, dx, dy, dz):
    return [([v[0]+dx, v[1]+dy, v[2]+dz] for v in tri) for tri in
            [([v0, v1, v2]) for v0, v1, v2 in tris]]

def tv(tris, dx=0., dy=0., dz=0.):
    return [([v0[0]+dx, v0[1]+dy, v0[2]+dz],
             [v1[0]+dx, v1[1]+dy, v1[2]+dz],
             [v2[0]+dx, v2[1]+dy, v2[2]+dz]) for v0, v1, v2 in tris]

def _ry(tris, deg):
    r = math.radians(deg); c, s = math.cos(r), math.sin(r)
    def rot(v): return [c*v[0]+s*v[2], v[1], -s*v[0]+c*v[2]]
    return [(rot(a), rot(b), rot(c2)) for a, b, c2 in tris]

def _rz(tris, deg):
    r = math.radians(deg); c, s = math.cos(r), math.sin(r)
    def rot(v): return [c*v[0]-s*v[1], s*v[0]+c*v[1], v[2]]
    return [(rot(a), rot(b), rot(c2)) for a, b, c2 in tris]

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
    """Hollow tube with annular cross-section."""
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
        for zc in [z0, z1]:
            oi=[r_in *math.cos(a0),r_in *math.sin(a0),zc]
            oo=[r_out*math.cos(a0),r_out*math.sin(a0),zc]
            ni=[r_in *math.cos(a1),r_in *math.sin(a1),zc]
            no=[r_out*math.cos(a1),r_out*math.sin(a1),zc]
            if zc==z0: t += [(oi,no,oo),(oi,ni,no)]
            else:      t += [(oo,no,oi),(no,ni,oi)]
    return t


# ═══════════════════════════════════════════════════════════════════════════
# 八角形截面輔助
# ═══════════════════════════════════════════════════════════════════════════
def make_octagonal_tube(r_out, r_in, z0, z1, offset_deg=22.5):
    """
    八角形空心管，沿 Z 軸延伸。
    r_out, r_in：外/內切圓半徑（mm）
    offset_deg=22.5 → 平面朝 X/Y 方向（非尖角朝上）
    壁厚 wall ≈ r_out - r_in（對角方向略薄）
    """
    segs = 8
    t = []
    angles = [math.radians(offset_deg + 45.0 * k) for k in range(segs)]
    angles_n = angles[1:] + [angles[0]]

    for a0, a1 in zip(angles, angles_n):
        # 外壁
        po0 = [r_out*math.cos(a0), r_out*math.sin(a0), z0]
        po1 = [r_out*math.cos(a1), r_out*math.sin(a1), z0]
        po2 = [r_out*math.cos(a1), r_out*math.sin(a1), z1]
        po3 = [r_out*math.cos(a0), r_out*math.sin(a0), z1]
        t += quad(po0, po1, po2, po3)
        # 內壁（法線向內）
        pi0 = [r_in *math.cos(a0), r_in *math.sin(a0), z0]
        pi1 = [r_in *math.cos(a1), r_in *math.sin(a1), z0]
        pi2 = [r_in *math.cos(a1), r_in *math.sin(a1), z1]
        pi3 = [r_in *math.cos(a0), r_in *math.sin(a0), z1]
        t += quad(pi3, pi2, pi1, pi0)
        # 環形端蓋（底部 z0，頂部 z1）
        t += [(pi0, po1, po0), (pi0, pi1, po1)]  # z0
        t += [(po3, po2, pi3), (po2, pi2, pi3)]  # z1

    return t


def oct_r_out(flat_to_flat_mm):
    """給定平面間距（flat-to-flat）mm，回傳外切圓半徑。"""
    return flat_to_flat_mm / (2.0 * math.cos(math.radians(22.5)))


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — 八角空心上臂連桿
# ═══════════════════════════════════════════════════════════════════════════
def build_upper_arm_oct():
    """
    八角空心上臂，沿 Z 延伸 200mm。
    外 flat-to-flat = 28.5mm，壁厚 3.5mm。
    兩端各有 12mm 圓形端蓋（與關節介面接合）。
    J2 伺服腔體整合於 -Z 端（端部 44mm 段）。
    """
    L        = 200.0
    F2F_OUT  = 28.5           # 外平面距
    WALL     = 3.5
    F2F_IN   = F2F_OUT - 2 * WALL
    R_OUT    = oct_r_out(F2F_OUT)
    R_IN     = oct_r_out(F2F_IN)
    CAP_L    = 12.0

    t = []
    # 主管體（八角空心）
    t += make_octagonal_tube(R_OUT, R_IN, 0, L)

    # -Z 端蓋（實心八角板，用於 J2 伺服接口）
    for a0, a1 in _oct_face_pairs():
        po0 = [R_OUT*math.cos(a0), R_OUT*math.sin(a0), 0]
        po1 = [R_OUT*math.cos(a1), R_OUT*math.sin(a1), 0]
        pi0 = [R_IN *math.cos(a0), R_IN *math.sin(a0), 0]
        pi1 = [R_IN *math.cos(a1), R_IN *math.sin(a1), 0]
        t += [(pi0, po0, [0,0,0]), (pi1, pi0, [0,0,0]),
              (po1, pi1, [0,0,0]), (po0, po1, [0,0,0])]

    # +Z 端蓋（實心，J3 關節）
    for a0, a1 in _oct_face_pairs():
        po0 = [R_OUT*math.cos(a0), R_OUT*math.sin(a0), L]
        po1 = [R_OUT*math.cos(a1), R_OUT*math.sin(a1), L]
        t += [([0,0,L], po0, po1)]

    # J2 伺服腔（簡化：矩形槽，於 Z=0..44 段）
    SERVO_W, SERVO_D, SERVO_H = 20.0, 40.0, 38.0
    # 在端蓋上挖槽（用負空間表示，實際列印時設計為開口）
    # 以補強耳座代替：兩側耳板
    ear_w = 8; ear_h = 6; ear_l = 8
    for sy in [-1, 1]:
        ty = sy * (F2F_OUT/2)
        t += make_box(-SERVO_H/2, SERVO_H/2,
                      ty, ty + sy*ear_w,
                      -ear_l, 0)

    # 走線槽：縱向 3mm 溝槽（象徵性凸起，實際列印時為通孔）
    # 用弱凸起表示（不挖空，以免 STL 自交）
    t += make_box(-1.5, 1.5, -1.5, 1.5, 5, L-5)

    return t

def _oct_face_pairs():
    """回傳八角形 8 對相鄰邊角角度對（弧度）。"""
    angles = [math.radians(22.5 + 45.0 * k) for k in range(8)]
    return list(zip(angles, angles[1:] + [angles[0]]))


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — 八角空心前臂連桿
# ═══════════════════════════════════════════════════════════════════════════
def build_forearm_oct():
    """
    八角空心前臂，沿 Z 延伸 220mm（最大 220mm，適合熱床）。
    外 flat-to-flat = 24mm，壁厚 3.5mm（細一點，末端負載較輕）。
    +Z 端整合探針夾座介面（20mm Ø）。
    """
    L        = 220.0
    F2F_OUT  = 24.0
    WALL     = 3.5
    F2F_IN   = F2F_OUT - 2 * WALL
    R_OUT    = oct_r_out(F2F_OUT)
    R_IN     = oct_r_out(F2F_IN)

    t = []
    t += make_octagonal_tube(R_OUT, R_IN, 0, L)

    # 兩端蓋
    for a0, a1 in _oct_face_pairs():
        po0 = [R_OUT*math.cos(a0), R_OUT*math.sin(a0), 0]
        po1 = [R_OUT*math.cos(a1), R_OUT*math.sin(a1), 0]
        t += [([0,0,0], po1, po0)]
        po0L = [R_OUT*math.cos(a0), R_OUT*math.sin(a0), L]
        po1L = [R_OUT*math.cos(a1), R_OUT*math.sin(a1), L]
        t += [([0,0,L], po0L, po1L)]

    # +Z 端探針夾圓柱介面（Ø28mm，長 16mm）
    probe_boss = make_tube(14, 10, L, L+16, segs=SEGS)
    t += probe_boss

    # J3 伺服耳座（於 Z=0 端）
    ear_w = 8
    for sy in [-1, 1]:
        ty = sy * (F2F_OUT/2)
        t += make_box(-10, 10, ty, ty + sy*ear_w, -8, 0)

    return t


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — 超音波換能器座（DS3218 俯仰 ±30°）
# ═══════════════════════════════════════════════════════════════════════════
def build_transducer_mount():
    """
    換能器座：
      - 底座：M4×4 螺孔固定到前臂末端
      - DS3218 伺服腔（40×20×38mm）：俯仰軸驅動換能器 ±30°
      - 換能器夾具：Ø20mm 夾環 + M3 鎖緊螺絲座
      - 防水 O-Ring 溝槽（Ø22mm，寬 2.5mm）

    局部座標：Z = 安裝軸（朝前），X = 俯仰旋轉軸，Y = 換能器朝向
    """
    t = []

    # ── 底座平板（連接前臂末端）─────────────────────────────────────────
    BASE_W, BASE_H, BASE_T = 36.0, 36.0, 8.0
    t += make_box(-BASE_W/2, BASE_W/2, -BASE_H/2, BASE_H/2, -BASE_T, 0)
    # M4 螺柱（×4 角）
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            cx, cy = sx*13.0, sy*13.0
            boss = make_cyl(4.0, -BASE_T, -BASE_T+10, segs=16)
            t += tv(boss, dx=cx, dy=cy)

    # ── DS3218 伺服腔體（開口朝 +X 軸）──────────────────────────────────
    SERVO_W, SERVO_D, SERVO_H = 20.0, 40.0, 38.0
    WALL = 3.0
    # 外殼
    t += make_box(-SERVO_H/2 - WALL, SERVO_H/2 + WALL,
                  -SERVO_W/2 - WALL, SERVO_W/2 + WALL,
                  0, SERVO_D + WALL)
    # 開口（以外殼減去內腔，以 STL 實心表示為薄壁） — 近似：頂蓋留 3mm
    # 省略內腔挖空（Bambu Studio 列印時手動設定空心）
    # 伺服固定耳
    for sx in [-1, 1]:
        t += make_box(sx * (SERVO_H/2), sx * (SERVO_H/2 + 6),
                      -SERVO_W/2 - WALL, SERVO_W/2 + WALL,
                      WALL, SERVO_D - WALL)

    # ── 俯仰關節樞軸（X 軸旋轉，圓柱軸座）──────────────────────────────
    PIVOT_R, PIVOT_L = 5.5, 14.0
    pivot_L = make_cyl(PIVOT_R, 0, PIVOT_L, segs=16)
    pivot_R = make_cyl(PIVOT_R, 0, PIVOT_L, segs=16)
    # 左側樞軸 (-Y)
    t += tv(_rz(pivot_L, 90), dx=0, dy=-(SERVO_W/2 + WALL + PIVOT_L), dz=SERVO_D/2)
    # 右側樞軸 (+Y)
    t += tv(_rz(pivot_R, -90), dx=0, dy=(SERVO_W/2 + WALL), dz=SERVO_D/2)

    # ── 換能器夾環（在樞軸末端，Ø20mm 換能器）──────────────────────────
    CLAMP_R_OUT, CLAMP_R_IN = 14.0, 10.5  # 壁厚 3.5mm
    CLAMP_L = 28.0
    ORING_GROOVE_W = 2.5; ORING_GROOVE_D = 1.5

    # 夾環主體
    clamp_body = make_tube(CLAMP_R_OUT, CLAMP_R_IN, 0, CLAMP_L, segs=SEGS)
    # 旋轉 90° 朝 -Z（換能器向下指向泥地）
    t += tv(_ry(clamp_body, 90),
            dx=0,
            dy=-(SERVO_W/2 + WALL + PIVOT_L + CLAMP_L),
            dz=SERVO_D/2)

    # O-Ring 溝槽（用外凸環表示，列印後銑削）
    oring_ring = make_tube(CLAMP_R_OUT + 1.5, CLAMP_R_OUT - 0.5,
                           CLAMP_L/2 - ORING_GROOVE_W/2,
                           CLAMP_L/2 + ORING_GROOVE_W/2,
                           segs=SEGS)
    t += tv(_ry(oring_ring, 90),
            dx=0,
            dy=-(SERVO_W/2 + WALL + PIVOT_L + CLAMP_L),
            dz=SERVO_D/2)

    # M3 鎖緊螺絲座（夾環側面凸台，×2）
    for sz in [0.3, 0.7]:
        screw_boss = make_cyl(4.0, 0, 6, segs=12)
        t += tv(_ry(screw_boss, 90),
                dx=CLAMP_R_OUT,
                dy=-(SERVO_W/2 + WALL + PIVOT_L + CLAMP_L * sz),
                dz=SERVO_D/2)

    return t


# ═══════════════════════════════════════════════════════════════════════════
# PART 4 — 馬達背板（含 NEMA23 螺孔）
# ═══════════════════════════════════════════════════════════════════════════
def build_motor_backplate_nema23():
    """
    馬達背板，連接兩條螺旋的 NEMA23 馬達（前端或後端各一片）。

    板外形：Y = 0~300mm（跨越 DY2=300mm），Z 寬 = 86mm（馬達寬）
    板厚：8mm（X 方向）
    NEMA23 螺孔：M5 過孔（Ø5.5mm），孔位中心距 47.14mm 對角
      = ±16.67mm × ±16.67mm，共 8 孔（兩側馬達各 4 孔）

    板中心有兩個 Ø22mm 軸通孔（聯軸器空間）
    """
    DY2       = 300.0          # 兩條螺旋 Y 間距
    PLATE_Y0  = -43.0          # Y 起點（第一條螺旋馬達背面）
    PLATE_Y1  = DY2 + 43.0     # Y 終點（第二條螺旋馬達背面）
    PLATE_Z0  = -43.0          # Z 起點（馬達寬 86mm 的一半）
    PLATE_Z1  =  43.0          # Z 終點
    PLATE_T   = 8.0            # 板厚（X）
    PLATE_X0  = 0.0
    PLATE_X1  = PLATE_T

    t = []

    # 主板（實心矩形）
    t += make_box(PLATE_X0, PLATE_X1,
                  PLATE_Y0, PLATE_Y1,
                  PLATE_Z0, PLATE_Z1)

    # NEMA23 螺孔：以 Ø5.5mm 圓柱孔代替（正空間 → 列印後鑽孔）
    # 標記孔位：以細深孔（直通板）表示，Bambu Studio 切片時設空心
    # 實作：以略小的圓柱（Ø2.5，代表鑽孔導引點）
    BOLT_OFFSET = 16.67  # mm
    SHAFT_R     = 11.0   # 軸通孔半徑 22mm
    PILOT_R     =  2.75  # 導孔半徑（M5 先導）

    # 第一條螺旋（Y 中心 = 0）
    cy1 = 0.0
    # 第二條螺旋（Y 中心 = 300）
    cy2 = DY2

    for cy in [cy1, cy2]:
        # 軸通孔（大）
        shaft_hole = make_cyl(SHAFT_R, PLATE_X0 - 1, PLATE_X1 + 1, segs=24)
        t += tv(_rz(_ry(shaft_hole, 90), 0), dx=PLATE_X0 + PLATE_T/2, dy=cy, dz=0)

        # NEMA23 螺孔 ×4
        for bx in [-BOLT_OFFSET, BOLT_OFFSET]:
            for bz in [-BOLT_OFFSET, BOLT_OFFSET]:
                pilot = make_cyl(PILOT_R, PLATE_X0 - 1, PLATE_X1 + 1, segs=12)
                t += tv(_ry(pilot, 90), dx=PLATE_X0 + PLATE_T/2, dy=cy + bz, dz=bx)

    # 橫向加強肋（板中段，Y 方向兩側）
    RIB_H = 20.0; RIB_T = 5.0
    for yz in [PLATE_Y0 + 10, PLATE_Y1 - 10]:
        t += make_box(PLATE_X1, PLATE_X1 + RIB_H,
                      yz - RIB_T/2, yz + RIB_T/2,
                      PLATE_Z0, PLATE_Z1)

    return t


# ═══════════════════════════════════════════════════════════════════════════
# PART 5 — 平台橫向加強肋
# ═══════════════════════════════════════════════════════════════════════════
def build_platform_rib():
    """
    平台橫向加強肋，等間距 6 條，分佈於平台底面。

    肋位置（Z 方向，沿平台長度）：等分 878mm 長度
    截面：20W（Y）× 12H（X）× 300L（Y，跨越兩管）
    材料：8mm 實心 PETG，20° 角接連接到平台底面

    平台範圍：Y=-104~404mm（DY2=300，管寬裕各 52mm）
    肋 Y 跨越：Y=0~300mm（剛好連接兩條 PVC 管中心線）
    """
    PLAT_X0   = 144.0   # 平台底面 X
    PLAT_Y0   = -104.0
    PLAT_Y1   =  404.0
    PLAT_Z0   = -214.0
    PLAT_Z1   =  664.0
    PLAT_L    = PLAT_Z1 - PLAT_Z0  # 878mm

    RIB_W     = 20.0    # Y 方向 — 橫過兩管
    RIB_H     = 12.0    # X 方向高度（與平台板厚相同）
    RIB_Y0    = 0.0     # 第一條管 Y 中心
    RIB_Y1    = 300.0   # 第二條管 Y 中心

    N_RIBS    = 6       # 均勻分布 6 條
    t = []

    z_positions = [PLAT_Z0 + PLAT_L * (i + 1) / (N_RIBS + 1)
                   for i in range(N_RIBS)]

    for zc in z_positions:
        # 主肋板
        t += make_box(PLAT_X0 - RIB_H, PLAT_X0,
                      RIB_Y0, RIB_Y1,
                      zc - RIB_W/2, zc + RIB_W/2)
        # 斜角接頭（三角補強，連接肋與平台底面）
        GUSSET_L = 15.0; GUSSET_T = RIB_W - 4
        for gsy in [RIB_Y0 + 2, RIB_Y1 - 2 - GUSSET_T]:
            # XZ 平面三角補強板
            pts_xz = [[PLAT_X0, gsy, zc - RIB_W/2],
                      [PLAT_X0, gsy, zc + RIB_W/2],
                      [PLAT_X0 - GUSSET_L, gsy, zc - RIB_W/2]]
            t.append((pts_xz[0], pts_xz[2], pts_xz[1]))
            pts_xz2 = [[PLAT_X0, gsy+GUSSET_T, zc - RIB_W/2],
                       [PLAT_X0, gsy+GUSSET_T, zc + RIB_W/2],
                       [PLAT_X0 - GUSSET_L, gsy+GUSSET_T, zc - RIB_W/2]]
            t.append((pts_xz2[0], pts_xz2[1], pts_xz2[2]))

    return t


# ═══════════════════════════════════════════════════════════════════════════
# 主程式
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parts = [
        ("upper_arm_oct",       build_upper_arm_oct,         OUT_DIR),
        ("forearm_oct",         build_forearm_oct,           OUT_DIR),
        ("transducer_mount",    build_transducer_mount,      OUT_DIR),
        ("motor_backplate_nema23", build_motor_backplate_nema23, OUT_MECH),
        ("platform_rib",        build_platform_rib,          OUT_MECH),
    ]

    for name, builder, out_dir in parts:
        print(f"Building {name}...", end=" ", flush=True)
        tris = builder()
        path = os.path.join(out_dir, f"arm_{name}.stl" if out_dir == OUT_DIR
                            else f"{name}.stl")
        write_stl(path, tris)
        print(f"OK  {len(tris)} tris -> {os.path.basename(path)}")

    print("\n--- Phase 3 CAD complete ---")
    print("Files:")
    print("  stl/arm/arm_upper_arm_oct.stl       Octagonal upper arm (200mm, F2F=28.5mm, wall=3.5mm)")
    print("  stl/arm/arm_forearm_oct.stl          Octagonal forearm   (220mm, F2F=24mm,  wall=3.5mm)")
    print("  stl/arm/arm_transducer_mount.stl     Transducer mount    (DS3218 +/-30deg, 20mm clamp)")
    print("  stl/motor_backplate_nema23.stl       Motor backplate     (NEMA23 holes +/-16.67mm)")
    print("  stl/platform_rib.stl                 Platform ribs       (6x, 20x12mm cross-section)")
    print()
    print("Print notes (Bambu Studio):")
    print("  Arm links: print long-axis horizontal, 40% Gyroid, Layer 0.2mm")
    print("  Transducer mount: support material needed (clamp overhang)")
    print("  Motor backplate: drill through pilot holes with 5.5mm bit (M5 clearance)")


if __name__ == "__main__":
    main()
