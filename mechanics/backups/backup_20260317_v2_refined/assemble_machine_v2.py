"""
整機組裝 v2
- 管長 900mm（400+500延伸）
- PVC 端蓋前後各一（厚 15mm，中心孔 22mm 直徑）
- 不鏽鋼中心軸（直徑 20mm，貫穿全長）
- 馬達支架（後端兩側）+ 頂部平台（放電池/主板）
- 暫時無培林、無機械手臂
"""
import numpy as np
import struct, os, math

OUT_DIR   = os.path.join(os.path.dirname(__file__), "stl")
PARTS_DIR = os.path.join(os.path.dirname(__file__), "parts")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 全局參數 ─────────────────────────────────────────────
PIPE_R    = 84.0    # PVC 管外壁半徑 mm
PIPE_T    = 6.0     # 管壁厚 mm
PIPE_LEN  = 450.0   # 管長 mm
CAP_T     = 15.0    # 端蓋厚度 mm
CAP_R     = PIPE_R  # 端蓋半徑（與管齊平）
ROD_R     = 10.0    # 不鏽鋼軸半徑 mm（直徑 20mm）
ROD_LEN   = PIPE_LEN + CAP_T*2 + 40  # 軸總長（兩端各伸出 20mm）
SEGS      = 64      # 圓形細分

# 支架參數
BRACKET_W   = 30.0   # 支架厚度 mm
BRACKET_H   = 120.0  # 支架高度 mm（從管中心算起）
PLATFORM_T  = 10.0   # 平台板厚度 mm
PLATFORM_W  = PIPE_R*2 + 60  # 平台寬（左右超出管各30mm）
PLATFORM_L  = 200.0  # 平台長度 mm（前後方向）

# ── STL 工具 ─────────────────────────────────────────────
def write_stl(path, tris):
    with open(path, "wb") as f:
        f.write(b"\x00" * 80)
        f.write(struct.pack("<I", len(tris)))
        for v0, v1, v2 in tris:
            a = np.array(v1)-np.array(v0)
            b = np.array(v2)-np.array(v0)
            n = np.cross(a, b)
            nm = np.linalg.norm(n)
            n = n/nm if nm > 1e-10 else np.array([0,0,1])
            f.write(struct.pack("<fff", *n))
            f.write(struct.pack("<fff", *v0))
            f.write(struct.pack("<fff", *v1))
            f.write(struct.pack("<fff", *v2))
            f.write(struct.pack("<H", 0))

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

def quad(v0,v1,v2,v3):
    return [(v0,v1,v2),(v0,v2,v3)]

def translate(tris, dx=0, dy=0, dz=0):
    return [([v0[0]+dx,v0[1]+dy,v0[2]+dz],
              [v1[0]+dx,v1[1]+dy,v1[2]+dz],
              [v2[0]+dx,v2[1]+dy,v2[2]+dz]) for v0,v1,v2 in tris]

def rotate_y(tris, deg):
    r = math.radians(deg)
    c,s = math.cos(r), math.sin(r)
    def ry(v): return [c*v[0]+s*v[2], v[1], -s*v[0]+c*v[2]]
    return [(ry(v0),ry(v1),ry(v2)) for v0,v1,v2 in tris]

# ── 圓柱生成（空心可選） ──────────────────────────────────
def make_cylinder(r, z0, z1, segs=SEGS, cap_bottom=True, cap_top=True):
    tris = []
    for i in range(segs):
        a0 = 2*np.pi*i/segs
        a1 = 2*np.pi*(i+1)/segs
        p00=[r*np.cos(a0),r*np.sin(a0),z0]
        p10=[r*np.cos(a1),r*np.sin(a1),z0]
        p01=[r*np.cos(a0),r*np.sin(a0),z1]
        p11=[r*np.cos(a1),r*np.sin(a1),z1]
        tris += quad(p00,p10,p11,p01)
        if cap_bottom: tris.append(([0,0,z0],p10,p00))
        if cap_top:    tris.append(([0,0,z1],p01,p11))
    return tris

def make_ring(r_out, r_in, z0, z1, segs=SEGS):
    """環形（中空圓柱）"""
    tris = []
    for i in range(segs):
        a0 = 2*np.pi*i/segs
        a1 = 2*np.pi*(i+1)/segs
        oo0=[r_out*np.cos(a0),r_out*np.sin(a0),z0]
        oo1=[r_out*np.cos(a1),r_out*np.sin(a1),z0]
        oi0=[r_out*np.cos(a0),r_out*np.sin(a0),z1]
        oi1=[r_out*np.cos(a1),r_out*np.sin(a1),z1]
        io0=[r_in*np.cos(a0), r_in*np.sin(a0), z0]
        io1=[r_in*np.cos(a1), r_in*np.sin(a1), z0]
        ii0=[r_in*np.cos(a0), r_in*np.sin(a0), z1]
        ii1=[r_in*np.cos(a1), r_in*np.sin(a1), z1]
        # 外壁
        tris += quad(oo0,oo1,oi1,oi0)
        # 內壁（反向）
        tris += quad(io1,ii1,ii0,io0)
        # 底環
        tris += quad(io0,io1,oo1,oo0)
        # 頂環
        tris += quad(oo0,oi0,ii0,io0) # 修正
        tris += quad(oi0,oi1,ii1,ii0)
    return tris

# ── O-ring 溝槽端蓋 ───────────────────────────────────────
def make_ring_oring(r_out, r_in, z0, z1, segs=SEGS,
                    groove_depth=2.4, groove_width=4.2):
    """
    帶 O-ring 溝槽的端蓋圓環（徑向密封）。
    溝槽規格：線徑 3.1mm → 深 2.4mm（77%），寬 4.2mm（1.35×）
    溝槽位置：外壁中心高度，向內凹 groove_depth。
    """
    tris = []
    z_mid  = (z0 + z1) / 2.0
    gz0    = z_mid - groove_width / 2.0   # 溝槽下緣
    gz1    = z_mid + groove_width / 2.0   # 溝槽上緣
    r_grv  = r_out - groove_depth          # 溝槽底半徑

    for i in range(segs):
        a0 = 2*np.pi*i/segs
        a1 = 2*np.pi*(i+1)/segs
        cos0, sin0 = np.cos(a0), np.sin(a0)
        cos1, sin1 = np.cos(a1), np.sin(a1)

        # 外壁分三段（溝槽上/溝槽凹/溝槽下）
        for za, zb, r in [(z0, gz0, r_out), (gz0, gz1, r_grv), (gz1, z1, r_out)]:
            p00=[r*cos0, r*sin0, za]; p10=[r*cos1, r*sin1, za]
            p01=[r*cos0, r*sin0, zb]; p11=[r*cos1, r*sin1, zb]
            tris += quad(p00, p10, p11, p01)
        # 溝槽台階（上下各連接全徑→溝槽徑）
        tris += quad([r_out*cos0,r_out*sin0,gz0],[r_out*cos1,r_out*sin1,gz0],
                     [r_grv*cos1,r_grv*sin1,gz0],[r_grv*cos0,r_grv*sin0,gz0])
        tris += quad([r_grv*cos0,r_grv*sin0,gz1],[r_grv*cos1,r_grv*sin1,gz1],
                     [r_out*cos1,r_out*sin1,gz1],[r_out*cos0,r_out*sin0,gz1])
        # 內壁
        io0=[r_in*cos0,r_in*sin0,z0]; io1=[r_in*cos1,r_in*sin1,z0]
        ii0=[r_in*cos0,r_in*sin0,z1]; ii1=[r_in*cos1,r_in*sin1,z1]
        tris += quad(ii0,ii1,io1,io0)
        # 底環
        tris += quad(io0,io1,[r_out*cos1,r_out*sin1,z0],[r_out*cos0,r_out*sin0,z0])
        # 頂環
        tris += quad([r_out*cos0,r_out*sin0,z1],[r_out*cos1,r_out*sin1,z1],ii1,ii0)
    return tris

# ── 零件生成 ─────────────────────────────────────────────

def make_pipe():
    """PVC 管體（空心圓柱）"""
    return make_ring(PIPE_R, PIPE_R-PIPE_T, 0, PIPE_LEN)

def make_cap(z_pos):
    """PVC 端蓋（環形板，中心有孔）"""
    # 端蓋 = 外半徑 PIPE_R，內孔半徑 ROD_R+2（略大於軸）
    z0 = z_pos
    z1 = z_pos + CAP_T if z_pos == 0 else z_pos - CAP_T
    return make_ring(PIPE_R, ROD_R+2, min(z0,z1), max(z0,z1))

def make_rod():
    """不鏽鋼中心軸"""
    z0 = -CAP_T - 20
    z1 = PIPE_LEN + CAP_T + 20
    return make_cylinder(ROD_R, z0, z1)

def make_bracket(side):
    """
    馬達支架：L 形金屬板
    side: +1 = 右側(+Y)，-1 = 左側(-Y)
    支架從管外壁延伸出去，高度超過管頂
    """
    tris = []
    bw = BRACKET_W
    bh = BRACKET_H
    # 垂直板（Y 方向，後端）
    y0 = side * PIPE_R
    y1 = side * (PIPE_R + bw)
    z0 = PIPE_LEN - 50
    z1 = PIPE_LEN + 20
    x0 = -bh
    x1 = 0
    # 垂直板 6 面
    verts_v = [
        [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1],
    ]
    faces = [
        (0,1,2,3),(7,6,5,4),(0,4,5,1),(3,2,6,7),(0,3,7,4),(1,5,6,2)
    ]
    for f in faces:
        v = [verts_v[i] for i in f]
        tris += quad(v[0],v[1],v[2],v[3])

    return tris

def make_platform():
    """
    頂部平台（橫跨左右支架上方）
    用來放電池、主板等
    """
    pw = PLATFORM_W
    pl = PLATFORM_L
    pt = PLATFORM_T
    top_z = BRACKET_H  # 平台在支架頂端
    # 平台中心在管正上方（x = -BRACKET_H），延伸到 z = PIPE_LEN
    x0 = -BRACKET_H
    x1 = x0 + pt
    y0 = -pw/2
    y1 =  pw/2
    z0 = PIPE_LEN - pl
    z1 = PIPE_LEN

    verts = [
        [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1],
    ]
    faces = [(0,1,2,3),(7,6,5,4),(0,4,5,1),(3,2,6,7),(0,3,7,4),(1,5,6,2)]
    tris = []
    for f in faces:
        v = [verts[i] for i in f]
        tris += quad(v[0],v[1],v[2],v[3])
    return tris

# ── 讀入螺旋（更新螺距後重新生成） ──────────────────────
def make_spiral_blades():
    """
    雙螺旋葉片：A/B 兩條螺旋 Z 範圍相同（0 ~ PIPE_LEN），
    B 只做角度旋轉 180°，不做 Z 偏移，確保與管對齊。
    """
    OUTER_R = 134.0
    INNER_R = 84.0
    PITCH   = 450.0 / 4  # ~112.5mm/圈
    SEG_DEG = 36.0
    N_SEGS  = 10
    N_TURNS = 4
    BLADE_T = 5.0
    zpd     = PITCH / 360.0

    tris = []
    for ang_offset in [0.0, 180.0]:   # A=0°, B=180° 角度旋轉
        for turn in range(N_TURNS):
            for seg in range(N_SEGS):
                gs = turn*N_SEGS + seg
                # Z 位置兩條螺旋完全一樣，只有角度不同
                z0 = gs * SEG_DEG * zpd
                z1 = (gs+1) * SEG_DEG * zpd
                a0 = np.radians(gs*SEG_DEG + ang_offset)
                a1 = np.radians((gs+1)*SEG_DEG + ang_offset)
                ht = BLADE_T/2
                def pt(r,a,z,dz): return [r*np.cos(a),r*np.sin(a),z+dz]
                ti0=pt(INNER_R,a0,z0,+ht); to0=pt(OUTER_R,a0,z0,+ht)
                ti1=pt(INNER_R,a1,z1,+ht); to1=pt(OUTER_R,a1,z1,+ht)
                bi0=pt(INNER_R,a0,z0,-ht); bo0=pt(OUTER_R,a0,z0,-ht)
                bi1=pt(INNER_R,a1,z1,-ht); bo1=pt(OUTER_R,a1,z1,-ht)
                tris += quad(ti0,to0,to1,ti1)
                tris += quad(bi1,bo1,bo0,bi0)
                tris += quad(to0,bo0,bo1,to1)
                tris += quad(ti1,bi1,bi0,ti0)
                tris += quad(ti0,to0,bo0,bi0)
                tris += quad(to1,ti1,bi1,bo1)
    return tris

# ── 組裝 ─────────────────────────────────────────────────
all_tris = []

# 管體
all_tris += make_pipe()

# 螺旋葉片（重新以900mm長、4圈計算）
all_tris += make_spiral_blades()

# 前端蓋（帶 O-ring 溝槽，線徑 3.1mm）
all_tris += make_ring_oring(PIPE_R, ROD_R+2, -CAP_T, 0)

# 後端蓋（帶 O-ring 溝槽）
all_tris += make_ring_oring(PIPE_R, ROD_R+2, PIPE_LEN, PIPE_LEN+CAP_T)

# 中心軸
all_tris += make_rod()

# NEMA34 馬達（一顆）
# 步驟1：放在管頂（+X 方向），Z 置中於管中點
# 步驟2：繞 Y 軸旋轉 90°，軸從朝 +Z 轉為朝 +X（立起來）
motor_raw = read_stl(os.path.join(PARTS_DIR, "NEMA34_86BHH156.stl"))

MOTOR_W = 86.0
MOTOR_L = 156.0
MID_Z   = PIPE_LEN / 2

# 馬達放在後端蓋外側，軸沿 Z 方向對準不鏽鋼棍
MOTOR_W = 86.0
MOTOR_L = 156.0
SHAFT_L = 38.0

# 軸朝 -Z（朝向管內），框架在端蓋外
# rotate_x(180°) 讓軸從 +Z 翻轉成 -Z
def rotate_x(tris, deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    def rx(v): return [v[0], c*v[1]-s*v[2], s*v[1]+c*v[2]]
    return [(rx(v0), rx(v1), rx(v2)) for v0, v1, v2 in tris]

# 1. x/y 已置中，只需 z 置中
motor = translate(motor_raw,
                  dx=0,
                  dy=0,
                  dz=-MOTOR_L/2)

# 2. 翻轉：軸改朝 -Z（插入棍）
motor = rotate_x(motor, 180)

# 3. 放在後端蓋外側（z = PIPE_LEN + CAP_T），軸對準棍心（x=0, y=0）
#    翻轉後軸尖在 z = -116，平移讓軸尖在 z = PIPE_LEN + CAP_T + 5
motor = translate(motor,
                  dx=0,
                  dy=0,
                  dz=PIPE_LEN + CAP_T + MOTOR_L/2 + SHAFT_L + 5)
all_tris += motor

# 第二顆馬達：前端蓋外側，軸朝 +Z（朝向管內）
motor2 = translate(motor_raw, dx=0, dy=0, dz=-MOTOR_L/2)
# 不需翻轉，軸本來就朝 +Z
motor2 = translate(motor2,
                   dx=0,
                   dy=0,
                   dz=-(CAP_T + MOTOR_L/2 + SHAFT_L + 5))
all_tris += motor2

# ── 聯軸器（Coupling）× 2 ─────────────────────────────
# 規格：OD=35mm，橋接馬達軸（14mm）與不鏽鋼棍（20mm）
# 後端：z 從馬達軸尖(462) 到棍端(485)
COUP_R = 17.5   # 聯軸器外半徑 mm

# 後端聯軸器
all_tris += make_cylinder(COUP_R,
                          PIPE_LEN + CAP_T + 5,   # z=470 馬達軸尖附近
                          PIPE_LEN + CAP_T + 20,  # z=485 棍端
                          segs=SEGS, cap_bottom=True, cap_top=True)

# 前端聯軸器
all_tris += make_cylinder(COUP_R,
                          -(CAP_T + 20),   # z=-35 棍端
                          -(CAP_T + 5),    # z=-20 馬達軸尖附近
                          segs=SEGS, cap_bottom=True, cap_top=True)

# 支架暫時移除

# ══════════════════════════════════════════════════════
# 第二條螺旋（對稱，Y 方向偏移 300mm）
# 完整複製：管體、螺旋、端蓋、中心軸、馬達、聯軸器、支架
# ══════════════════════════════════════════════════════
DY2 = 300.0   # 第二條螺旋的 Y 偏移量

def shift_y(tris, dy):
    return [([v0[0], v0[1]+dy, v0[2]],
              [v1[0], v1[1]+dy, v1[2]],
              [v2[0], v2[1]+dy, v2[2]]) for v0,v1,v2 in tris]

# 管體2
pipe2 = shift_y(make_pipe(), DY2)
all_tris += pipe2

# 螺旋2（反向：偏移 180° + Y 鏡像，用於對稱推進）
def make_spiral_blades_reverse():
    OUTER_R = 134.0; INNER_R = 84.0
    PITCH = 450.0/4; SEG_DEG = 36.0
    N_SEGS = 10; N_TURNS = 4; BLADE_T = 5.0
    zpd = PITCH/360.0
    tris = []
    for ang_offset in [0.0, 180.0]:
        for turn in range(N_TURNS):
            for seg in range(N_SEGS):
                gs = turn*N_SEGS+seg
                z0 = gs*SEG_DEG*zpd; z1 = (gs+1)*SEG_DEG*zpd
                # 反向：角度取負（鏡像旋轉方向）
                a0 = np.radians(-(gs*SEG_DEG + ang_offset))
                a1 = np.radians(-((gs+1)*SEG_DEG + ang_offset))
                ht = BLADE_T/2
                def pt(r,a,z,dz): return [r*np.cos(a), r*np.sin(a)+DY2, z+dz]
                ti0=pt(INNER_R,a0,z0,+ht); to0=pt(OUTER_R,a0,z0,+ht)
                ti1=pt(INNER_R,a1,z1,+ht); to1=pt(OUTER_R,a1,z1,+ht)
                bi0=pt(INNER_R,a0,z0,-ht); bo0=pt(OUTER_R,a0,z0,-ht)
                bi1=pt(INNER_R,a1,z1,-ht); bo1=pt(OUTER_R,a1,z1,-ht)
                tris += quad(ti0,to0,to1,ti1)
                tris += quad(bi1,bo1,bo0,bi0)
                tris += quad(to0,bo0,bo1,to1)
                tris += quad(ti1,bi1,bi0,ti0)
                tris += quad(ti0,to0,bo0,bi0)
                tris += quad(to1,ti1,bi1,bo1)
    return tris

all_tris += make_spiral_blades_reverse()

# 端蓋2（帶 O-ring 溝槽）+ 中心軸2
all_tris += shift_y(make_ring_oring(PIPE_R, ROD_R+2, -CAP_T, 0), DY2)
all_tris += shift_y(make_ring_oring(PIPE_R, ROD_R+2, PIPE_LEN, PIPE_LEN+CAP_T), DY2)
all_tris += shift_y(make_rod(), DY2)

# 馬達2（後端）
motor_r2 = translate(motor_raw, dx=0, dy=0, dz=-MOTOR_L/2)
motor_r2 = rotate_x(motor_r2, 180)
motor_r2 = translate(motor_r2, dx=0, dy=DY2,
                     dz=PIPE_LEN + CAP_T + MOTOR_L/2 + SHAFT_L + 5)
all_tris += motor_r2

# 馬達2（前端）
motor_f2 = translate(motor_raw, dx=0, dy=0, dz=-MOTOR_L/2)
motor_f2 = translate(motor_f2, dx=0, dy=DY2,
                     dz=-(CAP_T + MOTOR_L/2 + SHAFT_L + 5))
all_tris += motor_f2

# 聯軸器2
all_tris += shift_y(make_cylinder(COUP_R, PIPE_LEN+CAP_T+5, PIPE_LEN+CAP_T+20,
                                  segs=SEGS, cap_bottom=True, cap_top=True), DY2)
all_tris += shift_y(make_cylinder(COUP_R, -(CAP_T+20), -(CAP_T+5),
                                  segs=SEGS, cap_bottom=True, cap_top=True), DY2)

# ── 馬達背板（蓋在馬達機盒背面，連接兩顆馬達）─────────────
# 後端馬達背面 z = PIPE_LEN + CAP_T + MOTOR_L + SHAFT_L + 5
# = 450 + 15 + 156 + 38 + 5 = 664
# 前端馬達背面 z = -(CAP_T + MOTOR_L + SHAFT_L + 5)
# = -(15 + 156 + 38 + 5) = -214
BIND_T   = 8.0
MH       = MOTOR_W / 2   # = 43mm

REAR_BACK_Z  =  PIPE_LEN + CAP_T + MOTOR_L + SHAFT_L + 5   # 664
FRONT_BACK_Z = -(CAP_T + MOTOR_L + SHAFT_L + 5)             # -214

def make_motor_backplate(z0, z1, y_from, y_to):
    x0, x1 = -MH, MH
    verts = [
        [x0,y_from,z0],[x1,y_from,z0],[x1,y_to,z0],[x0,y_to,z0],
        [x0,y_from,z1],[x1,y_from,z1],[x1,y_to,z1],[x0,y_to,z1],
    ]
    faces = [(0,3,2,1),(4,5,6,7),(0,1,5,4),(2,3,7,6),(0,4,7,3),(1,2,6,5)]
    tris = []
    for f in faces:
        v = [verts[i] for i in f]
        tris += quad(v[0],v[1],v[2],v[3])
    return tris

# 後端背板（蓋住兩顆後端馬達背面）
all_tris += make_motor_backplate(REAR_BACK_Z, REAR_BACK_Z + BIND_T,
                                 -MH, DY2 + MH)
# 前端背板（蓋住兩顆前端馬達背面）
all_tris += make_motor_backplate(FRONT_BACK_Z - BIND_T, FRONT_BACK_Z,
                                 -MH, DY2 + MH)

# ── 平台板（蓋在兩管正上方）────────────────────────────────
# Y：從管1左緣到管2右緣（-84 ~ 384），加 20mm 邊距 → -104 ~ 404
# Z：管體範圍（0 ~ 450），加 10mm 邊距 → -10 ~ 460
# X：管頂 PIPE_R=84mm，板厚 12mm → 84 ~ 96
PLAT_Y0, PLAT_Y1 = -104.0, DY2 + PIPE_R + 20          # -104 ~ 404
PLAT_Z0, PLAT_Z1 = FRONT_BACK_Z, REAR_BACK_Z          # -214 ~ 664（與馬達底座齊）
PLAT_X0, PLAT_X1 =  PIPE_R + 60, PIPE_R + 72          #  144 ~ 156（再高）

def make_flat_plate(x0,x1,y0,y1,z0,z1):
    verts = [
        [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1],
    ]
    faces = [(0,3,2,1),(4,5,6,7),(0,1,5,4),(2,3,7,6),(0,4,7,3),(1,2,6,5)]
    tris = []
    for f in faces:
        v = [verts[i] for i in f]
        tris += quad(v[0],v[1],v[2],v[3])
    return tris

all_tris += make_flat_plate(PLAT_X0, PLAT_X1,
                             PLAT_Y0, PLAT_Y1,
                             PLAT_Z0, PLAT_Z1)
print(f"Platform: {PLAT_Y1-PLAT_Y0:.0f}mm wide x {PLAT_Z1-PLAT_Z0:.0f}mm long x {PLAT_X1-PLAT_X0:.0f}mm thick")

# ── 支撐架（馬達頂 → 平台底，螺旋管外側）──────────────────
# 垂直方柱：x 從馬達頂(43) 到平台底(PLAT_X0=144)
# Y 位置：兩側外緣（y=-43 左側，y=343 右側）
# Z 位置：前後馬達區段（螺旋管外，不進入 z=0~450 範圍）
SUPP_X0   = -MOTOR_W / 2       # -43mm（馬達底部，完整包覆馬達）
SUPP_X1   = PLAT_X0            # 144mm（平台底）
SUPP_W    = 20.0               # 方柱截面寬
SUPP_WALL = 3.0                # 壁厚（鋁擠料截面）

# 前馬達區段：z 從 FRONT_BACK_Z 到 -CAP_T（-214 ~ -15）
# 後馬達區段：z 從 PIPE_LEN+CAP_T 到 REAR_BACK_Z（465 ~ 664）
GAP = 25.0  # 與端蓋保持 25mm 間距，不碰螺旋
support_z_ranges = [
    (FRONT_BACK_Z,           -CAP_T - GAP),      # 前段（縮短，遠離螺旋）
    (PIPE_LEN + CAP_T + GAP,  REAR_BACK_Z),      # 後段（縮短，遠離螺旋）
]
# Y 位置：左外側（y=-43-SUPP_W ~ -43）和右外側（y=343 ~ 343+SUPP_W）
support_y_pairs = [
    (-MOTOR_W/2 - SUPP_W, -MOTOR_W/2),     # 左側
    (DY2 + MOTOR_W/2,      DY2 + MOTOR_W/2 + SUPP_W),  # 右側
]

PLATE_T = 8.0   # 板厚 mm

for z0_s, z1_s in support_z_ranges:
    # H 型：左腳 + 右腳 + 頂部橫梁

    # 左腳（外側垂直板）
    all_tris += make_flat_plate(SUPP_X0, SUPP_X1,
                                -MOTOR_W/2 - PLATE_T, -MOTOR_W/2,
                                z0_s, z1_s)
    # 右腳（外側垂直板）
    all_tris += make_flat_plate(SUPP_X0, SUPP_X1,
                                DY2 + MOTOR_W/2, DY2 + MOTOR_W/2 + PLATE_T,
                                z0_s, z1_s)
    # 頂部橫梁（連接左右腳，貼平台底）
    all_tris += make_flat_plate(SUPP_X1 - PLATE_T, SUPP_X1,
                                -MOTOR_W/2 - PLATE_T, DY2 + MOTOR_W/2 + PLATE_T,
                                z0_s, z1_s)

print(f"H-brackets: front+rear, x={SUPP_X0:.0f}~{SUPP_X1:.0f}mm")

# ── H 支架三角補強板（斜撐，X方向）──────────────────────────
# 在每個 H 支架近/遠端各加一片斜撐板（60mm 長，貼支架內側）
GUSSET_L = 60.0
GUSSET_T = 6.0
for z0_s, z1_s in support_z_ranges:
    for gy0, gy1 in [(-MOTOR_W/2 - PLATE_T - GUSSET_T, -MOTOR_W/2 - PLATE_T),
                     (DY2 + MOTOR_W/2 + PLATE_T,        DY2 + MOTOR_W/2 + PLATE_T + GUSSET_T)]:
        # 近端補強（貼近螺旋管端蓋側）
        all_tris += make_flat_plate(SUPP_X1 - GUSSET_T, SUPP_X1,
                                    gy0, gy1, z0_s, z0_s + GUSSET_L)
        # 遠端補強（貼近馬達背板側）
        all_tris += make_flat_plate(SUPP_X1 - GUSSET_T, SUPP_X1,
                                    gy0, gy1, z1_s - GUSSET_L, z1_s)
print(f"Gussets added: 4×2 = 8 reinforcement plates")

# ── 平台板 M4 螺栓定位凸塊（4×2 陣列）───────────────────────
# 凸塊：5×5×6mm 方形，平台頂面（X=PLAT_X1）向上凸出
BOLT_ZS = [PLAT_Z0 + 100, PLAT_Z0 + 220, PLAT_Z0 + 540, PLAT_Z0 + 660]
BOLT_YS = [PLAT_Y0 + 80,  PLAT_Y1 - 80]
for bz in BOLT_ZS:
    for by in BOLT_YS:
        all_tris += make_flat_plate(PLAT_X1, PLAT_X1 + 6,
                                    by - 2.5, by + 2.5,
                                    bz - 2.5, bz + 2.5)
print(f"Bolt bosses: 4×2 = 8 M4 positioning bosses on platform")

# 機械手臂暫時移除，沙灘測試後再重新安裝

out = os.path.join(OUT_DIR, "machine_v2_ASSEMBLY.stl")
write_stl(out, all_tris)
print(f"Done! triangles={len(all_tris)}")
print(f"Pipe: OD={PIPE_R*2:.0f}mm  L={PIPE_LEN:.0f}mm")
print(f"Rod:  D={ROD_R*2:.0f}mm  L={ROD_LEN:.0f}mm")
print(f"Saved: {os.path.abspath(out)}")
