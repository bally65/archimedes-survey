"""
馬達支架 + 鋁擠料長方體平台 STL 生成器

馬達支架：
  - NEMA34 安裝板（100×100mm，厚 8mm）
  - 中心孔 40mm（軸/聯軸器穿過）
  - 固定在端蓋外側

鋁擠料平台（4040 規格）：
  - 40×40mm 方管截面
  - 長方體框架：寬 250mm × 高 200mm × 長 700mm
  - 跨越整台機器（管體+馬達）
"""
import numpy as np, struct, os, math

OUT_DIR = os.path.join(os.path.dirname(__file__), "parts")
os.makedirs(OUT_DIR, exist_ok=True)

SEGS = 32

# ── STL 工具 ─────────────────────────────────────────────
def write_stl(path, tris):
    with open(path, "wb") as f:
        f.write(b"\x00" * 80)
        f.write(struct.pack("<I", len(tris)))
        for v0,v1,v2 in tris:
            a = np.array(v1)-np.array(v0)
            b = np.array(v2)-np.array(v0)
            n = np.cross(a,b)
            nm = np.linalg.norm(n)
            n = n/nm if nm>1e-10 else np.array([0,0,1])
            f.write(struct.pack("<fff",*n))
            f.write(struct.pack("<fff",*v0))
            f.write(struct.pack("<fff",*v1))
            f.write(struct.pack("<fff",*v2))
            f.write(struct.pack("<H",0))

def quad(v0,v1,v2,v3): return [(v0,v1,v2),(v0,v2,v3)]

# ── 方形管（鋁擠料截面）沿 Z 軸延伸 ──────────────────────
def make_square_tube(w, z0, z1, wall=3.0, cx=0, cy=0):
    """w=外寬，wall=壁厚，沿 Z 軸，中心 (cx,cy)"""
    ho = w/2
    hi = ho - wall
    tris = []
    corners_o = [(-ho,-ho),(ho,-ho),(ho,ho),(-ho,ho)]
    corners_i = [(-hi,-hi),(hi,-hi),(hi,hi),(-hi,hi)]

    def pt(x,y,z): return [x+cx, y+cy, z]

    for i in range(4):
        x0o,y0o = corners_o[i]
        x1o,y1o = corners_o[(i+1)%4]
        x0i,y0i = corners_i[i]
        x1i,y1i = corners_i[(i+1)%4]
        # 外壁
        tris += quad(pt(x0o,y0o,z0),pt(x1o,y1o,z0),
                     pt(x1o,y1o,z1),pt(x0o,y0o,z1))
        # 內壁（反向）
        tris += quad(pt(x0i,y0i,z1),pt(x1i,y1i,z1),
                     pt(x1i,y1i,z0),pt(x0i,y0i,z0))
        # 底端環
        tris += quad(pt(x0i,y0i,z0),pt(x1i,y1i,z0),
                     pt(x1o,y1o,z0),pt(x0o,y0o,z0))
        # 頂端環
        tris += quad(pt(x0o,y0o,z1),pt(x1o,y1o,z1),
                     pt(x1i,y1i,z1),pt(x0i,y0i,z1))
    return tris

# ── 沿 X 軸延伸的方形管（橫向橫梁） ──────────────────────
def make_square_tube_x(w, x0, x1, wall=3.0, cy=0, cz=0):
    """沿 X 軸方向延伸"""
    ho = w/2
    hi = ho - wall
    tris = []
    corners_o = [(-ho,-ho),(ho,-ho),(ho,ho),(-ho,ho)]
    corners_i = [(-hi,-hi),(hi,-hi),(hi,hi),(-hi,hi)]
    def pt(y,z,x): return [x, y+cy, z+cz]
    for i in range(4):
        y0o,z0o = corners_o[i]
        y1o,z1o = corners_o[(i+1)%4]
        y0i,z0i = corners_i[i]
        y1i,z1i = corners_i[(i+1)%4]
        tris += quad(pt(y0o,z0o,x0),pt(y1o,z1o,x0),
                     pt(y1o,z1o,x1),pt(y0o,z0o,x1))
        tris += quad(pt(y0i,z0i,x1),pt(y1i,z1i,x1),
                     pt(y1i,z1i,x0),pt(y0i,z0i,x0))
        tris += quad(pt(y0i,z0i,x0),pt(y1i,z1i,x0),
                     pt(y1o,z1o,x0),pt(y0o,z0o,x0))
        tris += quad(pt(y0o,z0o,x1),pt(y1o,z1o,x1),
                     pt(y1i,z1i,x1),pt(y0i,z0i,x1))
    return tris

# ═══════════════════════════════════════════════════════
# 1. 馬達支架板（前後各一，相同設計）
# ═══════════════════════════════════════════════════════
BRACKET_SIZE = 190.0   # 方形板邊長 mm（大於管 OD 168mm）
BRACKET_T    = 8.0     # 板厚 mm
SHAFT_HOLE_R = 20.0    # 中心孔半徑 mm（聯軸器穿過）
PIPE_R       = 84.0    # 用於定位

def make_motor_bracket(z_face, facing=+1):
    """
    z_face: 安裝面 Z 位置
    facing: +1 朝 +Z（後端），-1 朝 -Z（前端）
    """
    tris = []
    h  = BRACKET_SIZE/2
    z0 = z_face
    z1 = z_face + facing * BRACKET_T

    # 板的四個外角
    corners = [(-h,-h),(h,-h),(h,h),(-h,h)]

    # 正面（含中心孔：用多邊形環近似）
    def ring_face(z, normal_out):
        pts_out = [(-h,-h),(h,-h),(h,h),(-h,h)]  # 外框
        pts_in  = [(SHAFT_HOLE_R*np.cos(2*np.pi*i/SEGS),
                    SHAFT_HOLE_R*np.sin(2*np.pi*i/SEGS))
                   for i in range(SEGS)]           # 圓孔
        face_tris = []
        # 外框到孔之間的環狀面（近似：三角扇）
        for i in range(SEGS):
            p0 = list(pts_in[i]) + [z]
            p1 = list(pts_in[(i+1)%SEGS]) + [z]
            # 找最近外角
            cx = (pts_in[i][0]+pts_in[(i+1)%SEGS][0])/2
            cy = (pts_in[i][1]+pts_in[(i+1)%SEGS][1])/2
            # 四個外角三角形
            angle = math.atan2(cy,cx)
            idx = int((angle+math.pi)/(2*math.pi/4)) % 4
            pc = list(corners[idx]) + [z]
            if normal_out > 0:
                face_tris.append((pc, p0, p1))
            else:
                face_tris.append((pc, p1, p0))
        return face_tris

    tris += ring_face(z0, -facing)
    tris += ring_face(z1, +facing)

    # 外框四側面
    for i in range(4):
        x0c,y0c = corners[i]
        x1c,y1c = corners[(i+1)%4]
        tris += quad([x0c,y0c,z0],[x1c,y1c,z0],[x1c,y1c,z1],[x0c,y0c,z1])

    # 中心孔側壁
    for i in range(SEGS):
        a0 = 2*np.pi*i/SEGS
        a1 = 2*np.pi*(i+1)/SEGS
        p00=[SHAFT_HOLE_R*np.cos(a0),SHAFT_HOLE_R*np.sin(a0),z0]
        p10=[SHAFT_HOLE_R*np.cos(a1),SHAFT_HOLE_R*np.sin(a1),z0]
        p01=[SHAFT_HOLE_R*np.cos(a0),SHAFT_HOLE_R*np.sin(a0),z1]
        p11=[SHAFT_HOLE_R*np.cos(a1),SHAFT_HOLE_R*np.sin(a1),z1]
        tris += quad(p00,p01,p11,p10)

    return tris

# ═══════════════════════════════════════════════════════
# 2. 鋁擠料長方體框架平台
# ═══════════════════════════════════════════════════════
# 框架尺寸
PLAT_W  = 260.0   # 寬（Y 方向，跨越管體）
PLAT_H  = 220.0   # 高（X 方向，管中心到平台頂）
PLAT_L  = 720.0   # 長（Z 方向，含兩端馬達）
PLAT_Z0 = -230.0  # 框架 Z 起點（從前馬達外緣算）
PROF    = 40.0    # 4040 鋁擠料截面 40×40mm

# 縱向四根長梁（沿 Z 軸）
# 截面位置（四角）
long_beams = [
    ( PLAT_H,       -PLAT_W/2),   # 右上
    ( PLAT_H,        PLAT_W/2),   # 左上
    ( PLAT_H-PROF,  -PLAT_W/2),   # 右下（底層）
    ( PLAT_H-PROF,   PLAT_W/2),   # 左下
]

plat_tris = []
for cx, cy in long_beams:
    plat_tris += make_square_tube(PROF, PLAT_Z0, PLAT_Z0+PLAT_L,
                                  wall=3.0, cx=cx, cy=cy)

# 橫向橫梁（沿 X 方向連接左右，前中後各兩根）
cross_z_positions = [PLAT_Z0+20, PLAT_Z0+PLAT_L/2, PLAT_Z0+PLAT_L-20]
for cz in cross_z_positions:
    # 上橫梁
    plat_tris += make_square_tube_x(PROF,
                                    -PLAT_W/2, PLAT_W/2,
                                    wall=3.0,
                                    cy=0,
                                    cz=PLAT_H + PROF/2)
    # 下橫梁
    plat_tris += make_square_tube_x(PROF,
                                    -PLAT_W/2, PLAT_W/2,
                                    wall=3.0,
                                    cy=0,
                                    cz=PLAT_H - PROF/2)

# ── 輸出 ─────────────────────────────────────────────────
# 前後支架板
bracket_rear  = make_motor_bracket(z_face=465, facing=+1)   # 後端蓋外
bracket_front = make_motor_bracket(z_face=-15, facing=-1)   # 前端蓋外

write_stl(os.path.join(OUT_DIR, "motor_bracket_rear.stl"),  bracket_rear)
write_stl(os.path.join(OUT_DIR, "motor_bracket_front.stl"), bracket_front)
write_stl(os.path.join(OUT_DIR, "platform_frame.stl"),      plat_tris)

print("Done!")
print(f"  motor_bracket_rear/front.stl  : {BRACKET_SIZE}x{BRACKET_SIZE}mm, hole={SHAFT_HOLE_R*2:.0f}mm dia")
print(f"  platform_frame.stl            : {PLAT_W}x{PLAT_H}x{PLAT_L}mm, 4040 profile")
