"""
3-DOF 機械手臂佔位 STL 生成器
  J1：底座旋轉（Z 軸）
  J2：肩關節（上下俯仰）
  J3：肘關節（上下俯仰）
  末端：探針

安裝位置：平台頂面中央
"""
import numpy as np
import struct, os, math

OUT = os.path.join(os.path.dirname(__file__), "parts", "arm_3dof.stl")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

SEGS = 32

# ── 手臂幾何參數 ──────────────────────────────────────────
BASE_R      = 40.0   # 底座半徑 mm
BASE_H      = 25.0   # 底座高度 mm
JOINT_R     = 14.0   # 關節球半徑 mm
UPPER_L     = 150.0  # 上臂長度 mm
UPPER_R     = 12.0   # 上臂管半徑 mm
FORE_L      = 120.0  # 前臂長度 mm
FORE_R      = 10.0   # 前臂管半徑 mm
PROBE_L     = 80.0   # 探針長度 mm
PROBE_R     = 4.0    # 探針半徑 mm

# ── 姿態角度（展示用）────────────────────────────────────
J1_DEG = 0.0    # 底座旋轉（水平）
J2_DEG = 45.0   # 肩關節（向上 45°）
J3_DEG = -60.0  # 肘關節（向下 60°，手臂微彎）

# ── STL 工具 ─────────────────────────────────────────────
tris = []

def write_stl(path, triangles):
    with open(path, "wb") as f:
        f.write(b"\x00" * 80)
        f.write(struct.pack("<I", len(triangles)))
        for v0,v1,v2 in triangles:
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

# ── 基礎幾何：圓柱、球 ────────────────────────────────────
def cylinder(r, h, origin, axis_vec, segs=SEGS):
    """圓柱：從 origin 沿 axis_vec 延伸 h"""
    ax = np.array(axis_vec, dtype=float)
    ax = ax / np.linalg.norm(ax)
    # 建立垂直於 ax 的兩個基向量
    if abs(ax[0]) < 0.9:
        perp = np.cross(ax, [1,0,0])
    else:
        perp = np.cross(ax, [0,1,0])
    perp = perp / np.linalg.norm(perp)
    perp2 = np.cross(ax, perp)

    o = np.array(origin)
    top = o + ax * h
    pts_bot = [o + r*(math.cos(2*math.pi*i/segs)*perp +
                      math.sin(2*math.pi*i/segs)*perp2) for i in range(segs)]
    pts_top = [top + r*(math.cos(2*math.pi*i/segs)*perp +
                        math.sin(2*math.pi*i/segs)*perp2) for i in range(segs)]
    t = []
    for i in range(segs):
        ni = (i+1)%segs
        t += quad(pts_bot[i].tolist(), pts_bot[ni].tolist(),
                  pts_top[ni].tolist(), pts_top[i].tolist())
        t.append((o.tolist(), pts_bot[ni].tolist(), pts_bot[i].tolist()))
        t.append((top.tolist(), pts_top[i].tolist(), pts_top[ni].tolist()))
    return t

def sphere(r, center, segs=SEGS):
    """球（近似）"""
    c = np.array(center)
    t = []
    rings = segs // 2
    for i in range(rings):
        phi0 = math.pi * i / rings - math.pi/2
        phi1 = math.pi * (i+1) / rings - math.pi/2
        for j in range(segs):
            th0 = 2*math.pi * j / segs
            th1 = 2*math.pi * (j+1) / segs
            def pt(ph,th):
                return (c + r*np.array([math.cos(ph)*math.cos(th),
                                        math.cos(ph)*math.sin(th),
                                        math.sin(ph)])).tolist()
            v00=pt(phi0,th0); v01=pt(phi0,th1)
            v10=pt(phi1,th0); v11=pt(phi1,th1)
            t += quad(v00,v01,v11,v10)
    return t

# ── 組裝手臂（正向運動學）────────────────────────────────
# 安裝基點：平台頂面中央（由外部設定，這裡輸出在原點，組裝時平移）
# 座標系：X=上，Y=左，Z=前

j1 = math.radians(J1_DEG)
j2 = math.radians(J2_DEG)
j3 = math.radians(J3_DEG)

# 底座（圓柱，沿 X 軸向上）
tris += cylinder(BASE_R, BASE_H, [0,0,0], [1,0,0])

# J1 關節球（底座頂端）
j1_pos = np.array([BASE_H, 0, 0])
tris += sphere(JOINT_R, j1_pos)

# 上臂方向（J1 水平旋轉 + J2 俯仰）
upper_dir = np.array([
    math.sin(j2),
    math.sin(j1) * math.cos(j2),
    math.cos(j1) * math.cos(j2)
])
tris += cylinder(UPPER_R, UPPER_L, j1_pos.tolist(), upper_dir.tolist())

# J2 關節球（上臂末端）
j2_pos = j1_pos + upper_dir * UPPER_L
tris += sphere(JOINT_R, j2_pos)

# 前臂方向（J2 + J3 疊加）
fore_angle = j2 + j3
fore_dir = np.array([
    math.sin(fore_angle),
    math.sin(j1) * math.cos(fore_angle),
    math.cos(j1) * math.cos(fore_angle)
])
tris += cylinder(FORE_R, FORE_L, j2_pos.tolist(), fore_dir.tolist())

# 探針（前臂末端）
probe_pos = j2_pos + fore_dir * FORE_L
tris += sphere(JOINT_R * 0.8, probe_pos)
tris += cylinder(PROBE_R, PROBE_L, probe_pos.tolist(), fore_dir.tolist())

write_stl(OUT, tris)
print(f"Done: {os.path.abspath(OUT)}")
print(f"Triangles: {len(tris)}")
print(f"J1={J1_DEG}°  J2={J2_DEG}°  J3={J3_DEG}°")
print(f"Upper arm: {UPPER_L}mm  Forearm: {FORE_L}mm  Probe: {PROBE_L}mm")
