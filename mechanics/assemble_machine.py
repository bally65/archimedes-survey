"""
完整整機組裝 STL
  - PVC 管體（168mm OD，400mm 長）
  - 雙螺旋葉片 A+B（各 4 圈，螺距 100mm）
  - NEMA23 馬達 × 2（後端兩側，左右各一）
  - 6205 培林 × 4（前後各 2 個，支撐螺旋軸）
"""
import numpy as np
import struct, os, math

OUT_DIR   = os.path.join(os.path.dirname(__file__), "stl")
PARTS_DIR = os.path.join(os.path.dirname(__file__), "parts")
os.makedirs(OUT_DIR, exist_ok=True)

PIPE_LEN = 400.0   # 機身長 mm
PIPE_R   = 84.0    # PVC 管外壁半徑 mm
MOTOR_W  = 57.15   # NEMA23 方形邊長 mm
MOTOR_L  = 76.0    # NEMA23 機身長 mm
MOTOR_OFFSET = PIPE_R + 20  # 馬達距管中心距離（側邊）

# ── STL 讀取 ─────────────────────────────────────────────
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

# ── STL 寫入 ─────────────────────────────────────────────
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

# ── 幾何變換 ─────────────────────────────────────────────
def translate(tris, dx=0, dy=0, dz=0):
    return [([v[0]+dx, v[1]+dy, v[2]+dz] for v in tri) for tri in
            [((v0, v1, v2)) for v0, v1, v2 in tris]]

def translate(tris, dx=0, dy=0, dz=0):
    return [([v0[0]+dx, v0[1]+dy, v0[2]+dz],
              [v1[0]+dx, v1[1]+dy, v1[2]+dz],
              [v2[0]+dx, v2[1]+dy, v2[2]+dz]) for v0, v1, v2 in tris]

def rotate_y(tris, deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    def ry(v): return [c*v[0]+s*v[2], v[1], -s*v[0]+c*v[2]]
    return [(ry(v0), ry(v1), ry(v2)) for v0, v1, v2 in tris]

def rotate_z(tris, deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    def rz(v): return [c*v[0]-s*v[1], s*v[0]+c*v[1], v[2]]
    return [(rz(v0), rz(v1), rz(v2)) for v0, v1, v2 in tris]

# ── 主程式 ────────────────────────────────────────────────
all_tris = []

# 1. 螺旋 + 管體
spiral_tris = read_stl(os.path.join(OUT_DIR, "assembly_FULL.stl"))
all_tris += spiral_tris

# 2. 培林：讓 Z 軸成旋轉軸（原始 X 軸是寬度，旋轉 90° 校正）
bearing_raw = read_stl(os.path.join(PARTS_DIR, "6205 ball bearing.stl"))
bearing = rotate_y(bearing_raw, 90)   # 旋轉讓寬度朝 Z

# 前端培林 × 2（z = -20，左右各一）
for dy in [-PIPE_R - 15, PIPE_R + 15]:
    all_tris += translate(bearing, dy=dy, dz=-20)

# 後端培林 × 2（z = PIPE_LEN + 20）
for dy in [-PIPE_R - 15, PIPE_R + 15]:
    all_tris += translate(bearing, dy=dy, dz=PIPE_LEN + 20)

# 3. NEMA23 馬達：後端兩側，軸朝向管體（沿 Y 軸方向）
#    原始馬達軸朝 +Z，旋轉 90° 讓軸朝 Y 方向
motor_raw = read_stl(os.path.join(PARTS_DIR, "NEMA23_PL57H76.stl"))
motor_rotated = rotate_y(motor_raw, 90)   # 軸朝 +X 方向

# 左側馬達（-Y 側，軸朝 +Y）
motor_L = rotate_z(motor_rotated, 90)
motor_L = translate(motor_L,
                    dx=-MOTOR_W/2,
                    dy=-(MOTOR_OFFSET + MOTOR_L),
                    dz=PIPE_LEN - MOTOR_W/2)
all_tris += motor_L

# 右側馬達（+Y 側，軸朝 -Y）
motor_R = rotate_z(motor_rotated, -90)
motor_R = translate(motor_R,
                    dx=-MOTOR_W/2,
                    dy=MOTOR_OFFSET,
                    dz=PIPE_LEN - MOTOR_W/2)
all_tris += motor_R

out_path = os.path.join(OUT_DIR, "machine_FULL_ASSEMBLY.stl")
write_stl(out_path, all_tris)
print(f"Done! triangles={len(all_tris)}")
print(f"Pipe: OD={PIPE_R*2:.0f}mm  L={PIPE_LEN:.0f}mm")
print(f"Motors: rear sides, offset={MOTOR_OFFSET:.0f}mm from center")
print(f"Saved: {os.path.abspath(out_path)}")
