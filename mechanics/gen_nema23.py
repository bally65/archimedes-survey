"""
NEMA23 步進馬達佔位 STL 生成器
標準尺寸：57.15 x 57.15mm 方形，軸徑 8mm
PL57H76-D8 對應：機身長 76mm
"""
import numpy as np
import struct, os

OUT = os.path.join(os.path.dirname(__file__), "parts", "NEMA23_PL57H76.stl")

# 參數
W     = 57.15   # 方形邊長 mm
L     = 76.0    # 機身長 mm
SHAFT_D = 8.0   # 軸徑 mm
SHAFT_L = 21.0  # 軸伸出長 mm
SEGS  = 32      # 圓形細分

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

def quad(v0,v1,v2,v3):
    return [(v0,v1,v2),(v0,v2,v3)]

tris = []
h = W/2

# 機身六面（方形柱體，沿 Z 軸）
# 底面 z=0
tris += [([-h,-h,0],[h,-h,0],[h,h,0]),([-h,-h,0],[h,h,0],[-h,h,0])]
# 頂面 z=L
tris += [([h,-h,L],[-h,-h,L],[-h,h,L]),([h,-h,L],[-h,h,L],[h,h,L])]
# 四側面
for (x0,y0),(x1,y1) in [
    ((-h,-h),(h,-h)), ((h,-h),(h,h)),
    ((h,h),(-h,h)),  ((-h,h),(-h,-h))
]:
    tris += quad([x0,y0,0],[x1,y1,0],[x1,y1,L],[x0,y0,L])

# 前端面板（圓形孔省略，做實心面）
# 已包含在頂面

# 輸出軸（圓柱，從 z=L 延伸到 z=L+SHAFT_L）
r = SHAFT_D/2
for i in range(SEGS):
    a0 = 2*np.pi*i/SEGS
    a1 = 2*np.pi*(i+1)/SEGS
    z0, z1 = L, L+SHAFT_L
    p00 = [r*np.cos(a0), r*np.sin(a0), z0]
    p10 = [r*np.cos(a1), r*np.sin(a1), z0]
    p01 = [r*np.cos(a0), r*np.sin(a0), z1]
    p11 = [r*np.cos(a1), r*np.sin(a1), z1]
    tris += quad(p00, p10, p11, p01)
    # 軸端面
    tris.append(([0,0,z1], p11, p01))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
write_stl(OUT, tris)
print(f"Done: {os.path.abspath(OUT)}")
print(f"Body: {W}x{W}x{L}mm, Shaft: dia{SHAFT_D}mm x {SHAFT_L}mm")
