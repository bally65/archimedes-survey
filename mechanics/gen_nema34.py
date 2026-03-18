"""
NEMA34 步進馬達佔位 STL（86BHH156）
規格：86x86mm 框架，機身長 156mm，軸徑 14mm
"""
import numpy as np, struct, os, math

OUT = os.path.join(os.path.dirname(__file__), "parts", "NEMA34_86BHH156.stl")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

W      = 86.0   # 方形邊長 mm
L      = 156.0  # 機身長 mm
SHAFT_D = 14.0  # 軸徑 mm
SHAFT_L = 38.0  # 軸伸出長 mm
SEGS   = 48

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

tris = []
h = W/2

# 機身（方形柱，Z 軸方向）
tris += [  # 底面
    ([-h,-h,0],[h,-h,0],[h,h,0]),([-h,-h,0],[h,h,0],[-h,h,0])]
tris += [  # 頂面
    ([h,-h,L],[-h,-h,L],[-h,h,L]),([h,-h,L],[-h,h,L],[h,h,L])]
for (x0,y0),(x1,y1) in [((-h,-h),(h,-h)),((h,-h),(h,h)),
                          ((h,h),(-h,h)),((-h,h),(-h,-h))]:
    tris += quad([x0,y0,0],[x1,y1,0],[x1,y1,L],[x0,y0,L])

# 前端圓形法蘭（直徑 100mm）
FR = 50.0
for i in range(SEGS):
    a0=2*np.pi*i/SEGS; a1=2*np.pi*(i+1)/SEGS
    tris.append(([0,0,L],[FR*np.cos(a0),FR*np.sin(a0),L],[FR*np.cos(a1),FR*np.sin(a1),L]))
    tris += quad([FR*np.cos(a0),FR*np.sin(a0),L],[FR*np.cos(a1),FR*np.sin(a1),L],
                 [FR*np.cos(a1),FR*np.sin(a1),L+8],[FR*np.cos(a0),FR*np.sin(a0),L+8])

# 輸出軸
r = SHAFT_D/2
for i in range(SEGS):
    a0=2*np.pi*i/SEGS; a1=2*np.pi*(i+1)/SEGS
    z0,z1 = L+8, L+8+SHAFT_L
    tris += quad([r*np.cos(a0),r*np.sin(a0),z0],[r*np.cos(a1),r*np.sin(a1),z0],
                 [r*np.cos(a1),r*np.sin(a1),z1],[r*np.cos(a0),r*np.sin(a0),z1])
    tris.append(([0,0,z1],[r*np.cos(a1),r*np.sin(a1),z1],[r*np.cos(a0),r*np.sin(a0),z1]))

write_stl(OUT, tris)
print(f"Done: {os.path.abspath(OUT)}")
print(f"Frame: {W}x{W}mm  Body: {L}mm  Shaft: {SHAFT_D}mm dia x {SHAFT_L}mm")
