"""
Archimedes Survey Robot — 完整螺旋組裝 STL 生成器
把雙螺旋 A+B 全部 5 圈合併成一個 STL，加上 PVC 管體預覽
"""

import numpy as np
import struct
import os

# ── 參數 ────────────────────────────────────────────────
OUTER_R  = 134.0   # 葉片尖端半徑 mm
INNER_R  = 84.0    # PVC 管外壁半徑 mm
PIPE_R   = 84.0    # PVC 管外徑半徑
PIPE_T   = 5.0     # PVC 管壁厚 mm（估計）
PITCH    = 100.0   # 螺距 mm（400mm ÷ 4圈）
SEG_DEG  = 36.0    # 每段角度
N_SEGS   = 10      # 每圈片數
N_TURNS  = 4       # 圈數
BLADE_T  = 5.0     # 葉片厚度 mm
SPIRALS  = {"A": 0.0, "B": 180.0}

PIPE_LEN = PITCH * N_TURNS  # 管長 = 螺距 × 圈數 = 2500mm
PIPE_SEGS = 64               # 管圓周細分

OUT_DIR = os.path.join(os.path.dirname(__file__), "stl")
os.makedirs(OUT_DIR, exist_ok=True)

# ── STL 工具 ─────────────────────────────────────────────
def write_stl_binary(filepath, triangles):
    with open(filepath, "wb") as f:
        f.write(b"\x00" * 80)
        f.write(struct.pack("<I", len(triangles)))
        for v0, v1, v2 in triangles:
            a = np.array(v1) - np.array(v0)
            b = np.array(v2) - np.array(v0)
            n = np.cross(a, b)
            norm = np.linalg.norm(n)
            n = n / norm if norm > 1e-10 else np.array([0, 0, 1])
            f.write(struct.pack("<fff", *n))
            f.write(struct.pack("<fff", *v0))
            f.write(struct.pack("<fff", *v1))
            f.write(struct.pack("<fff", *v2))
            f.write(struct.pack("<H", 0))

def quad_tris(v0, v1, v2, v3):
    return [(v0, v1, v2), (v0, v2, v3)]

# ── 葉片段生成 ────────────────────────────────────────────
def make_blade_segment(seg_idx, offset_deg=0.0):
    a0 = np.radians(seg_idx * SEG_DEG + offset_deg)
    a1 = np.radians((seg_idx + 1) * SEG_DEG + offset_deg)
    z_per_deg = PITCH / 360.0
    z0 = (seg_idx * SEG_DEG + offset_deg) * z_per_deg
    z1 = ((seg_idx + 1) * SEG_DEG + offset_deg) * z_per_deg
    ht = BLADE_T / 2.0

    def pt(r, a, z, dz): return [r*np.cos(a), r*np.sin(a), z+dz]

    ti0 = pt(INNER_R, a0, z0, +ht); to0 = pt(OUTER_R, a0, z0, +ht)
    ti1 = pt(INNER_R, a1, z1, +ht); to1 = pt(OUTER_R, a1, z1, +ht)
    bi0 = pt(INNER_R, a0, z0, -ht); bo0 = pt(OUTER_R, a0, z0, -ht)
    bi1 = pt(INNER_R, a1, z1, -ht); bo1 = pt(OUTER_R, a1, z1, -ht)

    tris = []
    tris += quad_tris(ti0, to0, to1, ti1)   # 上面
    tris += quad_tris(bi1, bo1, bo0, bi0)   # 下面
    tris += quad_tris(to0, bo0, bo1, to1)   # 外緣
    tris += quad_tris(ti1, bi1, bi0, ti0)   # 內緣
    tris += quad_tris(ti0, to0, bo0, bi0)   # 起端
    tris += quad_tris(to1, ti1, bi1, bo1)   # 末端
    return tris

# ── PVC 管體（圓柱） ──────────────────────────────────────
def make_pipe():
    tris = []
    r_out = PIPE_R
    r_in  = PIPE_R - PIPE_T
    segs  = PIPE_SEGS
    z0, z1 = 0.0, PIPE_LEN

    for i in range(segs):
        a0 = 2*np.pi * i / segs
        a1 = 2*np.pi * (i+1) / segs
        # 外壁
        tris += quad_tris(
            [r_out*np.cos(a0), r_out*np.sin(a0), z0],
            [r_out*np.cos(a1), r_out*np.sin(a1), z0],
            [r_out*np.cos(a1), r_out*np.sin(a1), z1],
            [r_out*np.cos(a0), r_out*np.sin(a0), z1],
        )
        # 內壁
        tris += quad_tris(
            [r_in*np.cos(a0), r_in*np.sin(a0), z1],
            [r_in*np.cos(a1), r_in*np.sin(a1), z1],
            [r_in*np.cos(a1), r_in*np.sin(a1), z0],
            [r_in*np.cos(a0), r_in*np.sin(a0), z0],
        )
        # 底環
        tris += quad_tris(
            [r_in*np.cos(a0),  r_in*np.sin(a0),  z0],
            [r_in*np.cos(a1),  r_in*np.sin(a1),  z0],
            [r_out*np.cos(a1), r_out*np.sin(a1), z0],
            [r_out*np.cos(a0), r_out*np.sin(a0), z0],
        )
        # 頂環
        tris += quad_tris(
            [r_out*np.cos(a0), r_out*np.sin(a0), z1],
            [r_out*np.cos(a1), r_out*np.sin(a1), z1],
            [r_in*np.cos(a1),  r_in*np.sin(a1),  z1],
            [r_in*np.cos(a0),  r_in*np.sin(a0),  z1],
        )
    return tris

# ── 主程式 ────────────────────────────────────────────────
def main():
    all_tris = []

    # PVC 管
    all_tris += make_pipe()

    # 雙螺旋 A + B，全 5 圈
    for name, offset in SPIRALS.items():
        for turn in range(N_TURNS):
            for seg in range(N_SEGS):
                global_seg = turn * N_SEGS + seg
                all_tris += make_blade_segment(global_seg, offset_deg=offset)

    out_path = os.path.join(OUT_DIR, "assembly_FULL.stl")
    write_stl_binary(out_path, all_tris)
    print(f"Done! triangles={len(all_tris)}")
    print(f"Saved: {os.path.abspath(out_path)}")
    print(f"Pipe:   OD={PIPE_R*2:.0f}mm  L={PIPE_LEN:.0f}mm")
    print(f"Spiral: ID={INNER_R*2:.0f}mm  OD={OUTER_R*2:.0f}mm  turns={N_TURNS}")

if __name__ == "__main__":
    main()
