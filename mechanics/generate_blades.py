"""
Archimedes Survey Robot — 螺旋葉片 STL 生成器
參數：
  - 6 吋 PVC 管，外徑 168mm
  - 螺旋外徑 140mm，內徑（貼管）84mm
  - 每段 36°，10 片一圈
  - 葉片寬 50mm（沿軸方向），厚 5mm
  - 螺距 500mm（一圈走 500mm）
  - 雙螺旋 A/B（B 偏移 180°）
"""

import numpy as np
import os
import struct

# ── 參數 ────────────────────────────────────────────────
OUTER_R   = 134.0   # 葉片尖端半徑 mm（84 + 50 = 134mm，總直徑 268mm）
INNER_R   = 84.0    # PVC 管外壁半徑 mm（168mm 直徑）
PITCH     = 100.0   # 螺距 mm（400mm ÷ 4圈 = 100mm/圈）
SEG_DEG   = 36.0    # 每段角度
BLADE_T   = 5.0     # 葉片厚度 mm
N_SEGS    = 10      # 每圈片數
N_TURNS   = 4       # 圈數（機身 400mm）
SPIRALS   = {"A": 0, "B": 180}  # 雙螺旋，B 偏移 180°

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "mechanics", "stl")
os.makedirs(OUT_DIR, exist_ok=True)

# ── STL 工具函式 ─────────────────────────────────────────
def write_stl_binary(filepath, triangles):
    """triangles: list of (v0, v1, v2)，每頂點為 [x,y,z]"""
    with open(filepath, "wb") as f:
        f.write(b"\x00" * 80)           # header
        f.write(struct.pack("<I", len(triangles)))
        for v0, v1, v2 in triangles:
            # 計算法向量
            a = np.array(v1) - np.array(v0)
            b = np.array(v2) - np.array(v0)
            n = np.cross(a, b)
            norm = np.linalg.norm(n)
            n = n / norm if norm > 1e-10 else n
            f.write(struct.pack("<fff", *n))
            f.write(struct.pack("<fff", *v0))
            f.write(struct.pack("<fff", *v1))
            f.write(struct.pack("<fff", *v2))
            f.write(struct.pack("<H", 0))  # attribute

def quad_to_tris(v0, v1, v2, v3):
    """四邊形拆成兩個三角形"""
    return [(v0, v1, v2), (v0, v2, v3)]

# ── 單片葉片生成 ──────────────────────────────────────────
def make_blade_segment(seg_idx, offset_deg=0.0):
    """
    生成第 seg_idx 片葉片（0-based）的三角網格。
    螺旋沿 Z 軸，葉片從內徑延伸到外徑。
    """
    seg_deg   = SEG_DEG
    a0_deg    = seg_idx * seg_deg + offset_deg
    a1_deg    = a0_deg + seg_deg
    a0        = np.radians(a0_deg)
    a1        = np.radians(a1_deg)

    # 螺旋高度：每圈 PITCH，每度 PITCH/360
    z_per_deg = PITCH / 360.0
    z0        = a0_deg * z_per_deg
    z1        = a1_deg * z_per_deg
    half_t    = BLADE_T / 2.0
    # 外緣楔形倒角：外尖端縮薄為 2mm（每側 1.5mm），泥沙切入阻力更小
    half_t_out = max(0.5, half_t - 1.5)

    tris = []

    def pt(r, ang, z_base, dz):
        return [r * np.cos(ang), r * np.sin(ang), z_base + dz]

    # 內緣（根部）保持全厚 half_t，外緣（尖端）縮薄 half_t_out
    top_i0 = pt(INNER_R, a0, z0, +half_t)
    top_i1 = pt(INNER_R, a1, z1, +half_t)
    bot_i0 = pt(INNER_R, a0, z0, -half_t)
    bot_i1 = pt(INNER_R, a1, z1, -half_t)

    top_o0 = pt(OUTER_R, a0, z0, +half_t_out)
    top_o1 = pt(OUTER_R, a1, z1, +half_t_out)
    bot_o0 = pt(OUTER_R, a0, z0, -half_t_out)
    bot_o1 = pt(OUTER_R, a1, z1, -half_t_out)

    # 上表面（法向朝上）— 梯形面（內厚外薄）
    tris += quad_to_tris(top_i0, top_o0, top_o1, top_i1)
    # 下表面（法向朝下）
    tris += quad_to_tris(bot_i1, bot_o1, bot_o0, bot_i0)
    # 外緣側面（已楔形化，較窄）
    tris += quad_to_tris(top_o0, bot_o0, bot_o1, top_o1)
    # 內緣側面（全厚）
    tris += quad_to_tris(top_i1, bot_i1, bot_i0, top_i0)
    # 起始端面（梯形，連接內外不同厚度）
    tris += quad_to_tris(top_i0, top_o0, bot_o0, bot_i0)
    # 結束端面
    tris += quad_to_tris(top_o1, top_i1, bot_i1, bot_o1)

    return tris

# ── 完整螺旋預覽（一圈） ──────────────────────────────────
def make_full_preview(offset_deg=0.0):
    tris = []
    for i in range(N_SEGS):
        tris += make_blade_segment(i, offset_deg)
    return tris

# ── 主程式 ───────────────────────────────────────────────
def main():
    count = 0
    for spiral_name, offset in SPIRALS.items():
        for turn in range(N_TURNS):
            for seg in range(N_SEGS):
                global_seg = turn * N_SEGS + seg
                tris = make_blade_segment(global_seg, offset_deg=offset)
                fname = f"blade_{spiral_name}_seg{seg+1:02d}_turn{turn+1}_of_{N_SEGS}.stl"
                path  = os.path.join(OUT_DIR, fname)
                write_stl_binary(path, tris)
                count += 1

    # 完整預覽（第一圈 A 螺旋）
    preview_tris = make_full_preview(offset_deg=0)
    write_stl_binary(os.path.join(OUT_DIR, "blade_FULL_PREVIEW_1turn.stl"), preview_tris)
    count += 1

    print(f"Done! Generated {count} STL files")
    print(f"Location: {os.path.abspath(OUT_DIR)}")
    print(f"  blade_A_seg*: Spiral A, {N_TURNS * N_SEGS} pieces")
    print(f"  blade_B_seg*: Spiral B (offset 180 deg), {N_TURNS * N_SEGS} pieces")
    print(f"  blade_FULL_PREVIEW_1turn.stl: full 1-turn preview")

if __name__ == "__main__":
    main()
