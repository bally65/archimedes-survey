"""
gen_screw_printable.py
======================
阿基米德螺旋船 — 3D 列印版螺旋棍生成器

設計原則：
  - 每段 90°弧，4段一圈，共 4 圈 = 16 段/管（兩管共 32 段）
  - 材質：PETG 或 ASA（耐 UV、耐濕）
  - 固定方式：海洋級環氧樹脂（Marine Epoxy）黏合在 PVC 管外壁
  - 螺旋棒（blade）截面：矩形翼片，radial 厚 8mm，軸向寬 15mm
  - 葉片外徑：BLADE_R=134mm，PVC 管外徑：PIPE_R=84mm

座標系（與主組裝一致）：
  X：垂直（正=上），Y：側向，Z：縱向（螺旋前進方向）

輸出：
  stl/screw_seg_A_{n:02d}.stl   管 A 各段（0~15）
  stl/screw_seg_B_{n:02d}.stl   管 B 各段（0~15）
  stl/screw_assembly_A.stl      管 A 完整螺旋
  stl/screw_assembly_B.stl      管 B 完整螺旋
  stl/screw_FULL.stl            兩管合體（參考用）
"""

import numpy as np
import struct
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "stl")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 螺旋參數 ──────────────────────────────────────────────
PIPE_R    = 84.0     # PVC 管外半徑 mm
BLADE_R   = 134.0    # 葉片外徑 mm
BLADE_T   = 8.0      # 葉片徑向厚度（mm，列印壁厚）
BLADE_W   = 15.0     # 葉片軸向寬（沿 Z mm，每段基準寬）
PITCH     = 112.5    # 螺距（mm/轉）：450mm / 4 turns
N_TURNS   = 4        # 圈數
N_SEG     = 4        # 每圈段數（每段 90°）
N_TOTAL   = N_TURNS * N_SEG   # 總段數 = 16

PIPE_LEN  = 450.0    # 管長 mm
DY2       = 300.0    # 管 B 的 Y 中心偏移

# 管 A/B 中心
PIPE_A_CENTER = np.array([0.0, 0.0, 0.0])
PIPE_B_CENTER = np.array([0.0, DY2, 0.0])

# 列印設定建議
LAYER_H  = 0.2    # 層高 mm
NOZZLE   = 0.4    # 噴嘴 mm
INFILL   = 40     # 填充率 %
WALLS    = 4      # 壁厚層數
SUPPORT  = False  # 無需支撐（90°弧分段設計）


# ══════════════════════════════════════════════════════════
# STL 工具
# ══════════════════════════════════════════════════════════

def write_stl(path, tris):
    with open(path, "wb") as f:
        f.write(b"\x00" * 80)
        f.write(struct.pack("<I", len(tris)))
        for v0, v1, v2 in tris:
            a = np.asarray(v1, float) - np.asarray(v0, float)
            b = np.asarray(v2, float) - np.asarray(v0, float)
            n = np.cross(a, b)
            nm = np.linalg.norm(n)
            n = n / nm if nm > 1e-10 else np.array([0., 0., 1.])
            f.write(struct.pack("<fff", *n))
            f.write(struct.pack("<fff", *np.asarray(v0, float)))
            f.write(struct.pack("<fff", *np.asarray(v1, float)))
            f.write(struct.pack("<fff", *np.asarray(v2, float)))
            f.write(struct.pack("<H", 0))
    print(f"  OK {os.path.basename(path)}  ({len(tris)} tris)")


def quad(v0, v1, v2, v3):
    return [(v0, v1, v2), (v0, v2, v3)]


# ══════════════════════════════════════════════════════════
# 螺旋葉片段生成
# ══════════════════════════════════════════════════════════

def make_blade_segment(seg_idx, pipe_center, handed='R', n_arc=20):
    """
    生成一段 90° 螺旋葉片（實心掃掠體）

    seg_idx   : 0..N_TOTAL-1（0 = 最前段）
    pipe_center: [cx, cy, cz_offset] — 管中心
    handed    : 'R' 右旋，'L' 左旋（管 B 反旋，互相推水）
    n_arc     : 弧向分割數（影響圓滑度）

    返回三角形列表
    """
    sign = 1.0 if handed == 'R' else -1.0

    theta_start = sign * seg_idx * (np.pi / 2)          # 起始角（rad）
    theta_end   = sign * (seg_idx + 1) * (np.pi / 2)    # 終止角

    z_start = seg_idx       * PITCH / N_SEG + pipe_center[2]
    z_end   = (seg_idx + 1) * PITCH / N_SEG + pipe_center[2]

    cx, cy = pipe_center[0], pipe_center[1]

    # 內徑（貼 PVC 外壁）、外徑（葉片頂）
    r_inner = PIPE_R
    r_outer = BLADE_R

    # 葉片軸向半寬（讓相鄰段稍微重疊）
    half_w = BLADE_W / 2.0

    # 產生掃掠截面頂點（前後兩個面）
    # 每個截面：沿弧向在 n_arc+1 個角度取 4 個角點
    # 截面 shape：內底、外底、外頂、內頂（軸向 ±half_w）

    def profile_ring(theta, z_center):
        """給定旋轉角和 Z 中心，返回截面 4 個角點"""
        c, s = np.cos(theta), np.sin(theta)
        # 內壁點（在 XY 平面，Z = z_center ± half_w）
        ri_bot = np.array([cx + r_inner * c, cy + r_inner * s, z_center - half_w])
        ri_top = np.array([cx + r_inner * c, cy + r_inner * s, z_center + half_w])
        ro_bot = np.array([cx + r_outer * c, cy + r_outer * s, z_center - half_w])
        ro_top = np.array([cx + r_outer * c, cy + r_outer * s, z_center + half_w])
        return ri_bot, ro_bot, ro_top, ri_top

    tris = []
    thetas = np.linspace(theta_start, theta_end, n_arc + 1)
    zcs    = np.linspace(z_start, z_end, n_arc + 1)

    # 掃掠側面（主葉片面）
    for i in range(n_arc):
        t0, t1 = thetas[i], thetas[i + 1]
        z0, z1 = zcs[i], zcs[i + 1]

        ri0, ro0, ro0t, ri0t = profile_ring(t0, z0)
        ri1, ro1, ro1t, ri1t = profile_ring(t1, z1)

        # 外弧面（葉片頂）
        tris += quad(ro0, ro1, ro1t, ro0t)
        # 內弧面（貼管面，內法線）
        tris += quad(ri0t, ri1t, ri1, ri0)
        # 前側面（葉片前端 z-half_w）
        tris += quad(ri0, ro0, ro1, ri1)
        # 後側面（葉片後端 z+half_w）
        tris += quad(ri1t, ro1t, ro0t, ri0t)

    # 起始端帽（封閉段端面）
    t0 = thetas[0]; z0 = zcs[0]
    ri0, ro0, ro0t, ri0t = profile_ring(t0, z0)
    tris += quad(ri0, ri0t, ro0t, ro0)

    # 終止端帽
    t1 = thetas[-1]; z1 = zcs[-1]
    ri1, ro1, ro1t, ri1t = profile_ring(t1, z1)
    tris += quad(ri1, ro1, ro1t, ri1t)

    return tris


def make_collar(pipe_center, n_arc=32):
    """
    內圓環固定環（3mm 厚），方便海洋環氧定位
    套在 PVC 管外壁，每段葉片起點各一個
    """
    cx, cy, cz = pipe_center
    r_inner = PIPE_R
    r_outer = PIPE_R + 5.0   # 3mm 膠合肉厚 + 2mm 嵌位
    collar_h = 10.0

    thetas = np.linspace(0, 2 * np.pi, n_arc + 1)
    tris = []
    for i in range(n_arc):
        t0, t1 = thetas[i], thetas[i + 1]
        ci0 = [cx + r_inner * np.cos(t0), cy + r_inner * np.sin(t0), cz]
        ci1 = [cx + r_inner * np.cos(t1), cy + r_inner * np.sin(t1), cz]
        co0 = [cx + r_outer * np.cos(t0), cy + r_outer * np.sin(t0), cz]
        co1 = [cx + r_outer * np.cos(t1), cy + r_outer * np.sin(t1), cz]
        ci0h = [ci0[0], ci0[1], cz + collar_h]
        ci1h = [ci1[0], ci1[1], cz + collar_h]
        co0h = [co0[0], co0[1], cz + collar_h]
        co1h = [co1[0], co1[1], cz + collar_h]

        # 外側面
        tris += quad(co0, co1, co1h, co0h)
        # 內側面（反向）
        tris += quad(co0h, ci1h, ci1, ci0)
        tris += quad(ci0, co0h, co0, ci0)  # 連接
        # 底面
        tris += quad(ci0, ci1, co1, co0)
        # 頂面
        tris += quad(ci0h, co0h, co1h, ci1h)
    return tris


# ══════════════════════════════════════════════════════════
# 主函數
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 56)
    print("螺旋棍 3D 列印 STL 生成器")
    print(f"管外徑 {PIPE_R*2:.0f}mm  葉片外徑 {BLADE_R*2:.0f}mm  螺距 {PITCH:.1f}mm")
    print(f"每管 {N_TOTAL} 段（90°/段）× 2 管 = {N_TOTAL*2} 段總計")
    print("=" * 56)

    all_tris = []

    for pipe_label, pipe_center, handed in [
        ("A", PIPE_A_CENTER, "R"),
        ("B", PIPE_B_CENTER, "L"),
    ]:
        pipe_tris = []
        for seg in range(N_TOTAL):
            tris = make_blade_segment(seg, pipe_center, handed=handed)
            fname = f"screw_seg_{pipe_label}_{seg:02d}.stl"
            write_stl(os.path.join(OUT_DIR, fname), tris)
            pipe_tris += tris

        write_stl(os.path.join(OUT_DIR, f"screw_assembly_{pipe_label}.stl"), pipe_tris)
        all_tris += pipe_tris

    write_stl(os.path.join(OUT_DIR, "screw_FULL.stl"), all_tris)

    # 列印設定摘要
    seg_arc   = 90                     # 度
    seg_z     = PITCH / N_SEG         # mm
    blade_vol = (BLADE_R**2 - PIPE_R**2) * np.pi * BLADE_W * N_TOTAL  # rough
    print()
    print("── 列印參數建議 ─────────────────────────────────────────")
    print(f"  材質        : PETG（首選）或 ASA（戶外抗 UV）")
    print(f"  層高        : {LAYER_H}mm")
    print(f"  噴嘴        : {NOZZLE}mm")
    print(f"  壁厚        : {WALLS} 層（{WALLS*NOZZLE}mm）")
    print(f"  填充率      : {INFILL}%（六角形）")
    print(f"  支撐        : 不需要（每段獨立 ≤90°）")
    print(f"  每段尺寸    : 弧長 ≈ {int(BLADE_R*np.pi/2)}mm, Z={seg_z:.1f}mm")
    print()
    print("── 固定方式 ─────────────────────────────────────────────")
    print("  1. 砂紙打磨 PVC 管外壁 → 去油脂（丙酮）")
    print("  2. 刷塗海洋環氧樹脂（Marine Epoxy，24h 硬化）")
    print("  3. 套上螺旋段，對準螺距刻度線後夾緊")
    print("  4. 玻纖布（50g/m²）加強接縫處纏繞")
    print()
    print("── 材料估算（單管）──────────────────────────────────────")
    rough_g = 0.70 * INFILL / 100 * N_TOTAL * 25  # 粗估，每段約25g@40%
    print(f"  PETG 耗材   : 約 {rough_g:.0f}g / 管（兩管共 {rough_g*2:.0f}g）")
    print(f"  列印時間    : 約 {N_TOTAL * 1.5:.0f}h / 管（單台 0.4mm 噴嘴）")
    print("=" * 56)
    print(f"輸出目錄：{OUT_DIR}")


if __name__ == "__main__":
    main()
