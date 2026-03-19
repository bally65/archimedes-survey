"""
gen_aluminum_frame.py
=====================
阿基米德螺旋探勘船 — 鋁擠型骨架 STL 生成器

骨架規格（3030 T-slot 鋁擠型，30×30mm）：
  整機外框：長 878mm × 寬 508mm × 高 320mm（地面到平台頂）
  雙螺旋管中心距：300mm
  平台：508mm × 878mm，高 156mm（管中心以上）

座標系（與 assemble_machine_v2.py 一致）：
  X：垂直（正=上，管中心 X=0）
  Y：側向（管1 Y=0，管2 Y=300）
  Z：縱向（前後，Z=-214 前端，Z=664 後端）

輸出：
  stl/frame_bottom_rails.stl      底層縱梁 × 2
  stl/frame_bottom_cross.stl      底層橫梁 × 3
  stl/frame_vertical_posts.stl    四角立柱 × 4
  stl/frame_motor_end.stl         馬達端橫梁 × 2 + 立柱 × 4
  stl/frame_platform_rails.stl    平台縱梁 × 2
  stl/frame_platform_cross.stl    平台橫梁 × 4
  stl/frame_ASSEMBLY.stl          完整組裝
"""

import numpy as np
import struct
import os
import math

OUT_DIR = os.path.join(os.path.dirname(__file__), "stl")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 機器人關鍵尺寸（與主組裝檔一致）────────────────────────
PIPE_R   = 84.0    # PVC 管外徑半徑 mm
PIPE_LEN = 450.0   # 管長 mm
CAP_T    = 15.0    # 端蓋厚 mm
MOTOR_L  = 156.0   # NEMA23 長 mm
SHAFT_L  = 38.0    # 聯軸器+軸段 mm
DY2      = 300.0   # 第二管 Y 中心距 mm
BLADE_R  = 134.0   # 葉片外徑半徑 mm

# 整機 Z 範圍
Z_FRONT = -(CAP_T + MOTOR_L + SHAFT_L + 5)    # -214mm
Z_REAR  =  PIPE_LEN + CAP_T + MOTOR_L + SHAFT_L + 5  # +664mm

# 整機 Y 範圍（平台邊緣）
Y_LEFT  = -104.0
Y_RIGHT =  404.0

# ── 高度計算 ─────────────────────────────────────────────
BLADE_CLEARANCE = 30.0
GROUND_X   = -(BLADE_R + BLADE_CLEARANCE)  # = -164mm（地面）
PLAT_X_BOT =  PIPE_R + 60.0               # = 144mm（平台底）
PLAT_X_TOP =  PIPE_R + 72.0               # = 156mm（平台頂）
MOTOR_X    =  0.0                          # 管中心高（馬達轉軸高）

# ── 型材規格 ─────────────────────────────────────────────
S30 = 30.0  # 3030 型材邊長
S20 = 20.0  # 2020 型材邊長


# ═══════════════════════════════════════════════════════════
# STL 工具
# ═══════════════════════════════════════════════════════════

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
    print(f"  OK {os.path.basename(path)}  ({len(tris)} tris)")


def quad(v0, v1, v2, v3):
    return [(v0, v1, v2), (v0, v2, v3)]


def make_box(x0, y0, z0, x1, y1, z1):
    """實心矩形盒（6面 × 2三角形 = 12 tris）"""
    return (
        quad([x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0]) +  # -Z face
        quad([x0,y1,z1],[x1,y1,z1],[x1,y0,z1],[x0,y0,z1]) +  # +Z face
        quad([x0,y0,z1],[x1,y0,z1],[x1,y0,z0],[x0,y0,z0]) +  # -Y face
        quad([x0,y1,z0],[x1,y1,z0],[x1,y1,z1],[x0,y1,z1]) +  # +Y face
        quad([x0,y0,z0],[x0,y1,z0],[x0,y1,z1],[x0,y0,z1]) +  # -X face
        quad([x1,y0,z1],[x1,y1,z1],[x1,y1,z0],[x1,y0,z0])    # +X face
    )


def beam_z(xc, yc, z0, z1, s=S30):
    """Z 方向（縱梁），中心在 (xc, yc)"""
    h = s / 2
    return make_box(xc-h, yc-h, z0, xc+h, yc+h, z1)


def beam_y(xc, y0, y1, zc, s=S30):
    """Y 方向（橫梁），中心在 (xc, zc)"""
    h = s / 2
    return make_box(xc-h, y0, zc-h, xc+h, y1, zc+h)


def beam_x(x0, x1, yc, zc, s=S30):
    """X 方向（立柱），中心在 (yc, zc)"""
    h = s / 2
    return make_box(x0, yc-h, zc-h, x1, yc+h, zc+h)


def flat_plate(x_val, y0, y1, z0, z1, thick=3.0):
    """鋁板（馬達固定板，厚 thick mm，垂直 X 方向）"""
    return make_box(x_val - thick/2, y0, z0, x_val + thick/2, y1, z1)


# ═══════════════════════════════════════════════════════════
# 骨架各部件
# ═══════════════════════════════════════════════════════════

def make_bottom_rails():
    """底層縱梁 × 2（沿 Z，位於底層 X 高度）"""
    xc = GROUND_X + S30 / 2        # 梁中心 X = -164 + 15 = -149mm
    y_L = Y_LEFT  + S30 / 2        # 左梁中心 Y = -89mm
    y_R = Y_RIGHT - S30 / 2        # 右梁中心 Y = 389mm
    tris = []
    tris += beam_z(xc, y_L, Z_FRONT, Z_REAR)
    tris += beam_z(xc, y_R, Z_FRONT, Z_REAR)
    return tris


def make_bottom_cross():
    """底層橫梁 × 3（沿 Y，前/中/後）"""
    xc = GROUND_X + S30 / 2
    z_positions = [Z_FRONT + 15, 0.0, Z_REAR - 15]   # 前/中/後
    tris = []
    for zc in z_positions:
        tris += beam_y(xc, Y_LEFT, Y_RIGHT, zc)
    return tris


def make_vertical_posts():
    """四角立柱 × 4（沿 X，底→平台）"""
    x0 = GROUND_X + S30             # 底梁頂面
    x1 = PLAT_X_TOP                 # 平台頂面
    y_L = Y_LEFT  + S30 / 2
    y_R = Y_RIGHT - S30 / 2
    z_positions = [Z_FRONT + 15, Z_REAR - 15]   # 前/後
    tris = []
    for zc in z_positions:
        for yc in [y_L, y_R]:
            tris += beam_x(x0, x1, yc, zc)
    return tris


def make_motor_end_frames():
    """
    馬達端結構（前後各一）：
    - 管中心高橫梁（連接兩管馬達固定位）
    - 管中心高立柱（從底到管中心）
    - NEMA23 固定板（3mm 鋁板，86×86mm 位置）
    """
    tris = []
    x_bottom = GROUND_X + S30       # 底梁頂
    x_motor  = MOTOR_X              # 管中心（馬達轉軸高）

    for zc in [Z_FRONT + 15, Z_REAR - 15]:
        # 橫梁：從左管→右管，在管中心高
        tris += beam_y(x_motor, Y_LEFT, Y_RIGHT, zc)
        # 立柱：底→管中心（支撐橫梁）
        y_L = Y_LEFT  + S30 / 2
        y_R = Y_RIGHT - S30 / 2
        for yc in [y_L, y_R]:
            tris += beam_x(x_bottom, x_motor, yc, zc)

        # NEMA23 固定板（薄板代替，位於管1/管2外側）
        # 管1 (Y=0) 馬達在 Y 負方向外側
        tris += flat_plate(x_motor,
                           y0=-80, y1=80,    # ±80mm（NEMA23 86mm）
                           z0=zc-43, z1=zc+43, thick=3.0)

    return tris


def make_platform_rails():
    """平台縱梁 × 2（沿 Z，在平台頂高）"""
    xc = PLAT_X_TOP - S30 / 2      # = 141mm
    y_L = Y_LEFT  + S30 / 2
    y_R = Y_RIGHT - S30 / 2
    tris = []
    tris += beam_z(xc, y_L, Z_FRONT, Z_REAR)
    tris += beam_z(xc, y_R, Z_FRONT, Z_REAR)
    return tris


def make_platform_cross():
    """平台橫梁 × 4（沿 Y，均勻分布）"""
    xc = PLAT_X_TOP - S30 / 2
    total_z = Z_REAR - Z_FRONT
    z_positions = [Z_FRONT + total_z * i / 3 for i in range(4)]
    tris = []
    for zc in z_positions:
        tris += beam_y(xc, Y_LEFT, Y_RIGHT, zc)
    return tris


def make_mid_supports():
    """中間補強支撐（管正上方，2 個）"""
    tris = []
    x0 = GROUND_X + S30
    x1 = PLAT_X_BOT
    for yc in [S30 / 2, DY2 - S30 / 2]:   # 管1/管2 上方
        for zc in [0.0, PIPE_LEN]:          # 管前/後
            tris += beam_x(x0, x1, yc, zc, s=S20)
    return tris


# ═══════════════════════════════════════════════════════════
# 主函數
# ═══════════════════════════════════════════════════════════

PARTS = {
    "frame_bottom_rails":   make_bottom_rails,
    "frame_bottom_cross":   make_bottom_cross,
    "frame_vertical_posts": make_vertical_posts,
    "frame_motor_end":      make_motor_end_frames,
    "frame_platform_rails": make_platform_rails,
    "frame_platform_cross": make_platform_cross,
    "frame_mid_supports":   make_mid_supports,
}


def main():
    print("=" * 56)
    print("阿基米德骨架 STL 生成器")
    print(f"整機：{int(Z_REAR-Z_FRONT)}L × {int(Y_RIGHT-Y_LEFT)}W × "
          f"{int(PLAT_X_TOP-GROUND_X)}H mm")
    print(f"地面 X={GROUND_X:.0f}mm  管中心 X={MOTOR_X:.0f}mm  "
          f"平台 X={PLAT_X_TOP:.0f}mm")
    print("=" * 56)

    all_tris = []
    for name, func in PARTS.items():
        tris = func()
        write_stl(os.path.join(OUT_DIR, f"{name}.stl"), tris)
        all_tris += tris

    write_stl(os.path.join(OUT_DIR, "frame_ASSEMBLY.stl"), all_tris)

    print()
    print("── BOM 摘要 ────────────────────────────────────────────")
    bom = [
        ("A1", "底層縱梁 3030",    2, 878),
        ("A2", "底層橫梁 3030",    3, 508),
        ("A3", "四角立柱 3030",    4, 320),
        ("A4", "馬達端橫梁 3030",  2, 508),
        ("A5", "馬達端立柱 3030",  4, 164),
        ("A6", "平台縱梁 3030",    2, 878),
        ("A7", "平台橫梁 3030",    4, 508),
        ("B1", "管上支撐 2020",    4, 225),
    ]
    total_3030 = sum(qty * L for _, name, qty, L in bom if "3030" in name)
    total_2020 = sum(qty * L for _, name, qty, L in bom if "2020" in name)
    for code, name, qty, L in bom:
        print(f"  {code}  {name:20s}  ×{qty}  {L}mm  ={qty*L}mm")
    print(f"\n  3030 合計：{total_3030}mm ≈ {total_3030/1000:.1f}m")
    print(f"  2020 合計：{total_2020}mm ≈ {total_2020/1000:.1f}m")
    print(f"\n  建議採購：3030 × 4m × 3根（共 12m）")
    print(f"            2020 × 3m × 1根（共  3m）")
    print("=" * 56)
    print(f"輸出目錄：{OUT_DIR}")


if __name__ == "__main__":
    main()
