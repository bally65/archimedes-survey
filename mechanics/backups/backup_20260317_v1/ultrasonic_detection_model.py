# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
"""
Archimedes-Survey v2 -- Ultrasonic Subsurface Burrow Detection Model

Mantis shrimp burrow specs:
  - Diameter: 15-30 mm
  - Depth: 10-60 cm (typically 20-40 cm)
  - Shape: Y-shaped or L-shaped, vertical entry

Detection approach:
  - Ground-coupled low-frequency ultrasound
  - Pulse-echo: transducer touches (or is pressed to) wet sand surface
  - Burrow air cavity = strong reflector (acoustic impedance mismatch)
  - Signal analysis: ToF (time of flight), amplitude drop pattern
"""
import numpy as np
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as sig

DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")

# ── Acoustic properties ────────────────────────────────────────────────────────
# Medium             Speed (m/s)   Impedance Z (Pa*s/m = Rayl)
# Dry sand           300-600       ~0.5e6
# Wet sand (50%sat)  1500-1700     ~3.0e6
# Water-sat sand     1700-1900     ~3.5e6
# Air                343           ~415
# Water              1480          ~1.48e6

MEDIA = {
    "wet_sand_30":  {"v": 1400, "Z": 2.5e6, "alpha_dB_m": 8},   # dB/m at 40kHz
    "wet_sand_50":  {"v": 1600, "Z": 3.0e6, "alpha_dB_m": 6},
    "wet_sand_70":  {"v": 1750, "Z": 3.3e6, "alpha_dB_m": 5},
    "air_cavity":   {"v": 343,  "Z": 415,   "alpha_dB_m": 0.1},
}

FREQUENCIES_kHz = [20, 40, 80, 100, 200]  # transducer frequencies to evaluate

# Burrow parameters
BURROW_DIAMETERS_MM = [15, 20, 25, 30]
BURROW_DEPTHS_CM    = [10, 15, 20, 30, 40, 50, 60]

print("=" * 70)
print("Ultrasonic Burrow Detection Model -- Archimedes-Survey v2")
print("=" * 70)

# ── 1. Reflection coefficient at sand/air interface ───────────────────────────
print("\n-- Acoustic Impedance & Reflection --")
Z_wet_sand = MEDIA["wet_sand_50"]["Z"]  # ~3.0e6 Rayl
Z_air      = MEDIA["air_cavity"]["Z"]  # 415 Rayl

# Reflection coefficient (normal incidence)
R = (Z_air - Z_wet_sand) / (Z_air + Z_wet_sand)
R_intensity = R**2  # power reflection
T_intensity = 1 - R_intensity

print(f"\nWet sand (50% moisture)  Z = {Z_wet_sand:.2e} Rayl")
print(f"Air cavity               Z = {Z_air:.0f} Rayl")
print(f"Reflection coefficient R = {R:.4f}")
print(f"Reflected intensity      = {R_intensity*100:.1f}%  (near-total reflection!)")
print(f"Transmitted intensity    = {T_intensity*100:.1f}%")
print(f"=> Air-filled burrow is nearly perfect reflector in wet sand")

# ── 2. Frequency vs Resolution & Penetration ──────────────────────────────────
print("\n-- Frequency Trade-off: Resolution vs Penetration Depth --")
print(f"\n{'Freq(kHz)':>10}  {'Lambda(mm)':>11}  {'Min Resol(mm)':>14}  "
      f"{'Pen. depth(cm)':>15}  {'Detects 20mm?':>14}  {'Detects 60cm?':>14}")
print("-" * 80)

freq_analysis = []
v_sand = MEDIA["wet_sand_50"]["v"]
alpha  = MEDIA["wet_sand_50"]["alpha_dB_m"]  # dB/m

for f_kHz in FREQUENCIES_kHz:
    f_Hz     = f_kHz * 1000
    lam_mm   = v_sand / f_Hz * 1000         # wavelength in mm
    res_mm   = lam_mm / 2                   # axial resolution ~ lambda/2
    # Penetration: -40 dB limit (round trip = 2x alpha)
    pen_m    = 40 / (2 * alpha)             # meters at this frequency
    # alpha scales with frequency^1.5 (empirical for sand)
    alpha_f  = alpha * (f_kHz / 40)**1.5
    pen_f_m  = 40 / (2 * alpha_f)
    pen_f_cm = pen_f_m * 100

    detect_20mm = "YES" if res_mm <= 20 else "NO"
    detect_60cm = "YES" if pen_f_cm >= 60 else "NO (%.0fcm)" % pen_f_cm

    freq_analysis.append({
        "f_kHz": f_kHz,
        "wavelength_mm": round(lam_mm, 1),
        "resolution_mm": round(res_mm, 1),
        "penetration_cm": round(pen_f_cm, 1),
    })
    print(f"  {f_kHz:>8}  {lam_mm:>10.1f}  {res_mm:>13.1f}  "
          f"{pen_f_cm:>14.1f}  {detect_20mm:>14}  {detect_60cm:>14}")

# ── 3. Signal-to-Noise model for each depth/diameter ─────────────────────────
print("\n-- SNR Model: Can we detect burrow at various depths? --")
print("(Using 40 kHz, wet sand 50% moisture, pulse-echo mode)")
print(f"\n{'Depth(cm)':>10}  {'Diam(mm)':>10}  {'2way loss(dB)':>14}  "
      f"{'Refl. gain(dB)':>15}  {'Net SNR(dB)':>12}  {'Detectable':>12}")
print("-" * 75)

f_design  = 40e3   # Hz
alpha_40  = 6.0    # dB/m at 40kHz in wet sand
R_dB      = 20 * np.log10(abs(R))  # reflection strength (dB) = ~-0.012 dB (near 0 = full reflect)
TX_power  = 0      # reference

snr_results = []
for depth_cm in BURROW_DEPTHS_CM:
    for diam_mm in BURROW_DIAMETERS_MM:
        depth_m   = depth_cm / 100
        diam_m    = diam_mm / 1000

        # Two-way propagation loss
        two_way_loss = 2 * alpha_40 * depth_m  # dB

        # Geometric spreading: point source -> 1/r^2 -> -20log(2r)
        spread_loss  = 20 * np.log10(2 * depth_m + 1e-9)  # dB

        # Reflection gain: near-total reflection, but target size < beam
        lam = v_sand / f_design  # 0.04 m
        beam_area   = (lam * depth_m)**2 * np.pi  # rough beam area at depth
        target_area = np.pi * (diam_m/2)**2
        size_factor  = min(1.0, target_area / beam_area)
        size_dB      = 20 * np.log10(size_factor + 1e-9)

        # Near-total reflection at air boundary (R~1)
        reflect_dB = -0.1  # ~1.0 reflection coefficient (0.9999)

        # Net SNR (assume noise floor at -60 dB)
        noise_floor = -60
        rx_level    = TX_power - two_way_loss - spread_loss + reflect_dB + size_dB
        snr_dB      = rx_level - noise_floor

        detectable  = "YES" if snr_dB > 15 else ("MARGINAL" if snr_dB > 5 else "NO")

        snr_results.append({
            "depth_cm": depth_cm, "diam_mm": diam_mm,
            "snr_dB": round(snr_dB, 1), "detectable": detectable,
        })

        if diam_mm == 20:  # print only one diameter for readability
            print(f"  {depth_cm:>8}  {diam_mm:>9}  {two_way_loss+spread_loss:>13.1f}  "
                  f"{reflect_dB+size_dB:>14.1f}  {snr_dB:>11.1f}  {detectable:>12}")

# ── 4. Sensor recommendation ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SENSOR RECOMMENDATION")
print("=" * 70)

print("""
Optimal frequency: 40 kHz
  - Wavelength 40mm in wet sand -> resolves 20mm burrows
  - Penetration depth ~50 cm (adequate for typical 20-40cm burrows)
  - Standard transducers available at low cost

Detection method: Ground-coupled pulse-echo
  - Press transducer against sand surface (or mount on thin water-filled membrane)
  - Wet sand provides excellent acoustic coupling (no air gap needed)
  - Scan while machine moves: 1 A-scan per 2-5 cm travel

Sensor options:
  Option A: Waterproof 40kHz piezo transducer
    - e.g., Murata MA40MF14-0B (waterproof, IP67, 40kHz)
    - Price: ~NT$200-400 each; need 3-5 in linear array for cross-track scan
    - Interface: simple pulse-echo with STM32 timer capture

  Option B: Commercial waterproof ultrasonic module
    - e.g., MaxSonar WR series (weatherproof, 42kHz)
    - Price: ~NT$600-1,200 each
    - Interface: TTL serial or analog

  Option C: Medical/industrial PVDF film array (advanced)
    - Flexible array, higher resolution, expensive (~NT$5,000+)
    - Overkill for this application

RECOMMENDED: Option A x 5 (linear array, 3cm spacing)
  - Total cost: ~NT$1,500-2,000
  - Coverage: 15 cm wide scan strip per pass
  - Burrow detection confidence: HIGH at 20-40cm depth

Integration:
  - Mount in sealed housing under machine platform, face pointing DOWN
  - Add thin water-filled rubber membrane for coupling to dry sand
  - STM32 generates 40kHz burst, measures return ToF
  - RPi processes A-scan array -> detects void signature (strong reflection at depth)
  - GPS-tag all detections -> output burrow location map
""")

# ── 5. Detection workflow ──────────────────────────────────────────────────────
print("=" * 70)
print("DETECTION PIPELINE (revised with ultrasonic)")
print("=" * 70)
print("""
Traditional (visual only):
  Camera -> burrow opening visible at surface -> image recognition

Proposed (dual-mode):
  +--------------------------+    +---------------------------------+
  | Visual Detection         |    | Ultrasonic Subsurface Detection |
  | Pi Camera (RGB)          |    | 5x 40kHz transducer array       |
  | YOLOv8-nano model        |    | STM32 pulse-echo                |
  | Detects: surface opening |    | Detects: void at 10-60cm depth  |
  | Best at: 10-40% moisture |    | Best at: 30-80% moisture        |
  +-----------+--------------+    +----------------+----------------+
              |                                    |
              +-----------> Fusion Engine <---------+
                            (RPi4 Python)
                                  |
                         GPS-tagged burrow map
                         Confidence score per detection
""")

# Save
out_json = os.path.join(DOCS, "ultrasonic_detection_model.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump({"frequency_analysis": freq_analysis, "snr_results": snr_results}, f, indent=2)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Ultrasonic Burrow Detection -- Archimedes-Survey v2 (40kHz, wet sand)",
             fontsize=12, fontweight='bold')

# Left: Freq vs wavelength / penetration
ax = axes[0]
freqs = [r["f_kHz"] for r in freq_analysis]
resols = [r["resolution_mm"] for r in freq_analysis]
pens   = [r["penetration_cm"] for r in freq_analysis]
ax2 = ax.twinx()
ax.plot(freqs, resols, 'o-r', linewidth=2, label='Resolution (mm)', markersize=7)
ax2.plot(freqs, pens, 's--b', linewidth=2, label='Penetration (cm)', markersize=7)
ax.axhline(20, color='red', linestyle=':', alpha=0.7, label='Burrow dia 20mm')
ax.axvline(40, color='green', linestyle='--', alpha=0.7, label='Optimal 40kHz')
ax.set_xlabel("Frequency (kHz)"); ax.set_ylabel("Resolution (mm)", color='red')
ax2.set_ylabel("Penetration Depth (cm)", color='blue')
ax.set_title("Frequency vs Resolution & Penetration", fontweight='bold')
ax.legend(loc='upper left', fontsize=7)
ax2.legend(loc='upper right', fontsize=7)
ax.grid(True, alpha=0.3)

# Middle: SNR heatmap (depth vs diameter, 40kHz)
ax = axes[1]
depths_unique = sorted(set(r["depth_cm"] for r in snr_results))
diams_unique  = sorted(set(r["diam_mm"] for r in snr_results))
snr_matrix    = np.zeros((len(depths_unique), len(diams_unique)))
for r in snr_results:
    i = depths_unique.index(r["depth_cm"])
    j = diams_unique.index(r["diam_mm"])
    snr_matrix[i, j] = r["snr_dB"]

im = ax.imshow(snr_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=40,
               origin='lower')
ax.set_xticks(range(len(diams_unique)))
ax.set_xticklabels([f"{d}mm" for d in diams_unique])
ax.set_yticks(range(len(depths_unique)))
ax.set_yticklabels([f"{d}cm" for d in depths_unique])
ax.set_xlabel("Burrow Diameter"); ax.set_ylabel("Burrow Depth")
ax.set_title("SNR Map (dB) at 40kHz\n(green=detectable, red=blind)", fontweight='bold')
plt.colorbar(im, ax=ax, label='SNR (dB)')
for i in range(len(depths_unique)):
    for j in range(len(diams_unique)):
        ax.text(j, i, f"{snr_matrix[i,j]:.0f}", ha='center', va='center', fontsize=8,
                color='black' if 10 < snr_matrix[i,j] < 30 else 'white')

# Right: Simulated A-scan (burrow at 25cm depth)
ax = axes[2]
# Simulate time-domain A-scan
t_us   = np.linspace(0, 600, 6000)  # microseconds
depth  = 0.25    # m
v_s    = 1600    # m/s
t_burrow = 2 * depth / v_s * 1e6  # us round trip
# Transmitted pulse
pulse = np.zeros_like(t_us)
t_tx  = 10  # pulse at t=10us
f_sim = 40e3
pulse_width = 5  # periods
t_pulse = np.arange(0, pulse_width / f_sim * 1e6, t_us[1]-t_us[0])
tx_pulse = np.sin(2*np.pi*f_sim * t_pulse * 1e-6) * np.exp(-((t_pulse - pulse_width/(2*f_sim)*1e6)**2) / (pulse_width/(f_sim)*1e6)**2)
idx_start = int(t_tx / (t_us[1]-t_us[0]))
pulse[idx_start:idx_start+len(tx_pulse)] += tx_pulse * 1.0

# Ground reflection (surface clutter) at ~2us
t_surf = 2.0
idx_s = int(t_surf / (t_us[1]-t_us[0]))
surf_echo = tx_pulse * 0.15
pulse[idx_s:idx_s+len(surf_echo)] += surf_echo

# Burrow reflection
idx_b = int(t_burrow / (t_us[1]-t_us[0]))
burrow_echo = tx_pulse * 0.85 * np.exp(-alpha_40 * depth * 2 / 20)
if idx_b + len(burrow_echo) < len(pulse):
    pulse[idx_b:idx_b+len(burrow_echo)] += burrow_echo

# Add noise
np.random.seed(42)
pulse += np.random.randn(len(pulse)) * 0.015

ax.plot(t_us, pulse, 'b-', linewidth=0.8, alpha=0.7)
ax.axvline(t_burrow, color='red', linestyle='--', linewidth=2, label=f'Burrow echo @{t_burrow:.0f}us ({depth*100:.0f}cm)')
ax.axvline(t_surf, color='orange', linestyle=':', linewidth=1.5, label='Surface clutter')
ax.set_xlabel("Time (microseconds)"); ax.set_ylabel("Amplitude")
ax.set_title("Simulated A-scan\n(burrow at 25cm, 40kHz, wet sand)", fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_xlim(0, 400)

plt.tight_layout()
out_png = os.path.join(DOCS, "ultrasonic_detection.png")
plt.savefig(out_png, dpi=150, bbox_inches='tight')
print(f"Saved chart: {out_png}")
print(f"Saved JSON:  {out_json}")
print("\nDone.")
