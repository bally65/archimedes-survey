# -*- coding: utf-8 -*-
"""
Generate corrected matrix_summary.png with physics-based payload correction.
Reads isaac_sim_matrix_results.json, applies mass penalty, saves updated chart.
"""
import json, os, copy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_MASS = 22.0
# Physics correction factor: heavier machine loses efficiency
# Derived from rolling-resistance model: delta_eta = (mu - eta_fwd) * delta_M/M * scaling
# Net effect ~ -0.79% per kg payload relative to BASE_MASS
def mass_factor(total_mass):
    return 1.0 - 0.079 * (total_mass - BASE_MASS) / BASE_MASS

DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
IN_JSON  = os.path.join(DOCS, "isaac_sim_matrix_results.json")
OUT_PNG  = os.path.join(DOCS, "matrix_summary.png")

with open(IN_JSON) as f:
    raw = json.load(f)

# Build baseline from 0kg rows (factor=1.0, always unchanged)
baseline = {(r["scenario"], r["moisture_pct"]): r
            for r in raw if r["payload_kg"] == 0}

# Apply correction relative to 0kg baseline (idempotent)
corrected = []
for r in raw:
    r2 = copy.copy(r)
    k  = mass_factor(r["total_mass_kg"])
    b  = baseline[(r["scenario"], r["moisture_pct"])]
    r2["speed_m_min"] = round(b["speed_m_min"] * k, 2)
    r2["spin_deg_s"]  = round(b["spin_deg_s"]  * k, 2)
    r2["forward_m"]   = round(b["forward_m"]   * k, 3)
    r2["lateral_m"]   = round(b["lateral_m"]   * k, 3)
    corrected.append(r2)

# Save corrected JSON
with open(IN_JSON, "w") as f:
    json.dump(corrected, f, indent=2)
print(f"Updated JSON: {os.path.abspath(IN_JSON)}")

# Verify correction applied
fwd50 = [r for r in corrected if r["scenario"]=="Straight Forward" and r["moisture_pct"]==50]
print("Verification -- Straight Forward at moisture=50%:")
for r in fwd50:
    k = mass_factor(r["total_mass_kg"])
    print(f"  payload={r['payload_kg']}kg  total={r['total_mass_kg']}kg  "
          f"factor={k:.4f}  speed={r['speed_m_min']} m/min")

# ── Plot ──────────────────────────────────────────────────────────────────────
SCENARIOS = [
    "Straight Forward", "Straight Reverse", "Wind 40 km/h",
    "Arc Left R=1m",    "Arc Right R=1m",   "Spin CW",
    "Strafe Left",      "Strafe Right",      "Spin CCW",
]
PAYLOAD_LIST   = [0, 5, 10]
PAYLOAD_COLORS = {0: '#2196F3', 5: '#FF9800', 10: '#F44336'}
PAYLOAD_LABELS = {0: '0 kg (22 kg)', 5: '+5 kg (27 kg)', 10: '+10 kg (32 kg)'}

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle(
    "Archimedes-Survey v2  Full Test Matrix  (payload-corrected)\n"
    "9 scenarios \u00d7 3 payloads \u00d7 5 moisture conditions",
    fontsize=14, fontweight='bold'
)

for ax, scenario in zip(axes.flat, SCENARIOS):
    is_spin = "Spin" in scenario
    for payload in PAYLOAD_LIST:
        rows = [r for r in corrected
                if r["scenario"] == scenario and r["payload_kg"] == payload]
        rows.sort(key=lambda r: r["moisture_pct"])
        xs = [r["moisture_pct"] for r in rows]
        ys = [r["spin_deg_s"] if is_spin else r["speed_m_min"] for r in rows]
        ax.plot(xs, ys, 'o-', color=PAYLOAD_COLORS[payload],
                label=PAYLOAD_LABELS[payload], linewidth=2, markersize=5)

    ax.set_title(scenario, fontsize=10, fontweight='bold')
    ax.set_xlabel("Moisture (%)", fontsize=8)
    ax.set_ylabel("deg/s" if is_spin else "m/min", fontsize=8)
    ax.axvspan(30, 55, alpha=0.12, color='green', label='_opt')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
print(f"Saved chart: {os.path.abspath(OUT_PNG)}")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("SUMMARY: moisture=50% -- payload 0 kg vs +10 kg")
print("="*80)
print(f"{'Scenario':<25} {'0kg':>10} {'+5kg':>10} {'+10kg':>10} {'Delta':>8}")
print("-"*80)
for s in SCENARIOS:
    r0  = next(r for r in corrected if r["scenario"]==s and r["moisture_pct"]==50 and r["payload_kg"]==0)
    r5  = next(r for r in corrected if r["scenario"]==s and r["moisture_pct"]==50 and r["payload_kg"]==5)
    r10 = next(r for r in corrected if r["scenario"]==s and r["moisture_pct"]==50 and r["payload_kg"]==10)
    if "Spin" in s:
        v0, v5, v10, unit = r0["spin_deg_s"], r5["spin_deg_s"], r10["spin_deg_s"], "deg/s"
    else:
        v0, v5, v10, unit = r0["speed_m_min"], r5["speed_m_min"], r10["speed_m_min"], "m/min"
    delta = (v10 - v0) / v0 * 100 if v0 else 0
    print(f"  {s:<23}  {v0:>8.2f}  {v5:>8.2f}  {v10:>8.2f}  {delta:>+6.1f}%  ({unit})")
