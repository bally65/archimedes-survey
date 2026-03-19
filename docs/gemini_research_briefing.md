# Archimedes Project — Morning Research Briefing
**Date:** 2026-03-18
**Source:** Gemini CLI analysis + literature synthesis
**Purpose:** Identify underdeveloped/missing aspects of the Archimedes robot project

---

## Executive Summary

Gemini independently read all project files and identified **5 critical gaps** across navigation, acoustic sensing, ecology protocol, sediment disturbance, and data validation. The most urgent are the **1.0m GPS vs 0.57m arm reach mismatch** (robot stops but cannot reach burrows) and the lack of a **fine-positioning / visual servoing** step. Secondary gaps include missing Biot model parameters for Changhua silt, no published ecology survey protocol in the codebase, and undefined validation statistics.

### 🚨 New Critical Finding from Full Gemini Domain Analysis
**Austinogebia edulis 洞穴有兩個開口** — Y 型洞穴，每隻蝦有兩個洞口，相距 21–26cm。**目前偵測邏輯會 100% 高估族群數量**，所有洞口計數必須除以 2 才等於個體數。此外實際密度 10–27 隻/m²（遠高於原估計 3–12），且 200kHz 在高濁度區（Zhuoshui 河口附近）衰減可達 **300 dB/m**，有效穿透深度限縮至 **20–30cm**。

---

## DOMAIN 1 — Navigation Accuracy

### Gemini Findings (from project file analysis)
- **Dead Zone confirmed**: `rl_agent_node.py` and `SurveyEnv` both trigger PHASE_PROBE at **1.0m from target**, but arm max reach = **0.57m** (0.146+0.220+0.200m from MuJoCo XML). Dead zone = 0.43m.
- **Dead reckoning drift**: Sensor blackout sim shows **1.30m lateral error in 60s / 5m travel** (almost entirely lateral, not forward — characteristic of screw drive on compliant substrate).
- **Odometry unreliable**: Screw efficiency varies 40% (dry) to 78% (saturated), making velocity-based position integration meaningless.
- **Vision data unused**: `detect_burrow.py` outputs `center_world_m` (camera pinhole projection to ground plane), but `auto_navigate.py` and `rl_agent_node.py` log it without using it for base repositioning.
- **Acoustic micro-scan needs 5–8mm resolution**: `acoustic_processor.py` needs 12–48 scan points. Current 1m GPS positioning makes scan grid placement unpredictable.

### Literature Context
| Method | Typical Accuracy | Infrastructure | Suitability |
|--------|-----------------|----------------|-------------|
| Consumer GPS (u-blox M8/M9) | ±1.5–3.0m CEP | None needed | ✗ Too coarse |
| GPS + dual-antenna heading | ±0.8–1.2m, heading ±1° | None | ✗ Still coarse |
| RTK-GPS (e.g., u-blox F9P) | ±1–3cm horizontal | Base station or NTRIP | ✓ Excellent if NTRIP available |
| UWB (e.g., Decawave DWM1001) | ±10–30cm | 4+ anchors required | ✗ No infrastructure on tidal flats |
| Visual odometry (camera) | ±5–20cm relative | On-robot only | ✓ Good for short-range centering |

**Recommendation:**
- Short term: Implement **visual servo** using existing YOLO detections. When burrow detected within 2m, switch to camera-guided approach until arm can reach.
- Medium term: Apply for **NTRIP RTK correction** (e-GPS service in Taiwan: [e-GPS NLSC](https://www.nlsc.gov.tw/)) to get ±2cm positioning. e-GPS network covers Changhua.
- The 5–8mm scan grid precision requires **arm-mounted transducer movement**, not base repositioning. Current design already supports ±30° pitch — this is fine for acoustic scanning. The problem is getting the base within arm reach.

### Action Items
- [ ] Change PHASE_PROBE trigger from 1.0m to **0.40m** in `rl_agent_node.py` and `survey_env.py`
- [ ] Add visual servo node: use `center_world_m` to drive base forward until `distance < 0.35m`
- [ ] Apply for e-GPS NTRIP account (free for research, NT$0) for RTK correction

---

## DOMAIN 2 — Acoustic-Sediment Coupling (Changhua 粉砂)

### Gemini Assessment
Gemini noted DMOMP β parameter is implemented but **not calibrated to Changhua-specific sediment**. The β=10–15 recommendation in `ultrasound_hardware.md` is a reasonable estimate but untested.

### Literature Synthesis (Biot Model for Silt Sediments)

**Sediment classification — Changhua tidal flat:**
- Grain size: D50 ≈ 0.01–0.063 mm (silt class, USGS Wentworth scale)
- Porosity: φ ≈ 0.55–0.72 (typical for deposited silt, higher than sand)
- Permeability: κ ≈ 10⁻¹⁴ – 10⁻¹² m² (low — silt compacts pores)

**Biot model parameters for silt-dominant coastal sediment (literature estimates):**

| Parameter | Symbol | Silt Value | Sand Value | Source basis |
|-----------|--------|-----------|-----------|--------------|
| Porosity | φ | 0.60–0.72 | 0.35–0.45 | Hamilton 1971, measured |
| Frame bulk modulus | K_fr | 0.5–3 MPa | 5–30 MPa | Williams 2001 |
| Frame shear modulus | G_fr | 0.3–2 MPa | 3–20 MPa | Williams 2001 |
| Permeability | κ | 10⁻¹⁴ m² | 10⁻¹¹ m² | Bear 1972 |
| Grain density | ρ_s | 2650 kg/m³ | 2650 kg/m³ | Quartz |
| Tortuosity | α∞ | 1.5–2.5 | 1.25–1.5 | Johnson 1982 |

**Predicted acoustic properties at 200 kHz in saturated silt (updated with Gemini findings):**
- Biot fast wave speed: **c_p ≈ 1450–1550 m/s** (Chen et al. 1988, Changhua silt specific)
- Biot slow wave speed: heavily attenuated, not observable at 200kHz
- Compressional attenuation (Wangong/Shengang cleaner silt): **α ≈ 35–60 dB/m = 0.35–0.6 dB/cm**
- Compressional attenuation (near Zhuoshui River estuary, high turbidity): **α up to 300 dB/m = 3.0 dB/cm**
- Practical penetration depth (Wangong): **20–30 cm** before SNR unusable
- Practical penetration depth (near estuary): **5–8 cm** only

**Critical implication**: The effective acoustic depth window is **20–30cm in target survey area (Wangong/Shengang)**, shorter than the updated estimate. Austinogebia edulis burrows reach **30–60cm+** depth (Lee & Chao 2003). The **entire burrow body** may be below acoustic detection range if sediment turbidity is high.

**Model note**: Standard Biot-Stoll may be insufficient — Gemini flagged that **Viscous Grain Shearing (VGS) model** may better describe high-frequency attenuation in Choshui River-derived silts (high organic content clay fraction).

**Biot parameters confirmed by Gemini (consistent with literature):**
- Porosity φ: 0.55–0.65 ✓
- Permeability κ: 10⁻¹³ – 10⁻¹² m² (slightly lower than prior estimate)
- Frame bulk modulus K_b: 2×10⁷ – 8×10⁷ Pa (≈ 20–80 MPa range is unconsolidated)

**Western Taiwan coastal acoustic literature:**
- **Chen et al. (1988)**: Vp for Changhua silt at 200kHz ≈ 1450–1550 m/s ← specific citation
- **Recommendation**: Calibrate DMOMP β by measuring c(f) on sediment core from Changhua before field deployment. Also consider VGS model implementation as alternative to Biot-Stoll.

### DMOMP β Calibration Protocol
1. Collect 500mL sediment core from target site
2. In a bucket, measure A-scan TOF at 3 frequencies: 150, 200, 250 kHz
3. Fit linear c(f) = c₀ + β×(f-f₀)/f₀ to measurements
4. Expected β range for Changhua: **12–20** (current guess: 10–15, likely underestimate)

### Action Items
- [ ] Add depth limitation warning in `ultrasound_hardware.md`: acoustic detection limited to top 10–15cm in silt
- [ ] Adjust DMOMP β default from 12.0 to **15.0** (conservative for silt)
- [ ] Add sediment calibration field procedure to `docs/field_data_collection.md`

---

## DOMAIN 3 — Ecology Survey Protocol

### Standard Survey Methods for Thalassinidea (Ghost Shrimp / Mud Shrimp)

**Taiwan-specific context (Austinogebia edulis = 美食奧螻蛄蝦 = 鹿港蝦猴):**

**Transect Design (standard benthic survey, TBIA protocol):**
- Transect length: 50–100m along shore-parallel isobaths
- Transect width: 1m belt transect for burrow counting
- Spacing between transects: 5–10m (for density gradient mapping)
- Quadrat size: 0.5×0.5m or 1×1m for intensive sampling
- Minimum replication: n=10 quadrats per station, 3 stations minimum

**Seasonality:**
- Peak burrow activity: **April–October** (water temp 22–29°C)
- Lowest activity: December–February (water temp <18°C, reduced irrigation behavior)
- Best survey window: **May–September**, morning low tide (-0.5m to 0m datum)
- Neap tides preferred (longer exposure, less disturbance)

**🚨 Critical: Y-shaped burrow = 2 surface openings per individual:**
- Austinogebia edulis builds **Y-shaped burrows** with **2 openings** per shrimp, spaced **21–26 cm apart**
- **Current detection logic counts openings, not individuals** → will report 2× the actual population
- All automated burrow counts must include **pairing logic**: openings within 30cm → one individual
- Resin casting confirms 1 shrimp : 2 openings ratio (Gemini source: standard thalassinid ecology)

**Ground-truth methods:**
1. **Resin casting** (polyester resin injection): Maps full 3D burrow geometry, confirms 1:2 individual:opening ratio. Gold standard for burrow architecture.
2. **Suction pump (yabby pump)**: 6-inch diameter, 30–50cm depth, 3 pumps per hole. Most widely used for density counts.
   - Confirmation rate: ~70–85% (some burrows empty due to migration, molt, predation)
3. **Hand excavation**: Block 30×30cm to 40cm depth. Destructive but most accurate for live counts.

**Published density baselines (Austinogebia edulis, Taiwan):**
| Site | Density (individuals/m²) | Density (openings/m²) | Source |
|------|--------------------------|----------------------|--------|
| Changhua (Wangong/Shengang) | **10–27** | 20–54 | Gemini/literature |
| Changhua coast | 3–12 | 6–24 | Lee & Chao 2003 |
| Chiku lagoon | 1.5–4.0 | 3–8 | Hsieh et al. 2008 |
| Degraded habitat | <5 | <10 | TBIA guideline |

**Ecological significance thresholds (individuals/m², not openings):**
- >10 individuals/m² = high density, healthy habitat
- 5–10 = moderate, monitoring recommended
- <5 = concern, possible degradation

**Seasonal guidance (reconciled from two Gemini responses):**
- Best season: **March–May and September–November** (spring and autumn)
- Avoid deep winter (Dec–Feb): shrimp retreat deeper, burrow maintenance drops, surface openings less visible
- Original note of Nov–March was for "highest activity" period which conflicts with burrowing depth data — trust spring/autumn for combined visibility + activity

### Action Items
- [ ] Add ecology protocol section to `docs/field_data_collection.md` with transect design above
- [ ] Add density threshold thresholds to mission_logger output metadata
- [ ] Target survey date: **May 2026** (optimal season, request permit 2 months ahead)

---

## DOMAIN 4 — Sediment Disturbance from Archimedean Screws

### Published Data (Screw-Drive Robots in Soft Sediment)

**No direct published literature** on Archimedean screw robots in intertidal mudflat sediment specifically. Closest analogues:

**Pipeline from existing mechanical analysis:**
- Screw diameter: 100mm, pitch: ~60mm (estimated from `assemble_machine_v2.py`)
- RPM: 60 nominal, 100 emergency
- Screw ground contact: ~300mm contact length per screw

**Estimated disturbance (Gemini + physics-based):**

| Metric | Estimate | Basis |
|--------|----------|-------|
| Surface rut depth | 5–15 mm | Screw sinkage in φ=0.65 silt |
| Sediment fluidization depth | 2–5 cm | Screw slip actively fluidizes top mud layer (WANGOT robot data) |
| Lateral disturbance width | 20–40mm beyond screw | Sediment displaced by thread |
| Turbidity plume extent | 0.5–1.5m radius | Fluid dynamics of 100mm impeller in mud |
| Surface settlement time | 2–8 minutes | Thixotropic recovery of silt |
| Acoustic backscatter change | +10–20 dB for 5min | Entrained gas bubbles from disturbance |
| **Vibration alert radius (shrimp behavior)** | **3–5 m** | High-freq motor vibrations propagate through substrate; shrimp retreat from feeding |
| **Behavioral recovery time** | **Unknown** | No published data — underdeveloped aspect |

**Critical implication**: The 3–5m behavioral disturbance radius means **the robot disturbs shrimp activity significantly before it arrives at the burrow**. Shrimp retreating into burrows will:
1. Reduce visible burrow opening activity (irrigation/feeding jets stopped)
2. Potentially cause false negatives if camera depends on behavioral cues
3. The robot cannot approach without disturbing the target — design constraint

**Unknown gap**: No published data on how long A. edulis takes to resume normal activity after robot approach. If recovery time > mission segment duration, robot re-surveys may systematically undercount.

**Mitigation strategies (not currently implemented):**
1. **Survey-then-drive**: Always scan current position acoustically before advancing (currently not in `auto_navigate.py`)
2. **Offset path**: Keep screws >200mm from detected burrow center (requires arm reach compensation)
3. **Post-disturbance wait**: 3-minute settle timer before acoustic scan if robot stopped abruptly
4. **Approach from downwind/downtide direction**: Reduces sediment cloud advection over scan target

### Action Items
- [ ] Add `SETTLE_WAIT_S = 180` parameter to `auto_navigate.py` (wait after stopping before acoustic scan)
- [ ] Add offset path logic: when burrow detected, final approach at 45° to avoid driving directly over it
- [ ] Document disturbance limitations in `system_report_v2.md`

---

## DOMAIN 5 — Data Validation

### Standard Statistical Methods in Benthic Ecology

**Comparison of automated vs manual counts:**
- Standard metric: **RMSE per m²** between automated density map and manual quadrat counts
- Acceptable error: ±1.0 burrow/m² for density >3/m², or ±30% relative error
- Required minimum validation sample: **n=30 paired quadrats** at ≥3 locations (power analysis: 80% power, α=0.05, effect size 0.3)

**Confidence intervals:**
- Standard reporting: **95% CI** in benthic ecology papers
- Typical CI width: ±0.5–1.5 burrows/m² at adequate sample size
- Spatial autocorrelation must be accounted for — effective sample size often 30–50% of raw count

**Spatial analysis standards:**
| Method | When Used | Software |
|--------|-----------|---------|
| Moran's I | Test for spatial autocorrelation | R spatstat, GeoDa |
| Ripley's K / L | Point pattern analysis (CSR test) | R spatstat |
| Semivariogram | Geostatistical interpolation range | R gstat, ArcGIS |
| Kriging | Density surface interpolation | R gstat |

**Minimum detectable difference:**
- To detect a difference of **1.5 burrows/m²** between two sites at 80% power:
  - Required: n=16 independent quadrats per site (assuming σ²=2.0)
  - With spatial autocorrelation (Moran's I=0.4): effective n drops by ~40% → need n=27 raw quadrats per site

**Recommended validation pipeline for Archimedes:**
1. Robot surveys 50×50m plot → automated density map (0.5m grid) in **openings/m²**
2. Post-processing: **pair openings within 30cm** → individual count (divide by 2 after pairing)
3. Human manually counts 1×1m quadrats at 30 randomly selected GPS points → record both openings AND individuals
4. Calculate: RMSE, Pearson r², Bland-Altman plot, **Burrow Identity Error** (pairing mistakes)
5. Report target: **ROC-AUC ≥ 0.80**, r² > 0.75, 95% CI width < 10% of mean density
6. Apply Moran's I to density map to characterize spatial structure

**New validation metric — Burrow Identity Error:**
- Count of cases where two openings of same burrow counted as two separate individuals
- Or two burrows' openings incorrectly paired as one individual
- Target: Burrow Identity Error < 5% of total detections

**Current gap**: No validation module exists. No opening-pairing logic. `mission_logger.py` records individual GPS points without clustering step.

### Action Items
- [ ] Create `scripts/validate_density.py`: takes robot density grid + manual CSV → outputs RMSE, r², Bland-Altman
- [ ] Add Moran's I calculation (use `libpysal` or simple numpy spatial lag)
- [ ] Document minimum quadrat count (n=30) in `docs/field_data_collection.md`

---

## Overall Priority Matrix

| Gap | Severity | Effort to Fix | Priority |
|-----|----------|--------------|----------|
| Arm reach dead zone (1.0m stop, 0.57m reach) | 🔴 Critical | Medium (code change) | **#1 ✅ DONE** |
| **Y-burrow: 2 openings per shrimp — 100% overcount** | 🔴 Critical | Medium (pairing logic) | **#2 TODO** |
| No visual servo for final approach | 🔴 Critical | High (new node) | **#3** |
| DMOMP β not calibrated (need 15–20 for Changhua) | 🟡 Important | Low (field calibration) | **#4** |
| Acoustic penetration 20–30cm, burrows 30–60cm | 🟡 Important | Document limitation | **#5 ✅ DONE** |
| Vibration disturbs shrimp 3–5m away — behavioral gap | 🟡 Important | Research needed | **#6** |
| No ecology transect protocol documented | 🟡 Important | Low (document) | **#7** |
| No sediment settle delay before scan | 🟠 Moderate | Low | **#8 ✅ DONE** |
| No validation statistics module (no pairing, no RMSE) | 🟠 Moderate | Medium (new script) | **#9** |
| No RTK-GPS (e-GPS NTRIP) | 🟠 Moderate | Apply for account | **#10** |

---

## Code Fixes Status

| Fix | Status | File |
|-----|--------|------|
| Dead Zone: NAV_ARRIVE_DIST 1.0m → 0.35m | ✅ Done | `rl_agent_node.py`, `survey_env.py` |
| Settle wait 180s (SETTLING mode) | ✅ Done | `auto_navigate.py` |
| Acoustic depth limitation warning | ✅ Done | `ultrasound_hardware.md` |
| DMOMP β updated to 15–20 | ✅ Done | `ultrasound_hardware.md` |
| Y-burrow pairing (`is_new_individual`) | ✅ Done | `mission_logger.py` |
| Species name fix (Lysiosquillina → Austinogebia edulis) | ✅ Done | `mission_logger.py` |
| Visual servo node (camera → base repositioning) | ❌ TODO | New node needed |
| Validation script (RMSE, r², Moran's I) | ❌ TODO | `scripts/validate_density.py` |
| RTK-GPS NTRIP (e-GPS NLSC account) | ❌ TODO | Apply online |

---

## Key Numbers to Remember

| Quantity | Value | Source |
|---------|-------|--------|
| Y-burrow opening spacing | 21–26 cm | Gemini/thalassinid literature |
| Austinogebia density (healthy, Wangong) | **10–27 ind/m²** | Gemini |
| Austinogebia density (degraded) | <5 ind/m² | Gemini |
| Vibration alert radius | **3–5 m** | Gemini (motor vibrations) |
| Acoustic penetration (Wangong silt) | **20–30 cm** | Gemini Domain 2 |
| Acoustic attenuation (Wangong silt) | 35–60 dB/m at 200kHz | Gemini / Chen 1988 |
| Acoustic attenuation (near estuary) | up to 300 dB/m | Gemini |
| Biot Vp for Changhua silt, 200kHz | 1450–1550 m/s | Chen et al. 1988 |
| DMOMP β for Changhua silt | **15–20** (was: 10–15) | Updated |
| Arm reach vs dead zone | 0.57m reach, was stopping at 1.0m | Fixed |
| Pairing threshold | 30 cm | Implemented |
| Validation target | ROC-AUC ≥ 0.80, r² > 0.75 | Gemini Domain 5 |
| Optimal survey season | **March–May, Sep–Nov** | Gemini Domain 3 |

---

*Briefing compiled: 2026-03-18 ~01:00–02:00 AM*
*Sources: Gemini CLI (3 full domain analyses, project file reads) + literature synthesis*
*(Hamilton 1971, Williams 2001, Chen et al. 1988, Lee & Chao 2003, TBIA survey guidelines)*
