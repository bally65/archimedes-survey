# Archimedes Survey Robot -- System Integration Report v2.0
> Mantis Shrimp Habitat Survey Robot (Lysiosquillina), Updated 2026-03-16

---

## 01 System Overview

**Design Goal:** Autonomous beach traversal to detect mantis shrimp (*Lysiosquillina*) burrows, with probe sampling and environmental logging.

**Architecture:**
```
archimedes-survey/
├── mechanics/        # Screw propulsion physics, Isaac Sim simulation
├── arm/              # 3-DOF robotic arm kinematics
├── sensing/          # Burrow image recognition
├── simulation/       # Sand + hydrology simulation
├── navigation/       # Auto-traverse path planning
└── visualization/    # Population density map output
```

**Simulation Status:** Isaac Sim 4.5 kinematic simulation complete
- 135 runs: 9 scenarios × 3 payloads × 5 moisture conditions
- Output: `docs/isaac_sim_matrix_results.json`, `docs/matrix_summary.png`

---

## 02 Environment Analysis

**Soil Mechanics (Field-Calibrated)**
- Field test: 49.7 kg weight → 5 cm depth imprint
- Estimated foot pressure ≈ 1,393 Pa
- Bearing capacity (dry sand 10%): ~14 kPa
- Bearing capacity (saturated 90%): ~1.4 kPa (near-liquid)
- Hardest operation: 40–60% moisture (semi-hard, max probe resistance)

**Wind Conditions**
- Max design wind: 40 km/h (11.1 m/s) operational
- Typhoon anchor mode: screws reverse-lock into sand
- Operational wind limit: < 20 m/s
- Safety factor vs 40 km/h wind at 50% moisture: **55× (safe)**

---

## 03 Screw Propulsion — Simulation Results

### 3.1 Performance at Optimal Conditions (moisture=50%, 0 kg payload)

| Scenario | Speed | Notes |
|----------|-------|-------|
| Straight Forward/Reverse | **5.26 m/min** | Peak efficiency |
| Arc Turn L/R (R=1m) | **4.67 m/min** | 88.8% of forward speed |
| Spin CW/CCW (in-place) | **20.2 deg/s** | Full 360° in 17.8 s |
| Strafe Left/Right | **2.90 m/min** | 55% of forward speed |
| Wind 40 km/h | **5.26 m/min** | Zero lateral drift (anchor force 740 N >> wind 13.5 N) |

### 3.2 Moisture vs Speed (Straight Forward, 0 kg)

| Moisture | Speed | Efficiency | Condition |
|----------|-------|------------|-----------|
| 10% | 2.70 m/min | 40% | Dry, poor grip |
| 30% | 4.18 m/min | 62% | Damp |
| **50%** | **5.26 m/min** | **78%** | **Optimal** |
| 70% | 4.72 m/min | 70% | Wet |
| 90% | 3.71 m/min | 55% | Near-saturated |

**Best operating window: 30–55% moisture** (post-tide, ~30 min after low tide)

### 3.3 Payload Impact (moisture=50%)

| Payload | Speed | Spin | Delta |
|---------|-------|------|-------|
| 0 kg (22 kg total) | 5.26 m/min | 20.2 deg/s | baseline |
| +5 kg (27 kg total) | 5.17 m/min | 19.8 deg/s | -1.8% |
| +10 kg (32 kg total) | 5.07 m/min | 19.5 deg/s | -3.6% |

### 3.4 Slope Performance

| Moisture | Max Climbable Slope (0 kg) | Speed at 10° |
|----------|---------------------------|--------------|
| 10% (dry) | **2°** | 0 (cannot climb) |
| 30% | 20°+ | 3.79 m/min |
| 50% | 20°+ | **4.99 m/min** |
| 70% | 20°+ | 4.37 m/min |
| 90% | 20°+ | 3.05 m/min |

> **Key finding:** Dry sand (10% moisture) is the critical failure case — max slope only 2°.
> At 30%+ moisture, machine can handle slopes up to 20° with minimal speed loss.

---

## 04 Motor Specification

### 4.1 Force Budget (moisture=50%, flat beach)

| Parameter | Value |
|-----------|-------|
| Rolling resistance (total) | 97.1 N |
| Screw cohesion thrust | 643.2 N |
| Net excess force | **546.1 N** (5.6× safety margin) |
| Required torque at screw | 2.88 N*m per motor |
| With 10:1 gearbox, motor shaft torque | **0.29 N*m** |

### 4.2 Motor Selection

| Motor | Rated Torque | Power | Verdict |
|-------|-------------|-------|---------|
| NEMA23 (57mm) | 1.26 N*m | 60–100 W | **RECOMMENDED** |
| NEMA34 (86mm) | 3.0–4.5 N*m | 150–300 W | Overkill |

**Selected:** NEMA23 + 10:1 planetary gearbox × 4 motors
- Motor shaft torque needed: 0.29 N*m << NEMA23 rated 1.26 N*m (4.3× margin)
- Total system power at 60 RPM: **72.4 W**

### 4.3 RPM vs Performance

| RPM | Speed | Total Power |
|-----|-------|-------------|
| 40  | 3.51 m/min | 48 W |
| **60** | **5.27 m/min** | **72 W** ← design point |
| 80  | 7.02 m/min | 97 W |
| 100 | 8.77 m/min | 121 W |

### 4.4 Battery Runtime

| Motor Load | Total Draw | Runtime |
|-----------|-----------|---------|
| 25% (idle traverse) | 38 W | **12.6 h** |
| 50% (normal) | 56 W | **8.5 h** |
| 75% (rough terrain) | 74 W | **6.5 h** |
| 100% (max) | 92 W | **5.2 h** |

Battery: 480 Wh LiFePO4 (24V) — all scenarios exceed 5 h operational endurance.

---

## 05 Sensing System

### 5.1 Sensor Suite

| # | Sensor | Model (suggested) | Purpose | Cost (NT$) |
|---|--------|-------------------|---------|------------|
| 1 | RGB Camera | OV5647 (Pi Camera v2) | Burrow image recognition | ~400 |
| 2 | Soil Moisture | Capacitive v1.2 × 2 | Real-time sand condition | ~200 |
| 3 | Wind Speed | RS485 ultrasonic anemometer | Wind safety cutoff | ~1,500 |
| 4 | IMU/Compass | BNO085 (9-DOF) | Heading, slope, tilt | ~600 |
| 5 | GPS | NEO-M9N RTK-capable | Position logging | ~1,800 |
| 6 | Depth/Pressure | US-100 ultrasonic | Ground clearance check | ~200 |
| **Total** | | | | **~NT$4,700** |

### 5.2 Burrow Detection Pipeline

```
Camera → Raspberry Pi 4 → YOLOv8-nano model → Burrow bbox
  → GPS tag → population density map → JSON output
```

- Target: detect burrow openings ∅ 15–30 mm at 0.3–0.5 m range
- Optimal imaging moisture: 10–40% (clear surface, before tidal flooding)
- At 70%+ moisture: surface glare + mud reduce visibility; consider IR filter

### 5.3 Sensor Integration Notes

- All sensors on I2C / UART bus to Raspberry Pi 4
- STM32 handles real-time motor control (50 Hz loop), Pi handles vision + logging
- IP65 enclosure required for all external sensors
- Moisture sensors: mount flush with screw housing, angled 30° downward

---

## 06 Electrical System

**Power Architecture**

| Component | Voltage | Power | Notes |
|-----------|---------|-------|-------|
| 4× NEMA23 motors | 24V | 72 W (nominal) | Brushless, FOC driver |
| Raspberry Pi 4 | 5V (via DC-DC) | 8 W | Vision + navigation |
| STM32 MCU | 3.3V | 1 W | Motor control |
| Cameras + sensors | 3.3–5V | 6 W | |
| **Total** | | **~87 W** | |

**Battery:** 480 Wh, 24V LiFePO4 (e.g., 24V 20Ah pack)
**Runtime:** 5.2 h at full load, 8.5 h at 50% load (normal survey)
**Protection:** IP65 battery enclosure, BMS with over-temp cutoff

---

## 07 Risk Assessment

| # | Risk | Mitigation |
|---|------|-----------|
| 1 | Wind > friction (typhoon) | Screw anchor mode + 20 m/s op limit |
| 2 | Dry sand (10%) slope > 2° | Survey only at 30%+ moisture |
| 3 | Near-liquid sand (90%) | Screws still effective; limit to 15 m/min area |
| 4 | Sensor water ingress | IP65 enclosure + silicone seal |
| 5 | Battery swelling in heat | LiFePO4 + shaded enclosure |
| 6 | Motor overload on slope | 4.3× torque margin; slope limit 15° for dry sand |
| 7 | Burrow missed at high moisture | IR-supplemented camera; dual-angle lighting |

---

## 08 Development Roadmap

| Phase | Target | Key Milestone |
|-------|--------|---------------|
| Phase 1 | Screw drive prototype, beach mobility test | 5.26 m/min confirmed |
| Phase 2 | 3-DOF arm + probe insertion test | Soil sampling at 40–60% moisture |
| Phase 3 | Sensor integration, burrow detection | YOLOv8 live inference |
| Phase 4 | Autonomous traverse, full system test | GPS-tagged density map |

---

## 09 Cost Estimate (Updated)

| Category | Item | Cost (NT$) |
|----------|------|------------|
| **Screw / Structure** | 6" PVC pipe × 2, PETG filament (100 blades) | ~3,000 |
| **Motors** | NEMA23 × 4 + 10:1 gearbox × 4 | ~8,000 |
| **Motor Drivers** | FOC driver 24V × 4 | ~3,200 |
| **Battery** | 24V 20Ah LiFePO4 + BMS | ~6,000 |
| **Main Controller** | Raspberry Pi 4 (4GB) + STM32 dev board | ~3,500 |
| **Sensing** | Camera + moisture + wind + IMU + GPS + ultrasonic | ~4,700 |
| **Structure** | Aluminum extrusion, brackets, fasteners | ~2,500 |
| **Arm (Phase 2)** | NEMA23 shoulder + gearbox + linkage | ~5,000 |
| **Misc / IP65** | Enclosures, wiring, connectors, spare parts | ~2,000 |
| **Contingency 10%** | | ~3,800 |
| **Total** | | **~NT$41,700** |

---

## 10 Simulation Artifacts

| File | Description |
|------|-------------|
| `docs/isaac_sim_results.json` | 9 scenarios, baseline (moisture=50%, 0 kg) |
| `docs/isaac_sim_matrix_results.json` | 135 runs full matrix |
| `docs/matrix_summary.png` | 3×3 scenario chart, payload-differentiated |
| `docs/slope_analysis.json` | 9 angles × 5 moisture × 3 payloads |
| `docs/slope_analysis.png` | Slope performance chart |
| `docs/motor_analysis.json` | RPM vs torque/power table |
| `docs/motor_analysis.png` | Motor operating curve |
| `docs/beach_simulation.png` | Moisture vs efficiency (Python physics model) |

---

*Report version: v2.0 | Date: 2026-03-16 | Status: Simulation complete, prototype pending*
