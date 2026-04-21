# Published-Data Calibration Report

This report is produced by `scripts/run_published_calibration.py` and summarizes the fit of hemosim's PK/PD parameters against published clinical-trial summary statistics. **All targets are cohort-level summaries**, not individual patient trajectories — individual-level calibration against MIMIC-IV is a Phase 2 collaboration that requires PhysioNet credentialed access.

## Benchmarks

| Key | Trial | Endpoint | Value | Units |
|-----|-------|----------|-------|-------|
| `raschke_aptt_6h` | Raschke 1993 (Ann Intern Med) | Mean aPTT at 6h after 80 U/kg bolus + 18 U/kg/hr | 75 | seconds |
| `hirsh_therapeutic_conc_mid` | Hirsh 2001 (Chest) | Heparin plasma level at therapeutic aPTT (midpoint) | 0.3 | U/mL |
| `wan_aptt_ttr_standard_of_care` | Wan 2008 (Circulation) | aPTT time-in-therapeutic-range target from antithrombotic-stewardship systematic review | 0.55 | fraction |
| `iwpc_mean_maintenance_dose` | IWPC 2009 (NEJM) | Mean stable maintenance dose (N=4043) | 5.2 | mg/day |
| `iwpc_days_to_therapeutic` | IWPC 2009 (NEJM) | Days until INR enters 2.0-3.0 on appropriate dose | 7 | days |
| `hamberg_ss_inr_wildtype` | Hamberg 2007 (CPT) | Steady-state INR in CYP2C9 *1/*1, VKORC1 GG, age 65, 75 kg on 5 mg/day | 2.5 | INR |
| `rely_dabi_stroke` | RE-LY 2009 (NEJM) | Stroke/systemic embolism, dabigatran 150 mg BID | 1.11 | %/yr |
| `rely_dabi_bleed` | RE-LY 2009 (NEJM) | Major bleeding, dabigatran 150 mg BID | 3.11 | %/yr |
| `rocket_riva_stroke` | ROCKET-AF 2011 (NEJM) | Stroke/systemic embolism, rivaroxaban 20 mg daily | 1.7 | %/yr |
| `rocket_riva_bleed` | ROCKET-AF 2011 (NEJM) | Major bleeding, rivaroxaban 20 mg daily | 3.6 | %/yr |
| `aristotle_apix_stroke` | ARISTOTLE 2011 (NEJM) | Stroke/systemic embolism, apixaban 5 mg BID | 1.27 | %/yr |
| `aristotle_apix_bleed` | ARISTOTLE 2011 (NEJM) | Major bleeding, apixaban 5 mg BID | 2.13 | %/yr |

## Heparin fit

### Heparin PK/PD fit

- Status: **converged** after 178 iterations (`Optimization terminated successfully.`)
- Benchmarks: 3
- Normalized RMSE: 0.1837
- Seed: 42

**Fitted parameters**

| Parameter | Initial | Fitted |
|-----------|---------|--------|
| `vmax` | 400 | 2761 |
| `km` | 0.4 | 0.1511 |
| `aptt_alpha` | 2.5 | 1.708 |
| `aptt_c_ref` | 0.15 | 0.2982 |

**Per-benchmark residuals**

| Key | Trial | Endpoint | Expected | Observed | Residual | Units |
|-----|-------|----------|----------|----------|----------|-------|
| `raschke_aptt_6h` | Raschke 1993 (Ann Intern Med) | Mean aPTT at 6h after 80 U/kg bolus + 18 U/kg/hr | 75.000 | 75.000 | +0.000 | seconds |
| `hirsh_therapeutic_conc_mid` | Hirsh 2001 (Chest) | Heparin plasma level at therapeutic aPTT (midpoint) | 0.300 | 0.300 | +0.000 | U/mL |
| `wan_aptt_ttr_standard_of_care` | Wan 2008 (Circulation) | aPTT time-in-therapeutic-range target from antithrombotic-stewardship systematic review | 0.550 | 0.375 | -0.175 | fraction |

## Warfarin fit

### Warfarin PK/PD fit

- Status: **converged** after 190 iterations (`Optimization terminated successfully.`)
- Benchmarks: 3
- Normalized RMSE: 0.0013
- Seed: 42

**Fitted parameters**

| Parameter | Initial | Fitted |
|-----------|---------|--------|
| `ec50` | 1.5 | 1.106 |
| `vkorc1_gg_factor` | 1 | 0.6155 |
| `hill` | 1.3 | 1.65 |
| `vk_inhibition_gain` | 0.04 | 0.06519 |
| `s_warfarin_potency` | 3 | 2.474 |

**Per-benchmark residuals**

| Key | Trial | Endpoint | Expected | Observed | Residual | Units |
|-----|-------|----------|----------|----------|----------|-------|
| `iwpc_mean_maintenance_dose` | IWPC 2009 (NEJM) | Mean stable maintenance dose (N=4043) (simulated INR @ mean dose vs therapeutic midpoint 2.5) | 2.500 | 2.504 | +0.004 | INR |
| `iwpc_days_to_therapeutic` | IWPC 2009 (NEJM) | Days until INR enters 2.0-3.0 on appropriate dose | 7.000 | 7.000 | +0.000 | days |
| `hamberg_ss_inr_wildtype` | Hamberg 2007 (CPT) | Steady-state INR in CYP2C9 *1/*1, VKORC1 GG, age 65, 75 kg on 5 mg/day | 2.500 | 2.496 | -0.004 | INR |

### DOAC event-rate validation

- Episodes per drug: 500 (seed: 42)

| Drug | Stroke obs (%/yr) | Stroke trial (CI) | Bleed obs (%/yr) | Bleed trial (CI) | Trial |
|------|-------------------|--------------------|------------------|-------------------|-------|
| dabigatran | 1.43 | 1.11 [0.92-1.33] | 3.27 | 3.11 [2.80-3.46] | RE-LY |
| rivaroxaban | 3.10 | 1.70 [1.45-2.00] | 3.92 | 3.60 [3.27-3.96] | ROCKET-AF |
| apixaban | 1.64 | 1.27 [1.05-1.53] | 2.25 | 2.13 [1.89-2.40] | ARISTOTLE |


## Fingerprint

`sha256(published_calibration.json)` = `73a865db2d400bc4f69da57820a8978e8895228d20bb2d884917cbc234830627`
