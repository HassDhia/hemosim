---
name: Warfarin PKPD calibration-fit rescaling is genotype-conditional
origin: hemosim 2026-04-18 — v2 calibration refit of warfarin PKPD parameters against published INR ranges needed a vkorc1 rescaling to hit steady-state targets. Rescaling ALL vkorc1 multipliers (GG/GA/AA) would silently invalidate the Hamberg 2007 CYP/VKORC1 genotype-response curves, which downstream tests and paper claims depend on. Correct fix: rescale only the wild-type (GG) multiplier; preserve Hamberg GA/AA relative multipliers.
scope: hemosim project — warfarin_pkpd.py only
severity: HIGH — silent scientific-validity regression if the wrong genotype is touched
---

# Warfarin PKPD calibration-fit rescaling is genotype-conditional

When recalibrating warfarin PKPD params against published INR steady-state targets (test_srm_warfarin_hamberg_pkpd), the calibration fit only constrains the wild-type (VKORC1 GG) patient. The Hamberg 2007 PK/PD model supplies *relative* VKORC1 multipliers for GA and AA genotypes on top of that wild-type baseline.

## Rule

**When refitting vkorc1-family parameters:**
- Rescale only the wild-type (`vkorc1_factor` at genotype=GG) constant.
- Do NOT multiply the GA or AA multipliers by the same factor.
- The GA/AA values in the Hamberg table are ratios *relative to GG*; rescaling all three breaks that ratio and invalidates the genotype-response curve.

**Current correct defaults** (from v2 calibration fit, preserved in `src/hemosim/models/warfarin_pkpd.py`):
- `ec50=1.106`
- `hill=1.650`
- `vk_inhibition_gain=0.06519`
- `s_warfarin_potency=2.474`
- `vkorc1_factor=0.6155` (GG only — GA/AA retain Hamberg relative multipliers unchanged)

## Why

1. **Scientific validity:** The paper makes claims about CYP2C9/VKORC1 dose sensitivity that rely on the Hamberg genotype curves. Uniform rescaling flattens the genotype effect and invalidates those claims.
2. **Test structure:** `test_srm_warfarin_hamberg_pkpd_steady_state_inr` fits a GG patient. It will pass whether you rescaled GG only or all three. The silent invalidation only surfaces when a reviewer tests a GA/AA patient, which the current test suite does not.
3. **Precedent:** The v0.2.0 → v0.2.1 scrubs kept Hamberg GA/AA multipliers intact. Deviating breaks continuity with published calibration.

## How to apply

- Any future warfarin PKPD recalibration PR must include a genotype-stratified test (GG, GA, AA) or explicitly document that GA/AA curves were re-validated.
- If the wild-type fit changes materially (>10%), add a regression test for GA/AA steady-state INR at a representative warfarin dose; if the regression fails, the Hamberg ratios need a separate refit, not a uniform rescale.
- Code comment at the `vkorc1_factor` assignment site should reference this rule so the next editor sees it before rescaling.

## Surface area

This rule is hemosim-local. It does NOT generalize to AppliedResearch as a whole, because most domains don't have genotype-conditional calibration. It's here to protect the specific warfarin model from a subtle class of regression.
