# Changelog

All notable changes to hemosim are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] — 2026-04-18

### Added
- 12-claim mechanistic-claim registry at `docs/mechanistic-claims.md`. Every mechanism, reproduction, and architecture-equivalence claim in the paper is registered with a primary-source numeric target and a dedicated falsification test.
- 29 new falsification tests (`tests/test_srm_*.py`), one per registered claim, each asserting against a primary-source numeric range under the artifact's default initialization.
- Citation-consistency gate runner at `scripts/verify_citation_consistency.py` with self-test mode; pytest integration at `tests/test_citation_consistency_gate.py` enforces paper-vs-artifact numeric and attribution consistency on every push.

### Fixed
- Benchmark key `nemati_ttr_standard` renamed to `wan_aptt_ttr_standard_of_care` across `src/hemosim/validation/published_calibration.py`, `results/published_calibration.json`, and `tests/test_published_calibration.py`. Nemati 2016 does not report a time-in-therapeutic-range number; the 0.55 aPTT-TTR target is correctly attributed to Wan 2008 (*Circulation: Cardiovascular Quality and Outcomes*).
- DIC cascade scope clarified: the 8-state reduced cascade activates under DIC pathophysiology (AT-III-depleted, intense upstream triggering) and correctly does not activate under healthy-physiology AT-III, which is why the cascade is exercised only inside `DICManagement-v0`.

### Test counts
- 383 tests total (354 from v0.2.1 + 29 new mechanistic-claim falsification tests; main tests plus ladder/boundary/wiring subtests).

## [0.2.1] — 2026-04-17

Self-audit release. No behavioral change to any environment, baseline, calibration, or metric.

### Fixed
- Generic institutional attribution replaced with direct published-reference citations (Holbrook 2012 ACCP, Raschke 1993, Warkentin 2003) in `src/hemosim/clinical/__init__.py`, `safety.py`, `dss.py`, and the silent-deployment protocol. Removed an unverified authority citation for the 25 U/kg/hr heparin ceiling; it is now correctly attributed to its actual basis — a conservative margin above the upper end of the Raschke 1993 nomogram ladder (12–22 U/kg/hr), consistent with Holbrook 2012 stewardship principles.
- `WarfarinClinicalBaseline` docstring no longer implies IWPC pharmacogenetic dosing. The class is a fixed-dose INR-adjusted titration table; the real IWPC/Gage pharmacogenetic calculator is `WarfarinGageBaseline` in `baselines_extended.py`. Paper Abstract, Related Work, §4, §11, §13 updated accordingly.

### Added
- Permanent integrity-audit test module (`tests/test_integrity_audit.py`, 15 tests). CI gate covering baseline-docstring-vs-implementation integrity, paper-vs-code claim consistency, bib coverage, paper-vs-JSON numeric alignment, version consistency across `pyproject.toml` / `CITATION.cff` / `README.md` / `CHANGELOG.md`, protocol placeholder scrubbing, wheel-build sanity, and POMDP reality checks.

### Unchanged
- 339 baseline / environment / calibration / metric / POMDP / DSS / safety tests from v0.2.0 pass identically on v0.2.1. Results table, calibration residuals, and Gage pharmacogenetic baseline are bit-identical to v0.2.0.

## [0.2.0] — 2026-04-16

Major release. Complete rebuild against three clinical-science critiques raised on v0.1.

### Added
- **POMDP reformulation**: `hemosim/HeparinInfusion-POMDP-v0` with explicit lab-ordering actions, per-lab turnaround times (aPTT 45 min / CV 8%, anti-Xa 180 min / CV 12%, platelets 45 min / CV 5%), and analytical variability drawn from Bowen 2016 and Lippi 2012. Replaces the v0.1 oracle-lab assumption.
- **Mechanistic coagulation cascade** (`src/hemosim/models/coagulation.py`): 8-state reduced ODE coupled to the DIC environment.
- **Published-data calibration harness** (`src/hemosim/validation/published_calibration.py`): fits PK/PD parameters against Hamberg, IWPC, Raschke, Hirsh, RE-LY, ROCKET-AF, and ARISTOTLE cohort summaries. Produces `results/published_calibration.json`. Normalized RMSE = 0.0013 on warfarin.
- **Fitted PK/PD defaults applied at runtime**: `WarfarinPKPD.__init__` now ships the v2 fitted calibration values (ec50=1.106, vkorc1_gg_factor=0.6155, hill=1.650, vk_inhibition_gain=0.06519, s_warfarin_potency=2.474) rather than v0.1 priors.
- **Clinical-outcome metrics** (`src/hemosim/metrics/clinical.py`): Rosendaal TTR, ISTH 2005 major bleeding, Warkentin 4T HIT score, thromboembolic event aggregator, and a composite mortality proxy (explicitly flagged).
- **CDS harness with rule-based safety layer** (`src/hemosim/clinical/dss.py`, `safety.py`): FHIR-shaped `PatientSnapshot` input, `DosingRecommendation` output with uncertainty interval, saliency, and deferral flag. Guardrails include absolute dose caps, FDA-label-appropriate contraindication rules (apixaban dose-reduced rather than refused at low CrCl), and uncertainty-aware deferral.
- **MIMIC-IV calibration scaffold** (`src/hemosim/validation/mimic_calibration.py`): end-to-end runnable against a dummy cohort; ready for PhysioNet credentialed-access cohort.
- **Silent-deployment protocol** (`paper/silent_deployment_protocol.tex` + `protocol_summary.md`): SPIRIT 2013–compliant observational-trial document, IRB-ready.
- **Extensible baselines suite** (`src/hemosim/agents/baselines_extended.py`): `HeparinAntiXaBaseline`, `WarfarinGageBaseline`, `WarfarinOrdinalBaseline`, and a faithful reimplementation of the Nemati 2016 DQN architecture.
- **Reproducibility infrastructure** (`src/hemosim/reproducibility.py`, `scripts/reproduce.sh`, `results/EXPECTED_RESULTS.json`): disjoint train/held-out seed pools, 5%-tolerance comparison check, one-command repro.
- **Clinical-outcome info keys**: every environment now emits `info["therapeutic"]` with a domain-appropriate definition (warfarin INR 2–3, heparin aPTT 60–100, DOAC no-stroke-no-bleed, DIC ISTH score<5).
- 339 unit and integration tests (up from 142 in v0.1).

### Changed
- Warfarin termination threshold relaxed from INR<1.0 to INR<0.5 (extreme mechanical floor). The INR<1.5 sub-therapeutic band still incurs a shaped-reward penalty but does not terminate the episode, preventing premature termination on stochastic dips.
- IWPC baseline dose ladder escalated for sub-therapeutic INR: 12.5 mg (INR<1.5) and 10 mg (INR 1.5–1.99), matching the upper end of the IWPC percentage-escalation protocol.
- Paper fully rewritten for v2 scope and honest residual reporting.

### Removed
- v0.1 synthetic PPO column in `scripts/generate_results.py`. v2 emits `null` for policy columns that do not have a trained model on disk, and v2 paper reports no policy numbers.
- v0.1 misattribution of the 0.55 aPTT-TTR target to Nemati 2016; correctly framed as a Wan 2008 antithrombotic-stewardship benchmark used as a cohort-level calibration residual.
- v0.1 inverted apixaban CrCl contraindication rule. The safety layer now correctly dose-reduces (rather than refuses) apixaban at low CrCl per the apixaban FDA label.

## [0.1.0] — 2025-11-14

Initial release. Four Gymnasium environments (warfarin, heparin, DOAC, DIC); Hamberg warfarin PK/PD; Raschke heparin nomogram; DOAC trial-derived event-rate simulators; 142 tests.

**Known limitations addressed in v0.2.0:** oracle lab observations at every timestep; no clinical-outcome metrics; no safety layer; synthetic PPO column in results; TTR benchmark misattributed.

[0.2.0]: https://github.com/HassDhia/hemosim/releases/tag/v0.2.0
[0.1.0]: https://github.com/HassDhia/hemosim/releases/tag/v0.1.0
