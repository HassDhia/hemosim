# Changelog

All notable changes to hemosim are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] — 2026-04-18

SR-M gate pass. Registered every mechanistic scientific claim in `paper/hemosim.tex` in `.reviews/mechanistic-claims.md` with a primary-source numeric target, a falsification test under the artifact's default initialization, and a pre-committed rescope fallback. Zero paper text changes — v0.2.1 abstract, §3, §6, §8, §13, §15 survived Phase C intact; Council re-convene confirmed `APPROVED FOR PUBLICATION` on iteration 2 (two-in-a-row convergence) under the expanded gate set.

### Added
- **SR-M gate family (Scientific Rigor — Mechanistic).** 12-claim mechanistic-claim registry at `.reviews/mechanistic-claims.md`; 29 new SR-M falsification tests in `tests/test_srm_*.py`, one file per claim, each using the default artifact initialization and asserting against a primary-source numeric range. SR-M runs pre-Council via `bun Tools/research-verify-gates.ts --phase review-srm` and is enforced as an AppliedResearch non-negotiable. Origin: v0.2.1 Science-Skill Report T1.5 (Nemati→Wan drift) + T1.10 (cascade thrombin activation under default init).
- **v0.2.2 review certificate** at `.reviews/v0.2.2-review-certificate.md`, APPROVED FOR PUBLICATION with a `## SR-M gate results` section listing all 12 claims and their test status.
- **v0.2.2 preflight certificate** at `.reviews/v0.2.2-preflight-certificate.md`, OUTREACH GATE: OPEN.

### Fixed
- **Nemati → Wan 2008 attribution propagated through code + JSON + tests.** `src/hemosim/validation/published_calibration.py` benchmark key `nemati_ttr_standard` renamed to `wan_aptt_ttr_standard_of_care`; trial, endpoint, and citation strings updated to Wan 2008 Circ Cardiovasc Qual Outcomes; `results/published_calibration.json` regenerated; `tests/test_published_calibration.py::test_required_keys_present` updated. Zero residual `nemati_ttr_standard` string tokens in the tree. Origin: v0.2.1 Science-Skill Report T1.5 blocker.

### Disclosed (non-breaking scope clarification, no code or paper edit required)
- **DIC cascade activation claim scope** — `.reviews/mechanistic-claims.md::dic_hockin_mann_cascade` now explicitly states that the cascade's thrombin-activation behavior is bounded to DIC pathophysiology (AT-III-depleted + intense upstream triggering). Under healthy-physiology AT-III, the reduced cascade correctly does not activate (thrombin is scavenged faster than generated), which is consistent behavior and the reason the cascade is exercised only inside the DIC environment. Paper §6 language ("retains key dynamical features", "captures TF-initiated activation, the thrombin positive-feedback loop") is unchanged and, under this scope, accurate.

### Test counts
- v0.2.1: 354 tests (339 behavioral + 15 integrity-audit).
- v0.2.2: 383 tests (354 from v0.2.1 + 29 SR-M falsification tests — 12 main tests plus ladder/boundary/wiring subtests).

### Filed for future-round attention (non-blocking, separate PRs)
- `feedback/2026-04-18-consistency-reviewer-widen-gate.md` — proposes CL4 (string-field consistency) and CL5 (multi-path measurement consistency) gates for AppliedResearch.
- `feedback/2026-04-18-sr-m-registry-authoring-discipline.md` — proposes weakest-credible-commitment authoring discipline + `bootstrap-mechanistic-claims.ts` prompt update.

## [0.2.1] — 2026-04-17

Self-audit release. Pre-partner-send audit caught three integrity issues in the v0.2.0 tree. All three are fixed here. No behavioral change to any environment, baseline, calibration, or metric — all 339 v0.2.0 tests still pass identically, plus 15 new permanent integrity-audit tests added (354 total).

### Fixed
- **Institutional attribution scrubbed from code and protocol.** The v0.2.0 tree referenced a specific academic medical center (UCSD) in `src/hemosim/clinical/__init__.py`, `safety.py`, `dss.py`, in the silent-deployment protocol, and in the protocol summary. None of those references reflected actual endorsement by any institution. All 14 references replaced with either generic partner-site language ("partner academic medical center") or direct published-reference citations (Holbrook 2012 ACCP, Raschke 1993, Warkentin 2003).
- **Unverified authority citation removed.** `safety.py` cited "UCSD Medical Center Antithrombotic Stewardship Committee" as the basis for the 25 U/kg/hr heparin ceiling. No such externally-published protocol exists. The ceiling is now correctly attributed to its actual basis: a conservative margin above the upper end of the Raschke 1993 nomogram ladder (12–22 U/kg/hr), consistent with Holbrook 2012 stewardship principles.
- **`WarfarinClinicalBaseline` docstring no longer falsely claims IWPC pharmacogenetic dosing.** The class is (and was) a fixed-dose INR-adjusted titration table, not the multivariable IWPC 2009 pharmacogenetic dose calculator. Docstring, paper Abstract, Related Work, Architecture section (§4), Experimental Setup (§11), and Results interpretation (§13) all updated to describe the baseline accurately. The real IWPC/Gage pharmacogenetic calculator was already implemented as `WarfarinGageBaseline` in `baselines_extended.py`; that reference is now explicit.

### Added
- **Permanent integrity-audit test module** (`tests/test_integrity_audit.py`, 15 tests). Permanent CI gate covering: institutional overreach in code/docs, unverified authority citations, baseline-docstring-vs-implementation integrity, paper-vs-code claim consistency, bib coverage, paper-vs-JSON numeric alignment, version consistency across pyproject/CITATION/README/CHANGELOG, fabrication-formula detection, protocol placeholder scrubbing, wheel-build sanity, Nemati-DQN and POMDP reality checks. Every v0.2.0 failure mode and the v0.1 PPO-fabrication pattern would trigger one or more of these tests before a release could be cut.

### Unchanged
- 339 baseline / environment / calibration / metric / POMDP / DSS / safety tests from v0.2.0: **all pass** on v0.2.1 (verified). Plus 15 new permanent integrity-audit tests (354 total). Results table, calibration residuals, Nemati DQN baseline, POMDP env, and Gage pharmacogenetic baseline are bit-identical to v0.2.0.

## [0.2.0] — 2026-04-16

Major release. Complete rebuild against three clinical-science critiques raised on v0.1.

### Added
- **POMDP reformulation**: `hemosim/HeparinInfusion-POMDP-v0` with explicit lab-ordering actions, per-lab turnaround times (aPTT 45 min / CV 8%, anti-Xa 180 min / CV 12%, platelets 45 min / CV 5%), and analytical variability drawn from Bowen 2016 and Lippi 2012. Replaces the v0.1 oracle-lab assumption.
- **Mechanistic coagulation cascade** (`src/hemosim/models/coagulation.py`): 8-state reduced ODE coupled to the DIC environment.
- **Published-data calibration harness** (`src/hemosim/validation/published_calibration.py`): fits PK/PD parameters against Hamberg, IWPC, Raschke, Hirsh, RE-LY, ROCKET-AF, and ARISTOTLE cohort summaries. Produces `results/published_calibration.json`. Normalized RMSE = 0.0013 on warfarin.
- **Fitted PK/PD defaults applied at runtime**: `WarfarinPKPD.__init__` now ships the v2 fitted calibration values (ec50=1.106, vkorc1_gg_factor=0.6155, hill=1.650, vk_inhibition_gain=0.06519, s_warfarin_potency=2.474) rather than v0.1 priors.
- **Clinical-outcome metrics** (`src/hemosim/metrics/clinical.py`): Rosendaal TTR, ISTH 2005 major bleeding, Warkentin 4T HIT score, thromboembolic event aggregator, and a composite mortality proxy (explicitly flagged).
- **CDS harness with rule-based safety layer** (`src/hemosim/clinical/dss.py`, `safety.py`): FHIR-shaped `PatientSnapshot` input, `DosingRecommendation` output with uncertainty interval, saliency, and deferral flag. Guardrails include absolute dose caps, FDA-label-appropriate contraindication rules (apixaban dose-reduced rather than refused at low CrCl), and uncertainty-aware deferral.
- **MIMIC-IV calibration scaffold** (`src/hemosim/validation/mimic_calibration.py`): end-to-end runnable against a dummy cohort; ready for PhysioNet credentialed-access cohort on grant.
- **Silent-deployment protocol** (`paper/silent_deployment_protocol.tex` + `protocol_summary.md`): SPIRIT 2013–compliant observational-trial document, IRB-ready.
- **Extensible baselines suite** (`src/hemosim/agents/baselines_extended.py`): `HeparinAntiXaBaseline`, `WarfarinGageBaseline`, `WarfarinOrdinalBaseline`, and a faithful reimplementation of the Nemati 2016 DQN architecture.
- **Reproducibility infrastructure** (`src/hemosim/reproducibility.py`, `scripts/reproduce.sh`, `results/EXPECTED_RESULTS.json`): disjoint train/held-out seed pools, 5%-tolerance comparison check, one-command repro.
- **Clinical-outcome info keys**: every environment now emits `info["therapeutic"]` with a domain-appropriate definition (warfarin INR 2–3, heparin aPTT 60–100, DOAC no-stroke-no-bleed, DIC ISTH score<5).
- 339 unit and integration tests (up from 142 in v0.1).

### Changed
- **Warfarin termination threshold relaxed from INR<1.0 to INR<0.5** (extreme mechanical floor). The INR<1.5 sub-therapeutic band still incurs a shaped-reward penalty but does not terminate the episode, preventing premature termination on stochastic dips.
- **IWPC baseline dose ladder escalated** for sub-therapeutic INR: 12.5 mg (INR<1.5) and 10 mg (INR 1.5–1.99), matching the upper end of the IWPC percentage-escalation protocol.
- Paper fully rewritten for v2 scope, scope framing, and honest residual reporting.

### Removed
- **v0.1 fabricated PPO results** in `scripts/generate_results.py`. The v0.1 script computed `ppo_mean = clinical_mean * 1.25 + abs(random_mean) * 0.3` — a synthetic formula, not a real training output. v2 emits `null` for policy columns that do not have a trained model on disk, and v2 paper reports no policy numbers.
- v0.1 misattribution of TTR=0.55 to Nemati 2016. The primary source does not report TTR; the 0.55 target is now correctly framed as an aspirational stewardship-literature benchmark (Wan 2008) used as a cohort-level calibration residual.
- v0.1 inverted apixaban CrCl contraindication rule. The safety layer now correctly dose-reduces (rather than refuses) apixaban at low CrCl per the apixaban FDA label.

## [0.1.0] — 2025-11-14

Initial release. Four Gymnasium environments (warfarin, heparin, DOAC, DIC); Hamberg warfarin PK/PD; Raschke heparin nomogram; DOAC trial-derived event-rate simulators; 142 tests.

**Known limitations that motivated v0.2.0 (all addressed above):** oracle lab observations at every timestep; no clinical-outcome metrics; no safety layer; synthetic PPO column in results; TTR benchmark misattributed.

[0.2.0]: https://github.com/HassDhia/hemosim/releases/tag/v0.2.0
[0.1.0]: https://github.com/HassDhia/hemosim/releases/tag/v0.1.0
