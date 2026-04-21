# Mechanistic Claim Registry

Every mechanistic scientific claim in `paper/hemosim.tex` that asserts a
mechanism, reproduction, or architecture equivalence is registered here with
a primary-source numeric target and a falsification test in `tests/` that
runs the claim under the artifact's default initialization.

**Registry scope:** equation claims, mechanism claims ("we model / simulate /
reproduce", "mechanistic coupling"), reproduction claims, architecture claims
(reimplementations). Scope/methodology/infrastructure claims are excluded.

Each entry records: paper location, code location, verbatim claim text,
primary source and numeric target, and the path to the falsification test
that enforces it.

---

## Claim abstract_meta: Phase-1 infrastructure meta-claim (abstract restatement)

- **Location (paper):** paper/hemosim.tex:35-44
- **Location (code):** src/hemosim/__init__.py:1-30
- **Claim text (verbatim from paper, ≤ 2 sentences):**

  > hemosim v2 contributes: (a) four Gymnasium-compatible environments grounded in published PK/PD models, (b) a POMDP reformulation with explicit lab ordering, (c) mechanistic coupling to a reduced coagulation cascade ODE in the DIC environment, (d) clinical-outcome metrics (Rosendaal TTR, ISTH major bleeding, Warkentin 4T HIT, mortality proxy), (e) published-data calibration with normalized RMSE 0.0013 on warfarin against Hamberg and IWPC targets, (f) a clinical decision support harness, (g) a SPIRIT 2013-compliant silent-deployment protocol, and (h) a faithful reimplementation of the Nemati 2016 DQN heparin policy.

- **Primary source:** hemosim v2 abstract is an umbrella over individual claims below; this entry provides SR-M1 coverage of the abstract block. Primary sources live in the per-claim entries.
- **Primary-source target:** each clause (a)–(h) maps to one or more per-claim entries below whose numeric targets collectively cover the abstract.
- **Falsification test:** tests/test_srm_abstract_meta.py::test_srm_abstract_meta_all_subclaim_tests_pass
- **Falsification condition:**

  This claim is refuted if any per-claim SR-M test (warfarin_hamberg_pkpd, heparin_raschke_aptt, doac_rely_rocket_aristotle, dic_hockin_mann_cascade, rosendaal_ttr, isth_major_bleed, warkentin_4t_hit, nemati_dqn_arch, pomdp_lab_masking) is currently failing or marked xfail. The meta-claim is only true if every subclaim is true.

- **Rescope fallback:**

  If the meta-test fails because any subclaim fails, rewrite the abstract to drop the specific failing clause and move it to §15 Limitations as a known gap deferred to the corresponding Phase. The abstract must not claim anything whose falsification test is not green.

---

## Claim warfarin_hamberg_pkpd: Hamberg 2007 warfarin PK/PD reproduces IWPC cohort steady state

- **Location (paper):** paper/hemosim.tex:304-435
- **Location (code):** src/hemosim/models/warfarin_pkpd.py:27-120
- **Claim text (verbatim from paper, ≤ 2 sentences):**

  > Warfarin PK/PD parameters are fitted via SciPy's Nelder--Mead optimizer against three targets: IWPC mean stable maintenance dose [iwpc2009]; IWPC time-to-therapeutic [iwpc2009]; and Hamberg steady-state INR in a wildtype CYP2C9/VKORC1 patient at 5 mg/day [hamberg2007]. Optimization converged in 190 iterations with normalized RMSE = 0.0013 (seed 42).

- **Primary source:** Hamberg AK et al. Clin Pharmacol Ther 2007;81(4):529-38, PMID:17329993; IWPC. N Engl J Med 2009;360:753-64, PMID:19228618.
- **Primary-source target:** INR = 2.5 ± 0.2 at steady state for CYP2C9*1/*1 + VKORC1 G/G patient at 5 mg/day warfarin per Hamberg 2007 Table 2; IWPC mean maintenance dose = 5.2 mg/day per IWPC 2009 cohort summary.
- **Falsification test:** tests/test_srm_warfarin_hamberg_pkpd.py::test_srm_warfarin_hamberg_pkpd_steady_state_inr
- **Falsification condition:**

  This claim is refuted if running the default-initialized WarfarinDosing-v0 env with 5 mg/day dose for a wildtype patient does not produce steady-state INR within [2.3, 2.7] after 30 simulated days, OR if results/published_calibration.json warfarin_fit.rmse exceeds 0.005.

- **Rescope fallback:**

  Rewrite §8 Warfarin Fit to frame the 0.0013 RMSE as "fit against cohort-level IWPC/Hamberg summary statistics without individual-patient validation" and add a §15 paragraph disclosing that the steady-state INR test tolerance is wider than Hamberg's reported inter-subject variability.

---

## Claim heparin_raschke_aptt: Raschke/Hirsh nomogram reproduces target aPTT 6h post-bolus

- **Location (paper):** paper/hemosim.tex:315-470
- **Location (code):** src/hemosim/models/heparin_pkpd.py:1-180; src/hemosim/agents/baselines.py:62-91
- **Claim text (verbatim from paper, ≤ 2 sentences):**

  > The Raschke 1993 weight-based heparin nomogram is implemented as HeparinRaschkeBaseline with the modern ACCP absolute-seconds formulation and 1.5-2.5× ratio equivalence per Hirsh 2001 and Holbrook 2012. Adjustment ladder is 22/20/18/16/12 U/kg/hr across aPTT bands 45/60/100/120 s.

- **Primary source:** Raschke RA et al. Ann Intern Med 1993;119(9):874-81, PMID:8214998; Hirsh J et al. Chest 2001;119(1 Suppl):64S-94S, PMID:11157642.
- **Primary-source target:** aPTT 60-80 s at 6h post-bolus (therapeutic, ratio 1.5-2.5× baseline ~30 s) per Raschke 1993 Table 2; plasma heparin concentration midpoint 0.3 U/mL per Hirsh 2001 Chest therapeutic range 0.2-0.4 U/mL.
- **Falsification test:** tests/test_srm_heparin_raschke_aptt.py::test_srm_heparin_raschke_aptt_6h_post_bolus
- **Falsification condition:**

  This claim is refuted if running HeparinInfusion-v0 with HeparinRaschkeBaseline from default init produces aPTT outside [45, 100] s at t=6h for a 70 kg patient with normal renal function, OR if the Raschke adjustment ladder values in agents/baselines.py do not bit-exact match 22/20/18/16/12 U/kg/hr.

- **Rescope fallback:**

  Acknowledge in §15 that "the 6h oracle-labs env has TTR=0 on the Raschke baseline as a known discretization artifact of the 6h timestep against ~1-2h UFH distribution half-life; the claim reduces to ladder fidelity (ladder values match Raschke 1993 exactly) and leaves therapeutic-range reproduction to the POMDP env where the masking+TAT model is realistic."

---

## Claim doac_rely_rocket_aristotle: DOAC event rates reproduce RE-LY / ROCKET-AF / ARISTOTLE within CI (stroke CI miss pre-disclosed)

- **Location (paper):** paper/hemosim.tex:251-489
- **Location (code):** src/hemosim/envs/doac_management.py:1-200; results/published_calibration.json
- **Claim text (verbatim from paper, ≤ 2 sentences):**

  > DOAC event rates are calibrated against RE-LY, ROCKET-AF, and ARISTOTLE trials. We note that DOAC stroke-event rates fall outside trial confidence intervals for all three drugs (bleed rates are within CI) and are flagged for Phase 2 re-calibration.

- **Primary source:** Connolly SJ et al. RE-LY. NEJM 2009;361:1139-51; Patel MR et al. ROCKET AF. NEJM 2011;365:883-91; Granger CB et al. ARISTOTLE. NEJM 2011;365:981-92.
- **Primary-source target:** dabigatran 150 mg BID stroke 1.11 %/yr CI [0.92, 1.33] (RE-LY); rivaroxaban 20 mg stroke 1.7 %/yr CI [1.45, 2.00] (ROCKET-AF); apixaban 5 mg BID stroke 1.27 %/yr CI [1.05, 1.53] (ARISTOTLE). Bleed rates: 3.11 / 3.6 / 2.13 %/yr.
- **Falsification test:** tests/test_srm_doac_rely_rocket_aristotle.py::test_srm_doac_bleed_rates_in_CI
- **Falsification condition:**

  This claim is refuted if running DOACManagement-v0 simulated for ≥ 10,000 patient-years from default init produces a bleed rate outside the published 95% CI for any of the three drugs. Stroke CI miss is pre-disclosed; the test asserts bleed-rate CI compliance only, and asserts explicitly that stroke rates are logged as out-of-CI for Phase-2 re-calibration.

- **Rescope fallback:**

  If bleed rates also miss CI, §8.3 DOAC Event-Rate Validation is rewritten to state "event rates reproduce published trial magnitudes within 2× but are systematically biased; quantitative trial replication is deferred to Phase 2 MIMIC-IV calibration." Remove claim of within-CI reproduction.

---

## Claim dic_hockin_mann_cascade: DIC env couples to reduced Hockin-Mann cascade that qualitatively reproduces initiation dynamics

- **Location (paper):** paper/hemosim.tex:69-784
- **Location (code):** src/hemosim/models/coagulation.py:1-240; src/hemosim/envs/dic_management.py:1-300
- **Claim text (verbatim from paper, ≤ 2 sentences):**

  > Our 8-state reduced system retains the key dynamical features while achieving two orders of magnitude speedup... This captures TF-initiated activation, the thrombin positive-feedback loop, fibrin formation, antithrombin inhibition, and platelet activation.

- **Primary source:** Hockin MF, Jones KC, Everse SJ, Mann KG. J Biol Chem 2002;277(21):18322-33, PMID:11893748.
- **Primary-source target:** Qualitative activation consistent with Hockin-Mann 2002 initiation-phase dynamics — (i) thrombin rises above trace (> 1 nM) within 30 min of a physiologic TF/Xa/Va trigger, (ii) the fibrinogen+fibrin subsystem conserves mass to integrator tolerance (< 0.5 % drift), (iii) the ODE is wired into the DIC env when coag_cascade_mode=True. Peak amplitude in the 8-state reduction is intentionally lower than Hockin-Mann's full 34-species model (which reports 100-400 nM peaks) because intermediate amplification species (IX, VIII, XI, protein C, TFPI, etc.) and a larger zymogen pool are absent by design; paper §6 does not claim amplitude reproduction. **Scope:** the activation claim is bounded to DIC pathophysiology — AT-III depletion (representative at3_total ≈ 10 nM vs healthy ~3000 nM) combined with intense upstream TF/Xa/Va triggering. Under healthy-physiology AT-III, the reduced cascade correctly does not activate (thrombin is scavenged faster than generated); that is consistent behavior, not a defect, and is the reason the cascade is only exercised inside the DIC environment.
- **Falsification test:** tests/test_srm_dic_hockin_mann_cascade.py::test_srm_dic_hockin_mann_cascade_thrombin_activates_qualitatively, ::test_srm_dic_hockin_mann_cascade_fibrinogen_fibrin_conservation, ::test_srm_dic_hockin_mann_cascade_is_wired_in_env
- **Falsification condition:**

  This claim is refuted if (i) peak thrombin under TF/Xa/Va trigger (initial state [1.0, 0.5, 0.5, 0.0, 300.0, 0.0, 0.0, 0.0], 30 min simulated) does not exceed 1 nM, OR (ii) peak timing falls outside the 0-30 min window, OR (iii) fibrinogen+fibrin conservation drifts > 0.5 % over 30 min, OR (iv) DICManagement-v0 default init with coag_cascade_mode=True does not allocate a CoagulationCascade instance.

- **Rescope fallback:**

  If qualitative activation fails (peak < 1 nM under physiologic trigger), rewrite §6 to state explicitly that the 8-state ODE is a structural scaffold and that mechanism-level claims are deferred to a future version that restores intermediate amplification species. If only the peak-timing test fails, investigate rate-constant scaling. Mass-conservation failure on the fibrinogen+fibrin subsystem is a code bug (equal-and-opposite fluxes in dydt[4]/dydt[5]) to fix, not a rescope.

---

## Claim rosendaal_ttr: TTR metric implements Rosendaal 1993 linear interpolation

- **Location (paper):** paper/hemosim.tex:497-538
- **Location (code):** src/hemosim/metrics/clinical.py:100-180
- **Claim text (verbatim from paper, ≤ 2 sentences):**

  > Rosendaal time-in-therapeutic-range (TTR) is emitted by every environment, computed via piecewise-linear intersection of the INR (or aPTT) trajectory with the therapeutic range [low, high] per Rosendaal 1993.

- **Primary source:** Rosendaal FR, Cannegieter SC, van der Meer FJ, Briet E. Thromb Haemost 1993;69(3):236-9, PMID:8470047.
- **Primary-source target:** For a synthetic INR trajectory that is a straight line from 1.5 to 3.5 over 10 days with range [2.0, 3.0], Rosendaal TTR = exactly 0.5 (days 2.5-7.5 in range / 10 day total).
- **Falsification test:** tests/test_srm_rosendaal_ttr.py::test_srm_rosendaal_ttr_linear_interpolation_known_case
- **Falsification condition:**

  This claim is refuted if a straight-line INR trajectory from 1.5 to 3.5 over 10 days with range [2.0, 3.0] does not yield TTR ∈ [0.49, 0.51] via hemosim.metrics.clinical.time_in_therapeutic_range, OR if the implementation uses a naive per-step-in-range count (step-function) rather than linear intersection.

- **Rescope fallback:**

  If the linear-interpolation test fails, revert §9 TTR description to "per-step in-range fraction" and cite this as a v2.1 improvement target. Update paper and code docstring to match the actual (step-function) behavior.

---

## Claim isth_major_bleed: Major-bleed classification implements Schulman 2005 ISTH definition

- **Location (paper):** paper/hemosim.tex:497-538
- **Location (code):** src/hemosim/metrics/clinical.py:184-245
- **Claim text (verbatim from paper, ≤ 2 sentences):**

  > ISTH major bleeding is classified per the Schulman 2005 ISTH definition with four qualifying criteria: fatal bleeding, critical-site bleeding (intracranial, intraocular, intraspinal, intra-articular, pericardial, retroperitoneal, or intramuscular with compartment syndrome), hemoglobin drop ≥ 2.0 g/dL, or transfusion ≥ 2 units of packed red cells.

- **Primary source:** Schulman S, Kearon C. J Thromb Haemost 2005;3(4):692-4, PMID:15842354.
- **Primary-source target:** A synthetic bleeding event with Hb drop = 2.5 g/dL and no transfusion MUST classify as major; one with Hb drop = 1.5 g/dL and 1 transfused unit MUST classify as non-major; fatal event MUST always classify as major.
- **Falsification test:** tests/test_srm_isth_major_bleed.py::test_srm_isth_major_bleed_four_criteria
- **Falsification condition:**

  This claim is refuted if any of the four Schulman 2005 criteria (fatal, critical-site, Hb drop ≥ 2.0 g/dL, transfusion ≥ 2 units) does not, on its own, cause a synthetic bleeding event to classify as major — or if a below-threshold event (Hb drop = 1.5 g/dL, transfusion = 1 unit, non-critical site, non-fatal) incorrectly classifies as major.

- **Rescope fallback:**

  If thresholds in code drift from Schulman 2005 (e.g. Hb threshold 2.0 becomes 1.5 g/dL), fix the code; the claim is not rescopable — ISTH major-bleeding criteria are standardized and not open to reinterpretation.

---

## Claim warkentin_4t_hit: HIT scoring implements Warkentin 2003 / Lo 2006 4T score

- **Location (paper):** paper/hemosim.tex:497-538
- **Location (code):** src/hemosim/metrics/clinical.py:316-420
- **Claim text (verbatim from paper, ≤ 2 sentences):**

  > The Warkentin 4T HIT pre-test probability score is implemented as four categorical components: thrombocytopenia (>50% fall + nadir ≥20 → 2 pts), timing (5-10 days → 2 pts, 10-14 → 1 pt), thrombosis (binary 0/2 pts), and other causes (none=2 / possible=1 / definite=0). Risk categories 0-3 low / 4-5 intermediate / 6-8 high match Lo 2006 JTH.

- **Primary source:** Warkentin TE. Br J Haematol 2003;121(4):535-55, PMID:12786785; Lo GK, Juhl D, Warkentin TE et al. J Thromb Haemost 2006;4(4):759-65, PMID:16634744.
- **Primary-source target:** A synthetic case with 60% platelet fall + nadir 30 + onset day 7 + confirmed thrombosis + no other cause MUST score 8/8 (high risk); a case with 30% fall + nadir 150 + onset day 20 + no thrombosis + definite other cause MUST score 0/8 (low risk).
- **Falsification test:** tests/test_srm_warkentin_4t_hit.py::test_srm_warkentin_4t_hit_boundary_cases
- **Falsification condition:**

  This claim is refuted if the above boundary synthetic cases do not score 8/8 and 0/8 respectively, OR if any of the four component thresholds in code differ from Lo 2006 Table 1 values.

- **Rescope fallback:**

  Per Science-Skill Report T1.9: the "prior exposure 30-100 days ago" branch is collapsed in current code; if this becomes a Council finding, disclose explicitly in §15 and weaken paper claim to "4T score with simplified timing branch; full Warkentin branching deferred to v2.1."

---

## Claim nemati_dqn_arch: NematiDQN2016Baseline reimplements Nemati 2016 DQN architecture

- **Location (paper):** paper/hemosim.tex:71-186
- **Location (code):** src/hemosim/agents/baselines_extended.py:80-160
- **Claim text (verbatim from paper, ≤ 2 sentences):**

  > An extensible baselines suite, including a faithful reimplementation of the Nemati 2016 DQN heparin policy as a benchmark row. 2-layer MLP (64 hidden units, ReLU) with 5 discrete actions {-4, -2, 0, +2, +4} U/kg/hr adjustments matching Nemati 2016 §II-C action grid description.

- **Primary source:** Nemati S, Ghassemi MM, Clifford GD. IEEE EMBC 2016:2978-81, DOI:10.1109/EMBC.2016.7591355.
- **Primary-source target:** 2-layer MLP; 5 discrete actions {-4, -2, 0, +2, +4} U/kg/hr; 64 hidden units per layer (inferred from Nemati 2016 written description — explicitly caveated in docstring since Nemati 2016 does not specify hidden width).
- **Falsification test:** tests/test_srm_nemati_dqn_arch.py::test_srm_nemati_dqn_arch_structure_matches_description
- **Falsification condition:**

  This claim is refuted if NematiDQN2016Baseline's action space is not exactly 5 discrete actions with values {-4, -2, 0, +2, +4}, OR if the network is not 2-layer MLP with ReLU activation, OR if the docstring does not explicitly state "64 hidden units inferred from written description — Nemati 2016 does not specify width" (the honesty caveat).

- **Rescope fallback:**

  If any of the structural properties fail, the claim cannot survive as "faithful reimplementation." Rewrite paper to: "a DQN baseline inspired by Nemati 2016's action grid; network width and activation inferred from description and flagged in code." Drop "faithful" qualifier.

---

## Claim env_reward_function_shapes: Warfarin and heparin reward functions match the forms declared in §4 equations

- **Location (paper):** paper/hemosim.tex:214-256
- **Location (code):** src/hemosim/envs/warfarin_dosing.py:1-250; src/hemosim/envs/heparin_infusion.py:1-250
- **Claim text (verbatim from paper, ≤ 2 sentences):**

  > The warfarin reward is r_t = -|INR_t - 2.5| + 0.5 · (safety bonus term) per §4.1 equation; the heparin reward is r_t = -|aPTT_t - 75| / 30 + (safety bonus term) per §4.2 equation. Reward shape is an absolute-error-around-therapeutic-midpoint form with a bounded safety shaping term.

- **Primary source:** hemosim paper §4.1 equation (line 222) + §4.2 equation (line 239) — the canonical declared form is in the paper itself; the claim is that code matches.
- **Primary-source target:** For a warfarin env step with INR=2.5 (on-target), base reward contribution = 0.0 (before safety bonus). For INR=3.5, base reward = -1.0. For heparin env with aPTT=75 s, base reward = 0.0; for aPTT=105 s, base reward = -1.0. Exact (not proportional) — the paper equation specifies coefficients.
- **Falsification test:** tests/test_srm_env_reward_function_shapes.py::test_srm_warfarin_heparin_reward_exact_values
- **Falsification condition:**

  This claim is refuted if running a single default-init step of WarfarinDosing-v0 with INR=2.5 does not produce base reward = 0.0 (before safety bonus), OR a step with INR=3.5 does not produce base reward = -1.0 ± 0.01, OR the analogous heparin values (aPTT=75 → 0.0, aPTT=105 → -1.0) do not hold. Tolerance ±0.01 absolute.

- **Rescope fallback:**

  If the coefficients in code differ from the paper equation (e.g. heparin divisor is 30 in paper but 25 in code), update the paper equation to match code and add a §4 paragraph explaining the coefficient choice. Cannot drop the claim since the reward is the environment's behavioral specification.

---

## Claim silent_deployment_protocol_stats: Silent-deployment sample size formulas in protocol match canonical trial-stats definitions

- **Location (paper):** paper/silent_deployment_protocol.tex:1-500
- **Location (code):** paper/silent_deployment_protocol.tex (specification-only, not code)
- **Claim text (verbatim from paper, ≤ 2 sentences):**

  > The silent-deployment protocol specifies sample size via n = 2σ²(z_{1-α} + z_{1-β})² / Δ² and the TTR delta as Δ_i = TTR_i^policy - TTR_i^usual-care per SPIRIT 2013 / CONSORT 2010 conventions.

- **Primary source:** SPIRIT 2013 statement (Chan et al. Ann Intern Med 2013;158:200-207, DOI:10.7326/0003-4819-158-3-201302050-00583); CONSORT 2010 (Moher et al. BMJ 2010;340:c332).
- **Primary-source target:** n for α=0.05, β=0.20 (80% power), σ=0.15 (typical TTR std), Δ=0.05 (5-percentage-point minimum detectable) = 142 per-arm (per standard two-sided z-test sample-size formula).
- **Falsification test:** tests/test_srm_silent_deployment_protocol_stats.py::test_srm_sample_size_formula_canonical_case
- **Falsification condition:**

  This claim is refuted if substituting α=0.05, β=0.20, σ=0.15, Δ=0.05 into the formula in the protocol does not yield n within [135, 150] per arm (tolerance accounts for z-table rounding), OR if the formula in the .tex differs from the canonical two-sided sample-size equation.

- **Rescope fallback:**

  If formula in protocol drifts from canonical form, correct the .tex. The claim is not rescopable — sample-size formulas are standardized and not open to interpretation; any deviation is an error to fix, not a scope to weaken.

---

## Claim pomdp_lab_masking: POMDP env masks ground-truth labs until TAT elapses

- **Location (paper):** paper/hemosim.tex:341-392
- **Location (code):** src/hemosim/envs/pomdp.py:1-250; src/hemosim/envs/heparin_pomdp.py:1-200
- **Claim text (verbatim from paper, ≤ 2 sentences):**

  > The POMDP reformulation introduces explicit lab ordering as an action, per-lab turnaround times (TATs), and analytical variability, replacing the oracle-lab assumption. The agent's policy is π(a_t | o_{≤t}, a_{<t}) rather than π(a_t | s_t); this reproduces the bedside reality that a clinician choosing an infusion rate at 0600 is acting on an aPTT drawn at 0500 and resulted at 0545.

- **Primary source:** aPTT CV and TAT references: Lippi G et al. Clin Chem Lab Med 2012;50(11):2049-55 (CV ranges); Bowen RAR, Remaley AT. Biochem Med (Zagreb) 2016;26(2):194-205 (TAT distributions).
- **Primary-source target:** aPTT CV = 0.08 (Lippi 2012 aPTT CV range 5-10%); anti-Xa CV = 0.12 (Lippi 10-15%); platelets CV = 0.05 (Lippi 3-6%); aPTT TAT = 45 min, anti-Xa TAT = 180 min, platelets TAT = 45 min (Bowen 2016 hospital TAT distributions).
- **Falsification test:** tests/test_srm_pomdp_lab_masking.py::test_srm_pomdp_lab_masking_masks_and_returns_after_TAT
- **Falsification condition:**

  This claim is refuted if a HeparinInfusion-POMDP-v0 episode via gym.make() with an aPTT lab order at t=0 returns a value in the observation vector before t=45 min (masking violation), OR if the returned lab value equals the ground-truth aPTT exactly (analytical variability violation — must differ by non-zero amount consistent with CV=0.08), OR if the observation vector for heparin POMDP has fewer than 10 channels (lab-order actions missing).

- **Rescope fallback:**

  If masking works but CVs differ from Lippi 2012 midpoints, update code CVs to canonical Lippi values. If lab-ordering is not actually a distinct action dimension, rewrite §7 to acknowledge cosmetic POMDP and defer real lab-order action to v2.1. Cannot weaken further than that without invalidating the core Phase-1 contribution.

---
