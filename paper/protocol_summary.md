# SHADOW-HEP Protocol — Executive Summary

**Title:** Silent Deployment Observational Evaluation of a Reinforcement-Learning-Derived Heparin Dosing Policy Versus Usual Clinician-Directed Care in the Adult ICU

**Protocol Document:** `paper/silent_deployment_protocol.tex` (SPIRIT 2013 compliant, 33 checklist items addressed)
**Version:** 1.0 (draft) — **Status:** IRB pre-submission
**Proposed Site:** [partner academic medical center] adult ICU (placeholder pending PI confirmation)
**Target Enrollment:** 200–500 patients, planned 350 | **Study Period:** 6–12 months
**Primary Venue Target:** *The Lancet Digital Health* or *NEJM AI*

---

## The design in one paragraph

SHADOW-HEP is a prospective, single-arm, single-center, observational silent deployment study in which a Proximal Policy Optimization heparin dosing policy trained in the `hemosim` POMDP heparin environment runs in parallel with usual clinician-directed care in an adult ICU. The policy's recommendations are recorded in a research database but are never shown to clinicians and never influence patient care. The primary endpoint is within-patient non-inferiority on Rosendaal time-in-therapeutic-range (aPTT 60–100s) of the counterfactual simulated trajectory under policy-recommended doses versus the observed trajectory under actual care, with a pre-specified 10-percentage-point non-inferiority margin. A DSMB monitors for a safety signal defined as the policy recommending higher doses than actual care in the 24 hours preceding >5% of ISTH major bleeding events.

---

## Five key design choices

**1. Silent deployment (no patient-facing change).**
The policy produces recommendations that are recorded but never surfaced to bedside providers, never written to the EHR, and never influence dosing. This is the minimum-risk first-in-human study design for a novel dosing algorithm and supports a waiver of individual consent under 45 CFR 46.116(f). Every enrolled patient receives standard-of-care heparin management.

**2. Non-inferiority framing rather than superiority.**
The primary test is one-sided non-inferiority of counterfactual-policy TTR relative to actual-care TTR with a 10-percentage-point margin, α=0.025, power 0.80. With a baseline TTR of 55% (Wan 2008 antithrombotic-stewardship systematic review; Nemati 2016 itself does not report a TTR number — its primary outcome is accumulated reward under a sigmoid-shaped aPTT function — and is cited elsewhere in this protocol only for the wider RL-for-anticoagulation literature) and SD of 22 percentage points, the conservative independent-sample formula yields ~77 patients per arm; accounting for attrition and interim alpha spending, enrollment is planned at 350 with authority to extend to 500.

**3. Rosendaal TTR on aPTT as the primary endpoint.**
TTR is the most clinically meaningful continuous measure of heparin dosing quality and the metric most directly aligned with the off-policy claims in the prior RL-for-dosing literature (Nemati 2016, Raghu 2017, Komorowski 2018). Using the Rosendaal linear-interpolation method gives a rigorous, publication-standard computation. The counterfactual TTR uses the patient's own PK/PD parameters (fit through the `hemosim` calibration harness) with a pre-specified sensitivity and tipping-point analysis for parameter uncertainty.

**4. Independent DSMB with an explicit bleed-proximity stopping rule.**
The Data Safety Monitoring Board (senior critical-care physician chair, clinical pharmacologist, biostatistician, bioethicist; all free of conflicts with the sponsor or computational partner) reviews interim data at n=100 and n=250. The primary stopping trigger is counterfactual: if the policy would have recommended a higher dose than actual care in the 24 hours preceding >5% of observed major bleeds (CI lower bound above 5%), deployment is paused. This is the honest way to detect a safety signal in silent mode.

**5. Six-to-twelve-month horizon with a locked policy artifact.**
The specific PPO model file is cryptographically hashed (SHA-256 in the Manual of Operations) and cannot be retrained, fine-tuned, or substituted during the study without a formal IRB and DSMB amendment. Enrollment runs 6–12 months at one partner ICU; the full timeline from IRB approval to primary manuscript submission is ~18 months. Keeping the scope single-site and the policy locked is how this study earns the right to motivate a subsequent multi-site, patient-facing trial.

---

## Why this is drop-in-ready for Dr. Nemati's team

The protocol is written to SPIRIT 2013 specification, all 33 checklist items addressed, with explicit `[PLACEHOLDER]` callouts for PI name, IRB number, funding source, and site specifics. Every scientific choice — endpoint, margin, power, PK/PD counterfactual methodology, DSMB composition, stopping rules — is substantive and stands on its own. The IRB team can replace placeholders and submit; the biostatistician can operationalize the Statistical Analysis Plan outline (Appendix A) directly; the DSMB Charter outline (Appendix C) and CONSORT flow template (Appendix B) are ready for their Manual of Operations. Bibliography is inline (SPIRIT/Chan 2013, Nemati 2016, Raghu 2017, Komorowski 2018, Rosendaal 1993, Schulman/ISTH 2005, Lo/4T 2006, Raschke 1993, Hockin 2002, Hamberg 2007, Mueck 2007, Hirsh 2001) — no external `.bib` required, compiles cleanly with `pdflatex`.
