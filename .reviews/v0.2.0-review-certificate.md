# Review Certificate — hemosim v0.2.0 (paper, pre-Nemati-send)

**Project:** hemosim
**Version:** 0.2.0
**Workflow:** AppliedResearch::Review (pre-Nemati-send pass)
**Iterations:** 1
**Final status:** APPROVED FOR PUBLICATION (pending one user verification — see below)
**Reviewed by:** Domain Expert Simulator (clinical pharmacologist / BMI PI persona), Consistency Reviewer (programmatic grep + MV gate)
**Date:** 2026-04-16

This certificate supersedes the v0.1 certificate (dated 2026-04-14, APPROVED). The v0.1 approval was retracted when the fabricated-PPO generation was discovered; v2 is a full rewrite with no synthetic numbers.

## Step 0 — Metric Integrity Gate: PASS

- No `std_reward == 0` over N ≥ 10 episodes
- No identical means across policies within an environment
- No broken trained-vs-random ordering (PPO is `null` by design; deferred to Phase 2)
- No derived-metric-zero across all policies

## Resolved red flags (critical + high, all 8)

| # | Issue | Location | Fix applied |
|---|-------|----------|-------------|
| 1 | Test count "142" in abstract and architecture directory listing | §1 abstract; §4 architecture | Changed to `339 unit and integration tests (up from 142 in v0.1)` |
| 2 | DOAC Table 7 caption flagged only rivaroxaban; dabigatran and apixaban stroke rates also outside CIs | §8 Table 7 caption | Rewrote caption to explicitly state all three stroke rates are outside trial CIs; distinguished miscalibrated stroke simulator from within-CI bleed simulator |
| 3 | "UCSD antithrombotic stewardship practice" claim | §10 Safety Layer | Replaced with `ACCP antithrombotic therapy guidelines (Holbrook 2012) and general principles of antithrombotic stewardship practice` |
| 4 | "UCSD-class site" for IRB submission | §14 Discussion | Replaced with `partner academic medical center` |
| 5 | Fabricated pharmacist quote addition (`— the oracle labs at every step are not what any of us actually see`) | §7 POMDP | Removed the fabricated clause; kept only the actual email quote |
| 6 | Math error: "25 U/kg/hr (Raschke plus a 25% ceiling)" — 18 × 1.25 = 22.5, not 25 | §10 Safety Layer | Replaced with `a conservative ceiling above the extremes of the standard Raschke adjustment ladder (12–22 U/kg/hr depending on aPTT response)`, cited Raschke 1993 |
| 7 | "Could not, by design, release a shared simulation environment" reads as Nemati 2016 criticism | §3 Related Work | Softened to `The paper's principled focus was learning from retrospective institutionally-held data; releasing a reusable simulation environment was outside the scope of that work.` |
| 8 | "The foundational RL anticoagulation paper" is fawning | §3 Related Work | Changed to `An early and frequently-cited RL-for-anticoagulation paper.` |
| 9 | "Shared benchmark the field has been missing since Nemati 2016" overclaim | §16 Conclusion | Changed to `the shared benchmark that has been absent in RL anticoagulation research since Nemati 2016` |

## Outstanding — USER VERIFICATION REQUIRED (not auto-resolvable, not blocking)

**Nemati 2016 TTR = 0.55 target verification.** The calibration benchmark cites `Nemati TTR on Raschke (MIMIC-II) = 0.55`. Confirm this number against the actual Nemati et al. 2016 IEEE EMBC paper ("Optimal medication dosing from suboptimal clinical examples: A deep reinforcement learning approach") before send. If the number is correct, no action. If not, the target value, Table 8 row, and related prose need updating to whatever Nemati 2016 actually reports.

**How to verify:** open the Nemati 2016 PDF, search for "time in therapeutic range" / "TTR" / "55%" / "Raschke". The relevant number should appear in the Results section or a table reporting cohort-level baseline performance.

## Verification commands (all currently passing)

```bash
# No red-flag remnants
grep -cE "UCSD|foundational RL|142 unit|oracle labs at every step are not" paper/hemosim.tex
# Expected: 0 ✓

# Paper compiles cleanly
cd paper && pdflatex -interaction=nonstopmode hemosim.tex
# Expected: 27-page PDF, zero undefined citations ✓

# Test count is honest
.venv/bin/python -m pytest tests/ -q | tail -1
# Expected: 339 passed ✓

# Metric integrity
.venv/bin/python -c "import json; r=json.load(open('results/training_results.json')); assert all(m.get('std_reward',1)!=0 for e in r['environments'].values() for m in e.values() if isinstance(m,dict) and m.get('episodes',0)>=10)"
# Expected: no exception ✓
```

## Gate decision

**APPROVED FOR PUBLICATION** contingent on user verification of the Nemati 2016 TTR figure.

The paper now reads correctly to a clinical pharmacologist and to Dr. Nemati personally:
- Test count honest (339, with v0.1 delta disclosed)
- DOAC stroke miscalibration honestly documented for all three drugs
- No UCSD institutional overreach
- No fabricated pharmacist quote
- Safety-layer dosing math correct and cited
- Attribution to Nemati 2016 accurate in scope and tone
- Conclusion scoped to anticoagulation (not all of clinical RL)

## Outreach gate

This certificate approves publication (GitHub + PyPI). It does NOT by itself open the Outreach gate. Per AppliedResearch standards §13, outreach requires a separate `preflight-certificate.md` from the PreFlight workflow with "OUTREACH GATE: OPEN" — which has not yet been run. No email should be sent to Dr. Nemati until PreFlight has produced that certificate.
