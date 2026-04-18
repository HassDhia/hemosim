# PostFeedback — Widen Consistency Reviewer coverage to catch string-field drift and calibration-vs-test path drift

**Date:** 2026-04-18
**Origin:** hemosim v0.2.1 Science-Skill Report T1.5 + SR-M Phase C investigation (this session)
**Severity:** HIGH (mechanism of capture for two classes of defect that shipped through 4 review passes)

## Two missed defects, same root cause

### Defect A: Nemati → Wan string-field drift across code + JSON (T1.5 from v0.2.1 Science report)

Across v0.2.0 → v0.2.3 Council passes, the calibration harness source at `src/hemosim/validation/published_calibration.py` and its emitted `results/published_calibration.json` both continued to carry `trial="Nemati 2016 (EMBC)"`, `endpoint="TTR on standard Raschke nomogram in MIMIC-II cohort"`, and a full Nemati citation on what the paper §8 had (correctly) reframed as a Wan-2008 antithrombotic-stewardship target. Council's Consistency Reviewer did not flag this because its prompt focuses on **numeric-field** cross-deliverable consistency (test count, version string, metric values, URLs). The drift was in **string fields** (trial, endpoint, citation), and string fields were not in the Consistency Reviewer's scan surface.

### Defect B: Calibration-harness vs SR-M-test measurement-path divergence (this session)

During SR-M Phase B test authoring, the initial Phase B test for `heparin_raschke_aptt` used `gym.make("hemosim/HeparinInfusion-v0")` + `HeparinRaschkeBaseline.predict(obs)` + one 6h env.step, measuring aPTT from info dict. This path produced aPTT = 193 s vs the calibration harness's `_simulate_raschke()` path which uses `HeparinPKPD(weight=80)` with fitted params + 1h dt integration, producing aPTT = 75 s (residual 0.000). Same declared claim, different measurement paths, different answers. Neither Council nor the SR-M2 structural check caught this because both focused on test existence, not measurement-path coherence with the calibration harness.

## Proposed new gate family: CL (Consistency Long) — string-field + measurement-path scans

### CL4 — String-field consistency across code + JSON + paper

For every `PublishedBenchmark` in calibration-harness source and emitted JSON, grep the paper for mentions of the `trial=` and `citation=` strings. Any calibration benchmark whose `trial` string is not referenced in the paper (or whose referenced string differs from the code's) is a CL4 finding. Equivalent check on the opposite direction: any citation the paper makes to a calibration target that does not appear in the calibration harness source.

```typescript
const CL_GATES: Gate[] = [
  // ... existing CL1..CL3
  {
    id: "CL4",
    name: "Calibration benchmark trial/citation strings match paper narrative",
    severity: "HIGH",
    phase: ["review", "publish", "all"],
    run: (ctx) => {
      // 1. Parse all PublishedBenchmark(...) invocations from validation/*.py
      // 2. Extract trial= and citation= string literals
      // 3. For each, grep the paper .tex for a match or synonym
      // 4. Report mismatches as CL4 findings
      ...
    },
  },
];
```

### CL5 — Multi-path measurement consistency

For any declared claim that can be measured via more than one code path (calibration harness, gym env step, unit-test ad-hoc construction, reproduce script, notebook demo, …), the gate verifies that **all extant measurement paths produce the same numeric answer within a declared tolerance**. The heparin v0.2.1 Phase-C finding was the canonical case — `published_calibration.py::_simulate_raschke()` produced aPTT = 75 s while `gym.make("hemosim/HeparinInfusion-v0").step()` produced 193 s for what the registry called the same claim — but the gate generalizes beyond calibration harnesses. The canonical measurement path for any given claim is the one tied to the published numeric (usually the calibration harness or the `results/*.json` emitter); other paths must either reproduce that answer or be declared non-canonical in code comments.

Heuristic implementation: for each claim, enumerate every code path that computes the declared quantity (grep for the quantity name in `src/`, `tests/`, `scripts/`, `notebooks/`), execute each, and assert pairwise consistency within tolerance. Flag any path that differs from the canonical by > declared tolerance.

## Not in scope for this entry

- This is not an SR-M gate. SR-M is mechanistic-claim falsification. CL4/CL5 are cross-deliverable consistency — they belong in the CL (cross-linking / consistency) family that already owns CL1–CL3.
- This does not retrospectively re-open v0.2.2. The Nemati→Wan drift was fixed in Phase C of this SR-M pass; CL4/CL5 is the prospective gate that would have caught it.

## Next steps

1. Implement CL4 + CL5 in `Tools/research-verify-gates.ts` as part of AppliedResearch skill's Phase G commit OR as a follow-up PR.
2. Update `ReviewChecklist.md` §Cross-Linking (CL) with CL4 + CL5 entries.
3. Update Consistency Reviewer prompt (Review.md Agent 4) to explicitly call out string-field + calibration-vs-test path coverage.
