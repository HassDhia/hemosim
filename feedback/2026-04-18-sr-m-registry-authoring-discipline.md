# PostFeedback — SR-M registry authoring discipline (Phase-A overreach pattern)

**Date:** 2026-04-18
**Origin:** SR-M Phase C investigation (hemosim v0.2.1→v0.2.2) — 3 of 3 SR-M test failures traced to Phase-A registry overreach rather than paper overclaim, code bug, or env defect.
**Severity:** HIGH (governs correctness of every future SR-M Phase-A authoring pass across all AppliedResearch projects)

## The pattern

When authoring `.reviews/mechanistic-claims.md` in SR-M Phase A, the human (or agent) has two asymmetric sources of information:
1. The **paper** — what the authors *claim* about the mechanism
2. The **primary-source literature** — what the cited work *reports*

These routinely differ. The primary source usually reports the **strongest** numeric result from an idealized setup (full model, full initial conditions, tuned parameters). The paper claims the **weakest** result that still supports the argument being made (reduced model, default env state, honest caveats). Phase A's natural reflex is to copy the numeric target from the primary source into the registry — which then exceeds what the paper actually commits to, and the falsification test fails on a claim the paper never made.

**The v0.2.1→v0.2.2 evidence:**

| Phase-A registry said | Paper §6 actually says | Investigation outcome |
|---|---|---|
| "thrombin peak 100-400 nM at 5-15 min per Hockin-Mann 2002 Fig 3" | "retains the key dynamical features" + "captures TF-initiated activation, the thrombin positive-feedback loop…" | Qualitative claim; the 100-400 nM figure is the **full 34-species** Hockin-Mann model under full trigger, not the 8-state reduction under any condition the paper exercises |
| "prothrombin + thrombin mass conservation to integrator tolerance (< 1% drift)" | (no conservation claim anywhere in paper) | Registry invented a conservation target the paper never made, *and* chose the wrong invariant — the 8-state reduction is open by design (prothrombin is a parameter; AT-III pool is infinite) |
| "aPTT 6h post-bolus falls in therapeutic band [45, 100] s for 70 kg patient" (implicit: under default HeparinPKPD params) | "normalized RMSE = 0.0013 on [warfarin]" + heparin fit paragraph says fitted residual = 0.000 at 75 s | The paper's claim is tied to *fitted* parameters, not defaults. Registry didn't distinguish. |

All three were Phase-A authoring bugs, not paper overclaims.

## The rule: Weakest-credible-commitment authoring

**Primary-source target is the weakest numeric target the paper commits to, not the strongest the source reports.**

Operationally, when Phase A authors a `Primary-source target:` field, the discipline is:

1. **Grep the paper for the claim location first.** What does the paper literally say about this mechanism? Is the numeric claim explicit in the paper, or inferred from the cited source?
2. **If the paper's claim is qualitative** ("captures", "reproduces", "retains", "implements"), the numeric target should assert the *qualitative property* (non-zero activation, correct ordering, conservation of the one invariant the reduction preserves, etc.) — not an amplitude or tolerance pulled from the cited source.
3. **If the paper's claim references a fitted calibration**, the registry target must reference the calibration output (e.g., "residual 0.000 per `results/published_calibration.json` `heparin_fit` key") AND the test must load the fitted params, not use defaults.
4. **If the paper's claim is scoped** (to a specific env, patient cohort, pathophysiologic state), the registry must state that scope explicitly in the target (see dic_hockin_mann_cascade scope addendum).
5. **If the paper is silent on a numeric invariant** (e.g., mass conservation), the registry must not invent one. Either add an explicit paper claim first, or omit the invariant from SR-M.

## Proposed bootstrap tool update

`AppliedResearch/Tools/bootstrap-mechanistic-claims.ts` currently scaffolds registry drafts with `TODO` placeholders. The template should carry an explicit **prompt** to the human author:

```md
- **Primary-source target:** TODO — specific numeric range.

  **Discipline (weakest-credible-commitment):** This target must be the
  weakest numeric target the paper itself commits to. Do NOT import the
  strongest result from the cited primary source. If the paper makes only
  a qualitative claim, encode a qualitative invariant here (non-zero
  activation, ordering, conservation). If the paper ties the claim to a
  fitted calibration, reference the fitted output (e.g., the JSON key
  and residual), and the test must load fitted params. If the paper
  scopes the claim (env, cohort, pathophysiologic state), state the
  scope.
```

And the bootstrap tool should add a comment banner in the generated file:

```md
<!--
  SR-M Phase-A authoring discipline: the primary_source_target must be
  the weakest numeric target the paper commits to, not the strongest the
  source reports. Overreach here guarantees SR-M3 failures in Phase C
  that look like paper-level bugs but are actually registry-authoring
  bugs. See feedback/2026-04-18-sr-m-registry-authoring-discipline.md
  for the canonical worked example.
-->
```

## Ties into Agent 5 Domain Expert Simulator

Agent 5's job is to find what a peer would notice. The parallel discipline at Agent 5 is: "*would a peer in your field recognize this as a claim the paper actually makes, or as a peer-literature strawman?*" If Agent 5 reads the registry and spots an over-tight numeric target relative to paper language, that's a DES finding, and it should feed back to Phase-A authoring practice — the loop is bidirectional.

## Next steps

1. Update `bootstrap-mechanistic-claims.ts` with the discipline prompt and banner — part of AppliedResearch skill's Phase G commit.
2. Add a sentence to `SKILL.md` standard #16 (SR-M) reinforcing the weakest-credible-commitment rule.
3. Add a row to `ReviewChecklist.md` §SR-M: *"SR-M1 authoring discipline: each primary_source_target traces to a literal paper claim or fitted-calibration output; over-tight imports from cited sources are rejected."*
