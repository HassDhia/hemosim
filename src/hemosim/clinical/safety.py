"""Clinical safety layer for the hemosim DSS (ISC-11).

This module sits between a trained RL policy (or clinical baseline
fallback) and any simulated patient action. Its single purpose is to
make sure that *nothing* leaves the DSS that would be unsafe on a real
patient per published anticoagulation guidance — dose bounds,
contraindications, and uncertainty-aware deferral to a clinician.

The layer is implemented as three cooperating objects:

* :class:`SafetyBounds` — per-drug absolute dose bounds + the
  platelet / INR contraindication thresholds.
* :class:`SafetyGuard` — applies the bounds to a
  :class:`~hemosim.clinical.dss.DosingRecommendation` against a
  :class:`~hemosim.clinical.dss.PatientSnapshot` and returns a
  :class:`SafetyCheckResult` describing whether the action is safe,
  what was violated, what the adjusted (clipped) recommendation looks
  like, and whether the system should defer to a clinician.
* :class:`SafeDSS` — convenience wrapper. ``SafeDSS(HeparinDSS(...))``
  exposes the same ``.recommend(snapshot)`` method but always returns
  a recommendation that has already been safety-checked.

Clinical sources
----------------

* Garcia DA, Baglin TP, Weitz JI, Samama MM. *Parenteral
  anticoagulants: Antithrombotic Therapy and Prevention of Thrombosis,
  9th ed: ACCP Evidence-Based Clinical Practice Guidelines.* Chest
  2012;141(2 Suppl):e24S-e43S. Anti-Xa target 0.3–0.7 IU/mL,
  weight-based Raschke nomogram. Initial UFH bolus 80 U/kg, maximum
  intermittent bolus generally not to exceed that ceiling per
  institutional protocols.
* Holbrook A, Schulman S, Witt DM, et al. *Evidence-based management
  of anticoagulant therapy.* Chest 2012;141(2 Suppl):e152S-e184S.
  Warfarin target INR 2.0–3.0 for most indications; hold-dose
  thresholds at INR > 4.5–5.0.
* Raschke RA, Reilly BM, Guidry JR, et al. *The weight-based heparin
  dosing nomogram.* Ann Intern Med 1993;119:874-881. Typical
  infusions 12–22 U/kg/hr; our absolute ceiling of 25 U/kg/hr is a
  conservative margin above the upper end of the Raschke nomogram
  ladder (not an institutional committee ceiling), consistent with
  Holbrook 2012 stewardship principles for absolute-dose guardrails.
* Warkentin TE. *Heparin-induced thrombocytopenia.* Br J Haematol
  2003;121:535-555. HIT suspected at platelets <100 x10^9/L with
  clinical features; platelets <50 x10^9/L is a commonly-used
  hold threshold for continued heparin in the absence of an
  explicit override.

Design notes
------------

1. **Clip, don't silently pass.** When a recommendation exceeds a
   dose bound the guard both clips the dose *and* emits a violation
   string so downstream audit logs are complete.
2. **Contraindications default-fail-safe.** Platelets < 50 x10^9/L
   with a heparin order triggers ``defer_to_clinician`` and sets the
   adjusted dose to zero unless ``override_hit_contraindication`` is
   passed in the snapshot's ``extra`` dict.
3. **Uncertainty-aware deferral.** If the recommendation's confidence
   is below ``min_confidence`` (default 0.60) *or* the uncertainty
   interval spans more than ``max_interval_range_fraction`` (default
   0.50) of the clinical dose range, the guard sets
   ``defer_to_clinician`` True. This matches the "decision hygiene"
   posture assumed by ACCP guidance — uncertain policies defer to a
   human.
4. **No bolus stacking.** If the prior decision was a bolus
   (``snapshot.current_rate_u_per_hr > 0`` AND the snapshot extras
   include ``last_bolus_hours_ago`` below a documented refractory
   window of 1 hour) AND the new recommendation also requests a
   bolus, the guard strips the new bolus with a violation message.
   The motivation is the well-known pharmacologic risk of stacking a
   second 80 U/kg bolus within the distribution half-life of the
   first (Hirsh et al. Chest 2001;119:64S).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from hemosim.clinical.dss import (
    BaseDSS,
    DosingRecommendation,
    PatientSnapshot,
)

# ---------------------------------------------------------------------------
# SafetyBounds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SafetyBounds:
    """Per-drug absolute dose bounds and contraindication thresholds.

    All fields default to ``None`` where not applicable to the drug in
    question. For instance, ``max_daily_mg`` is meaningful for
    warfarin but not for heparin; the two pre-built instances at the
    bottom of this module fill in the appropriate subset.

    Attributes
    ----------
    drug
        Canonical drug name (``"heparin"``, ``"warfarin"``, ...).
    dose_unit
        Human-readable unit string (``"U/hr"``, ``"mg"``).
    clinical_dose_range
        ``(min, max)`` for the dose/rate field of
        :class:`DosingRecommendation`, used by the uncertainty-interval
        heuristic to decide when to defer to a clinician.
    max_infusion_u_per_kg_hr
        Heparin only — absolute ceiling on U/kg/hr (conservative margin
        above the upper end of the Raschke 1993 nomogram ladder).
    max_bolus_u_per_kg
        Heparin only — absolute ceiling on bolus dose in U/kg.
    bolus_refractory_hours
        Heparin only — minimum time between 80 U/kg boluses. Defaults
        to 1.0 h based on the initial distribution half-life of UFH
        (Hirsh 2001).
    max_daily_mg
        Warfarin only — absolute ceiling on daily oral dose in mg.
    platelet_contraindication_threshold
        Heparin — platelet count (x10^3/uL) below which heparin
        continuation is contraindicated absent an explicit override.
    inr_contraindication_threshold_upper
        Warfarin — INR value above which further warfarin
        administration is held per Holbrook 2012.
    min_confidence
        Minimum confidence required to *not* defer to a clinician.
    max_interval_range_fraction
        Maximum uncertainty-interval width (as a fraction of
        ``clinical_dose_range``) allowed before deferring.
    """

    drug: str
    dose_unit: str
    clinical_dose_range: tuple[float, float]
    max_infusion_u_per_kg_hr: float | None = None
    max_bolus_u_per_kg: float | None = None
    bolus_refractory_hours: float | None = None
    max_daily_mg: float | None = None
    platelet_contraindication_threshold: float | None = None
    inr_contraindication_threshold_upper: float | None = None
    min_confidence: float = 0.60
    max_interval_range_fraction: float = 0.50


# Public, shared instances. These mirror the thresholds referenced in
# the module docstring and in the associated envs (e.g. HeparinInfusion
# env's MAX_INFUSION_RATE = 2500 U/hr, platelet termination at 50).
HEPARIN_SAFETY_BOUNDS = SafetyBounds(
    drug="heparin",
    dose_unit="U/hr",
    clinical_dose_range=(0.0, 2500.0),
    max_infusion_u_per_kg_hr=25.0,
    max_bolus_u_per_kg=80.0,
    bolus_refractory_hours=1.0,
    platelet_contraindication_threshold=50.0,
    min_confidence=0.60,
    max_interval_range_fraction=0.50,
)

WARFARIN_SAFETY_BOUNDS = SafetyBounds(
    drug="warfarin",
    dose_unit="mg",
    clinical_dose_range=(0.0, 15.0),
    max_daily_mg=15.0,
    inr_contraindication_threshold_upper=5.0,
    min_confidence=0.60,
    max_interval_range_fraction=0.50,
)


# ---------------------------------------------------------------------------
# SafetyCheckResult
# ---------------------------------------------------------------------------


@dataclass
class SafetyCheckResult:
    """Outcome of :meth:`SafetyGuard.check`.

    Attributes
    ----------
    is_safe
        True iff the recommendation passed every check without
        adjustment. A clipped-but-still-safe recommendation sets
        ``is_safe=False`` and returns the clipped version in
        ``adjusted_recommendation``.
    violations
        Ordered list of human-readable strings describing each failed
        check. Empty if and only if ``is_safe`` is True.
    adjusted_recommendation
        The safest recommendation the guard could produce from the
        input. If the input was already safe, this is ``None`` and the
        caller should continue to use the original recommendation.
    defer_to_clinician
        True when the guard recommends that a human clinician approve
        the dose before administration. Set by contraindications and
        by uncertainty-based triggers.
    confidence_threshold_exceeded
        True when the uncertainty-interval and/or confidence were
        *inside* safe limits (i.e. confidence >= ``min_confidence`` and
        interval fraction <= ``max_interval_range_fraction``). Named
        in the positive so a downstream dashboard can surface "policy
        is sure enough to act automatically" as a single boolean.
    """

    is_safe: bool
    violations: list[str] = field(default_factory=list)
    adjusted_recommendation: DosingRecommendation | None = None
    defer_to_clinician: bool = False
    confidence_threshold_exceeded: bool = False


# ---------------------------------------------------------------------------
# SafetyGuard
# ---------------------------------------------------------------------------


class SafetyGuard:
    """Applies a :class:`SafetyBounds` policy to a DSS recommendation.

    The guard is drug-aware: the drug identity is taken from the
    ``action_taken`` string on the recommendation if present, else
    from ``snapshot.drug``. Unknown drugs are routed to a
    ``"defer_to_clinician=True"`` result with an explicit violation
    string; no dose is ever passed through unchecked.
    """

    DRUG_BOUNDS: dict[str, SafetyBounds] = {
        "heparin": HEPARIN_SAFETY_BOUNDS,
        "warfarin": WARFARIN_SAFETY_BOUNDS,
    }

    def __init__(self, bounds: SafetyBounds | None = None) -> None:
        # Explicit bounds override drug routing (useful for tests and
        # institution-specific tweaks). When ``None`` the guard picks
        # bounds from DRUG_BOUNDS at ``check`` time.
        self._bounds = bounds

    # ------------------------------------------------------------------
    def _resolve_bounds(
        self,
        recommendation: DosingRecommendation,
        snapshot: PatientSnapshot,
    ) -> SafetyBounds | None:
        if self._bounds is not None:
            return self._bounds
        drug = _drug_of(recommendation, snapshot)
        return self.DRUG_BOUNDS.get(drug)

    # ------------------------------------------------------------------
    def check(
        self,
        recommendation: DosingRecommendation,
        snapshot: PatientSnapshot,
    ) -> SafetyCheckResult:
        """Validate ``recommendation`` against ``snapshot`` and return outcome."""
        bounds = self._resolve_bounds(recommendation, snapshot)
        if bounds is None:
            return SafetyCheckResult(
                is_safe=False,
                violations=[
                    f"unknown_drug: cannot safety-check "
                    f"action_taken={recommendation.action_taken!r}"
                ],
                adjusted_recommendation=_zero_out(recommendation),
                defer_to_clinician=True,
                confidence_threshold_exceeded=False,
            )

        violations: list[str] = []
        # Start from a copy that we can mutate; we will decide at the
        # end whether to return it as ``adjusted_recommendation``.
        adjusted_dose = float(recommendation.dose_or_rate)
        adjusted_bolus_given = bool(recommendation.bolus_given)
        adjusted_bolus_u_per_kg = float(recommendation.bolus_u_per_kg)
        defer = False

        weight_kg = float(snapshot.weight_kg) if snapshot.weight_kg else 80.0

        # (a) absolute dose bounds — clip and flag.
        dose_min, dose_max = bounds.clinical_dose_range
        if adjusted_dose < dose_min:
            violations.append(
                f"dose_below_min: {adjusted_dose:.3f} {bounds.dose_unit} "
                f"< min {dose_min:.3f} {bounds.dose_unit}"
            )
            adjusted_dose = dose_min
        if adjusted_dose > dose_max:
            violations.append(
                f"dose_above_max: {adjusted_dose:.3f} {bounds.dose_unit} "
                f"> max {dose_max:.3f} {bounds.dose_unit}"
            )
            adjusted_dose = dose_max

        # Heparin-specific: U/kg/hr ceiling and bolus checks.
        if bounds.drug == "heparin":
            max_rate_u_per_hr = (
                bounds.max_infusion_u_per_kg_hr * weight_kg
                if bounds.max_infusion_u_per_kg_hr is not None
                else dose_max
            )
            if adjusted_dose > max_rate_u_per_hr:
                violations.append(
                    f"heparin_rate_above_u_per_kg_hr_ceiling: "
                    f"{adjusted_dose:.1f} U/hr > "
                    f"{max_rate_u_per_hr:.1f} U/hr "
                    f"(ceiling {bounds.max_infusion_u_per_kg_hr:.1f} U/kg/hr)"
                )
                adjusted_dose = max_rate_u_per_hr

            if adjusted_bolus_given:
                if (
                    bounds.max_bolus_u_per_kg is not None
                    and adjusted_bolus_u_per_kg > bounds.max_bolus_u_per_kg
                ):
                    violations.append(
                        f"heparin_bolus_above_u_per_kg_ceiling: "
                        f"{adjusted_bolus_u_per_kg:.1f} U/kg > "
                        f"{bounds.max_bolus_u_per_kg:.1f} U/kg"
                    )
                    adjusted_bolus_u_per_kg = float(bounds.max_bolus_u_per_kg)

                # No-bolus-stacking rule: if we were just bolused, do
                # not re-bolus within the refractory window unless the
                # existing rate is genuinely subtherapeutic. hemosim
                # snapshots mark subtherapeutic via
                # ``snapshot.extra["prior_bolus_still_subtherapeutic"]``
                # — when absent we assume the safer default and strip
                # the new bolus.
                refr = bounds.bolus_refractory_hours or 0.0
                last_bolus_hours_ago = _extras_float(
                    snapshot, "last_bolus_hours_ago", math.inf
                )
                prior_bolus_still_subtherapeutic = _extras_bool(
                    snapshot, "prior_bolus_still_subtherapeutic", False
                )
                if (
                    last_bolus_hours_ago < refr
                    and not prior_bolus_still_subtherapeutic
                ):
                    violations.append(
                        f"heparin_bolus_stacking_blocked: "
                        f"last bolus {last_bolus_hours_ago:.2f} h ago < "
                        f"refractory {refr:.2f} h (no subtherapeutic override)"
                    )
                    adjusted_bolus_given = False
                    adjusted_bolus_u_per_kg = 0.0

            # Contraindication: platelets below HIT threshold.
            plt = snapshot.platelets_k_per_ul
            if (
                bounds.platelet_contraindication_threshold is not None
                and plt is not None
                and float(plt) < float(bounds.platelet_contraindication_threshold)
            ):
                override = _extras_bool(
                    snapshot, "override_hit_contraindication", False
                )
                if not override:
                    violations.append(
                        f"heparin_contraindicated_low_platelets: "
                        f"{float(plt):.1f} x10^3/uL < "
                        f"{bounds.platelet_contraindication_threshold:.1f} "
                        f"(HIT/thrombocytopenia; no override flag set)"
                    )
                    adjusted_dose = 0.0
                    adjusted_bolus_given = False
                    adjusted_bolus_u_per_kg = 0.0
                    defer = True

        # Warfarin-specific.
        if bounds.drug == "warfarin":
            if (
                bounds.max_daily_mg is not None
                and adjusted_dose > bounds.max_daily_mg
            ):
                violations.append(
                    f"warfarin_above_max_daily_mg: "
                    f"{adjusted_dose:.2f} mg > {bounds.max_daily_mg:.2f} mg"
                )
                adjusted_dose = float(bounds.max_daily_mg)

            inr = snapshot.inr
            if (
                bounds.inr_contraindication_threshold_upper is not None
                and inr is not None
                and float(inr) >= float(
                    bounds.inr_contraindication_threshold_upper
                )
            ):
                violations.append(
                    f"warfarin_contraindicated_high_inr: "
                    f"INR {float(inr):.2f} >= "
                    f"{bounds.inr_contraindication_threshold_upper:.2f} "
                    f"(hold warfarin per Holbrook 2012)"
                )
                adjusted_dose = 0.0
                defer = True

        # Uncertainty-aware deferral.
        lo, hi = recommendation.uncertainty_interval
        dose_range_width = max(dose_max - dose_min, 1e-9)
        interval_frac = max(hi - lo, 0.0) / dose_range_width
        confidence_ok = (
            recommendation.confidence >= bounds.min_confidence
            and interval_frac <= bounds.max_interval_range_fraction
        )
        if not confidence_ok:
            defer = True
            if recommendation.confidence < bounds.min_confidence:
                violations.append(
                    f"low_confidence: {recommendation.confidence:.3f} < "
                    f"{bounds.min_confidence:.3f}"
                )
            if interval_frac > bounds.max_interval_range_fraction:
                violations.append(
                    f"wide_uncertainty_interval: "
                    f"{interval_frac:.3f} of clinical range > "
                    f"{bounds.max_interval_range_fraction:.3f}"
                )

        any_adjustment = (
            adjusted_dose != recommendation.dose_or_rate
            or adjusted_bolus_given != recommendation.bolus_given
            or adjusted_bolus_u_per_kg != recommendation.bolus_u_per_kg
        )
        is_safe = (not violations) and (not any_adjustment) and (not defer)

        if any_adjustment or defer or violations:
            adjusted = DosingRecommendation(
                action_taken=recommendation.action_taken,
                dose_or_rate=adjusted_dose,
                uncertainty_interval=recommendation.uncertainty_interval,
                top_feature_contributions=list(
                    recommendation.top_feature_contributions
                ),
                confidence=recommendation.confidence,
                rationale=(
                    f"{recommendation.rationale} | safety_guard:"
                    f"{'adjusted' if any_adjustment else 'flagged'}"
                ),
                bolus_given=adjusted_bolus_given,
                bolus_u_per_kg=adjusted_bolus_u_per_kg,
            )
        else:
            adjusted = None

        return SafetyCheckResult(
            is_safe=is_safe,
            violations=violations,
            adjusted_recommendation=adjusted,
            defer_to_clinician=defer,
            confidence_threshold_exceeded=confidence_ok,
        )


# ---------------------------------------------------------------------------
# SafeDSS
# ---------------------------------------------------------------------------


class SafeDSS:
    """Wraps a :class:`BaseDSS` with a :class:`SafetyGuard`.

    ``SafeDSS(HeparinDSS(...)).recommend(snapshot)`` returns a
    :class:`DosingRecommendation` that is guaranteed to have been
    run through the safety guard. When the guard produced an
    adjusted recommendation, that adjusted version is returned; when
    the guard flagged an uncertainty-based deferral but did not
    change the dose, the returned recommendation's ``rationale`` is
    annotated ``| safety_guard:flagged``. The safety check itself is
    available via :meth:`check_last`.

    Also exposes :meth:`recommend_with_check` for callers who want
    the ``SafetyCheckResult`` alongside the recommendation in one
    call — the typical pattern for ICU audit logs.
    """

    def __init__(
        self,
        dss: BaseDSS,
        guard: SafetyGuard | None = None,
    ) -> None:
        self._dss = dss
        self._guard = guard if guard is not None else SafetyGuard()
        self._last_check: SafetyCheckResult | None = None

    # ------------------------------------------------------------------
    @property
    def dss(self) -> BaseDSS:
        """The underlying un-guarded DSS (read-only)."""
        return self._dss

    @property
    def guard(self) -> SafetyGuard:
        """The safety guard instance (read-only)."""
        return self._guard

    # ------------------------------------------------------------------
    def check_last(self) -> SafetyCheckResult | None:
        """Return the most recent :class:`SafetyCheckResult`."""
        return self._last_check

    # ------------------------------------------------------------------
    def recommend(self, snapshot: PatientSnapshot) -> DosingRecommendation:
        rec, _ = self.recommend_with_check(snapshot)
        return rec

    # ------------------------------------------------------------------
    def recommend_with_check(
        self,
        snapshot: PatientSnapshot,
    ) -> tuple[DosingRecommendation, SafetyCheckResult]:
        """Run the DSS then the guard and return both artifacts.

        If the guard produced an ``adjusted_recommendation``, that is
        what is returned as the first tuple element; otherwise the
        raw DSS recommendation is returned. The :class:`SafetyCheckResult`
        is always returned and should be logged for audit.
        """
        raw = self._dss.recommend(snapshot)
        result = self._guard.check(raw, snapshot)
        self._last_check = result
        if result.adjusted_recommendation is not None:
            return result.adjusted_recommendation, result
        return raw, result

    # ------------------------------------------------------------------
    def explain(
        self,
        snapshot: PatientSnapshot,
        recommendation: DosingRecommendation | None = None,
    ) -> list[tuple[str, float]]:
        """Proxy to the underlying DSS's ``explain``."""
        return self._dss.explain(snapshot, recommendation)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _drug_of(
    recommendation: DosingRecommendation,
    snapshot: PatientSnapshot,
) -> str:
    """Resolve drug identity from recommendation.action_taken or snapshot.drug."""
    token = str(recommendation.action_taken or "").lower()
    if "heparin" in token:
        return "heparin"
    if "warfarin" in token:
        return "warfarin"
    if snapshot.drug:
        return str(snapshot.drug).lower()
    return "unknown"


def _zero_out(recommendation: DosingRecommendation) -> DosingRecommendation:
    """Return a zeroed copy of ``recommendation`` used for unknown-drug fallback."""
    return DosingRecommendation(
        action_taken=recommendation.action_taken,
        dose_or_rate=0.0,
        uncertainty_interval=recommendation.uncertainty_interval,
        top_feature_contributions=list(recommendation.top_feature_contributions),
        confidence=recommendation.confidence,
        rationale=f"{recommendation.rationale} | safety_guard:unknown_drug",
        bolus_given=False,
        bolus_u_per_kg=0.0,
    )


def _extras_float(snapshot: PatientSnapshot, key: str, default: float) -> float:
    extras: dict[str, Any] = snapshot.extra or {}
    if key in extras:
        try:
            return float(extras[key])
        except (TypeError, ValueError):
            return default
    return default


def _extras_bool(snapshot: PatientSnapshot, key: str, default: bool) -> bool:
    extras: dict[str, Any] = snapshot.extra or {}
    if key in extras:
        return bool(extras[key])
    return default
