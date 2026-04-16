"""Tests for :mod:`hemosim.clinical.safety` (ISC-11).

Exercises the safety layer against the fallback clinical-baseline DSS
(Raschke 1993 / IWPC titration) — the only path available without
trained PPO weights on this machine. The layer is env-independent,
so tests here use the public API (``PatientSnapshot``,
``DosingRecommendation``) without touching the Gymnasium envs.
"""

from __future__ import annotations

import pytest

from hemosim.clinical.dss import (
    DosingRecommendation,
    HeparinDSS,
    PatientSnapshot,
    WarfarinDSS,
)
from hemosim.clinical.safety import (
    HEPARIN_SAFETY_BOUNDS,
    SafeDSS,
    SafetyBounds,
    SafetyCheckResult,
    SafetyGuard,
    WARFARIN_SAFETY_BOUNDS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rec(
    action_taken: str,
    dose: float,
    *,
    interval: tuple[float, float] | None = None,
    confidence: float = 0.9,
    bolus_given: bool = False,
    bolus_u_per_kg: float = 0.0,
    rationale: str = "test",
) -> DosingRecommendation:
    """Build a synthetic DosingRecommendation for guard testing."""
    if interval is None:
        # Narrow band around the dose so confidence is high by default.
        interval = (max(dose - 10.0, 0.0), dose + 10.0)
    return DosingRecommendation(
        action_taken=action_taken,
        dose_or_rate=dose,
        uncertainty_interval=interval,
        top_feature_contributions=[("aptt_norm", 0.1)],
        confidence=confidence,
        rationale=rationale,
        bolus_given=bolus_given,
        bolus_u_per_kg=bolus_u_per_kg,
    )


# ---------------------------------------------------------------------------
# SafetyBounds
# ---------------------------------------------------------------------------


class TestSafetyBounds:
    def test_heparin_bounds_have_expected_values(self) -> None:
        b = HEPARIN_SAFETY_BOUNDS
        assert b.drug == "heparin"
        assert b.dose_unit == "U/hr"
        assert b.max_infusion_u_per_kg_hr == pytest.approx(25.0)
        assert b.max_bolus_u_per_kg == pytest.approx(80.0)
        assert b.platelet_contraindication_threshold == pytest.approx(50.0)
        assert b.clinical_dose_range == (0.0, 2500.0)
        assert 0.0 < b.min_confidence < 1.0
        assert 0.0 < b.max_interval_range_fraction < 1.0

    def test_warfarin_bounds_have_expected_values(self) -> None:
        b = WARFARIN_SAFETY_BOUNDS
        assert b.drug == "warfarin"
        assert b.dose_unit == "mg"
        assert b.max_daily_mg == pytest.approx(15.0)
        assert b.inr_contraindication_threshold_upper == pytest.approx(5.0)
        assert b.clinical_dose_range == (0.0, 15.0)

    def test_bounds_are_frozen(self) -> None:
        """SafetyBounds is a frozen dataclass — prevent accidental mutation."""
        with pytest.raises(Exception):
            HEPARIN_SAFETY_BOUNDS.max_infusion_u_per_kg_hr = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SafetyGuard — clipping
# ---------------------------------------------------------------------------


class TestSafetyGuardClipping:
    def test_clips_heparin_rate_above_weight_adjusted_ceiling(self) -> None:
        """Rate above 25 U/kg/hr * weight must be clipped and flagged."""
        guard = SafetyGuard()
        snap = PatientSnapshot(
            drug="heparin",
            weight_kg=80.0,
            aptt_seconds=30.0,
            platelets_k_per_ul=200.0,
        )
        # 2200 U/hr on an 80 kg patient = 27.5 U/kg/hr, above the 25 ceiling.
        over = _rec("heparin_infusion", 2200.0)
        result = guard.check(over, snap)
        assert not result.is_safe
        assert result.adjusted_recommendation is not None
        # Clipped to 25 * 80 = 2000 U/hr.
        assert result.adjusted_recommendation.dose_or_rate == pytest.approx(2000.0)
        assert any("heparin_rate_above" in v for v in result.violations)

    def test_clips_heparin_rate_above_absolute_max(self) -> None:
        """Rate above the 2500 U/hr absolute ceiling must be clipped."""
        guard = SafetyGuard()
        snap = PatientSnapshot(
            drug="heparin",
            weight_kg=200.0,   # large patient: 25 U/kg/hr = 5000, well above 2500
            aptt_seconds=30.0,
            platelets_k_per_ul=200.0,
        )
        over = _rec("heparin_infusion", 4000.0)
        result = guard.check(over, snap)
        assert not result.is_safe
        assert result.adjusted_recommendation is not None
        assert result.adjusted_recommendation.dose_or_rate <= 2500.0 + 1e-6

    def test_clips_warfarin_above_max_daily(self) -> None:
        guard = SafetyGuard()
        snap = PatientSnapshot(
            drug="warfarin",
            inr=1.3,
            age_years=60.0,
        )
        # 20 mg exceeds both the 15 mg clinical cap AND the 15 mg
        # ``max_daily_mg`` bound; any clipping violation is acceptable
        # (the generic ``dose_above_max`` fires first on the
        # ``clinical_dose_range``). The important invariant is that
        # the adjusted dose is clipped to <=15 mg.
        over = _rec("warfarin_oral", 20.0)
        result = guard.check(over, snap)
        assert not result.is_safe
        assert result.adjusted_recommendation is not None
        assert result.adjusted_recommendation.dose_or_rate == pytest.approx(15.0)
        assert any(
            "dose_above_max" in v or "warfarin_above_max_daily_mg" in v
            for v in result.violations
        )

    def test_clips_warfarin_just_above_max_daily_cites_warfarin_rule(self) -> None:
        """A dose above the warfarin-specific cap but within the generic cap
        must emit the warfarin-specific violation tag (used for audit logs)."""
        # Use a custom bound that decouples clinical_dose_range from max_daily_mg.
        bounds = SafetyBounds(
            drug="warfarin",
            dose_unit="mg",
            clinical_dose_range=(0.0, 30.0),   # permissive upper
            max_daily_mg=15.0,                  # stricter cap
            inr_contraindication_threshold_upper=5.0,
        )
        guard = SafetyGuard(bounds=bounds)
        snap = PatientSnapshot(drug="warfarin", inr=1.3, age_years=60.0)
        over = _rec("warfarin_oral", 20.0)
        result = guard.check(over, snap)
        assert not result.is_safe
        assert result.adjusted_recommendation is not None
        assert result.adjusted_recommendation.dose_or_rate == pytest.approx(15.0)
        assert any("warfarin_above_max_daily_mg" in v for v in result.violations)

    def test_safe_recommendation_passes_unchanged(self) -> None:
        guard = SafetyGuard()
        snap = PatientSnapshot(
            drug="heparin",
            weight_kg=80.0,
            aptt_seconds=65.0,
            platelets_k_per_ul=200.0,
        )
        safe = _rec("heparin_infusion", 1440.0, confidence=0.9)
        result = guard.check(safe, snap)
        assert result.is_safe is True
        assert result.adjusted_recommendation is None
        assert result.defer_to_clinician is False


# ---------------------------------------------------------------------------
# SafetyGuard — contraindications
# ---------------------------------------------------------------------------


class TestSafetyGuardContraindications:
    def test_platelet_contraindication_for_heparin(self) -> None:
        """HIT: platelets < 50 must zero the dose and defer to clinician."""
        guard = SafetyGuard()
        snap = PatientSnapshot(
            drug="heparin",
            weight_kg=80.0,
            aptt_seconds=45.0,
            platelets_k_per_ul=30.0,  # below threshold
        )
        rec = _rec("heparin_infusion", 1440.0, bolus_given=True, bolus_u_per_kg=80.0)
        result = guard.check(rec, snap)
        assert result.defer_to_clinician is True
        assert result.adjusted_recommendation is not None
        assert result.adjusted_recommendation.dose_or_rate == 0.0
        assert result.adjusted_recommendation.bolus_given is False
        assert any("contraindicated_low_platelets" in v for v in result.violations)

    def test_platelet_override_flag_bypasses_contraindication(self) -> None:
        """Explicit override in extras must allow heparin even at low platelets."""
        guard = SafetyGuard()
        snap = PatientSnapshot(
            drug="heparin",
            weight_kg=80.0,
            aptt_seconds=45.0,
            platelets_k_per_ul=30.0,
            extra={"override_hit_contraindication": True},
        )
        rec = _rec("heparin_infusion", 1000.0)
        result = guard.check(rec, snap)
        # Override prevents contraindication deferral; other checks may
        # still apply but dose must not be zeroed.
        assert result.adjusted_recommendation is None or (
            result.adjusted_recommendation.dose_or_rate > 0.0
        )
        assert not any(
            "contraindicated_low_platelets" in v for v in result.violations
        )

    def test_inr_contraindication_for_warfarin(self) -> None:
        guard = SafetyGuard()
        snap = PatientSnapshot(
            drug="warfarin",
            inr=5.5,   # above 5.0 hold threshold
            age_years=65.0,
        )
        rec = _rec("warfarin_oral", 5.0)
        result = guard.check(rec, snap)
        assert result.defer_to_clinician is True
        assert result.adjusted_recommendation is not None
        assert result.adjusted_recommendation.dose_or_rate == 0.0
        assert any(
            "contraindicated_high_inr" in v for v in result.violations
        )

    def test_unknown_drug_defers_and_zeroes(self) -> None:
        guard = SafetyGuard()
        snap = PatientSnapshot(drug="aspirin")
        rec = _rec("aspirin_oral", 81.0)
        result = guard.check(rec, snap)
        assert result.defer_to_clinician is True
        assert result.adjusted_recommendation is not None
        assert result.adjusted_recommendation.dose_or_rate == 0.0
        assert any("unknown_drug" in v for v in result.violations)


# ---------------------------------------------------------------------------
# SafetyGuard — uncertainty
# ---------------------------------------------------------------------------


class TestSafetyGuardUncertainty:
    def test_low_confidence_triggers_deferral(self) -> None:
        guard = SafetyGuard()
        snap = PatientSnapshot(
            drug="heparin",
            weight_kg=80.0,
            aptt_seconds=60.0,
            platelets_k_per_ul=200.0,
        )
        rec = _rec("heparin_infusion", 1440.0, confidence=0.3)
        result = guard.check(rec, snap)
        assert result.defer_to_clinician is True
        assert result.confidence_threshold_exceeded is False
        assert any("low_confidence" in v for v in result.violations)

    def test_wide_interval_triggers_deferral(self) -> None:
        guard = SafetyGuard()
        snap = PatientSnapshot(
            drug="heparin",
            weight_kg=80.0,
            aptt_seconds=60.0,
            platelets_k_per_ul=200.0,
        )
        # Interval spanning 1500 / 2500 = 60% of clinical range -> defer.
        rec = _rec(
            "heparin_infusion",
            1440.0,
            interval=(500.0, 2000.0),
            confidence=0.95,
        )
        result = guard.check(rec, snap)
        assert result.defer_to_clinician is True
        assert any("wide_uncertainty_interval" in v for v in result.violations)

    def test_narrow_high_confidence_passes(self) -> None:
        guard = SafetyGuard()
        snap = PatientSnapshot(
            drug="heparin",
            weight_kg=80.0,
            aptt_seconds=60.0,
            platelets_k_per_ul=200.0,
        )
        rec = _rec(
            "heparin_infusion",
            1440.0,
            interval=(1400.0, 1480.0),
            confidence=0.95,
        )
        result = guard.check(rec, snap)
        assert result.confidence_threshold_exceeded is True
        assert result.defer_to_clinician is False


# ---------------------------------------------------------------------------
# Bolus stacking
# ---------------------------------------------------------------------------


class TestBolusStacking:
    def test_bolus_stripped_when_recent_bolus_given(self) -> None:
        """No-stacking rule: recent bolus strips a new bolus request."""
        guard = SafetyGuard()
        snap = PatientSnapshot(
            drug="heparin",
            weight_kg=80.0,
            aptt_seconds=45.0,
            platelets_k_per_ul=200.0,
            extra={
                "last_bolus_hours_ago": 0.25,   # 15 min ago
                "prior_bolus_still_subtherapeutic": False,
            },
        )
        rec = _rec(
            "heparin_infusion",
            1440.0,
            bolus_given=True,
            bolus_u_per_kg=80.0,
            confidence=0.9,
        )
        result = guard.check(rec, snap)
        assert result.adjusted_recommendation is not None
        assert result.adjusted_recommendation.bolus_given is False
        assert result.adjusted_recommendation.bolus_u_per_kg == 0.0
        assert any("heparin_bolus_stacking_blocked" in v for v in result.violations)

    def test_bolus_allowed_when_subtherapeutic_override_set(self) -> None:
        """Clinician override lets bolus pass even within refractory window."""
        guard = SafetyGuard()
        snap = PatientSnapshot(
            drug="heparin",
            weight_kg=80.0,
            aptt_seconds=35.0,
            platelets_k_per_ul=200.0,
            extra={
                "last_bolus_hours_ago": 0.25,
                "prior_bolus_still_subtherapeutic": True,
            },
        )
        rec = _rec(
            "heparin_infusion",
            1440.0,
            bolus_given=True,
            bolus_u_per_kg=80.0,
            confidence=0.9,
        )
        result = guard.check(rec, snap)
        stacking = [v for v in result.violations if "bolus_stacking" in v]
        assert stacking == []


# ---------------------------------------------------------------------------
# SafeDSS end-to-end
# ---------------------------------------------------------------------------


class TestSafeDSSEndToEnd:
    def test_heparin_safedss_recommend_and_check(self) -> None:
        safe = SafeDSS(HeparinDSS())
        snap = PatientSnapshot(
            drug="heparin",
            aptt_seconds=45.0,
            weight_kg=80.0,
            platelets_k_per_ul=200.0,
            renal_function=1.0,
            hours_on_therapy=0.0,
        )
        rec, check = safe.recommend_with_check(snap)
        assert isinstance(check, SafetyCheckResult)
        # Fallback baseline is well-inside bounds; expect the raw
        # recommendation to come through safely.
        assert rec.dose_or_rate > 0.0
        assert rec.dose_or_rate <= 2500.0
        # Baseline Raschke gives 18 U/kg/hr with bolus at t=0.
        assert rec.bolus_given is True
        assert check.defer_to_clinician is False or check.is_safe

    def test_heparin_safedss_defers_on_hit(self) -> None:
        safe = SafeDSS(HeparinDSS())
        snap = PatientSnapshot(
            drug="heparin",
            aptt_seconds=45.0,
            weight_kg=80.0,
            platelets_k_per_ul=30.0,   # HIT suspicion
            renal_function=1.0,
            hours_on_therapy=0.0,
        )
        rec, check = safe.recommend_with_check(snap)
        assert check.defer_to_clinician is True
        assert rec.dose_or_rate == 0.0
        assert rec.bolus_given is False

    def test_warfarin_safedss_clips_high_dose(self) -> None:
        safe = SafeDSS(WarfarinDSS())
        # Subtherapeutic INR -> IWPC baseline suggests 10 mg which is
        # within the 15 mg cap. Re-verify cap is enforced.
        snap = PatientSnapshot(
            drug="warfarin",
            inr=1.2,
            age_years=60.0,
            weight_kg=75.0,
            cyp2c9="*1/*1",
            vkorc1="GG",
            days_on_therapy=5,
        )
        rec, check = safe.recommend_with_check(snap)
        assert rec.dose_or_rate <= 15.0 + 1e-6
        assert rec.dose_or_rate >= 0.0

    def test_warfarin_safedss_defers_on_inr_over_5(self) -> None:
        safe = SafeDSS(WarfarinDSS())
        snap = PatientSnapshot(
            drug="warfarin",
            inr=6.0,                 # well above hold threshold
            age_years=60.0,
            cyp2c9="*1/*1",
            vkorc1="GG",
            days_on_therapy=20,
        )
        rec, check = safe.recommend_with_check(snap)
        assert check.defer_to_clinician is True
        assert rec.dose_or_rate == 0.0


# ---------------------------------------------------------------------------
# Custom bounds injection
# ---------------------------------------------------------------------------


class TestCustomBounds:
    def test_custom_bounds_override_heparin_max_rate(self) -> None:
        """Site-specific guard with a tighter ceiling should clip more aggressively."""
        strict_bounds = SafetyBounds(
            drug="heparin",
            dose_unit="U/hr",
            clinical_dose_range=(0.0, 2500.0),
            max_infusion_u_per_kg_hr=15.0,   # stricter than default
            max_bolus_u_per_kg=80.0,
            bolus_refractory_hours=1.0,
            platelet_contraindication_threshold=50.0,
        )
        guard = SafetyGuard(bounds=strict_bounds)
        snap = PatientSnapshot(
            drug="heparin",
            weight_kg=80.0,
            aptt_seconds=40.0,
            platelets_k_per_ul=200.0,
        )
        rec = _rec("heparin_infusion", 1440.0)   # 18 U/kg/hr
        result = guard.check(rec, snap)
        assert not result.is_safe
        assert result.adjusted_recommendation is not None
        # 15 * 80 = 1200 U/hr ceiling.
        assert result.adjusted_recommendation.dose_or_rate <= 1200.0 + 1e-6
