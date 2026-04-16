"""Tests for the clinical outcomes metrics module.

Covers the pure functions in ``hemosim.metrics.clinical``:
Rosendaal TTR, ISTH 2005 major bleeding, thromboembolic events,
Warkentin 4T HIT score, mortality proxy, and the episode summary.
"""

from __future__ import annotations

import pytest

from hemosim.metrics import clinical
from hemosim.metrics.clinical import (
    hit_4t_score,
    isth_major_bleeding,
    mortality_proxy,
    patient_outcome_summary,
    thromboembolic_events,
    time_in_therapeutic_range,
)


# ---------------------------------------------------------------------------
# Rosendaal time-in-therapeutic-range
# ---------------------------------------------------------------------------


class TestTimeInTherapeuticRange:
    def test_constant_in_range_is_one(self):
        """A trajectory entirely inside [low, high] must yield TTR == 1.0."""
        values = [2.0, 2.2, 2.5, 2.4, 2.1]
        times = [0.0, 24.0, 48.0, 72.0, 96.0]
        ttr = time_in_therapeutic_range(values, times, low=2.0, high=3.0)
        assert ttr == pytest.approx(1.0, abs=1e-9)

    def test_constant_out_of_range_is_zero(self):
        """A trajectory entirely outside [low, high] must yield TTR == 0.0."""
        values = [1.0, 1.1, 1.2, 1.3]
        times = [0.0, 24.0, 48.0, 72.0]
        ttr = time_in_therapeutic_range(values, times, low=2.0, high=3.0)
        assert ttr == pytest.approx(0.0, abs=1e-9)

    def test_linear_crossing_at_midpoint(self):
        """Linear ramp from below-low to in-range crossing at t=0.5.

        Values move from INR 1.0 to INR 3.0 linearly over 24 h. The lower
        bound of 2.0 is crossed at exactly the midpoint, so TTR = 0.5.
        """
        values = [1.0, 3.0]
        times = [0.0, 24.0]
        ttr = time_in_therapeutic_range(values, times, low=2.0, high=3.0)
        assert ttr == pytest.approx(0.5, abs=1e-9)

    def test_crossing_through_full_range(self):
        """Ramp from 1.0 to 4.0: enters range at 33%, exits at 67%, TTR = 1/3."""
        values = [1.0, 4.0]
        times = [0.0, 30.0]
        ttr = time_in_therapeutic_range(values, times, low=2.0, high=3.0)
        assert ttr == pytest.approx(1.0 / 3.0, rel=1e-9)

    def test_single_measurement_returns_indicator(self):
        """With a single measurement, TTR is 1.0 if in-range else 0.0."""
        assert time_in_therapeutic_range([2.5], [0.0], 2.0, 3.0) == 1.0
        assert time_in_therapeutic_range([5.0], [0.0], 2.0, 3.0) == 0.0

    def test_empty_input_returns_zero(self):
        """Empty sequences return 0.0 (no measured time)."""
        assert time_in_therapeutic_range([], [], 2.0, 3.0) == 0.0

    def test_mismatched_lengths_raises(self):
        """values and times must have matching lengths."""
        with pytest.raises(ValueError):
            time_in_therapeutic_range([2.0, 2.5], [0.0], 2.0, 3.0)

    def test_invalid_range_raises(self):
        """low must be strictly less than high."""
        with pytest.raises(ValueError):
            time_in_therapeutic_range([2.0], [0.0], low=3.0, high=2.0)


# ---------------------------------------------------------------------------
# ISTH 2005 major bleeding
# ---------------------------------------------------------------------------


class TestISTHMajorBleeding:
    def test_fatal_event_counts(self):
        events = [{"type": "GI", "hb_drop_g_dl": 0.0, "units_transfused": 0,
                   "critical_site": False, "fatal": True}]
        result = isth_major_bleeding(events)
        assert result["count"] == 1
        assert result["rate_per_100_patient_years"] > 0

    def test_critical_site_counts(self):
        """Intracranial bleed without other criteria still qualifies."""
        events = [{"type": "intracranial", "hb_drop_g_dl": 0.0,
                   "units_transfused": 0, "critical_site": True, "fatal": False}]
        result = isth_major_bleeding(events)
        assert result["count"] == 1

    def test_hb_drop_threshold_boundary(self):
        """Hb drop of exactly 2.0 g/dL qualifies (>= threshold)."""
        at_threshold = [{"type": "GI", "hb_drop_g_dl": 2.0, "units_transfused": 0,
                         "critical_site": False, "fatal": False}]
        below = [{"type": "GI", "hb_drop_g_dl": 1.99, "units_transfused": 0,
                  "critical_site": False, "fatal": False}]
        assert isth_major_bleeding(at_threshold)["count"] == 1
        assert isth_major_bleeding(below)["count"] == 0

    def test_transfusion_threshold_boundary(self):
        """2 units transfused qualifies; 1 unit does not."""
        two_units = [{"type": "GI", "hb_drop_g_dl": 0.0, "units_transfused": 2,
                      "critical_site": False, "fatal": False}]
        one_unit = [{"type": "GI", "hb_drop_g_dl": 0.0, "units_transfused": 1,
                     "critical_site": False, "fatal": False}]
        assert isth_major_bleeding(two_units)["count"] == 1
        assert isth_major_bleeding(one_unit)["count"] == 0

    def test_minor_bleed_not_counted(self):
        events = [{"type": "epistaxis", "hb_drop_g_dl": 0.5, "units_transfused": 0,
                   "critical_site": False, "fatal": False}]
        result = isth_major_bleeding(events)
        assert result["count"] == 0

    def test_empty_events(self):
        result = isth_major_bleeding([])
        assert result["count"] == 0
        assert result["rate_per_100_patient_years"] == 0.0
        assert result["events"] == []

    def test_rate_computation_with_exposure(self):
        """When patient_years given, rate scales to per-100-patient-years."""
        events = [{"type": "GI", "hb_drop_g_dl": 3.0, "units_transfused": 0,
                   "critical_site": False, "fatal": False}]
        result = isth_major_bleeding(events, patient_years=2.0)
        # 1 event / 2 py * 100 = 50 per 100 py
        assert result["rate_per_100_patient_years"] == pytest.approx(50.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Thromboembolic events
# ---------------------------------------------------------------------------


class TestThromboembolicEvents:
    def test_aggregates_by_type(self):
        events = [
            {"type": "stroke"},
            {"type": "stroke"},
            {"type": "DVT"},
            {"type": "PE"},
        ]
        result = thromboembolic_events(events)
        assert result["count"] == 4
        assert result["by_type"]["stroke"] == 2
        assert result["by_type"]["DVT"] == 1
        assert result["by_type"]["PE"] == 1

    def test_rate_scales_to_patient_years(self):
        events = [{"type": "stroke"}]
        result = thromboembolic_events(events, patient_years=4.0)
        # 1 / 4 py * 100 = 25 per 100 py
        assert result["rate_per_100_patient_years"] == pytest.approx(25.0, rel=1e-6)

    def test_empty_events(self):
        result = thromboembolic_events([])
        assert result["count"] == 0
        assert result["rate_per_100_patient_years"] == 0.0
        assert result["by_type"] == {}


# ---------------------------------------------------------------------------
# Warkentin 4T HIT score
# ---------------------------------------------------------------------------


class TestHIT4TScore:
    def test_maximum_score_high_risk(self):
        """Drop >50%, onset day 5-10, thrombosis present, no other cause = 8/8."""
        platelets = [200.0] * 5 + [80.0] * 5  # 60% fall
        result = hit_4t_score(
            platelet_trajectory=platelets,
            timing_days_from_heparin=7.0,
            other_cause="none",
            thrombosis=True,
        )
        assert result["total_score"] == 8
        assert result["risk_category"] == "high"

    def test_minimum_score_low_risk(self):
        """No thrombocytopenia, <4 days, no thrombosis, other cause present = 0."""
        platelets = [200.0, 195.0, 198.0]
        result = hit_4t_score(
            platelet_trajectory=platelets,
            timing_days_from_heparin=1.0,
            other_cause="definite",
            thrombosis=False,
        )
        assert result["total_score"] == 0
        assert result["risk_category"] == "low"

    def test_intermediate_risk_boundary(self):
        """Total = 4 is the boundary between low (<=3) and intermediate (4-5)."""
        # 30-50% fall (1 pt), 5-10 days onset (2 pts), progressive thrombosis (2 pts -> 2 for def),
        # But we pick simpler combo. Use drop = 40% (1 pt), 5-10 days (2 pts),
        # no thrombosis (0 pts), possible other cause (1 pt) -> 4
        platelets = [200.0, 200.0, 120.0]  # 40% fall
        result = hit_4t_score(
            platelet_trajectory=platelets,
            timing_days_from_heparin=6.0,
            other_cause="possible",
            thrombosis=False,
        )
        assert result["total_score"] == 4
        assert result["risk_category"] == "intermediate"

    def test_components_reported(self):
        platelets = [200.0, 100.0]  # 50% drop
        result = hit_4t_score(
            platelet_trajectory=platelets,
            timing_days_from_heparin=7.0,
            other_cause="none",
            thrombosis=True,
        )
        assert set(result["components"]) == {
            "thrombocytopenia", "timing", "thrombosis", "other_causes"
        }
        # Each component is 0, 1, or 2 points
        for pts in result["components"].values():
            assert 0 <= pts <= 2

    def test_empty_trajectory_scores_zero_thrombocytopenia(self):
        """With no platelet data, thrombocytopenia component is 0."""
        result = hit_4t_score(
            platelet_trajectory=[],
            timing_days_from_heparin=6.0,
            other_cause="none",
            thrombosis=False,
        )
        assert result["components"]["thrombocytopenia"] == 0


# ---------------------------------------------------------------------------
# Mortality proxy
# ---------------------------------------------------------------------------


class TestMortalityProxy:
    def test_output_in_unit_interval(self):
        p = mortality_proxy(
            organ_function=0.5, hemorrhage_severity=0.5,
            thrombosis=False, duration_hours=24.0,
        )
        assert 0.0 <= p <= 1.0

    def test_healthy_state_low_mortality(self):
        p = mortality_proxy(
            organ_function=1.0, hemorrhage_severity=0.0,
            thrombosis=False, duration_hours=24.0,
        )
        assert p < 0.1

    def test_severe_state_high_mortality(self):
        p = mortality_proxy(
            organ_function=0.1, hemorrhage_severity=1.0,
            thrombosis=True, duration_hours=168.0,
        )
        assert p > 0.5

    def test_thrombosis_increases_risk(self):
        base = mortality_proxy(0.6, 0.3, thrombosis=False, duration_hours=48.0)
        with_thrombus = mortality_proxy(0.6, 0.3, thrombosis=True, duration_hours=48.0)
        assert with_thrombus > base


# ---------------------------------------------------------------------------
# Episode summary
# ---------------------------------------------------------------------------


class TestPatientOutcomeSummary:
    def test_summary_has_all_sections(self):
        info_trajectory = [
            {
                "time_hours": t,
                "inr": 2.0 + 0.01 * t,
                "inr_low": 2.0,
                "inr_high": 3.0,
                "platelets": 200.0 - 1.0 * t,
                "bleeding_events": [],
                "thromboembolic_events": [],
                "organ_function": 0.9,
                "hemorrhage_severity": 0.0,
                "active_thrombosis": False,
            }
            for t in range(0, 96, 24)
        ]
        summary = patient_outcome_summary(info_trajectory)
        assert "ttr" in summary
        assert "major_bleeding" in summary
        assert "thromboembolic" in summary
        assert "mortality_proxy" in summary
        assert "duration_hours" in summary
        assert summary["duration_hours"] == pytest.approx(72.0, abs=1e-9)

    def test_empty_trajectory_returns_null_summary(self):
        summary = patient_outcome_summary([])
        assert summary["duration_hours"] == 0.0
        assert summary["major_bleeding"]["count"] == 0
        assert summary["thromboembolic"]["count"] == 0


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_module_exposes_expected_functions():
    """The metrics package exports the six documented functions."""
    for name in (
        "time_in_therapeutic_range",
        "isth_major_bleeding",
        "thromboembolic_events",
        "hit_4t_score",
        "mortality_proxy",
        "patient_outcome_summary",
    ):
        assert hasattr(clinical, name)
