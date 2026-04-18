"""SR-M claim: warkentin_4t_hit

Falsification test for the Warkentin 2003 / Lo 2006 4T HIT score claim.
Constructs boundary synthetic cases: 8/8 high-risk and 0/8 low-risk, and
asserts hit_4t_score returns the exact primary-source-target scores.
"""

from __future__ import annotations

from hemosim.metrics.clinical import hit_4t_score


def test_srm_warkentin_4t_hit_boundary_cases():
    """8/8 high-risk and 0/8 low-risk boundary cases score correctly."""
    # High-risk: >50% platelet fall + nadir >= 20 (2 pts), onset day 7 (2),
    # confirmed thrombosis (2), no other cause (2) → 8/8
    high = hit_4t_score(
        platelet_trajectory=[250.0, 150.0, 80.0, 60.0],  # 76% fall, nadir 60
        timing_days_from_heparin=7.0,
        other_cause="none",
        thrombosis=True,
    )
    assert high["total_score"] == 8, (
        f"High-risk boundary case scored {high['total_score']}/8, expected 8/8. "
        f"Components: {high['components']}. Registry rescope_fallback permits "
        f"collapsing the 'prior exposure 30-100d' branch in §15 but not the "
        f"core 8/8 boundary scoring."
    )
    assert high["risk_category"] == "high", (
        f"High-risk case mapped to category {high['risk_category']!r}, "
        f"expected 'high' per Lo 2006."
    )

    # Low-risk: <30% fall + nadir 150 (0 pts), day 20 (0), no thrombosis (0),
    # definite other cause (0) → 0/8
    low = hit_4t_score(
        platelet_trajectory=[200.0, 180.0, 160.0, 150.0],  # 25% fall, nadir 150
        timing_days_from_heparin=20.0,
        other_cause="definite",
        thrombosis=False,
    )
    assert low["total_score"] == 0, (
        f"Low-risk boundary case scored {low['total_score']}/8, expected 0/8. "
        f"Components: {low['components']}."
    )
    assert low["risk_category"] == "low", (
        f"Low-risk case mapped to category {low['risk_category']!r}, "
        f"expected 'low' per Lo 2006."
    )


def test_srm_warkentin_4t_hit_score_range_bounded():
    """Total score is always in numeric range [0, 8]."""
    for timing in (0.0, 7.0, 20.0):
        r = hit_4t_score(
            platelet_trajectory=[250.0, 100.0],
            timing_days_from_heparin=timing,
            other_cause="possible",
            thrombosis=True,
        )
        assert 0 <= r["total_score"] <= 8, (
            f"Score {r['total_score']} outside [0, 8] for timing={timing}"
        )
