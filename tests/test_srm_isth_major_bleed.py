"""SR-M claim: isth_major_bleed

Falsification test for the Schulman 2005 ISTH major-bleeding
classification claim. Constructs synthetic bleeding events and asserts
each of the four Schulman criteria (fatal, critical-site, Hb drop ≥
2.0 g/dL, transfusion ≥ 2 units) independently triggers "major," and
that below-threshold events do not.
"""

from __future__ import annotations

from hemosim.metrics.clinical import isth_major_bleeding


def test_srm_isth_major_bleed_four_criteria():
    """Each Schulman 2005 criterion independently triggers major classification."""
    # 1) Hb drop 2.5 g/dL alone → major (>= 2.0 threshold)
    r = isth_major_bleeding([
        {"type": "gi", "hb_drop_g_dl": 2.5, "units_transfused": 0,
         "critical_site": False, "fatal": False}
    ])
    assert r["count"] == 1, (
        f"Hb drop 2.5 g/dL should classify as major (Schulman threshold "
        f"2.0); got count={r['count']}. Registry: not rescopable."
    )
    assert "hb_drop_>=2_g_dl" in r["events"][0]["criteria_met"]

    # 2) Transfusion 2 units alone → major
    r = isth_major_bleeding([
        {"type": "gu", "hb_drop_g_dl": 0.5, "units_transfused": 2,
         "critical_site": False, "fatal": False}
    ])
    assert r["count"] == 1, (
        f"Transfusion >= 2 units should trigger major per Schulman 2005; got {r['count']}."
    )

    # 3) Critical site alone → major
    r = isth_major_bleeding([
        {"type": "intracranial", "hb_drop_g_dl": 0.5, "units_transfused": 0,
         "critical_site": True, "fatal": False}
    ])
    assert r["count"] == 1, "Critical-site bleed must classify as major."

    # 4) Fatal alone → major
    r = isth_major_bleeding([
        {"type": "exsanguination", "hb_drop_g_dl": 0.5, "units_transfused": 0,
         "critical_site": False, "fatal": True}
    ])
    assert r["count"] == 1, "Fatal bleed must classify as major."


def test_srm_isth_major_bleed_below_threshold_not_major():
    """Below-threshold event (Hb=1.5, 1 unit, non-critical, non-fatal) does NOT qualify."""
    r = isth_major_bleeding([
        {"type": "gi", "hb_drop_g_dl": 1.5, "units_transfused": 1,
         "critical_site": False, "fatal": False}
    ])
    # Primary-source target (numeric range): count must equal 0 for this case.
    assert r["count"] == 0, (
        f"Event with Hb drop 1.5 g/dL + 1 unit transfused + non-critical + "
        f"non-fatal must NOT classify as major per Schulman 2005; got "
        f"count={r['count']}. If threshold drifted (e.g. Hb 2.0 → 1.5), fix "
        f"code — claim is not rescopable (ISTH definition is standardized)."
    )
