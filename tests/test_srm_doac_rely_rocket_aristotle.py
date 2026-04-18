"""SR-M claim: doac_rely_rocket_aristotle

Falsification test for the DOAC event-rate trial-reproduction claim.
Reads the shipped results/published_calibration.json (generated from a
large-cohort DOACManagement-v0 simulation) and asserts bleed rates are
within the published 95 % CIs for all three drugs. Stroke CI miss is
pre-disclosed in the registry and tested as an explicit logged miss,
not as a pass-condition.
"""

from __future__ import annotations

import json
from pathlib import Path


CALIBRATION_PATH = (
    Path(__file__).parent.parent / "results" / "published_calibration.json"
)


def _load_doac_block():
    assert CALIBRATION_PATH.is_file(), (
        f"results/published_calibration.json missing at {CALIBRATION_PATH}; "
        f"DOAC trial-replication claim requires this artifact."
    )
    data = json.loads(CALIBRATION_PATH.read_text())
    # Known layout: data["doac_validation"]["drugs"] = {apixaban, dabigatran, rivaroxaban}.
    if (
        isinstance(data.get("doac_validation"), dict)
        and isinstance(data["doac_validation"].get("drugs"), dict)
    ):
        return data["doac_validation"]["drugs"]
    # Back-compat probes.
    for candidate in (
        "doac_trial_replication",
        "doac",
        "doac_management",
    ):
        if candidate in data:
            return data[candidate]
    # Fall back: recursively scan for a dict containing all three drug keys.
    def _find(obj):
        if isinstance(obj, dict):
            if {"apixaban", "dabigatran", "rivaroxaban"}.issubset(obj.keys()):
                return obj
            for v in obj.values():
                r = _find(v)
                if r is not None:
                    return r
        return None

    found = _find(data)
    if found is not None:
        return found
    raise AssertionError(
        "No DOAC trial-replication block found in published_calibration.json; "
        "claim cannot be tested."
    )


def test_srm_doac_bleed_rates_in_CI():
    """Bleed rates for all 3 DOACs fall within their trial 95 % CIs."""
    block = _load_doac_block()
    for drug in ("apixaban", "dabigatran", "rivaroxaban"):
        entry = block[drug]
        bleed = float(entry["bleed_rate_per_100py"])
        ci_low, ci_high = (float(x) for x in entry["ci_bleed"])
        assert ci_low <= bleed <= ci_high, (
            f"{drug} bleed rate {bleed:.3f}%/yr outside primary-source CI "
            f"[{ci_low}, {ci_high}] per registry. Per registry rescope_fallback, "
            f"bleed-CI miss invalidates the within-CI claim."
        )


def test_srm_doac_stroke_ci_miss_disclosed():
    """Stroke CI miss is explicitly logged (pre-disclosed in registry + paper §8.3)."""
    block = _load_doac_block()
    # Per registry, stroke CI miss is pre-disclosed. We assert the miss is
    # *observable in the calibration JSON* (i.e. stroke_within_ci is False for
    # ≥ 1 drug), which is the falsification test of "we flagged it honestly."
    stroke_misses = [
        drug for drug in ("apixaban", "dabigatran", "rivaroxaban")
        if block[drug].get("stroke_within_ci") is False
    ]
    assert len(stroke_misses) >= 1, (
        "Paper discloses stroke-rate CI miss for all three DOACs, but "
        "calibration JSON reports stroke_within_ci=True for every drug. "
        "Honesty-disclosure claim refuted: paper says we failed, artifact "
        "says we succeeded. At least 1 drug must show stroke_within_ci=False."
    )
