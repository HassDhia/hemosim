"""SR-M claim: rosendaal_ttr

Falsification test for the Rosendaal 1993 TTR implementation claim.
Constructs a synthetic straight-line INR trajectory from 1.5 to 3.5
over 10 days, calls time_in_therapeutic_range(low=2.0, high=3.0), and
asserts TTR ∈ [0.49, 0.51] per registry primary-source target
(exactly 0.5 = days 2.5-7.5 in range / 10 day total).
"""

from __future__ import annotations

import numpy as np

from hemosim.metrics.clinical import time_in_therapeutic_range


def test_srm_rosendaal_ttr_linear_interpolation_known_case():
    """Straight-line INR 1.5→3.5 over 10 days yields TTR = 0.50 (Rosendaal 1993)."""
    n_points = 11  # 11 points over 10 days = inclusive endpoints
    values = np.linspace(1.5, 3.5, n_points).tolist()
    times_hours = np.linspace(0.0, 10.0 * 24.0, n_points).tolist()

    ttr = time_in_therapeutic_range(
        values=values,
        times_hours=times_hours,
        low=2.0,
        high=3.0,
    )
    # Registry primary-source target: TTR exactly 0.5 on this synthetic case.
    assert 0.49 <= ttr <= 0.51, (
        f"Rosendaal TTR = {ttr:.4f} outside primary-source target [0.49, 0.51] "
        f"for straight-line INR 1.5→3.5 over 10d, range [2.0, 3.0]. Expected "
        f"exact 0.5 = 5d-in-range/10d-total per linear-interpolation Rosendaal "
        f"1993 definition. Registry rescope_fallback: if implementation is a "
        f"naive per-step count, revert §9 TTR description to 'per-step "
        f"in-range fraction' and flag v2.1 fix."
    )


def test_srm_rosendaal_ttr_full_in_range():
    """Trajectory entirely inside [low, high] yields TTR = 1.0."""
    ttr = time_in_therapeutic_range(
        values=[2.0, 2.5, 2.8, 3.0],
        times_hours=[0.0, 24.0, 48.0, 72.0],
        low=2.0,
        high=3.0,
    )
    # Numeric range check with comparator: TTR should be 1.0 (±0.001).
    assert 0.999 <= ttr <= 1.001, (
        f"TTR for trajectory fully in [2.0, 3.0] = {ttr}; expected ~1.0."
    )


def test_srm_rosendaal_ttr_full_out_of_range():
    """Trajectory entirely outside [low, high] yields TTR = 0.0."""
    ttr = time_in_therapeutic_range(
        values=[1.0, 1.2, 1.4, 1.5],
        times_hours=[0.0, 24.0, 48.0, 72.0],
        low=2.0,
        high=3.0,
    )
    assert 0.0 <= ttr <= 0.001, (
        f"TTR for trajectory fully below 2.0 = {ttr}; expected 0.0."
    )
