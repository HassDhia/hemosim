"""Validation and calibration utilities for hemosim PK/PD models.

Modules:
    published_calibration: Fit PK/PD parameters against published clinical
        trial summary statistics (Raschke 1993, Hirsh 2001, Nemati 2016,
        IWPC 2009, Hamberg 2007, RE-LY, ROCKET-AF, ARISTOTLE).

    mimic_calibration: (scaffold — ISC-7) individual-patient-level fitting
        against MIMIC-IV cohort. Requires PhysioNet credentialed access.
"""

from __future__ import annotations

from hemosim.validation.mimic_calibration import (
    CalibrationResult,
    MIMICHeparinCohort,
    calibrate_heparin_pkpd,
)
from hemosim.validation.published_calibration import (
    BENCHMARKS,
    FitResult,
    PublishedBenchmark,
    fit_heparin_pkpd,
    fit_warfarin_pkpd,
    validate_doac_rates,
)

__all__ = [
    "BENCHMARKS",
    "CalibrationResult",
    "FitResult",
    "MIMICHeparinCohort",
    "PublishedBenchmark",
    "calibrate_heparin_pkpd",
    "fit_heparin_pkpd",
    "fit_warfarin_pkpd",
    "validate_doac_rates",
]
