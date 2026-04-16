"""Tests for `hemosim.validation.mimic_calibration` (ISC-7).

Covers:
    * Synthetic cohort shape and provenance labeling
    * CSV round-trip (synthetic -> CSV -> reload -> equality within float tol)
    * Calibration runs to convergence on a synthetic cohort
    * Fitted params finite and physiologic
    * CalibrationResult JSON and Markdown rendering
    * Input validation (bad cohort shape; bad CSV columns; unsupported method)
    * Non-regression: module import does not break existing API surface
"""

from __future__ import annotations

import csv
import io
import json
import math
import tempfile
from pathlib import Path

import pytest

from hemosim.validation.mimic_calibration import (
    CalibrationResult,
    MIMICHeparinCohort,
    _parse_csv_string,
    calibrate_heparin_pkpd,
    cohort_summary,
    write_cohort_csv,
)


# ---------------------------------------------------------------------------
# Synthetic cohort shape
# ---------------------------------------------------------------------------


class TestSyntheticCohort:
    """The synthetic generator is the only path the harness can run in CI,
    so its shape is load-bearing for every downstream test."""

    def test_synthetic_cohort_size_and_fields(self):
        """Requesting N patients must return exactly N objects with the
        right dataclass fields populated and parallel-length time series."""
        cohort = MIMICHeparinCohort.synthetic_cohort(n_patients=7, seed=123)
        assert len(cohort) == 7
        for patient in cohort:
            assert isinstance(patient, MIMICHeparinCohort)
            assert patient.patient_id.startswith("synthetic_")
            assert patient.weight_kg > 0
            assert 0 < patient.renal_function <= 1.5
            assert patient.baseline_aptt >= 20.0
            n = len(patient.time_hours)
            assert n >= 3, "need enough observations to calibrate"
            # Parallel arrays: all same length.
            assert len(patient.aptt_observations) == n
            assert len(patient.heparin_rate_u_per_hr) == n
            assert len(patient.heparin_bolus_u) == n
            # Monotone times.
            for i in range(1, n):
                assert patient.time_hours[i] >= patient.time_hours[i - 1]

    def test_synthetic_cohort_is_seed_reproducible(self):
        """Same seed must yield identical cohorts — a must-have for the
        calibration regression test in reproduce.sh."""
        a = MIMICHeparinCohort.synthetic_cohort(n_patients=4, seed=777)
        b = MIMICHeparinCohort.synthetic_cohort(n_patients=4, seed=777)
        assert len(a) == len(b)
        for pa, pb in zip(a, b, strict=True):
            assert pa.patient_id == pb.patient_id
            assert pa.weight_kg == pytest.approx(pb.weight_kg)
            assert pa.aptt_observations == pytest.approx(pb.aptt_observations)
            assert pa.heparin_rate_u_per_hr == pytest.approx(
                pb.heparin_rate_u_per_hr
            )

    def test_synthetic_cohort_rejects_nonpositive_n(self):
        with pytest.raises(ValueError):
            MIMICHeparinCohort.synthetic_cohort(n_patients=0)
        with pytest.raises(ValueError):
            MIMICHeparinCohort.synthetic_cohort(n_patients=-3)


# ---------------------------------------------------------------------------
# CSV round-trip
# ---------------------------------------------------------------------------


class TestCSVRoundTrip:
    def test_synthetic_to_csv_roundtrip(self, tmp_path: Path):
        """synthetic -> write CSV -> from_csv -> field-level equality
        within the 6-decimal precision the writer uses."""
        original = MIMICHeparinCohort.synthetic_cohort(n_patients=4, seed=42)
        csv_path = tmp_path / "mimic_heparin.csv"
        write_cohort_csv(original, csv_path)

        loaded = MIMICHeparinCohort.from_csv(csv_path)
        assert len(loaded) == len(original)
        for o, lhs in zip(original, loaded, strict=True):
            assert lhs.patient_id == o.patient_id
            assert lhs.weight_kg == pytest.approx(o.weight_kg, abs=1e-4)
            # renal_function round-trips via the crcl_recent column — the
            # writer reverses the / _DEFAULT_CRCL transform.
            assert lhs.renal_function == pytest.approx(o.renal_function, abs=1e-4)
            assert lhs.baseline_aptt == pytest.approx(o.baseline_aptt, abs=1e-4)
            assert len(lhs.time_hours) == len(o.time_hours)
            for orig_v, load_v in zip(
                o.aptt_observations, lhs.aptt_observations, strict=True
            ):
                assert load_v == pytest.approx(orig_v, abs=1e-4)

    def test_from_csv_requires_all_columns(self, tmp_path: Path):
        """Missing a required column must raise a clear KeyError."""
        bad = tmp_path / "missing.csv"
        bad.write_text(
            "subject_id,time_hours,aptt\n"
            "p1,0.0,30.0\n",
            encoding="utf-8",
        )
        with pytest.raises(KeyError):
            MIMICHeparinCohort.from_csv(bad)

    def test_from_csv_handles_empty_crcl_and_baseline(self):
        """Empty crcl/baseline cells should fall back to sensible
        defaults (normal renal function; first aPTT as baseline)."""
        csv_text = (
            "subject_id,time_hours,aptt,current_rate_u_per_hr,"
            "bolus_in_last_hour_u,weight_kg,crcl_recent,baseline_aptt\n"
            "p1,0.0,29.8,0,0,70.0,,\n"
            "p1,6.0,72.1,1400,5000,70.0,,\n"
        )
        cohorts = _parse_csv_string(csv_text)
        assert len(cohorts) == 1
        p = cohorts[0]
        assert p.renal_function == pytest.approx(1.0)
        assert p.baseline_aptt == pytest.approx(29.8)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestCohortValidation:
    def test_mismatched_series_lengths_rejected(self):
        with pytest.raises(ValueError):
            MIMICHeparinCohort(
                patient_id="bad",
                weight_kg=70.0,
                renal_function=1.0,
                baseline_aptt=30.0,
                time_hours=[0.0, 6.0],
                aptt_observations=[30.0],
                heparin_rate_u_per_hr=[0.0, 1400.0],
                heparin_bolus_u=[0.0, 0.0],
            )

    def test_non_monotone_times_rejected(self):
        with pytest.raises(ValueError):
            MIMICHeparinCohort(
                patient_id="bad",
                weight_kg=70.0,
                renal_function=1.0,
                baseline_aptt=30.0,
                time_hours=[6.0, 0.0],
                aptt_observations=[70.0, 30.0],
                heparin_rate_u_per_hr=[1400.0, 0.0],
                heparin_bolus_u=[0.0, 0.0],
            )

    def test_empty_cohort_rejected(self):
        with pytest.raises(ValueError):
            MIMICHeparinCohort(
                patient_id="bad",
                weight_kg=70.0,
                renal_function=1.0,
                baseline_aptt=30.0,
                time_hours=[],
                aptt_observations=[],
                heparin_rate_u_per_hr=[],
                heparin_bolus_u=[],
            )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


class TestCalibrateHeparinPKPD:
    def test_runs_to_convergence_on_synthetic(self):
        """End-to-end: synthetic cohort + Nelder-Mead must report
        convergence (optimizer.success is True) within a modest iteration
        budget, and RMSE must be finite and small (< 10 s)."""
        cohort = MIMICHeparinCohort.synthetic_cohort(n_patients=8, seed=42)
        result = calibrate_heparin_pkpd(
            cohort,
            max_iter=400,
            data_source="synthetic",
            seed=42,
        )
        assert isinstance(result, CalibrationResult)
        assert result.convergence_info["converged"] is True, (
            f"optimizer did not converge: {result.convergence_info}"
        )
        assert math.isfinite(result.overall_rmse)
        # Synthetic data was generated from the defaults plus N(0, 5 s)
        # noise, so a well-converged fit should reach RMSE in single
        # digits. Fail loudly if we regress above 10s.
        assert result.overall_rmse < 10.0, (
            f"overall RMSE too high: {result.overall_rmse}"
        )
        assert result.n_patients == 8
        assert result.n_observations > 0
        assert result.data_source == "synthetic"

    def test_fitted_params_finite_and_physiologic(self):
        """Every fitted parameter must be finite, positive, and within
        the physiologic bounds the loss enforces."""
        cohort = MIMICHeparinCohort.synthetic_cohort(n_patients=6, seed=42)
        result = calibrate_heparin_pkpd(cohort, max_iter=300, seed=42)
        for name in ("vmax", "km", "aptt_alpha", "aptt_c_ref"):
            assert name in result.fitted_params
            value = result.fitted_params[name]
            assert math.isfinite(value)
            assert value > 0.0
        # Per the loss bounds.
        assert result.fitted_params["vmax"] < 5000.0
        assert result.fitted_params["km"] < 5.0
        assert result.fitted_params["aptt_alpha"] < 20.0
        assert result.fitted_params["aptt_c_ref"] < 2.0

    def test_respects_custom_initial_params(self):
        """`initial_params` should override defaults and appear verbatim
        in the returned `CalibrationResult.initial_params`."""
        cohort = MIMICHeparinCohort.synthetic_cohort(n_patients=3, seed=42)
        custom = {"vmax": 500.0, "km": 0.5, "aptt_alpha": 3.0, "aptt_c_ref": 0.2}
        result = calibrate_heparin_pkpd(
            cohort, initial_params=custom, max_iter=50
        )
        assert result.initial_params == custom

    def test_rejects_empty_cohort(self):
        with pytest.raises(ValueError):
            calibrate_heparin_pkpd([])

    def test_rejects_unsupported_method(self):
        cohort = MIMICHeparinCohort.synthetic_cohort(n_patients=2, seed=42)
        with pytest.raises(ValueError):
            calibrate_heparin_pkpd(cohort, method="powell")  # type: ignore[arg-type]

    def test_l_bfgs_b_method_runs(self):
        """Second supported method must also produce a finite RMSE."""
        cohort = MIMICHeparinCohort.synthetic_cohort(n_patients=3, seed=42)
        result = calibrate_heparin_pkpd(
            cohort, method="L-BFGS-B", max_iter=100
        )
        assert math.isfinite(result.overall_rmse)
        assert result.convergence_info["method"] == "L-BFGS-B"


# ---------------------------------------------------------------------------
# CalibrationResult serialization + report
# ---------------------------------------------------------------------------


class TestCalibrationResult:
    def _result(self) -> CalibrationResult:
        cohort = MIMICHeparinCohort.synthetic_cohort(n_patients=4, seed=42)
        return calibrate_heparin_pkpd(
            cohort, max_iter=80, data_source="synthetic", seed=42
        )

    def test_to_json_roundtrip(self):
        r = self._result()
        s = r.to_json()
        payload = json.loads(s)
        assert payload["data_source"] == "synthetic"
        assert payload["n_patients"] == 4
        assert "fitted_params" in payload
        assert "convergence_info" in payload
        # Ensure all floats are actually JSON numbers (no NaN leakage).
        for name, value in payload["fitted_params"].items():
            assert isinstance(value, (int, float)), name

    def test_markdown_report_has_required_sections(self):
        r = self._result()
        md = r.markdown_report()
        assert "## MIMIC-IV heparin PK/PD calibration" in md
        assert "Data source: **synthetic**" in md
        # The "Scaffold notice" warning must appear for synthetic runs so
        # the paper can't quietly misrepresent provenance.
        assert "Scaffold notice" in md
        assert "Fitted parameters" in md
        assert "Per-patient RMSE" in md
        # Every patient should show up in the per-patient table.
        for row in r.rmse_per_patient:
            assert row["patient_id"] in md

    def test_markdown_report_drops_scaffold_notice_for_real_data(self):
        """If data_source == 'mimic_iv' the scaffold-notice block must not
        appear — that block is *only* for synthetic runs."""
        cohort = MIMICHeparinCohort.synthetic_cohort(n_patients=2, seed=42)
        r = calibrate_heparin_pkpd(
            cohort, max_iter=20, data_source="mimic_iv", seed=42
        )
        md = r.markdown_report()
        assert "Data source: **mimic_iv**" in md
        assert "Scaffold notice" not in md


# ---------------------------------------------------------------------------
# cohort_summary
# ---------------------------------------------------------------------------


class TestCohortSummary:
    def test_summary_of_empty_cohort(self):
        s = cohort_summary([])
        assert s == {
            "n_patients": 0,
            "n_observations": 0,
            "mean_weight_kg": 0.0,
            "mean_baseline_aptt": 0.0,
        }

    def test_summary_counts_observations_correctly(self):
        cohort = MIMICHeparinCohort.synthetic_cohort(n_patients=3, seed=42)
        expected_obs = sum(len(c.time_hours) for c in cohort)
        s = cohort_summary(cohort)
        assert s["n_patients"] == 3
        assert s["n_observations"] == expected_obs
        assert s["mean_weight_kg"] > 0
        assert s["mean_baseline_aptt"] > 0


# ---------------------------------------------------------------------------
# Non-regression
# ---------------------------------------------------------------------------


class TestNonRegression:
    def test_mimic_module_does_not_break_published_calibration_api(self):
        """Adding the MIMIC harness must not disturb the existing
        published_calibration public API — a direct smoke import."""
        from hemosim.validation import (
            BENCHMARKS,
            FitResult,
            PublishedBenchmark,
            fit_heparin_pkpd,
            fit_warfarin_pkpd,
            validate_doac_rates,
        )

        assert isinstance(BENCHMARKS, dict)
        assert BENCHMARKS  # non-empty
        # Spot-check one entry we know v0.1 exposed.
        assert "raschke_aptt_6h" in BENCHMARKS
        assert callable(fit_heparin_pkpd)
        assert callable(fit_warfarin_pkpd)
        assert callable(validate_doac_rates)
        assert PublishedBenchmark is not None
        assert FitResult is not None

    def test_mimic_calibration_exports_match_init(self):
        """The validation package must re-export the new MIMIC symbols."""
        from hemosim.validation import (
            CalibrationResult as CR,
            MIMICHeparinCohort as MC,
            calibrate_heparin_pkpd as calib,
        )

        assert CR is CalibrationResult
        assert MC is MIMICHeparinCohort
        assert calib is calibrate_heparin_pkpd
