"""Tests for `hemosim.validation.published_calibration` (ISC-12).

Covers: benchmark well-formedness, FitResult JSON round-trip, heparin and
warfarin fitters running to convergence, DOAC rate validator smoke test,
and Markdown-report rendering.
"""

from __future__ import annotations

import json
from dataclasses import fields

import numpy as np
import pytest

from hemosim.validation.published_calibration import (
    BENCHMARKS,
    FitResult,
    PublishedBenchmark,
    _simulate_raschke,
    _simulate_warfarin,
    doac_report_markdown,
    fit_heparin_pkpd,
    fit_warfarin_pkpd,
    full_report_markdown,
    validate_doac_rates,
)


# ---------------------------------------------------------------------------
# Benchmarks well-formed
# ---------------------------------------------------------------------------


class TestBenchmarks:
    """Structural checks on the BENCHMARKS dict."""

    def test_required_keys_present(self):
        """All endpoints referenced in the calibration brief must exist."""
        required = {
            # heparin
            "raschke_aptt_6h",
            "hirsh_therapeutic_conc_mid",
            "nemati_ttr_standard",
            # warfarin
            "iwpc_mean_maintenance_dose",
            "iwpc_days_to_therapeutic",
            "hamberg_ss_inr_wildtype",
            # DOAC
            "rely_dabi_stroke",
            "rely_dabi_bleed",
            "rocket_riva_stroke",
            "rocket_riva_bleed",
            "aristotle_apix_stroke",
            "aristotle_apix_bleed",
        }
        assert required <= set(BENCHMARKS.keys())

    def test_benchmarks_well_formed(self):
        """Every entry must have non-empty trial/endpoint/citation, a
        positive value, and a CI that is None or a valid (lo, hi) range.
        """
        for key, b in BENCHMARKS.items():
            assert isinstance(b, PublishedBenchmark), key
            assert b.key == key
            assert b.trial.strip(), key
            assert b.endpoint.strip(), key
            assert b.citation.strip(), key
            assert b.value > 0, key
            assert b.units.strip(), key
            if b.ci is not None:
                assert len(b.ci) == 2, key
                lo, hi = b.ci
                assert lo <= b.value <= hi, (
                    f"{key}: value {b.value} outside CI {b.ci}"
                )

    def test_citations_contain_doi_or_journal(self):
        """Each citation must include a DOI token or a journal abbreviation
        so reviewers can trace the source.
        """
        for key, b in BENCHMARKS.items():
            citation = b.citation.lower()
            ok = (
                "doi:" in citation
                or "n engl j med" in citation
                or "chest" in citation
                or "ann intern med" in citation
                or "clin pharmacol ther" in citation
                or "conf proc ieee" in citation
            )
            assert ok, f"{key}: citation has no DOI or journal: {b.citation!r}"

    def test_benchmark_to_dict_roundtrips_json(self):
        """PublishedBenchmark.to_dict() must be JSON-serializable."""
        for b in BENCHMARKS.values():
            payload = b.to_dict()
            dumped = json.dumps(payload)
            loaded = json.loads(dumped)
            assert loaded["key"] == b.key
            assert loaded["value"] == b.value
            # CI round-trips as list (json has no tuples)
            if b.ci is not None:
                assert loaded["ci"] == list(b.ci)


# ---------------------------------------------------------------------------
# FitResult serialization
# ---------------------------------------------------------------------------


class TestFitResult:
    def _minimal(self) -> FitResult:
        return FitResult(
            model="heparin",
            fitted_params={"vmax": 400.0, "km": 0.4},
            initial_params={"vmax": 400.0, "km": 0.4},
            residuals=[
                {
                    "key": "raschke_aptt_6h",
                    "trial": "Raschke 1993",
                    "endpoint": "aPTT 6h",
                    "expected": 75.0,
                    "observed": 75.0,
                    "residual": 0.0,
                    "units": "seconds",
                }
            ],
            rmse=0.0,
            n_benchmarks=1,
            n_iterations=10,
            converged=True,
            message="ok",
            seed=42,
            benchmark_keys=["raschke_aptt_6h"],
        )

    def test_to_json_roundtrip(self):
        """to_json() output must be JSON-loadable and reconstructable."""
        r = self._minimal()
        s = r.to_json()
        loaded = json.loads(s)
        assert loaded["model"] == "heparin"
        assert loaded["rmse"] == 0.0
        assert loaded["converged"] is True
        assert loaded["fitted_params"]["vmax"] == 400.0
        assert loaded["residuals"][0]["key"] == "raschke_aptt_6h"

    def test_has_all_required_fields(self):
        r = self._minimal()
        names = {f.name for f in fields(r)}
        required = {
            "model",
            "fitted_params",
            "initial_params",
            "residuals",
            "rmse",
            "n_benchmarks",
            "n_iterations",
            "converged",
            "message",
            "seed",
            "benchmark_keys",
        }
        assert required <= names

    def test_markdown_renders(self):
        """to_markdown() must produce a non-empty string with section header,
        fitted-params table, and residuals table.
        """
        r = self._minimal()
        md = r.to_markdown()
        assert "### Heparin PK/PD fit" in md
        assert "Fitted parameters" in md
        assert "Per-benchmark residuals" in md
        assert "raschke_aptt_6h" in md


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------


class TestSimulateRaschke:
    def test_default_params_produce_nontrivial_output(self):
        """Calling _simulate_raschke with v0.1 defaults must return
        plausible (aPTT, conc, TTR) without raising.
        """
        aptt, conc, ttr = _simulate_raschke(
            vmax=400.0, km=0.4, aptt_alpha=2.5, aptt_c_ref=0.15
        )
        assert 20.0 < aptt < 300.0
        assert 0.0 <= conc <= 10.0
        assert 0.0 <= ttr <= 1.0

    def test_response_is_deterministic(self):
        """Same params -> same outputs (the PK/PD model is a deterministic ODE)."""
        r1 = _simulate_raschke(400.0, 0.4, 2.5, 0.15)
        r2 = _simulate_raschke(400.0, 0.4, 2.5, 0.15)
        assert r1 == pytest.approx(r2)


class TestSimulateWarfarin:
    def test_higher_potency_raises_inr(self):
        """Sanity: boosting inhibition gain must raise steady-state INR."""
        inr_low, _ = _simulate_warfarin(
            ec50=1.5,
            vkorc1_gg_factor=1.0,
            hill=1.3,
            vk_inhibition_gain=0.04,
            s_warfarin_potency=3.0,
            dose_mg_per_day=5.0,
        )
        inr_hi, _ = _simulate_warfarin(
            ec50=1.5,
            vkorc1_gg_factor=1.0,
            hill=1.3,
            vk_inhibition_gain=0.10,  # stronger inhibition
            s_warfarin_potency=3.0,
            dose_mg_per_day=5.0,
        )
        assert inr_hi > inr_low


# ---------------------------------------------------------------------------
# Fitters
# ---------------------------------------------------------------------------


class TestFitHeparin:
    def test_runs_to_convergence_trivial_case(self):
        """Fit with generous iteration budget should converge on trivial case."""
        result = fit_heparin_pkpd(max_iter=500)
        assert result.model == "heparin"
        assert result.converged
        # All four expected parameters fitted.
        for key in ("vmax", "km", "aptt_alpha", "aptt_c_ref"):
            assert key in result.fitted_params
            assert result.fitted_params[key] > 0
        # RMSE should be well below the initial miss (initial default
        # already overshoots aPTT 6h to ~193s; fit must dramatically
        # improve).
        assert result.rmse < 1.0
        # Each residual record is complete.
        for r in result.residuals:
            for field in (
                "key",
                "trial",
                "endpoint",
                "expected",
                "observed",
                "residual",
                "units",
            ):
                assert field in r

    def test_missing_benchmark_raises(self):
        """If a required benchmark is missing, the fitter raises KeyError."""
        broken = {k: v for k, v in BENCHMARKS.items() if k != "raschke_aptt_6h"}
        with pytest.raises(KeyError):
            fit_heparin_pkpd(benchmarks=broken, max_iter=10)

    def test_fit_result_is_json_serializable(self):
        result = fit_heparin_pkpd(max_iter=50)
        s = result.to_json()
        loaded = json.loads(s)
        assert loaded["model"] == "heparin"
        assert isinstance(loaded["residuals"], list)


class TestFitWarfarin:
    def test_runs_to_convergence(self):
        result = fit_warfarin_pkpd(max_iter=500)
        assert result.model == "warfarin"
        assert result.converged
        for key in (
            "ec50",
            "vkorc1_gg_factor",
            "hill",
            "vk_inhibition_gain",
            "s_warfarin_potency",
        ):
            assert key in result.fitted_params
            assert result.fitted_params[key] > 0
        # The INR-at-IWPC-mean-dose target is 2.5, the baseline model
        # caps at ~1.85 with defaults. A successful fit must at least
        # halve that gap.
        ss_inr = next(
            r["observed"]
            for r in result.residuals
            if r["key"] == "iwpc_mean_maintenance_dose"
        )
        assert ss_inr > 2.0, f"Fitted model didn't reach therapeutic INR: {ss_inr}"

    def test_missing_benchmark_raises(self):
        broken = {k: v for k, v in BENCHMARKS.items() if k != "hamberg_ss_inr_wildtype"}
        with pytest.raises(KeyError):
            fit_warfarin_pkpd(benchmarks=broken, max_iter=10)


# ---------------------------------------------------------------------------
# DOAC event-rate validator
# ---------------------------------------------------------------------------


class TestValidateDoacRates:
    def test_returns_all_three_drugs(self):
        """Validator must report for dabigatran, rivaroxaban, apixaban."""
        # Use a small n_episodes for test speed; rates will be noisy but
        # structure is what we check here.
        out = validate_doac_rates(n_episodes=20, seed=123)
        assert set(out["drugs"].keys()) == {"dabigatran", "rivaroxaban", "apixaban"}
        for drug, d in out["drugs"].items():
            for field in (
                "stroke_rate_per_100py",
                "bleed_rate_per_100py",
                "expected_stroke",
                "expected_bleed",
                "stroke_residual",
                "bleed_residual",
                "patient_years",
                "stroke_events",
                "bleed_events",
            ):
                assert field in d, f"{drug} missing {field}"
            # Expected values match the benchmarks exactly.
            if drug == "dabigatran":
                assert d["expected_stroke"] == pytest.approx(1.11)
                assert d["expected_bleed"] == pytest.approx(3.11)
            elif drug == "apixaban":
                assert d["expected_stroke"] == pytest.approx(1.27)
                assert d["expected_bleed"] == pytest.approx(2.13)

    def test_result_is_json_serializable(self):
        out = validate_doac_rates(n_episodes=10, seed=7)
        s = json.dumps(out)
        loaded = json.loads(s)
        assert loaded["n_episodes_per_drug"] == 10

    def test_deterministic_for_same_seed(self):
        """Two runs with the same seed must produce the same event counts."""
        a = validate_doac_rates(n_episodes=15, seed=99)
        b = validate_doac_rates(n_episodes=15, seed=99)
        for drug in a["drugs"]:
            assert (
                a["drugs"][drug]["stroke_events"]
                == b["drugs"][drug]["stroke_events"]
            )
            assert (
                a["drugs"][drug]["bleed_events"]
                == b["drugs"][drug]["bleed_events"]
            )


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


class TestReportRendering:
    def test_doac_report_markdown(self):
        out = validate_doac_rates(n_episodes=5, seed=1)
        md = doac_report_markdown(out)
        assert "DOAC event-rate validation" in md
        assert "dabigatran" in md
        assert "RE-LY" in md

    def test_full_report_contains_all_sections(self):
        h = fit_heparin_pkpd(max_iter=30)
        w = fit_warfarin_pkpd(max_iter=30)
        d = validate_doac_rates(n_episodes=5, seed=1)
        md = full_report_markdown(h, w, d)
        assert "# Published-Data Calibration Report" in md
        assert "## Benchmarks" in md
        assert "## Heparin fit" in md
        assert "## Warfarin fit" in md
        assert "DOAC event-rate validation" in md
        assert "## Fingerprint" in md
        # Fingerprint line is a sha256 hex digest (64 chars)
        assert any("sha256" in line and "=" in line for line in md.splitlines())


# ---------------------------------------------------------------------------
# Non-regression guard
# ---------------------------------------------------------------------------


class TestNonRegression:
    def test_warfarin_pkpd_backward_compatible_defaults(self):
        """New fittable attributes must default to the v0.1 values so
        downstream tests and the env baselines do not see a behavior change
        unless the validation harness explicitly sets them.
        """
        from hemosim.models.warfarin_pkpd import WarfarinPKPD

        m = WarfarinPKPD()
        assert m.vk_inhibition_gain == pytest.approx(0.04)
        assert m.s_warfarin_potency == pytest.approx(3.0)
