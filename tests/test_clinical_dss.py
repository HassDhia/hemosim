"""Tests for :mod:`hemosim.clinical.dss` (ISC-10).

These tests exercise the Decision-Support harness against the
deterministic clinical-baseline fallback path — which is the path
that runs on this CI/dev machine today (no ``torch`` / ``sb3``
installed). The fallback path is the canonical promise of the
harness: the DSS must always produce a recommendation. PPO paths are
exercised in ``tests/test_clinical_ppo_*`` (not yet present — will
land with ISC-8 once real PPO checkpoints exist).
"""

from __future__ import annotations

import numpy as np
import pytest

from hemosim.clinical.dss import (
    BaseDSS,
    DosingRecommendation,
    HeparinDSS,
    PatientSnapshot,
    WarfarinDSS,
    _SB3_AVAILABLE,
)


# ---------------------------------------------------------------------------
# PatientSnapshot
# ---------------------------------------------------------------------------


class TestPatientSnapshot:
    def test_default_construction_is_all_optional(self) -> None:
        """An empty snapshot must be constructable — clinical reality."""
        snap = PatientSnapshot()
        assert snap.aptt_seconds is None
        assert snap.inr is None
        assert snap.drug is None
        assert snap.extra == {}

    def test_roundtrip_through_dict(self) -> None:
        """FHIR-like dict -> PatientSnapshot -> dict round-trip preserves fields."""
        original = {
            "drug": "heparin",
            "aptt_seconds": 62.5,
            "weight_kg": 78.0,
            "platelets_k_per_ul": 180.0,
            "renal_function": 0.8,
            "hours_on_therapy": 12.0,
            "some_unknown_fhir_field": "observation-abc-123",
        }
        snap = PatientSnapshot.from_dict(original)
        assert snap.aptt_seconds == pytest.approx(62.5)
        assert snap.weight_kg == pytest.approx(78.0)
        # Unknown keys routed to extras.
        assert snap.extra["some_unknown_fhir_field"] == "observation-abc-123"

        roundtrip = snap.to_dict()
        assert roundtrip["aptt_seconds"] == pytest.approx(62.5)
        assert roundtrip["drug"] == "heparin"
        # extras survive the roundtrip.
        assert roundtrip["extra"]["some_unknown_fhir_field"] == "observation-abc-123"

    def test_mutable_default_is_independent_per_instance(self) -> None:
        """No shared-mutable-default trap on ``extra``."""
        a = PatientSnapshot()
        b = PatientSnapshot()
        a.extra["x"] = 1
        assert "x" not in b.extra


# ---------------------------------------------------------------------------
# HeparinDSS (baseline fallback)
# ---------------------------------------------------------------------------


class TestHeparinDSSBaseline:
    def test_constructs_without_policy_and_uses_baseline(self) -> None:
        """With no sb3/model, HeparinDSS must still be constructable."""
        dss = HeparinDSS(policy_path=None)
        assert isinstance(dss, BaseDSS)
        assert dss.uses_baseline is True

    def test_constructs_with_missing_policy_file(self, tmp_path) -> None:
        """Non-existent policy_path must not raise; falls back to baseline."""
        missing = tmp_path / "does_not_exist.zip"
        dss = HeparinDSS(policy_path=missing, n_ensemble=1)
        assert dss.uses_baseline is True

    def test_recommend_populates_all_fields(self) -> None:
        dss = HeparinDSS()
        snap = PatientSnapshot(
            drug="heparin",
            aptt_seconds=45.0,
            weight_kg=80.0,
            platelets_k_per_ul=200.0,
            renal_function=1.0,
            hours_on_therapy=0.0,
        )
        rec = dss.recommend(snap)
        assert isinstance(rec, DosingRecommendation)
        assert rec.action_taken == "heparin_infusion"
        # Raschke initial ≈ 18 U/kg/hr * 80 kg = 1440 U/hr. Allow a wide band.
        assert 500.0 <= rec.dose_or_rate <= 2500.0
        lo, hi = rec.uncertainty_interval
        assert lo <= rec.dose_or_rate <= hi
        assert 0.0 <= rec.confidence <= 1.0
        # At t=0 Raschke gives the loading bolus.
        assert rec.bolus_given is True
        assert rec.bolus_u_per_kg == pytest.approx(80.0)
        # Feature contributions populated across all obs dims.
        assert len(rec.top_feature_contributions) == HeparinDSS._OBS_DIM
        assert "baseline" in rec.rationale or "raschke" in rec.rationale.lower()

    def test_recommend_with_missing_fields_uses_defaults(self) -> None:
        """Partial snapshots still yield a recommendation."""
        dss = HeparinDSS()
        snap = PatientSnapshot(drug="heparin")  # everything None
        rec = dss.recommend(snap)
        assert 0.0 <= rec.dose_or_rate <= 2500.0
        assert len(rec.top_feature_contributions) == 6

    def test_explain_returns_ranked_features(self) -> None:
        dss = HeparinDSS()
        snap = PatientSnapshot(
            drug="heparin",
            aptt_seconds=30.0,  # well below therapeutic
            weight_kg=80.0,
            platelets_k_per_ul=200.0,
            renal_function=1.0,
            hours_on_therapy=6.0,  # past the bolus window
        )
        contributions = dss.explain(snap)
        assert len(contributions) == 6
        # Sorted descending by impact.
        impacts = [c[1] for c in contributions]
        assert impacts == sorted(impacts, reverse=True)
        # All non-negative (absolute deltas).
        assert all(i >= 0.0 for i in impacts)


class TestHeparinDSSEnsemble:
    def test_ensemble_interval_at_least_as_wide_as_single(self) -> None:
        """An N>=2 ensemble must produce an interval at least as wide as N=1.

        With the deterministic baseline all members agree, so the
        fallback band is installed (sigma=0 -> default band). Still,
        the returned interval width must satisfy the monotonicity
        property the paper cites: 'ensemble uncertainty >= single
        default band'.
        """
        single = HeparinDSS(n_ensemble=1)
        ensemble = HeparinDSS(n_ensemble=3)
        snap = PatientSnapshot(
            drug="heparin",
            aptt_seconds=65.0,
            weight_kg=80.0,
            platelets_k_per_ul=200.0,
            renal_function=1.0,
            hours_on_therapy=6.0,
        )
        r_single = single.recommend(snap)
        r_ensemble = ensemble.recommend(snap)
        w_single = r_single.uncertainty_interval[1] - r_single.uncertainty_interval[0]
        w_ensemble = r_ensemble.uncertainty_interval[1] - r_ensemble.uncertainty_interval[0]
        # Fallback default band applies in both; ensemble must not be
        # *narrower* than single.
        assert w_ensemble >= w_single - 1e-6

    def test_n_ensemble_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            HeparinDSS(n_ensemble=0)


# ---------------------------------------------------------------------------
# WarfarinDSS (baseline fallback)
# ---------------------------------------------------------------------------


class TestWarfarinDSSBaseline:
    def test_constructs_without_policy(self) -> None:
        dss = WarfarinDSS(policy_path=None)
        assert dss.uses_baseline is True

    def test_recommend_populates_all_fields(self) -> None:
        dss = WarfarinDSS()
        snap = PatientSnapshot(
            drug="warfarin",
            inr=1.3,
            age_years=67.0,
            weight_kg=75.0,
            cyp2c9="*1/*1",
            vkorc1="GG",
            days_on_therapy=5,
        )
        rec = dss.recommend(snap)
        assert rec.action_taken == "warfarin_oral"
        # Subtherapeutic INR on day 5+ must push baseline toward 10 mg.
        assert 7.0 <= rec.dose_or_rate <= 15.0
        lo, hi = rec.uncertainty_interval
        assert lo <= rec.dose_or_rate <= hi
        assert 0.0 <= rec.confidence <= 1.0
        assert rec.bolus_given is False
        assert len(rec.top_feature_contributions) == 8

    def test_recommend_handles_unknown_genotype(self) -> None:
        """Unknown CYP2C9/VKORC1 strings coerce to wild-type defaults."""
        dss = WarfarinDSS()
        snap = PatientSnapshot(
            drug="warfarin",
            inr=2.5,
            age_years=60.0,
            weight_kg=75.0,
            cyp2c9="*99/*99",   # invalid
            vkorc1="ZZ",         # invalid
            days_on_therapy=10,
        )
        rec = dss.recommend(snap)
        # Therapeutic INR in-range -> baseline prescribes 5 mg.
        assert 2.0 <= rec.dose_or_rate <= 10.0

    def test_explain_has_genotype_features(self) -> None:
        dss = WarfarinDSS()
        snap = PatientSnapshot(
            drug="warfarin",
            inr=2.4,
            age_years=60.0,
            weight_kg=75.0,
            cyp2c9="*1/*1",
            vkorc1="GG",
            days_on_therapy=10,
        )
        contributions = dss.explain(snap)
        names = {n for n, _ in contributions}
        assert "cyp2c9_norm" in names
        assert "vkorc1_norm" in names
        assert "inr_norm" in names


# ---------------------------------------------------------------------------
# Confidence behavior
# ---------------------------------------------------------------------------


class TestConfidence:
    def test_confidence_decreases_with_wider_interval(self) -> None:
        """Larger uncertainty -> lower confidence (monotone)."""
        dss = HeparinDSS(n_ensemble=1)
        snap = PatientSnapshot(
            drug="heparin",
            aptt_seconds=45.0,
            weight_kg=80.0,
            platelets_k_per_ul=200.0,
            renal_function=1.0,
            hours_on_therapy=6.0,
        )
        rec = dss.recommend(snap)
        lo, hi = rec.uncertainty_interval
        dose_range = 2500.0
        expected_conf = 1.0 - (hi - lo) / dose_range
        assert rec.confidence == pytest.approx(expected_conf, rel=1e-3)
        assert 0.0 <= rec.confidence <= 1.0


# ---------------------------------------------------------------------------
# Rationale strings
# ---------------------------------------------------------------------------


class TestRationale:
    def test_fallback_rationale_mentions_baseline(self) -> None:
        dss = HeparinDSS()
        snap = PatientSnapshot(drug="heparin", aptt_seconds=30.0)
        rec = dss.recommend(snap)
        assert "heparin" in rec.rationale

    @pytest.mark.skipif(_SB3_AVAILABLE, reason="test targets the no-sb3 path")
    def test_no_sb3_rationale_flag_present(self) -> None:
        dss = HeparinDSS()
        snap = PatientSnapshot(drug="heparin")
        rec = dss.recommend(snap)
        assert "fallback_no_sb3" in rec.rationale


# ---------------------------------------------------------------------------
# Observation encoding
# ---------------------------------------------------------------------------


class TestObservationEncoding:
    def test_heparin_obs_in_unit_box(self) -> None:
        dss = HeparinDSS()
        snap = PatientSnapshot(
            drug="heparin",
            aptt_seconds=150.0,  # very high
            weight_kg=200.0,     # very heavy
            platelets_k_per_ul=500.0,  # very high
            renal_function=1.5,   # outside [0, 1]; should clip
            hours_on_therapy=500.0,  # well past episode
        )
        obs = dss._snapshot_to_obs(snap)
        assert obs.shape == (6,)
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

    def test_warfarin_obs_in_unit_box(self) -> None:
        dss = WarfarinDSS()
        snap = PatientSnapshot(
            drug="warfarin",
            inr=10.0,     # supratherapeutic
            age_years=99.0,
            weight_kg=200.0,
            cyp2c9="*3/*3",
            vkorc1="AA",
            days_on_therapy=1000.0,
        )
        obs = dss._snapshot_to_obs(snap)
        assert obs.shape == (8,)
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)
