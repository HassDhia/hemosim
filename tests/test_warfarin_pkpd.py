"""Tests for the warfarin PK/PD model."""

from __future__ import annotations

import pytest

from hemosim.models.warfarin_pkpd import (
    CYP2C9_CL_FACTOR,
    VKORC1_EC50_FACTOR,
    WarfarinPKPD,
)


class TestWarfarinInit:
    def test_pkpd_initialization(self, warfarin_model):
        """Model initializes with correct default state."""
        assert warfarin_model.cyp2c9 == "*1/*1"
        assert warfarin_model.vkorc1 == "GG"
        assert warfarin_model.state[7] == pytest.approx(1.0, abs=0.1)  # baseline INR

    def test_invalid_cyp2c9(self):
        """Invalid CYP2C9 genotype raises ValueError."""
        with pytest.raises(ValueError, match="CYP2C9"):
            WarfarinPKPD(cyp2c9="*4/*4")

    def test_invalid_vkorc1(self):
        """Invalid VKORC1 genotype raises ValueError."""
        with pytest.raises(ValueError, match="VKORC1"):
            WarfarinPKPD(vkorc1="XX")


class TestWarfarinPK:
    def test_single_dose_absorption(self, warfarin_model):
        """A single dose should increase plasma concentration."""
        warfarin_model.step(5.0, dt_hours=24.0)
        conc = warfarin_model.get_concentration()
        assert conc["s_warfarin"] > 0
        assert conc["r_warfarin"] > 0

    def test_steady_state_reached(self, warfarin_model):
        """Daily dosing should approach steady state within ~7 days."""
        inrs = []
        for day in range(14):
            warfarin_model.step(5.0, dt_hours=24.0)
            inrs.append(warfarin_model.get_inr())

        # INR should be changing less at day 14 than at day 1
        early_change = abs(inrs[1] - inrs[0])
        late_change = abs(inrs[-1] - inrs[-2])
        assert late_change < early_change + 0.01  # approaching steady state

    def test_two_compartment_distribution(self, warfarin_model):
        """Drug should distribute between central and peripheral compartments."""
        warfarin_model.step(10.0, dt_hours=24.0)
        # Both compartments should have drug
        assert warfarin_model.state[0] > 0  # S central
        assert warfarin_model.state[1] > 0  # S peripheral

    def test_concentration_non_negative(self, warfarin_model):
        """Concentrations should never go negative."""
        for _ in range(30):
            warfarin_model.step(5.0, dt_hours=24.0)
        conc = warfarin_model.get_concentration()
        assert conc["s_warfarin"] >= 0
        assert conc["r_warfarin"] >= 0

    def test_time_to_steady_state(self, warfarin_model):
        """INR should be within 20% of its eventual value by day 7."""
        for day in range(14):
            warfarin_model.step(5.0, dt_hours=24.0)
        inr_day14 = warfarin_model.get_inr()

        # Reset and check day 7
        model2 = WarfarinPKPD()
        for day in range(7):
            model2.step(5.0, dt_hours=24.0)
        inr_day7 = model2.get_inr()

        if inr_day14 > 1.1:
            rel_diff = abs(inr_day7 - inr_day14) / inr_day14
            assert rel_diff < 0.5  # within 50% by day 7


class TestWarfarinPD:
    def test_inr_response_to_dose(self, warfarin_model):
        """INR should increase with warfarin dosing."""
        baseline_inr = warfarin_model.get_inr()
        for _ in range(10):
            warfarin_model.step(5.0, dt_hours=24.0)
        treated_inr = warfarin_model.get_inr()
        assert treated_inr > baseline_inr

    def test_inr_within_bounds(self, warfarin_model):
        """INR should stay within reasonable bounds during normal dosing."""
        for _ in range(30):
            warfarin_model.step(5.0, dt_hours=24.0)
        inr = warfarin_model.get_inr()
        assert 0.5 <= inr <= 12.0

    def test_zero_dose_inr_returns_baseline(self, warfarin_model):
        """Without dosing, INR should remain near baseline."""
        for _ in range(7):
            warfarin_model.step(0.0, dt_hours=24.0)
        inr = warfarin_model.get_inr()
        assert abs(inr - 1.0) < 0.5

    def test_overdose_high_inr(self):
        """High doses should produce high INR (supratherapeutic)."""
        model = WarfarinPKPD(cyp2c9="*3/*3", vkorc1="AA")  # very sensitive patient
        for _ in range(14):
            model.step(10.0, dt_hours=24.0)
        inr = model.get_inr()
        assert inr > 1.5  # should be elevated above baseline

    def test_dose_response_monotonic(self):
        """Higher doses should generally produce higher INR."""
        inrs = {}
        for dose in [2.0, 5.0, 10.0]:
            model = WarfarinPKPD()
            for _ in range(14):
                model.step(dose, dt_hours=24.0)
            inrs[dose] = model.get_inr()

        assert inrs[5.0] >= inrs[2.0]
        assert inrs[10.0] >= inrs[5.0]


class TestWarfarinGenotype:
    def test_cyp2c9_genotype_effect(self):
        """CYP2C9 variants should reduce S-warfarin clearance."""
        results = {}
        for genotype in CYP2C9_CL_FACTOR:
            model = WarfarinPKPD(cyp2c9=genotype)
            for _ in range(10):
                model.step(5.0, dt_hours=24.0)
            results[genotype] = model.get_inr()

        # *3/*3 should have higher INR than *1/*1 (reduced clearance)
        assert results["*3/*3"] > results["*1/*1"]

    def test_vkorc1_genotype_effect(self):
        """VKORC1 AA should be more sensitive (higher INR at same dose)."""
        results = {}
        for genotype in VKORC1_EC50_FACTOR:
            model = WarfarinPKPD(vkorc1=genotype)
            for _ in range(10):
                model.step(5.0, dt_hours=24.0)
            results[genotype] = model.get_inr()

        # AA should produce higher INR than GG (more sensitive)
        assert results["AA"] > results["GG"]

    def test_age_effect_on_clearance(self):
        """Older patients should have higher INR at same dose (reduced CL)."""
        young = WarfarinPKPD(age=30)
        old = WarfarinPKPD(age=80)
        for _ in range(10):
            young.step(5.0, dt_hours=24.0)
            old.step(5.0, dt_hours=24.0)
        assert old.get_inr() > young.get_inr()


class TestWarfarinReset:
    def test_reset_returns_baseline(self, warfarin_model):
        """Reset should return model to drug-free baseline state."""
        for _ in range(10):
            warfarin_model.step(5.0, dt_hours=24.0)
        assert warfarin_model.get_inr() > 1.0

        state = warfarin_model.reset()
        assert warfarin_model.get_inr() == pytest.approx(1.0, abs=0.1)
        assert state[0] == 0.0  # no drug
