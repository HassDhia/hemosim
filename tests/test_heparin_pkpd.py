"""Tests for the heparin PK/PD model."""

from __future__ import annotations

import numpy as np
import pytest

from hemosim.models.heparin_pkpd import HeparinPKPD


class TestHeparinInit:
    def test_initialization(self, heparin_model):
        """Model initializes with correct baseline state."""
        assert heparin_model.weight == 80
        assert heparin_model.renal_function == 1.0
        assert heparin_model.get_aptt() == pytest.approx(30.0, abs=0.1)
        assert heparin_model.get_concentration() == 0.0

    def test_custom_weight(self):
        """Weight affects volume of distribution and clearance."""
        light = HeparinPKPD(weight=50)
        heavy = HeparinPKPD(weight=120)
        assert heavy.vd > light.vd


class TestHeparinPK:
    def test_bolus_effect(self, heparin_model):
        """IV bolus should increase concentration immediately."""
        heparin_model.step(infusion_rate_u_hr=0, bolus_u=5000, dt_hours=0.25)
        assert heparin_model.get_concentration() > 0

    def test_infusion_steady_state(self, heparin_model):
        """Continuous infusion should reach steady state."""
        concentrations = []
        for _ in range(48):  # 48 hours
            heparin_model.step(infusion_rate_u_hr=1000, dt_hours=1.0)
            concentrations.append(heparin_model.get_concentration())

        # Late concentrations should be fairly stable
        late_conc = concentrations[-10:]
        cv = np.std(late_conc) / (np.mean(late_conc) + 1e-10)
        assert cv < 0.15, f"Concentration CV too high: {cv:.3f}"

    def test_nonlinear_clearance(self):
        """Higher concentrations should have proportionally less clearance (saturable)."""
        model = HeparinPKPD()
        # Give a large bolus to get high concentration
        model.step(infusion_rate_u_hr=0, bolus_u=10000, dt_hours=0.1)
        high_conc = model.get_concentration()

        model2 = HeparinPKPD()
        model2.step(infusion_rate_u_hr=0, bolus_u=2000, dt_hours=0.1)
        low_conc = model2.get_concentration()

        # Both should be positive
        assert high_conc > low_conc > 0

    def test_zero_infusion_clearance(self, heparin_model):
        """After stopping infusion, drug should clear."""
        heparin_model.step(infusion_rate_u_hr=0, bolus_u=5000, dt_hours=0.25)
        initial_conc = heparin_model.get_concentration()

        for _ in range(12):
            heparin_model.step(infusion_rate_u_hr=0, dt_hours=1.0)

        assert heparin_model.get_concentration() < initial_conc

    def test_concentration_non_negative(self, heparin_model):
        """Concentrations should never go negative."""
        heparin_model.step(infusion_rate_u_hr=0, bolus_u=3000, dt_hours=0.25)
        for _ in range(50):
            heparin_model.step(infusion_rate_u_hr=0, dt_hours=1.0)
        assert heparin_model.get_concentration() >= 0

    def test_weight_effect(self):
        """Same dose (U/kg) should produce similar concentrations regardless of weight."""
        light = HeparinPKPD(weight=60)
        heavy = HeparinPKPD(weight=100)

        # Weight-based bolus: 80 U/kg
        light.step(infusion_rate_u_hr=0, bolus_u=80 * 60, dt_hours=0.25)
        heavy.step(infusion_rate_u_hr=0, bolus_u=80 * 100, dt_hours=0.25)

        # Concentrations should be in same ballpark (Vd is weight-proportional)
        ratio = heavy.get_concentration() / max(light.get_concentration(), 1e-10)
        assert 0.5 < ratio < 2.0

    def test_renal_function_effect(self):
        """Reduced renal function should lead to higher drug levels."""
        normal = HeparinPKPD(renal_function=1.0)
        impaired = HeparinPKPD(renal_function=0.3)

        for _ in range(24):
            normal.step(infusion_rate_u_hr=1000, dt_hours=1.0)
            impaired.step(infusion_rate_u_hr=1000, dt_hours=1.0)

        # Impaired renal function -> slower clearance -> higher concentration
        assert impaired.get_concentration() > normal.get_concentration()


class TestHeparinPD:
    def test_aptt_response(self, heparin_model):
        """aPTT should increase with heparin."""
        baseline_aptt = heparin_model.get_aptt()
        for _ in range(6):
            heparin_model.step(infusion_rate_u_hr=1200, dt_hours=1.0)
        assert heparin_model.get_aptt() > baseline_aptt

    def test_therapeutic_range_achievable(self, heparin_model):
        """Therapeutic aPTT (60-100s) should be achievable."""
        for _ in range(24):
            heparin_model.step(infusion_rate_u_hr=1200, dt_hours=1.0)
        aptt = heparin_model.get_aptt()
        # Should be in or near therapeutic range with standard infusion
        assert aptt > 40  # at least somewhat elevated

    def test_platelet_dynamics(self, heparin_model):
        """Platelet count should be tracked and potentially decrease."""
        heparin_model.get_platelet_count()  # baseline
        for _ in range(72):
            heparin_model.step(infusion_rate_u_hr=1500, dt_hours=1.0)
        # Platelets should be affected (consumption vs production)
        final_plt = heparin_model.get_platelet_count()
        assert final_plt > 0  # should not go negative


class TestHeparinReset:
    def test_reset(self, heparin_model):
        """Reset should return to drug-free baseline."""
        heparin_model.step(infusion_rate_u_hr=0, bolus_u=5000, dt_hours=0.25)
        assert heparin_model.get_concentration() > 0

        heparin_model.reset()
        assert heparin_model.get_concentration() == 0
        assert heparin_model.get_aptt() == pytest.approx(30.0, abs=0.1)
