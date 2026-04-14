"""Tests for the DOAC PK/PD models."""

from __future__ import annotations

import numpy as np
import pytest

from hemosim.models.doac_pkpd import DOACPKPD


class TestDOACInit:
    def test_rivaroxaban_pk(self, rivaroxaban_model):
        """Rivaroxaban model initializes correctly."""
        assert rivaroxaban_model.drug == "rivaroxaban"
        assert rivaroxaban_model.get_concentration() == 0.0

    def test_dabigatran_pk(self, dabigatran_model):
        """Dabigatran model initializes correctly."""
        assert dabigatran_model.drug == "dabigatran"
        assert dabigatran_model.get_concentration() == 0.0

    def test_apixaban_pk(self, apixaban_model):
        """Apixaban model initializes correctly."""
        assert apixaban_model.drug == "apixaban"

    def test_unknown_drug(self):
        """Unknown drug name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown drug"):
            DOACPKPD(drug="aspirin")


class TestDOACPharmacokinetics:
    def test_oral_absorption_profile(self, rivaroxaban_model):
        """Drug should absorb from gut and appear in plasma."""
        rivaroxaban_model.step(20.0, dt_hours=4.0)
        assert rivaroxaban_model.get_concentration() > 0

    def test_peak_concentration(self, rivaroxaban_model):
        """Peak concentration should occur within hours of dosing."""
        concs = []
        rivaroxaban_model.step(20.0, dt_hours=0.0)  # dose without time advance
        for h in range(24):
            rivaroxaban_model.step(0.0, dt_hours=1.0)  # advance without dose
            concs.append(rivaroxaban_model.get_concentration())

        peak_hour = np.argmax(concs)
        assert 0 <= peak_hour < 12  # peak within 12 hours

    def test_trough_concentration(self, rivaroxaban_model):
        """Trough concentration (just before next dose) should be lower than peak."""
        # Give a dose and measure at different times
        rivaroxaban_model.step(20.0, dt_hours=4.0)
        peak_conc = rivaroxaban_model.get_concentration()

        rivaroxaban_model.step(0.0, dt_hours=20.0)  # 24h total, no new dose
        trough_conc = rivaroxaban_model.get_concentration()

        assert trough_conc < peak_conc

    def test_crcl_effect_on_clearance(self):
        """Lower CrCl should result in higher drug levels for renally cleared drugs."""
        normal = DOACPKPD(drug="dabigatran", crcl=90)
        impaired = DOACPKPD(drug="dabigatran", crcl=30)

        for _ in range(5):
            normal.step(150.0, dt_hours=12.0)
            impaired.step(150.0, dt_hours=12.0)

        # Dabigatran is 80% renally cleared - impaired should have higher levels
        assert impaired.get_concentration() > normal.get_concentration()

    def test_renal_dose_adjustment(self):
        """Model should recommend lower dose for impaired renal function."""
        normal = DOACPKPD(drug="rivaroxaban", crcl=90)
        impaired = DOACPKPD(drug="rivaroxaban", crcl=30)

        assert impaired.get_dose_for_renal() < normal.get_dose_for_renal()

    def test_antixa_activity(self, rivaroxaban_model):
        """Rivaroxaban should produce measurable anti-Xa activity."""
        rivaroxaban_model.step(20.0, dt_hours=4.0)
        antixa = rivaroxaban_model.get_antixa_activity()
        assert antixa > 0

    def test_dabigatran_no_antixa(self, dabigatran_model):
        """Dabigatran (thrombin inhibitor) should have no anti-Xa activity."""
        dabigatran_model.step(150.0, dt_hours=4.0)
        antixa = dabigatran_model.get_antixa_activity()
        assert antixa == 0.0

    def test_drug_switching(self):
        """Creating a new model with different drug should work cleanly."""
        riva = DOACPKPD(drug="rivaroxaban")
        riva.step(20.0, dt_hours=4.0)

        api = DOACPKPD(drug="apixaban")
        api.step(5.0, dt_hours=4.0)

        assert riva.drug != api.drug
        assert riva.get_concentration() > 0
        assert api.get_concentration() > 0

    def test_concentration_non_negative(self, rivaroxaban_model):
        """Concentrations should never go negative."""
        rivaroxaban_model.step(20.0, dt_hours=4.0)
        for _ in range(100):
            rivaroxaban_model.step(0.0, dt_hours=12.0)
        assert rivaroxaban_model.get_concentration() >= 0


class TestDOACReset:
    def test_reset(self, rivaroxaban_model):
        """Reset should return to drug-free state."""
        rivaroxaban_model.step(20.0, dt_hours=4.0)
        assert rivaroxaban_model.get_concentration() > 0

        rivaroxaban_model.reset()
        assert rivaroxaban_model.get_concentration() == 0.0
        assert np.all(rivaroxaban_model.state == 0)
