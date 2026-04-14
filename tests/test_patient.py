"""Tests for the patient generator."""

from __future__ import annotations


from hemosim.models.patient import (
    CYP2C9_GENOTYPES,
    VKORC1_GENOTYPES,
    PatientGenerator,
)


class TestWarfarinPatient:
    def test_warfarin_patient_generation(self, patient_gen):
        """Warfarin patient should have all required fields."""
        patient = patient_gen.generate_warfarin_patient()
        required = ["age", "weight", "cyp2c9", "vkorc1", "target_inr",
                     "inr_range_low", "inr_range_high", "initial_dose_mg", "indication"]
        for field in required:
            assert field in patient, f"Missing field: {field}"

    def test_demographic_ranges(self, patient_gen):
        """Demographics should be within physiological ranges."""
        patient = patient_gen.generate_warfarin_patient()
        assert 20 <= patient["age"] <= 95
        assert 40 <= patient["weight"] <= 150

    def test_genotype_frequencies(self):
        """Generated genotype distribution should roughly match population frequencies."""
        gen = PatientGenerator(seed=123)
        cyp2c9_counts = {g: 0 for g in CYP2C9_GENOTYPES}
        vkorc1_counts = {g: 0 for g in VKORC1_GENOTYPES}

        n = 2000
        for i in range(n):
            p = gen.generate_warfarin_patient()
            cyp2c9_counts[p["cyp2c9"]] += 1
            vkorc1_counts[p["vkorc1"]] += 1

        # Check that *1/*1 is most common (should be ~65%)
        assert cyp2c9_counts["*1/*1"] / n > 0.55
        assert cyp2c9_counts["*1/*1"] / n < 0.75

        # Check VKORC1 GA is most common (~47%)
        assert vkorc1_counts["GA"] / n > 0.37
        assert vkorc1_counts["GA"] / n < 0.57


class TestHeparinPatient:
    def test_heparin_patient_generation(self, patient_gen):
        """Heparin patient should have required fields."""
        patient = patient_gen.generate_heparin_patient()
        assert "weight" in patient
        assert "renal_function" in patient
        assert "baseline_aptt" in patient
        assert "bleeding_risk" in patient
        assert 0.2 <= patient["renal_function"] <= 1.0


class TestDOACPatient:
    def test_doac_patient_generation(self, patient_gen):
        """DOAC patient should have all required fields."""
        patient = patient_gen.generate_doac_patient()
        required = ["age", "weight", "crcl", "cha2ds2_vasc", "has_bled",
                     "indication", "initial_drug"]
        for field in required:
            assert field in patient

    def test_cha2ds2vasc_range(self, patient_gen):
        """CHA2DS2-VASc should be in valid range."""
        patient = patient_gen.generate_doac_patient()
        assert 0 <= patient["cha2ds2_vasc"] <= 9


class TestDICPatient:
    def test_dic_patient_generation(self, patient_gen):
        """DIC patient should have required fields with valid ranges."""
        patient = patient_gen.generate_dic_patient()
        assert patient["cause"] in ["sepsis", "trauma", "malignancy", "obstetric", "vascular"]
        assert patient["severity"] in ["mild", "moderate", "severe"]
        assert patient["platelet_count"] > 0
        assert patient["fibrinogen"] > 0
        assert patient["pt"] > 0
        assert patient["d_dimer"] > 0
        assert 0 <= patient["isth_dic_score"] <= 8


class TestReproducibility:
    def test_reproducibility_with_seed(self):
        """Same seed should produce identical patients."""
        p1 = PatientGenerator(seed=99).generate_warfarin_patient()
        p2 = PatientGenerator(seed=99).generate_warfarin_patient()
        assert p1["age"] == p2["age"]
        assert p1["cyp2c9"] == p2["cyp2c9"]
        assert p1["vkorc1"] == p2["vkorc1"]

    def test_different_seeds_different_patients(self):
        """Different seeds should produce different patients."""
        p1 = PatientGenerator(seed=1).generate_warfarin_patient()
        p2 = PatientGenerator(seed=2).generate_warfarin_patient()
        # At least one field should differ (extremely unlikely to be identical)
        different = (
            p1["age"] != p2["age"]
            or p1["weight"] != p2["weight"]
            or p1["cyp2c9"] != p2["cyp2c9"]
        )
        assert different

    def test_patient_fields_complete(self, patient_gen):
        """All patient types should return dicts with string keys and valid values."""
        for method in [
            patient_gen.generate_warfarin_patient,
            patient_gen.generate_heparin_patient,
            patient_gen.generate_doac_patient,
            patient_gen.generate_dic_patient,
        ]:
            patient = method()
            assert isinstance(patient, dict)
            assert len(patient) > 0
            for key, value in patient.items():
                assert isinstance(key, str)
                assert value is not None
