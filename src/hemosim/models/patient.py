"""Virtual patient generator for hemostasis simulation environments.

Generates realistic patient profiles with demographics, genetics, and clinical
parameters drawn from published population distributions.

Population frequencies for pharmacogenomics are from:
- CYP2C9: Scordo et al. (2001), European population
- VKORC1: Rieder et al. (2005), European population
"""

from __future__ import annotations

import numpy as np


# CYP2C9 allele frequencies (European population)
CYP2C9_GENOTYPES = ["*1/*1", "*1/*2", "*1/*3", "*2/*2", "*2/*3", "*3/*3"]
CYP2C9_FREQUENCIES = [0.654, 0.222, 0.078, 0.019, 0.015, 0.012]

# VKORC1 -1639G>A genotype frequencies (European population)
VKORC1_GENOTYPES = ["GG", "GA", "AA"]
VKORC1_FREQUENCIES = [0.37, 0.47, 0.16]

# DIC underlying causes and their severity distributions
DIC_CAUSES = ["sepsis", "trauma", "malignancy", "obstetric", "vascular"]


class PatientGenerator:
    """Generate virtual patients with realistic clinical profiles.

    Uses numpy random generators for reproducibility (not global state).
    All parameters drawn from published population distributions.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def generate_warfarin_patient(self, seed: int | None = None) -> dict:
        """Generate a patient profile for warfarin dosing.

        Demographics and genetics drawn from population distributions.
        Target INR 2.5 (range 2.0-3.0) for standard AF indication.

        Returns:
            Dict with patient parameters for WarfarinPKPD model.
        """
        rng = np.random.default_rng(seed) if seed is not None else self._rng

        age = float(np.clip(rng.normal(65, 12), 20, 95))
        weight = float(np.clip(rng.normal(80, 18), 40, 150))

        cyp2c9 = rng.choice(CYP2C9_GENOTYPES, p=CYP2C9_FREQUENCIES)
        vkorc1 = rng.choice(VKORC1_GENOTYPES, p=VKORC1_FREQUENCIES)

        return {
            "age": age,
            "weight": weight,
            "cyp2c9": str(cyp2c9),
            "vkorc1": str(vkorc1),
            "target_inr": 2.5,
            "inr_range_low": 2.0,
            "inr_range_high": 3.0,
            "initial_dose_mg": 5.0,
            "indication": "atrial_fibrillation",
        }

    def generate_heparin_patient(self, seed: int | None = None) -> dict:
        """Generate a patient profile for heparin infusion management.

        Returns:
            Dict with patient parameters for HeparinPKPD model.
        """
        rng = np.random.default_rng(seed) if seed is not None else self._rng

        weight = float(np.clip(rng.normal(80, 20), 40, 180))
        renal_function = float(np.clip(rng.normal(0.85, 0.2), 0.2, 1.0))
        baseline_aptt = float(np.clip(rng.normal(30, 5), 22, 40))

        # Bleeding risk score (simplified HAS-BLED-like)
        bleeding_risk = float(np.clip(rng.normal(2.0, 1.5), 0, 6))

        return {
            "weight": weight,
            "renal_function": renal_function,
            "baseline_aptt": baseline_aptt,
            "bleeding_risk": bleeding_risk,
            "target_aptt_low": 60.0,
            "target_aptt_high": 100.0,
            "indication": rng.choice(["dvt", "pe", "acs", "af"]),
        }

    def generate_doac_patient(self, seed: int | None = None) -> dict:
        """Generate an atrial fibrillation patient for DOAC management.

        Includes CHA2DS2-VASc and HAS-BLED risk scores.

        Returns:
            Dict with patient parameters for DOAC environment.
        """
        rng = np.random.default_rng(seed) if seed is not None else self._rng

        age = float(np.clip(rng.normal(72, 10), 40, 95))
        weight = float(np.clip(rng.normal(82, 18), 45, 150))
        crcl = float(np.clip(rng.normal(75, 25), 15, 130))

        # CHA2DS2-VASc score components (simplified generation)
        cha2ds2_vasc = int(np.clip(rng.poisson(3.0), 0, 9))

        # HAS-BLED score
        has_bled = int(np.clip(rng.poisson(2.0), 0, 9))

        return {
            "age": age,
            "weight": weight,
            "crcl": crcl,
            "cha2ds2_vasc": cha2ds2_vasc,
            "has_bled": has_bled,
            "indication": "atrial_fibrillation",
            "initial_drug": rng.choice(["rivaroxaban", "dabigatran", "apixaban"]),
        }

    def generate_dic_patient(self, seed: int | None = None) -> dict:
        """Generate a DIC patient with underlying cause and baseline labs.

        DIC severity based on ISTH scoring system.

        Returns:
            Dict with patient parameters for DIC environment.
        """
        rng = np.random.default_rng(seed) if seed is not None else self._rng

        cause = rng.choice(DIC_CAUSES)

        # Severity determines initial lab derangements
        # ISTH DIC score: 0-8, >=5 is overt DIC
        severity = rng.choice(["mild", "moderate", "severe"], p=[0.3, 0.45, 0.25])

        severity_multiplier = {"mild": 1.0, "moderate": 1.5, "severe": 2.0}[severity]

        # Baseline labs (already deranged based on severity)
        platelet_count = float(np.clip(
            rng.normal(120 / severity_multiplier, 30), 20, 250
        ))  # x10^3/uL
        fibrinogen = float(np.clip(
            rng.normal(250 / severity_multiplier, 50), 50, 400
        ))  # mg/dL
        pt = float(np.clip(
            rng.normal(14 * severity_multiplier, 2), 10, 40
        ))  # seconds
        d_dimer = float(np.clip(
            rng.exponential(2.0 * severity_multiplier), 0.5, 30
        ))  # ug/mL FEU

        # Calculate initial ISTH DIC score
        isth_score = _calculate_isth_score(platelet_count, fibrinogen, pt, d_dimer)

        return {
            "cause": str(cause),
            "severity": str(severity),
            "platelet_count": platelet_count,
            "fibrinogen": fibrinogen,
            "pt": pt,
            "d_dimer": d_dimer,
            "isth_dic_score": isth_score,
            "organ_function": float(np.clip(1.0 / severity_multiplier, 0.3, 1.0)),
            "age": float(np.clip(rng.normal(60, 15), 20, 90)),
            "weight": float(np.clip(rng.normal(75, 18), 40, 150)),
        }


def _calculate_isth_score(
    platelets: float,
    fibrinogen: float,
    pt: float,
    d_dimer: float,
) -> int:
    """Calculate ISTH DIC score (0-8).

    Based on the International Society on Thrombosis and Haemostasis
    scoring system for disseminated intravascular coagulation.
    """
    score = 0

    # Platelet count (x10^3/uL)
    if platelets < 50:
        score += 2
    elif platelets < 100:
        score += 1

    # Fibrinogen (mg/dL)
    if fibrinogen < 100:
        score += 1

    # PT prolongation (seconds above normal ~12s)
    pt_prolongation = pt - 12.0
    if pt_prolongation > 6:
        score += 2
    elif pt_prolongation > 3:
        score += 1

    # D-dimer (ug/mL FEU)
    if d_dimer > 4.0:
        score += 3
    elif d_dimer > 1.0:
        score += 2

    return score
