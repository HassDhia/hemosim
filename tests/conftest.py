"""Shared test fixtures for hemosim test suite."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import hemosim  # noqa: F401 - triggers env registration
from hemosim.models.coagulation import CoagulationCascade
from hemosim.models.doac_pkpd import DOACPKPD
from hemosim.models.heparin_pkpd import HeparinPKPD
from hemosim.models.patient import PatientGenerator
from hemosim.models.warfarin_pkpd import WarfarinPKPD


@pytest.fixture
def coag_model():
    """Default coagulation cascade model."""
    return CoagulationCascade()


@pytest.fixture
def warfarin_model():
    """Default warfarin PK/PD model (wild-type genotype)."""
    return WarfarinPKPD(cyp2c9="*1/*1", vkorc1="GG", age=60, weight=75)


@pytest.fixture
def heparin_model():
    """Default heparin PK/PD model."""
    return HeparinPKPD(weight=80, renal_function=1.0, baseline_aptt=30)


@pytest.fixture
def rivaroxaban_model():
    """Default rivaroxaban PK model."""
    return DOACPKPD(drug="rivaroxaban", crcl=90, age=65, weight=75)


@pytest.fixture
def dabigatran_model():
    """Default dabigatran PK model."""
    return DOACPKPD(drug="dabigatran", crcl=90, age=65, weight=75)


@pytest.fixture
def apixaban_model():
    """Default apixaban PK model."""
    return DOACPKPD(drug="apixaban", crcl=90, age=65, weight=75)


@pytest.fixture
def patient_gen():
    """Patient generator with fixed seed."""
    return PatientGenerator(seed=42)


@pytest.fixture
def warfarin_env():
    """WarfarinDosing environment instance."""
    env = gym.make("hemosim/WarfarinDosing-v0")
    yield env
    env.close()


@pytest.fixture
def heparin_env():
    """HeparinInfusion environment instance."""
    env = gym.make("hemosim/HeparinInfusion-v0")
    yield env
    env.close()


@pytest.fixture
def doac_env():
    """DOACManagement environment instance."""
    env = gym.make("hemosim/DOACManagement-v0")
    yield env
    env.close()


@pytest.fixture
def dic_env():
    """DICManagement environment instance."""
    env = gym.make("hemosim/DICManagement-v0")
    yield env
    env.close()


@pytest.fixture
def initial_coag_state():
    """Standard initial coagulation state with TF initiation."""
    return np.array([
        1.0,    # TF_VIIa = 1.0 nM (triggered)
        0.0,    # Xa
        0.0,    # Va
        0.0,    # IIa (thrombin)
        300.0,  # fibrinogen (mg/dL)
        0.0,    # fibrin
        0.0,    # AT_III_bound
        0.0,    # platelet_activated
    ])


@pytest.fixture
def zero_coag_state():
    """Coagulation state with no TF trigger."""
    return np.array([0.0, 0.0, 0.0, 0.0, 300.0, 0.0, 0.0, 0.0])
