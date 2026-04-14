"""Warfarin dosing Gymnasium environment.

Simulates 90 days of warfarin therapy with daily dose decisions.
The agent must titrate warfarin to maintain INR within therapeutic range (2.0-3.0)
while accounting for patient-specific pharmacogenomics and PK/PD variability.

Observation space (8,):
    0: Current INR (normalized)
    1: S-warfarin plasma concentration (normalized)
    2: R-warfarin plasma concentration (normalized)
    3: Age (normalized to 0-1)
    4: Weight (normalized to 0-1)
    5: CYP2C9 genotype encoded (0-5 ordinal)
    6: VKORC1 genotype encoded (0-2 ordinal)
    7: Days on therapy (normalized to 0-1)

Action space (1,):
    Continuous dose scaled [0, 1] -> [0, 15] mg warfarin

Episode:
    90 days, one decision per day. Terminated early if INR > 6.0 or < 1.0.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hemosim.models.patient import PatientGenerator
from hemosim.models.warfarin_pkpd import WarfarinPKPD

# CYP2C9 genotype to ordinal encoding
_CYP2C9_ENCODE = {"*1/*1": 0, "*1/*2": 1, "*1/*3": 2, "*2/*2": 3, "*2/*3": 4, "*3/*3": 5}
_VKORC1_ENCODE = {"GG": 0, "GA": 1, "AA": 2}

MAX_DOSE_MG = 15.0
EPISODE_DAYS = 90


class WarfarinDosingEnv(gym.Env):
    """Warfarin dose titration environment.

    Difficulty tiers:
        easy:   Only *1/*1 + GG genotypes, stable patients
        medium: All genotypes, standard variability
        hard:   All genotypes + drug interactions + illness variability
    """

    metadata = {"render_modes": []}

    def __init__(self, difficulty: str = "medium", **kwargs) -> None:
        super().__init__()
        self.difficulty = difficulty

        self.observation_space = spaces.Box(
            low=np.zeros(8, dtype=np.float32),
            high=np.ones(8, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._patient_gen = PatientGenerator()
        self._model: WarfarinPKPD | None = None
        self._patient: dict | None = None
        self._day = 0
        self._np_random: np.random.Generator | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Use gymnasium-managed np_random for deterministic seeding
        self._np_random = self.np_random

        # Generate patient using a seed derived from np_random for determinism
        patient_seed = int(self.np_random.integers(0, 2**31))
        gen = PatientGenerator(seed=patient_seed)
        patient = gen.generate_warfarin_patient()

        # Apply difficulty constraints
        if self.difficulty == "easy":
            patient["cyp2c9"] = "*1/*1"
            patient["vkorc1"] = "GG"
            patient["age"] = float(np.clip(patient["age"], 50, 70))

        self._patient = patient
        self._model = WarfarinPKPD(
            cyp2c9=patient["cyp2c9"],
            vkorc1=patient["vkorc1"],
            age=patient["age"],
            weight=patient["weight"],
        )
        self._model.reset()
        self._day = 0

        obs = self._get_obs()
        info = self._get_info(dose_mg=0.0)
        return obs, info

    def step(self, action: np.ndarray):
        # Decode action: [0, 1] -> [0, MAX_DOSE_MG]
        dose_mg = float(np.clip(action[0], 0.0, 1.0)) * MAX_DOSE_MG

        # Advance PK/PD model by 24 hours
        self._model.step(dose_mg, dt_hours=24.0)
        self._day += 1

        # Add noise for hard difficulty (illness, drug interactions)
        if self.difficulty == "hard" and self._np_random is not None:
            # Random INR perturbation (simulating diet changes, illness, amiodarone)
            noise = self._np_random.normal(0, 0.1)
            self._model.state[7] = max(self._model.state[7] + noise, 0.5)

        inr = self._model.get_inr()
        target = self._patient["target_inr"]

        # Reward shaping
        reward = -abs(inr - target)  # distance from target
        if 2.0 <= inr <= 3.0:
            reward += 0.5  # bonus for therapeutic range
        if inr > 4.0:
            reward -= 10.0  # supratherapeutic penalty (bleeding risk)
        if inr < 1.5:
            reward -= 5.0  # subtherapeutic penalty (clotting risk)

        # Termination conditions
        terminated = False
        if inr > 6.0:
            terminated = True  # severe bleeding risk
            reward -= 20.0
        if inr < 1.0:
            terminated = True  # no anticoagulation effect
            reward -= 10.0

        truncated = self._day >= EPISODE_DAYS

        obs = self._get_obs()
        info = self._get_info(dose_mg=dose_mg)

        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        inr = self._model.get_inr()
        conc = self._model.get_concentration()
        p = self._patient

        obs = np.array([
            np.clip(inr / 6.0, 0.0, 1.0),                       # INR normalized
            np.clip(conc["s_warfarin"] / 5.0, 0.0, 1.0),        # S-warfarin conc
            np.clip(conc["r_warfarin"] / 5.0, 0.0, 1.0),        # R-warfarin conc
            np.clip((p["age"] - 20) / 75, 0.0, 1.0),            # age normalized
            np.clip((p["weight"] - 40) / 110, 0.0, 1.0),        # weight normalized
            _CYP2C9_ENCODE[p["cyp2c9"]] / 5.0,                  # CYP2C9 encoded
            _VKORC1_ENCODE[p["vkorc1"]] / 2.0,                  # VKORC1 encoded
            np.clip(self._day / EPISODE_DAYS, 0.0, 1.0),        # days normalized
        ], dtype=np.float32)

        return obs

    def _get_info(self, dose_mg: float) -> dict:
        inr = self._model.get_inr()
        return {
            "inr": inr,
            "dose_mg": dose_mg,
            "therapeutic": 2.0 <= inr <= 3.0,
            "day": self._day,
            "patient": self._patient,
        }
