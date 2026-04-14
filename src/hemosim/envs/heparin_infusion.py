"""Heparin infusion management Gymnasium environment.

Simulates 120 hours (5 days) of unfractionated heparin therapy with
infusion rate decisions every 6 hours. The agent must maintain aPTT
in the therapeutic range (60-100 seconds) using bolus and infusion adjustments.

Observation space (6,):
    0: aPTT normalized (0-1, mapped from 20-200s)
    1: Heparin concentration (normalized)
    2: Weight (normalized)
    3: Renal function (0-1)
    4: Platelet count (normalized)
    5: Hours since start (normalized)

Action space (2,):
    0: Infusion rate scaled [0, 1] -> [0, 2500] U/hr
    1: Bolus flag: > 0.5 means give 80 U/kg bolus

Episode:
    120 hours, decisions every 6 hours (20 steps).
    Terminated if platelet_count < 50000 or aPTT > 150.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hemosim.models.heparin_pkpd import HeparinPKPD
from hemosim.models.patient import PatientGenerator

MAX_INFUSION_RATE = 2500.0  # U/hr
EPISODE_HOURS = 120
STEP_HOURS = 6
MAX_STEPS = EPISODE_HOURS // STEP_HOURS  # 20


class HeparinInfusionEnv(gym.Env):
    """Heparin infusion titration environment.

    Difficulty tiers:
        easy:   Stable clearance, no complications
        medium: Variable renal function
        hard:   HIT risk + bleeding events + renal variability
    """

    metadata = {"render_modes": []}

    def __init__(self, difficulty: str = "medium", **kwargs) -> None:
        super().__init__()
        self.difficulty = difficulty

        self.observation_space = spaces.Box(
            low=np.zeros(6, dtype=np.float32),
            high=np.ones(6, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._patient_gen = PatientGenerator()
        self._model: HeparinPKPD | None = None
        self._patient: dict | None = None
        self._step_count = 0
        self._hours_elapsed = 0.0
        self._np_random: np.random.Generator | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._np_random = self.np_random

        patient_seed = int(self.np_random.integers(0, 2**31))
        gen = PatientGenerator(seed=patient_seed)
        patient = gen.generate_heparin_patient()

        if self.difficulty == "easy":
            patient["renal_function"] = 1.0
            patient["weight"] = float(np.clip(patient["weight"], 60, 100))

        self._patient = patient
        self._model = HeparinPKPD(
            weight=patient["weight"],
            renal_function=patient["renal_function"],
            baseline_aptt=patient["baseline_aptt"],
        )
        self._model.reset()
        self._step_count = 0
        self._hours_elapsed = 0.0

        obs = self._get_obs()
        info = self._get_info(infusion_rate=0.0, bolus=False)
        return obs, info

    def step(self, action: np.ndarray):
        # Decode actions
        infusion_rate = float(np.clip(action[0], 0.0, 1.0)) * MAX_INFUSION_RATE
        give_bolus = float(action[1]) > 0.5
        bolus_u = 80.0 * self._patient["weight"] if give_bolus else 0.0

        # Advance model by STEP_HOURS
        self._model.step(
            infusion_rate_u_hr=infusion_rate,
            bolus_u=bolus_u,
            dt_hours=STEP_HOURS,
        )
        self._step_count += 1
        self._hours_elapsed += STEP_HOURS

        # Add variability for hard difficulty
        if self.difficulty == "hard" and self._np_random is not None:
            # Random renal function changes
            rf_change = self._np_random.normal(0, 0.02)
            self._model.renal_function = float(np.clip(
                self._model.renal_function + rf_change, 0.2, 1.0
            ))
            self._model.cl_renal_factor = 0.85 + 0.15 * self._model.renal_function

            # HIT risk: random platelet drop
            if self._hours_elapsed > 48 and self._np_random.random() < 0.02:
                self._model.state[3] *= 0.7  # sudden platelet drop

        aptt = self._model.get_aptt()
        platelet_count = self._model.get_platelet_count()

        # Reward shaping
        reward = -abs(aptt - 75.0) / 30.0  # distance from midpoint of therapeutic range
        if 60.0 <= aptt <= 100.0:
            reward += 0.5  # therapeutic range bonus
        if aptt > 120.0:
            reward -= 5.0  # bleeding risk
        if aptt < 45.0:
            reward -= 3.0  # subtherapeutic

        # Termination conditions
        terminated = False
        if platelet_count < 50.0:  # x10^3/uL - HIT threshold
            terminated = True
            reward -= 20.0
        if aptt > 150.0:
            terminated = True
            reward -= 15.0

        truncated = self._step_count >= MAX_STEPS

        obs = self._get_obs()
        info = self._get_info(infusion_rate=infusion_rate, bolus=give_bolus)

        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        aptt = self._model.get_aptt()
        conc = self._model.get_concentration()
        p = self._patient

        obs = np.array([
            np.clip((aptt - 20.0) / 180.0, 0.0, 1.0),              # aPTT normalized
            np.clip(conc / 1.0, 0.0, 1.0),                          # heparin conc
            np.clip((p["weight"] - 40) / 140, 0.0, 1.0),            # weight normalized
            np.clip(p["renal_function"], 0.0, 1.0),                  # renal function
            np.clip(self._model.get_platelet_count() / 400.0, 0.0, 1.0),  # platelets
            np.clip(self._hours_elapsed / EPISODE_HOURS, 0.0, 1.0), # time normalized
        ], dtype=np.float32)

        return obs

    def _get_info(self, infusion_rate: float, bolus: bool) -> dict:
        aptt = self._model.get_aptt()
        return {
            "aptt": aptt,
            "infusion_rate": infusion_rate,
            "bolus_given": bolus,
            "therapeutic": 60.0 <= aptt <= 100.0,
            "platelet_count": self._model.get_platelet_count(),
            "hours_elapsed": self._hours_elapsed,
            "patient": self._patient,
        }
