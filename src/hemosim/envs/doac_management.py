"""DOAC management Gymnasium environment.

Simulates 365 days of direct oral anticoagulant therapy for atrial fibrillation,
with drug and dose decisions every 30 days (12 steps per episode).

The agent must select the appropriate DOAC and dose level based on patient
characteristics, stroke risk (CHA2DS2-VASc), and bleeding risk (HAS-BLED),
while monitoring for renal function changes.

Observation space (8,):
    0: Drug concentration (normalized)
    1: CrCl (normalized)
    2: Age (normalized)
    3: Weight (normalized)
    4: CHA2DS2-VASc (normalized)
    5: HAS-BLED (normalized)
    6: Days on therapy (normalized)
    7: Current drug encoded (0-2)

Action space: MultiDiscrete([3, 3])
    0: Drug choice (0=rivaroxaban, 1=dabigatran, 2=apixaban)
    1: Dose level (0=low, 1=standard, 2=high)

Episode:
    365 days, decisions every 30 days. Terminated on stroke or fatal bleed.

Event probabilities derived from:
    RE-LY (dabigatran), ROCKET-AF (rivaroxaban), ARISTOTLE (apixaban) trials.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hemosim.models.doac_pkpd import DOACPKPD, DRUG_PARAMS
from hemosim.models.patient import PatientGenerator

DRUGS = ["rivaroxaban", "dabigatran", "apixaban"]
EPISODE_DAYS = 365
STEP_DAYS = 30
MAX_STEPS = EPISODE_DAYS // STEP_DAYS  # 12

# Annual event rates from landmark trials (per 100 patient-years)
# Stroke/systemic embolism rates (standard dose)
STROKE_RATES = {
    "rivaroxaban": 2.1,  # ROCKET-AF
    "dabigatran": 1.1,   # RE-LY 150mg
    "apixaban": 1.27,    # ARISTOTLE
}

# Major bleeding rates (standard dose, per 100 patient-years)
BLEED_RATES = {
    "rivaroxaban": 3.6,  # ROCKET-AF
    "dabigatran": 3.1,   # RE-LY 150mg
    "apixaban": 2.13,    # ARISTOTLE
}


class DOACManagementEnv(gym.Env):
    """DOAC selection and dosing environment for atrial fibrillation.

    Difficulty tiers:
        easy:   Stable renal function, no complications
        medium: Declining CrCl over time
        hard:   Drug interactions + perioperative management + renal decline
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
        self.action_space = spaces.MultiDiscrete([3, 3])

        self._patient_gen = PatientGenerator()
        self._model: DOACPKPD | None = None
        self._patient: dict | None = None
        self._current_drug_idx = 0
        self._step_count = 0
        self._days_elapsed = 0
        self._np_random: np.random.Generator | None = None
        self._current_crcl = 90.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._np_random = self.np_random

        patient_seed = int(self.np_random.integers(0, 2**31))
        gen = PatientGenerator(seed=patient_seed)
        patient = gen.generate_doac_patient()

        if self.difficulty == "easy":
            patient["crcl"] = float(np.clip(patient["crcl"], 60, 120))

        self._patient = patient
        self._current_crcl = patient["crcl"]
        self._current_drug_idx = DRUGS.index(patient["initial_drug"])

        self._model = DOACPKPD(
            drug=patient["initial_drug"],
            crcl=patient["crcl"],
            age=patient["age"],
            weight=patient["weight"],
        )
        self._model.reset()
        self._step_count = 0
        self._days_elapsed = 0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray):
        drug_idx = int(action[0])
        dose_level = int(action[1])

        drug_name = DRUGS[drug_idx]
        drug_params = DRUG_PARAMS[drug_name]

        # Determine dose from level
        doses = [drug_params["low_dose"], drug_params["standard_dose"], drug_params["high_dose"]]
        dose_mg = doses[dose_level]

        # Check for drug switch
        drug_switched = drug_idx != self._current_drug_idx

        # If drug changed, create new PK model
        if drug_switched:
            self._model = DOACPKPD(
                drug=drug_name,
                crcl=self._current_crcl,
                age=self._patient["age"],
                weight=self._patient["weight"],
            )
            self._model.reset()
            self._current_drug_idx = drug_idx

        # Simulate 30 days of therapy (dose twice daily for most DOACs)
        # Simplified: give daily equivalent dose in one step per day
        for day in range(STEP_DAYS):
            self._model.step(dose_mg, dt_hours=24.0)

        self._step_count += 1
        self._days_elapsed += STEP_DAYS

        # Renal function changes
        if self.difficulty in ("medium", "hard"):
            # CrCl declines ~1-3 mL/min per year for elderly
            crcl_decline = self._np_random.normal(0.25, 0.15) * (STEP_DAYS / 30)
            self._current_crcl = max(self._current_crcl - crcl_decline, 10.0)
            self._model.crcl = self._current_crcl

        # Event simulation for this 30-day period
        # Convert annual rates to 30-day probabilities
        cha2ds2 = self._patient["cha2ds2_vasc"]
        has_bled = self._patient["has_bled"]

        # Stroke probability: base rate * CHA2DS2-VASc modifier
        stroke_base = STROKE_RATES[drug_name] / 100.0  # annual rate
        stroke_modifier = 1.0 + 0.3 * max(cha2ds2 - 2, 0)  # higher score = more risk
        stroke_30d_prob = 1.0 - (1.0 - stroke_base * stroke_modifier) ** (30 / 365)
        stroke_event = self._np_random.random() < stroke_30d_prob

        # Bleeding probability: base rate * HAS-BLED modifier * dose modifier
        bleed_base = BLEED_RATES[drug_name] / 100.0
        bleed_modifier = 1.0 + 0.25 * max(has_bled - 2, 0)
        dose_bleed_modifier = [0.7, 1.0, 1.3][dose_level]  # low/standard/high
        bleed_30d_prob = 1.0 - (1.0 - bleed_base * bleed_modifier * dose_bleed_modifier) ** (
            30 / 365
        )
        bleed_event = self._np_random.random() < bleed_30d_prob

        # Hard difficulty: additional drug interaction risk
        if self.difficulty == "hard" and self._np_random.random() < 0.05:
            # Drug interaction event increases bleeding risk
            bleed_event = bleed_event or (self._np_random.random() < 0.1)

        # Reward calculation
        reward = 1.0  # base reward for surviving 30 days
        if stroke_event:
            reward -= 20.0
        if bleed_event:
            reward -= 10.0
        if drug_switched:
            reward -= 5.0  # transition cost

        # Renal dose appropriateness bonus
        recommended_dose = self._model.get_dose_for_renal()
        if dose_mg == recommended_dose:
            reward += 0.5

        # Termination
        terminated = stroke_event or (bleed_event and self._np_random.random() < 0.15)
        truncated = self._step_count >= MAX_STEPS

        obs = self._get_obs()
        info = self._get_info(
            stroke=stroke_event,
            bleed=bleed_event,
            drug_switched=drug_switched,
        )

        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        conc = self._model.get_concentration()
        p = self._patient

        obs = np.array([
            np.clip(conc / 0.5, 0.0, 1.0),                              # drug concentration
            np.clip(self._current_crcl / 130.0, 0.0, 1.0),              # CrCl
            np.clip((p["age"] - 40) / 55, 0.0, 1.0),                    # age
            np.clip((p["weight"] - 45) / 105, 0.0, 1.0),                # weight
            np.clip(p["cha2ds2_vasc"] / 9.0, 0.0, 1.0),                 # CHA2DS2-VASc
            np.clip(p["has_bled"] / 9.0, 0.0, 1.0),                     # HAS-BLED
            np.clip(self._days_elapsed / EPISODE_DAYS, 0.0, 1.0),       # time
            self._current_drug_idx / 2.0,                                # drug encoded
        ], dtype=np.float32)

        return obs

    def _get_info(self, stroke: bool = False, bleed: bool = False,
                  drug_switched: bool = False) -> dict:
        return {
            "drug": DRUGS[self._current_drug_idx],
            "concentration": self._model.get_concentration(),
            "crcl": self._current_crcl,
            "stroke_event": stroke,
            "bleed_event": bleed,
            "drug_switched": drug_switched,
            "days_elapsed": self._days_elapsed,
            "patient": self._patient,
        }
