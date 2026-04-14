"""DIC (Disseminated Intravascular Coagulation) management Gymnasium environment.

Simulates 168 hours (7 days) of DIC management with treatment decisions every
4 hours (42 steps per episode). The agent must manage blood product transfusions
and heparin therapy to control DIC while preventing organ failure.

Observation space (8,):
    0: ISTH DIC score (normalized 0-8 -> 0-1)
    1: Platelet count (normalized)
    2: Fibrinogen (normalized)
    3: PT prolongation (normalized)
    4: D-dimer (normalized)
    5: Organ function (0-1, 1=normal)
    6: Hours elapsed (normalized)
    7: Hemorrhage severity (0-1)

Action space: MultiDiscrete([4, 4, 3, 3])
    0: Platelet transfusion (0=none, 1=1unit, 2=2units, 3=4units)
    1: FFP dose (0=none, 1=2units, 2=4units, 3=6units)
    2: Cryoprecipitate (0=none, 1=5units, 2=10units)
    3: Heparin (0=none, 1=low dose, 2=therapeutic)

Episode:
    168 hours, decisions every 4 hours. Terminated if organ failure score
    exceeds threshold or platelets < 10,000.

References:
    Levi M, Toh CH, Thachil J, Watson HG. Guidelines for the diagnosis and
    management of disseminated intravascular coagulation. Br J Haematol. 2009.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hemosim.models.patient import PatientGenerator, _calculate_isth_score

EPISODE_HOURS = 168
STEP_HOURS = 4
MAX_STEPS = EPISODE_HOURS // STEP_HOURS  # 42

# Treatment effect parameters (simplified from clinical data)
# Platelet transfusion: ~30-60 x10^3/uL increase per unit (in non-DIC)
PLATELET_PER_UNIT = 15.0  # reduced in DIC due to consumption

# FFP: provides clotting factors, reduces PT
FFP_PT_REDUCTION_PER_UNIT = 0.8  # seconds PT reduction per unit

# Cryoprecipitate: ~50-70 mg/dL fibrinogen increase per 5 units
CRYO_FIBRINOGEN_PER_5U = 50.0

# Heparin doses
HEPARIN_LOW_RATE = 500.0        # U/hr - prophylactic
HEPARIN_THERAPEUTIC_RATE = 1000.0  # U/hr - therapeutic


class DICManagementEnv(gym.Env):
    """DIC management environment with blood product and heparin decisions.

    Difficulty tiers:
        easy:   Single cause, mild DIC, slow progression
        medium: Multi-organ involvement, moderate DIC
        hard:   Fulminant DIC with concurrent hemorrhage + thrombosis
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
        self.action_space = spaces.MultiDiscrete([4, 4, 3, 3])

        self._patient: dict | None = None
        self._step_count = 0
        self._hours_elapsed = 0.0
        self._np_random: np.random.Generator | None = None

        # DIC state variables (direct tracking, not using coag ODE model)
        self._platelet_count = 100.0   # x10^3/uL
        self._fibrinogen = 200.0       # mg/dL
        self._pt = 16.0                # seconds
        self._d_dimer = 4.0            # ug/mL FEU
        self._organ_function = 0.8     # 0-1, 1=normal
        self._hemorrhage_severity = 0.0  # 0-1
        self._heparin_active = False
        self._prev_dic_score = 5

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._np_random = self.np_random

        patient_seed = int(self.np_random.integers(0, 2**31))
        gen = PatientGenerator(seed=patient_seed)
        patient = gen.generate_dic_patient()

        if self.difficulty == "easy":
            patient["severity"] = "mild"
            patient["platelet_count"] = max(patient["platelet_count"], 80)
            patient["fibrinogen"] = max(patient["fibrinogen"], 150)
            patient["pt"] = min(patient["pt"], 18)

        self._patient = patient
        self._platelet_count = patient["platelet_count"]
        self._fibrinogen = patient["fibrinogen"]
        self._pt = patient["pt"]
        self._d_dimer = patient["d_dimer"]
        self._organ_function = patient["organ_function"]
        self._hemorrhage_severity = 0.0
        self._heparin_active = False
        self._step_count = 0
        self._hours_elapsed = 0.0

        self._prev_dic_score = _calculate_isth_score(
            self._platelet_count, self._fibrinogen, self._pt, self._d_dimer
        )

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray):
        # Decode actions
        platelet_units = [0, 1, 2, 4][int(action[0])]
        ffp_units = [0, 2, 4, 6][int(action[1])]
        cryo_units = [0, 5, 10][int(action[2])]
        heparin_level = int(action[3])  # 0=none, 1=low, 2=therapeutic

        # --- Apply treatments ---

        # Platelet transfusion effect (reduced in DIC due to ongoing consumption)
        if platelet_units > 0:
            increase = platelet_units * PLATELET_PER_UNIT
            self._platelet_count += increase

        # FFP effect (provides clotting factors)
        if ffp_units > 0:
            self._pt = max(self._pt - ffp_units * FFP_PT_REDUCTION_PER_UNIT, 11.0)
            # FFP also modestly increases fibrinogen
            self._fibrinogen += ffp_units * 5.0

        # Cryoprecipitate (concentrated fibrinogen source)
        if cryo_units > 0:
            self._fibrinogen += (cryo_units / 5.0) * CRYO_FIBRINOGEN_PER_5U

        # Heparin effect on microvascular thrombosis
        self._heparin_active = heparin_level > 0

        # --- Simulate DIC progression over STEP_HOURS ---

        severity_factor = {"mild": 0.5, "moderate": 1.0, "severe": 1.5}.get(
            self._patient["severity"], 1.0
        )
        if self.difficulty == "hard":
            severity_factor *= 1.3

        # Consumption: platelets and fibrinogen are consumed
        consumption_rate = severity_factor * (1.0 - 0.3 * self._heparin_active)
        self._platelet_count -= consumption_rate * 3.0 * STEP_HOURS / 4.0
        self._fibrinogen -= consumption_rate * 5.0 * STEP_HOURS / 4.0

        # PT worsens as factors are consumed
        self._pt += consumption_rate * 0.3 * STEP_HOURS / 4.0

        # D-dimer rises with fibrinolysis
        self._d_dimer += consumption_rate * 0.2 * STEP_HOURS / 4.0

        # Heparin can reduce microvascular thrombosis and slow organ damage
        if self._heparin_active:
            self._d_dimer *= 0.95  # mild reduction
            # But increases bleeding risk
            if heparin_level == 2:  # therapeutic
                self._hemorrhage_severity += self._np_random.uniform(0, 0.05)

        # Natural recovery (bone marrow, liver synthesis)
        self._platelet_count += 1.5 * STEP_HOURS / 4.0  # marrow production
        self._fibrinogen += 2.0 * STEP_HOURS / 4.0       # hepatic synthesis
        self._pt -= 0.1 * STEP_HOURS / 4.0               # factor recovery

        # Organ function dynamics
        # Low platelets + high D-dimer -> organ damage
        if self._platelet_count < 50 or self._d_dimer > 8.0:
            organ_damage = severity_factor * 0.01 * STEP_HOURS / 4.0
            self._organ_function -= organ_damage
        else:
            # Slow recovery
            self._organ_function += 0.005 * STEP_HOURS / 4.0

        # Hemorrhage dynamics
        if self._platelet_count < 30 or self._fibrinogen < 80:
            self._hemorrhage_severity += self._np_random.uniform(0, 0.08)
        else:
            self._hemorrhage_severity *= 0.9  # gradual resolution

        # Random noise
        if self._np_random is not None:
            self._platelet_count += self._np_random.normal(0, 2)
            self._fibrinogen += self._np_random.normal(0, 3)
            self._pt += self._np_random.normal(0, 0.2)

        # Clamp values to physiological bounds
        self._platelet_count = float(np.clip(self._platelet_count, 0, 500))
        self._fibrinogen = float(np.clip(self._fibrinogen, 10, 600))
        self._pt = float(np.clip(self._pt, 10, 60))
        self._d_dimer = float(np.clip(self._d_dimer, 0.1, 50))
        self._organ_function = float(np.clip(self._organ_function, 0, 1))
        self._hemorrhage_severity = float(np.clip(self._hemorrhage_severity, 0, 1))

        self._step_count += 1
        self._hours_elapsed += STEP_HOURS

        # Calculate ISTH DIC score
        dic_score = _calculate_isth_score(
            self._platelet_count, self._fibrinogen, self._pt, self._d_dimer
        )

        # --- Reward ---
        reward = -dic_score  # minimize DIC score

        # Bonus for improving DIC score
        if dic_score < self._prev_dic_score:
            reward += 2.0

        # Organ failure penalty
        if self._organ_function < 0.3:
            reward -= 5.0

        # Transfusion cost (resource utilization)
        transfusion_cost = (
            platelet_units * 0.1 + ffp_units * 0.1 + cryo_units * 0.05
        )
        reward -= transfusion_cost

        # Hemorrhage penalty
        if self._hemorrhage_severity > 0.5:
            reward -= 3.0

        self._prev_dic_score = dic_score

        # --- Termination ---
        terminated = False
        if self._organ_function < 0.15:
            terminated = True  # multi-organ failure
            reward -= 20.0
        if self._platelet_count < 10:
            terminated = True  # critical thrombocytopenia
            reward -= 15.0

        truncated = self._step_count >= MAX_STEPS

        obs = self._get_obs()
        info = self._get_info()

        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        dic_score = _calculate_isth_score(
            self._platelet_count, self._fibrinogen, self._pt, self._d_dimer
        )

        obs = np.array([
            np.clip(dic_score / 8.0, 0.0, 1.0),                           # ISTH DIC score
            np.clip(self._platelet_count / 300.0, 0.0, 1.0),              # platelets
            np.clip(self._fibrinogen / 400.0, 0.0, 1.0),                  # fibrinogen
            np.clip((self._pt - 10) / 30.0, 0.0, 1.0),                   # PT
            np.clip(self._d_dimer / 20.0, 0.0, 1.0),                     # D-dimer
            np.clip(self._organ_function, 0.0, 1.0),                      # organ function
            np.clip(self._hours_elapsed / EPISODE_HOURS, 0.0, 1.0),      # time
            np.clip(self._hemorrhage_severity, 0.0, 1.0),                 # hemorrhage
        ], dtype=np.float32)

        return obs

    def _get_info(self) -> dict:
        dic_score = _calculate_isth_score(
            self._platelet_count, self._fibrinogen, self._pt, self._d_dimer
        )
        return {
            "isth_dic_score": dic_score,
            "platelet_count": self._platelet_count,
            "fibrinogen": self._fibrinogen,
            "pt": self._pt,
            "d_dimer": self._d_dimer,
            "organ_function": self._organ_function,
            "hemorrhage_severity": self._hemorrhage_severity,
            "heparin_active": self._heparin_active,
            "hours_elapsed": self._hours_elapsed,
            "patient": self._patient,
        }
