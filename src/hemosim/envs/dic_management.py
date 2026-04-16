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

from hemosim.models.coagulation import CoagulationCascade
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

    def __init__(
        self,
        difficulty: str = "medium",
        coag_cascade_mode: bool = False,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        difficulty
            Patient-severity tier: ``"easy"``, ``"medium"``, or ``"hard"``.
        coag_cascade_mode
            When ``True``, the Hockin-Mann reduced coagulation cascade ODE
            (``hemosim.models.coagulation.CoagulationCascade``) runs alongside
            the flat algebraic dynamics and provides mechanistic fibrinogen
            evolution. When ``False`` (default, v0.1 behaviour), only the
            flat dynamics run. Enables downstream ablations and reviewer
            reassurance that the coagulation ODE is actually wired.
            Reference: Hockin MF et al. J Biol Chem 2002.
        """
        super().__init__()
        self.difficulty = difficulty
        self.coag_cascade_mode = bool(coag_cascade_mode)

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

        # Flat DIC state variables (primary state).
        self._platelet_count = 100.0   # x10^3/uL
        self._fibrinogen = 200.0       # mg/dL
        self._pt = 16.0                # seconds
        self._d_dimer = 4.0            # ug/mL FEU
        self._organ_function = 0.8     # 0-1, 1=normal
        self._hemorrhage_severity = 0.0  # 0-1
        self._heparin_active = False
        self._prev_dic_score = 5

        # Optional mechanistic coagulation cascade (ISC-3). The cascade is
        # allocated once per env instance and its state is reset on each
        # episode. Integration is silent when `coag_cascade_mode=False`.
        self._cascade: CoagulationCascade | None = (
            CoagulationCascade() if self.coag_cascade_mode else None
        )
        self._cascade_state: np.ndarray | None = None

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

        # Initialize cascade state when mode enabled. Baseline physiological
        # state perturbed by DIC severity: more activation (TF:VIIa, Xa,
        # thrombin), lower fibrinogen, higher AT-III consumption.
        if self._cascade is not None:
            severity_mult = {
                "mild": 1.0, "moderate": 1.5, "severe": 2.0,
            }.get(self._patient["severity"], 1.0)
            self._cascade_state = np.array([
                0.5 * severity_mult,               # TF:VIIa initial activation
                2.0 * severity_mult,               # Xa already elevated
                5.0 * severity_mult,               # Va co-activated
                10.0 * severity_mult,              # thrombin (nM)
                self._fibrinogen,                  # fibrinogen seed from patient
                severity_mult * 20.0,              # fibrin already formed
                200.0 * severity_mult,             # AT-III consumed
                0.15 * severity_mult,              # platelet activation frac
            ], dtype=np.float64)

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

        # --- Advance CoagulationCascade ODE (optional, ISC-3) ---
        #
        # Run the Hockin-Mann reduced cascade for the STEP_HOURS window and
        # blend the cascade-derived fibrinogen into the flat-dynamics value.
        # The cascade provides mechanistic consumption; flat dynamics still
        # provide the coarse-grained recovery / treatment-response terms.
        cascade_fibrinogen_delta = 0.0
        if self._cascade is not None and self._cascade_state is not None:
            # Heparin dampens the coagulation cascade by enhancing AT-III
            # activity — reflect this by transiently boosting AT-III.
            if self._heparin_active:
                heparin_at3_boost = 200.0 if heparin_level == 1 else 500.0
                self._cascade_state = self._cascade_state.copy()
                self._cascade_state[6] = min(
                    self._cascade_state[6] + heparin_at3_boost,
                    self._cascade.params["at3_total"] * 0.9,
                )
            # Transfused fibrinogen (from cryo + FFP) is already reflected in
            # self._fibrinogen above; sync the cascade to that value so the
            # ODE sees the treatment effect.
            pre_fib = float(self._cascade_state[4])
            self._cascade_state[4] = self._fibrinogen
            try:
                t_end_min = STEP_HOURS * 60.0
                _t, traj = self._cascade.simulate(
                    self._cascade_state,
                    t_span=(0.0, t_end_min),
                    dt=5.0,  # 5-min resolution; 4h window = 48 samples
                )
                self._cascade_state = np.asarray(traj[-1], dtype=np.float64)
                post_fib = float(self._cascade_state[4])
                cascade_fibrinogen_delta = post_fib - pre_fib
            except Exception:
                # ODE integration failures are non-fatal: cascade stalls but
                # env continues on flat dynamics. Never kill an episode
                # because of a stiff-ODE edge case.
                cascade_fibrinogen_delta = 0.0

        # --- Simulate DIC progression over STEP_HOURS ---

        severity_factor = {"mild": 0.5, "moderate": 1.0, "severe": 1.5}.get(
            self._patient["severity"], 1.0
        )
        if self.difficulty == "hard":
            severity_factor *= 1.3

        # Consumption: platelets and fibrinogen are consumed.
        # When the cascade ODE is active, half of the fibrinogen consumption
        # signal comes from the mechanistic cascade delta (negative when
        # cascade burns fibrinogen faster than thrombin generation can be
        # replenished). Keeps the flat-dynamics baseline as the ground
        # truth when cascade is off.
        consumption_rate = severity_factor * (1.0 - 0.3 * self._heparin_active)
        self._platelet_count -= consumption_rate * 3.0 * STEP_HOURS / 4.0
        flat_fibrinogen_consumption = consumption_rate * 5.0 * STEP_HOURS / 4.0
        if self._cascade is not None and cascade_fibrinogen_delta != 0.0:
            # Blend: 50% cascade-derived, 50% flat. Cascade delta is already
            # a signed change over the same window.
            self._fibrinogen += 0.5 * cascade_fibrinogen_delta
            self._fibrinogen -= 0.5 * flat_fibrinogen_consumption
        else:
            self._fibrinogen -= flat_fibrinogen_consumption

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
        info = {
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
            "coag_cascade_mode": self.coag_cascade_mode,
        }
        if self._cascade is not None and self._cascade_state is not None:
            # Expose cascade state for ablations and sensitivity analyses.
            # Order matches CoagulationCascade state vector.
            info["cascade_state"] = {
                "tf_viia": float(self._cascade_state[0]),
                "xa": float(self._cascade_state[1]),
                "va": float(self._cascade_state[2]),
                "thrombin": float(self._cascade_state[3]),
                "fibrinogen_cascade": float(self._cascade_state[4]),
                "fibrin": float(self._cascade_state[5]),
                "at3_bound": float(self._cascade_state[6]),
                "platelet_act_frac": float(self._cascade_state[7]),
            }
        return info
