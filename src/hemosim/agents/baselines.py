"""Clinical baseline agents for comparison with RL policies.

Each baseline implements a standard clinical protocol or guideline-based
approach for the corresponding hemosim environment.
"""

from __future__ import annotations

import numpy as np


class WarfarinClinicalBaseline:
    """IWPC pharmacogenetic-guided warfarin dosing algorithm.

    Simplified fixed-dose protocol: start 5mg, adjust by INR.
    Based on the International Warfarin Pharmacogenetics Consortium algorithm.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Predict warfarin dose from observation.

        Args:
            obs: Observation vector from WarfarinDosingEnv.

        Returns:
            Action array [dose_scaled] where dose_scaled is in [0, 1].
        """
        inr = obs[0] * 6.0  # denormalize INR
        days = obs[7] * 90  # denormalize days

        # Initial dose: 5mg (scaled: 5/15 = 0.333)
        if days < 3:
            dose_mg = 5.0
        elif inr < 1.5:
            dose_mg = 7.5  # increase dose
        elif inr < 2.0:
            dose_mg = 5.0  # slight increase
        elif inr <= 3.0:
            dose_mg = 3.0  # therapeutic - maintain
        elif inr <= 4.0:
            dose_mg = 1.5  # reduce
        elif inr <= 5.0:
            dose_mg = 0.5  # hold/minimal
        else:
            dose_mg = 0.0  # hold

        dose_scaled = np.clip(dose_mg / 15.0, 0.0, 1.0)
        return np.array([dose_scaled], dtype=np.float32)


class HeparinRaschkeBaseline:
    """Raschke weight-based heparin dosing nomogram.

    Protocol: 80 U/kg bolus + 18 U/kg/hr initial, then adjust by aPTT.

    Reference: Raschke RA, et al. Ann Intern Med. 1993.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._first_step = True

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Predict heparin infusion rate and bolus decision.

        Args:
            obs: Observation vector from HeparinInfusionEnv.

        Returns:
            Action array [infusion_scaled, bolus_flag].
        """
        aptt = obs[0] * 180.0 + 20.0  # denormalize aPTT
        weight_norm = obs[2]
        weight = weight_norm * 140.0 + 40.0  # denormalize weight
        hours = obs[5] * 120.0

        # Initial: bolus + 18 U/kg/hr
        if hours < 1:
            infusion_rate = 18.0 * weight
            give_bolus = 1.0
            self._first_step = False
        else:
            give_bolus = 0.0

            # Raschke nomogram adjustments by aPTT
            if aptt < 45:
                # Re-bolus 80 U/kg, increase rate by 4 U/kg/hr
                infusion_rate = 22.0 * weight
                give_bolus = 1.0
            elif aptt < 60:
                # Increase rate by 2 U/kg/hr
                infusion_rate = 20.0 * weight
            elif aptt <= 100:
                # Therapeutic - maintain at 18 U/kg/hr
                infusion_rate = 18.0 * weight
            elif aptt <= 120:
                # Decrease rate by 2 U/kg/hr
                infusion_rate = 16.0 * weight
            else:
                # Hold for 1h then decrease by 3 U/kg/hr
                infusion_rate = 12.0 * weight

        infusion_scaled = np.clip(infusion_rate / 2500.0, 0.0, 1.0)
        return np.array([infusion_scaled, give_bolus], dtype=np.float32)


class DOACGuidelineBaseline:
    """Guideline-based DOAC selection for atrial fibrillation.

    Selects drug and dose based on CrCl, CHA2DS2-VASc, and HAS-BLED scores.
    Based on ESC/AHA guideline recommendations.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Predict DOAC drug and dose from observation.

        Args:
            obs: Observation vector from DOACManagementEnv.

        Returns:
            Action array [drug_choice, dose_level].
        """
        crcl = obs[1] * 130.0          # denormalize CrCl
        has_bled = obs[5] * 9.0         # denormalize HAS-BLED

        # Drug selection based on renal function
        if crcl < 25:
            drug = 2  # apixaban - safest in renal impairment
        elif crcl < 50:
            drug = 2  # apixaban preferred
        elif has_bled >= 3:
            drug = 2  # apixaban - lowest bleeding risk (ARISTOTLE)
        else:
            drug = 0  # rivaroxaban - once daily convenience

        # Dose selection based on renal function
        if crcl < 30:
            dose = 0  # low dose
        elif crcl < 50:
            dose = 0  # low dose
        else:
            dose = 1  # standard dose

        return np.array([drug, dose], dtype=np.int64)


class DICProtocolBaseline:
    """BCSH/ISTH guideline-based DIC management.

    Transfuse based on laboratory thresholds per ISTH guidelines:
    - Platelets < 50: transfuse if bleeding or high risk
    - Fibrinogen < 100: give cryoprecipitate
    - PT > 1.5x normal: give FFP
    - Heparin: consider for predominantly thrombotic DIC

    Reference: Levi M, et al. Br J Haematol. 2009.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Predict DIC treatment actions from observation.

        Args:
            obs: Observation vector from DICManagementEnv.

        Returns:
            Action array [platelet_tx, ffp_dose, cryo, heparin].
        """
        platelets = obs[1] * 300.0      # denormalize
        fibrinogen = obs[2] * 400.0     # denormalize
        pt = obs[3] * 30.0 + 10.0       # denormalize
        hemorrhage = obs[7]

        # Platelet transfusion decision
        if platelets < 20:
            plt_action = 3  # 4 units - critical
        elif platelets < 50 and hemorrhage > 0.3:
            plt_action = 2  # 2 units
        elif platelets < 50:
            plt_action = 1  # 1 unit
        else:
            plt_action = 0  # none

        # FFP decision (PT > 18s = ~1.5x normal 12s)
        if pt > 24:
            ffp_action = 3  # 6 units
        elif pt > 18:
            ffp_action = 2  # 4 units
        elif pt > 15:
            ffp_action = 1  # 2 units
        else:
            ffp_action = 0  # none

        # Cryoprecipitate (fibrinogen < 100 mg/dL)
        if fibrinogen < 80:
            cryo_action = 2  # 10 units
        elif fibrinogen < 100:
            cryo_action = 1  # 5 units
        else:
            cryo_action = 0  # none

        # Heparin decision
        # Low-dose heparin if predominantly thrombotic (low hemorrhage + high D-dimer)
        if hemorrhage < 0.2:
            hep_action = 1  # low dose
        else:
            hep_action = 0  # none if bleeding

        return np.array([plt_action, ffp_action, cryo_action, hep_action], dtype=np.int64)


class RandomBaseline:
    """Random action selection baseline.

    Uniformly samples from the action space. Used as a lower bound
    for performance comparison.
    """

    def __init__(self, action_space, seed: int | None = None) -> None:
        self.action_space = action_space
        self._rng = np.random.default_rng(seed)
        # Seed the gym action space too
        self.action_space.seed(seed if seed is not None else 42)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Sample a random action from the environment's action space."""
        return self.action_space.sample()
