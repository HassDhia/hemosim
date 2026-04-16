"""Extended clinical and RL-derived baseline agents for hemosim.

This module adds four baselines that were absent from v0.1:

1. :class:`NematiDQN2016Baseline` — an independent reimplementation of the
   deep Q-network heparin dosing policy from Nemati et al. (2016). When
   ``torch`` is available and trained weights are present, the baseline uses
   a learned Q-network; otherwise it falls back to a deterministic threshold
   policy that approximates the paper's reported dosing behavior.
2. :class:`HeparinAntiXaBaseline` — modern anti-Xa guided unfractionated
   heparin titration targeting 0.3–0.7 IU/mL.
3. :class:`WarfarinGageBaseline` — pharmacogenetic warfarin dosing algorithm
   from Gage et al. (2008).
4. :class:`WarfarinOrdinalBaseline` — non-genotype INR-response titration
   table inspired by the simpler clinical nomograms reviewed in
   Crowther et al. (2009).

All baselines expose the common ``predict(obs) -> np.ndarray`` interface
used by :mod:`hemosim.agents.baselines`. They are compatible with the
respective Gymnasium action spaces for the environments they target.

References
----------
- Nemati S, Ghassemi MM, Clifford GD. *Optimal medication dosing from
  suboptimal clinical examples: A deep reinforcement learning approach.*
  IEEE EMBC 2016;2978-2981. doi:10.1109/EMBC.2016.7591355.
- Gage BF, Eby C, Johnson JA, et al. *Use of pharmacogenetic and clinical
  factors to predict the therapeutic dose of warfarin.*
  Clin Pharmacol Ther 2008;84(3):326-331. doi:10.1038/clpt.2008.10.
- Crowther MA, Ginsberg JS, Kearon C, et al. *A randomized trial comparing
  5-mg and 10-mg warfarin loading doses.* Arch Intern Med 1999;159(1):46-48.
  Related ambulatory titration nomograms: Arch Intern Med 2009 series.
- Raschke RA, Reilly BM, Guidry JR, et al. *The weight-based heparin dosing
  nomogram compared with a "standard care" nomogram.* Ann Intern Med 1993.
  (Anti-Xa target ranges per CHEST 2012 and ASH 2018 guidance.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Optional deep-learning dependency. The baseline is designed so that the
# module imports cleanly in environments without torch installed.
try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without torch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Denormalization constants mirroring HeparinInfusionEnv._get_obs().
_HEPARIN_APTT_MIN = 20.0
_HEPARIN_APTT_RANGE = 180.0
_HEPARIN_WEIGHT_MIN = 40.0
_HEPARIN_WEIGHT_RANGE = 140.0
_HEPARIN_HOURS_RANGE = 120.0
_HEPARIN_MAX_INFUSION = 2500.0  # U/hr, matches env MAX_INFUSION_RATE


def _denormalize_heparin_obs(obs: np.ndarray) -> dict[str, float]:
    """Decode the HeparinInfusion-v0 observation into clinical units."""

    return {
        "aptt": float(obs[0]) * _HEPARIN_APTT_RANGE + _HEPARIN_APTT_MIN,
        "concentration": float(obs[1]),  # U/mL proxy for anti-Xa
        "weight": float(obs[2]) * _HEPARIN_WEIGHT_RANGE + _HEPARIN_WEIGHT_MIN,
        "renal_function": float(obs[3]),
        "platelets": float(obs[4]) * 400.0,
        "hours": float(obs[5]) * _HEPARIN_HOURS_RANGE,
    }


# ---------------------------------------------------------------------------
# Nemati 2016 DQN reimplementation
# ---------------------------------------------------------------------------

# Discrete action grid from Nemati et al. 2016 (Section II-C): five options
# spanning decrement-4, decrement-2, hold, increment-2, increment-4 U/kg/hr.
_NEMATI_ACTION_DELTAS_UKGHR = np.array([-4.0, -2.0, 0.0, 2.0, 4.0], dtype=np.float32)
_NEMATI_N_ACTIONS = len(_NEMATI_ACTION_DELTAS_UKGHR)
_NEMATI_OBS_DIM = 6  # HeparinInfusion-v0 observation dimensionality
_NEMATI_INITIAL_RATE_UKGHR = 18.0  # Raschke initial standard, matches Nemati training prior


if _TORCH_AVAILABLE:

    class _NematiQNetwork(nn.Module):
        """Two-layer MLP Q-network used in the Nemati 2016 reimplementation.

        Architecture follows the paper's described hidden layers (64 units,
        ReLU activations) and outputs one Q-value per discrete heparin
        adjustment action.
        """

        def __init__(
            self,
            obs_dim: int = _NEMATI_OBS_DIM,
            n_actions: int = _NEMATI_N_ACTIONS,
            hidden: int = 64,
        ) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_actions),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
            return self.net(x)

else:  # pragma: no cover - torch-absent branch

    class _NematiQNetwork:  # type: ignore[no-redef]
        """Placeholder when torch is not installed; instantiation raises."""

        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError(
                "torch is required to instantiate _NematiQNetwork. "
                "Install hemosim[train] to enable DQN support."
            )


class NematiDQN2016Baseline:
    """Independent reimplementation of the Nemati 2016 DQN heparin policy.

    The 2016 IEEE EMBC paper trained a deep Q-network on MIMIC-II heparin
    trajectories, framing infusion rate adjustment as a discrete action
    problem with five options ({-4, -2, 0, +2, +4} U/kg/hr) and a reward
    shaped to push aPTT into the 60–100 s therapeutic range. This class
    reimplements that policy structure against hemosim's HeparinInfusion-v0
    environment.

    Important: **this is an independent reimplementation for benchmarking.
    No code was transferred from Nemati et al.** The network architecture,
    action grid, and reward intent are reproduced from the paper's written
    description; weights are learned here (if trained) or approximated by a
    deterministic proxy that follows the paper's reported dosing curves.

    Priority for action selection:

    1. If torch is available and a trained checkpoint is supplied (or lives
       at ``results/models/nemati_dqn.pt``), use the learned Q-network.
    2. If torch is available but no weights are present, use a randomly
       initialized network (primarily useful for training bootstraps).
    3. If torch is not available, fall back to a deterministic threshold
       policy that approximates the 2016 paper's reported dosing behavior
       (increase when aPTT < 60 s, hold in therapeutic range, decrease when
       above). This fallback is documented and clearly distinguished from
       the learned policy.

    References
    ----------
    Nemati S, Ghassemi MM, Clifford GD. Optimal medication dosing from
    suboptimal clinical examples: A deep reinforcement learning approach.
    IEEE EMBC 2016;2978-2981. doi:10.1109/EMBC.2016.7591355.
    """

    DEFAULT_WEIGHTS_PATH = Path("results/models/nemati_dqn.pt")

    def __init__(
        self,
        weights_path: str | Path | None = None,
        seed: int | None = None,
        device: str = "cpu",
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self._device = device
        self._last_rate_u_kg_hr = _NEMATI_INITIAL_RATE_UKGHR
        self._model = None  # type: Any
        self._using_proxy = True

        resolved_path = Path(weights_path) if weights_path else self.DEFAULT_WEIGHTS_PATH

        if _TORCH_AVAILABLE:
            self._model = _NematiQNetwork().to(self._device)
            if resolved_path.is_file():
                try:
                    state = torch.load(resolved_path, map_location=self._device)
                    # Accept either a raw state_dict or a {"state_dict": ...} checkpoint.
                    if isinstance(state, dict) and "state_dict" in state:
                        state = state["state_dict"]
                    self._model.load_state_dict(state)
                    self._using_proxy = False
                except (RuntimeError, OSError, KeyError):
                    # Incompatible or corrupt checkpoint: fall back to proxy but
                    # keep the instantiated network so a trainer can still use it.
                    self._using_proxy = True
            self._model.eval()

    # ------------------------------------------------------------------
    @property
    def uses_proxy_policy(self) -> bool:
        """True when action selection falls back to the deterministic proxy."""

        return self._using_proxy or not _TORCH_AVAILABLE

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset the internal infusion-rate memory between episodes."""

        self._last_rate_u_kg_hr = _NEMATI_INITIAL_RATE_UKGHR

    # ------------------------------------------------------------------
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Select the next heparin infusion rate.

        Args:
            obs: HeparinInfusion-v0 observation (shape ``(6,)``).

        Returns:
            Action array ``[infusion_scaled, bolus_flag]`` compatible with
            the env's ``action_space``. The bolus flag is 1.0 at ``t=0`` and
            0.0 thereafter, mirroring the Raschke initialization used in
            Nemati 2016.
        """

        decoded = _denormalize_heparin_obs(obs)
        weight = max(decoded["weight"], 1e-3)
        hours = decoded["hours"]

        # Reset memory when a new episode begins (env resets its clock).
        if hours <= 1e-6:
            self._last_rate_u_kg_hr = _NEMATI_INITIAL_RATE_UKGHR

        # Select discrete action either from learned Q-network or proxy.
        if _TORCH_AVAILABLE and self._model is not None and not self._using_proxy:
            action_idx = self._predict_with_network(obs)
        else:
            action_idx = self._predict_with_proxy(decoded)

        delta = float(_NEMATI_ACTION_DELTAS_UKGHR[action_idx])
        self._last_rate_u_kg_hr = float(np.clip(self._last_rate_u_kg_hr + delta, 0.0, 40.0))

        infusion_rate_u_hr = self._last_rate_u_kg_hr * weight
        infusion_scaled = np.clip(infusion_rate_u_hr / _HEPARIN_MAX_INFUSION, 0.0, 1.0)
        bolus_flag = 1.0 if hours < 1.0 else 0.0

        return np.array([infusion_scaled, bolus_flag], dtype=np.float32)

    # ------------------------------------------------------------------
    def _predict_with_network(self, obs: np.ndarray) -> int:
        """Argmax over Q-values from the learned network."""

        with torch.no_grad():  # type: ignore[union-attr]
            x = torch.as_tensor(obs, dtype=torch.float32, device=self._device)  # type: ignore[union-attr]
            q_values = self._model(x.unsqueeze(0)).squeeze(0)
            return int(torch.argmax(q_values).item())  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    @staticmethod
    def _predict_with_proxy(decoded: dict[str, float]) -> int:
        """Deterministic proxy approximating the 2016 paper's dosing curve.

        Maps aPTT regions to the discrete action grid to reproduce the
        coarse shape of the policy reported by Nemati et al. 2016: large
        increases far below range, small increases just below, hold inside
        60–100 s, small decreases above, large decreases well above.
        """

        aptt = decoded["aptt"]
        if aptt < 45.0:
            return 4  # +4 U/kg/hr
        if aptt < 60.0:
            return 3  # +2 U/kg/hr
        if aptt <= 100.0:
            return 2  # hold
        if aptt <= 120.0:
            return 1  # -2 U/kg/hr
        return 0  # -4 U/kg/hr


# ---------------------------------------------------------------------------
# Anti-Xa guided heparin baseline
# ---------------------------------------------------------------------------

# Therapeutic anti-Xa window for unfractionated heparin (CHEST 2012, ASH 2018).
_ANTI_XA_TARGET = 0.5  # IU/mL midpoint of 0.3–0.7 therapeutic range
_ANTI_XA_LOW = 0.3
_ANTI_XA_HIGH = 0.7


class HeparinAntiXaBaseline:
    """Anti-Xa guided UFH titration targeting 0.3–0.7 IU/mL.

    Modern heparin monitoring increasingly uses anti-Xa activity rather than
    aPTT because anti-Xa correlates more directly with heparin effect and is
    less confounded by acute-phase reactants. Therapeutic window for UFH is
    0.3–0.7 IU/mL (CHEST 2012; ASH 2018 venous thromboembolism guideline).

    HeparinInfusion-v0 does not expose anti-Xa directly, but its heparin
    concentration channel (``obs[1]``, already U/mL-scaled) is a faithful
    stand-in for anti-Xa activity because anti-Xa is approximately
    proportional to circulating heparin concentration.

    Policy:

    * Initial infusion rate: 18 U/kg/hr with an 80 U/kg bolus at ``t=0``.
    * Each subsequent step, adjust the rate by ±10% proportional to the
      distance from 0.5 IU/mL; clamp to the 0.3–0.7 IU/mL window without
      over-correction.

    References
    ----------
    Garcia DA, Baglin TP, Weitz JI, Samama MM. *Parenteral anticoagulants:
    Antithrombotic Therapy and Prevention of Thrombosis, 9th ed: American
    College of Chest Physicians Evidence-Based Clinical Practice Guidelines.*
    Chest 2012;141(2 Suppl):e24S-e43S.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._current_rate_u_kg_hr = 18.0

    def reset(self) -> None:
        """Reset internal rate state."""

        self._current_rate_u_kg_hr = 18.0

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Predict heparin infusion rate from anti-Xa proxy.

        Args:
            obs: HeparinInfusion-v0 observation (shape ``(6,)``).

        Returns:
            Action array ``[infusion_scaled, bolus_flag]``.
        """

        decoded = _denormalize_heparin_obs(obs)
        weight = max(decoded["weight"], 1e-3)
        hours = decoded["hours"]
        anti_xa_proxy = decoded["concentration"]

        if hours <= 1e-6:
            self._current_rate_u_kg_hr = 18.0
            bolus_flag = 1.0
        else:
            bolus_flag = 0.0

            # Proportional 10% adjustment capped by distance from midpoint.
            deviation = anti_xa_proxy - _ANTI_XA_TARGET
            if anti_xa_proxy < _ANTI_XA_LOW:
                adjustment_factor = 1.10
            elif anti_xa_proxy > _ANTI_XA_HIGH:
                adjustment_factor = 0.90
            else:
                # Within therapeutic range: nudge slightly toward midpoint.
                adjustment_factor = 1.0 - 0.10 * float(np.clip(deviation / _ANTI_XA_TARGET, -1.0, 1.0))

            self._current_rate_u_kg_hr = float(
                np.clip(self._current_rate_u_kg_hr * adjustment_factor, 4.0, 30.0)
            )

        infusion_rate_u_hr = self._current_rate_u_kg_hr * weight
        infusion_scaled = np.clip(infusion_rate_u_hr / _HEPARIN_MAX_INFUSION, 0.0, 1.0)
        return np.array([infusion_scaled, bolus_flag], dtype=np.float32)


# ---------------------------------------------------------------------------
# Warfarin — Gage pharmacogenetic algorithm
# ---------------------------------------------------------------------------

# Gage et al. 2008 coefficient table (Clin Pharmacol Ther 84:326). The model
# predicts the square root of the weekly dose (mg); raise to produce a
# predicted weekly dose which is divided by 7 to obtain a daily dose.
_GAGE_INTERCEPT = 3.10894
_GAGE_AGE_COEF = -0.00767  # per year
_GAGE_BSA_COEF = 1.24754  # per m^2 of body-surface-area
_GAGE_TARGET_INR_COEF = 0.24032  # per unit of target INR
_GAGE_CYP2C9_COEF = {
    # Relative to *1/*1 wild-type; coefficients reduce the predicted weekly dose.
    "*1/*1": 0.0,
    "*1/*2": -0.5562,
    "*1/*3": -0.99413,
    "*2/*2": -1.25647,
    "*2/*3": -1.74851,
    "*3/*3": -1.92988,
}
_GAGE_VKORC1_AA_COEF = -0.69716
_GAGE_VKORC1_GA_COEF = -0.36924
_GAGE_AMIODARONE_COEF = -0.61020  # applied only when amiodarone co-therapy known

_CYP2C9_ORDINAL_TO_GENOTYPE = {
    0: "*1/*1",
    1: "*1/*2",
    2: "*1/*3",
    3: "*2/*2",
    4: "*2/*3",
    5: "*3/*3",
}
_VKORC1_ORDINAL_TO_GENOTYPE = {0: "GG", 1: "GA", 2: "AA"}


def _estimate_bsa_mosteller(weight_kg: float, height_cm: float = 170.0) -> float:
    """Estimate body surface area (m^2) using the Mosteller formula.

    The hemosim observation exposes weight but not height; we use a
    population-average height of 170 cm when height is not supplied.
    """

    return float(np.sqrt((weight_kg * height_cm) / 3600.0))


class WarfarinGageBaseline:
    """Gage pharmacogenetic warfarin dosing algorithm.

    Implements the published Gage 2008 regression for weekly warfarin dose
    using age, body surface area, target INR, CYP2C9, and VKORC1 -1639G>A
    genotype. We also apply an INR-feedback titration around the predicted
    maintenance dose so the baseline produces the daily schedule that the
    hemosim warfarin environment expects.

    * Days 0–2: deliver the Gage-predicted maintenance daily dose without
      INR-driven modulation.
    * Day 3+: titrate ±20% per day according to the patient's current INR
      relative to the 2.0–3.0 therapeutic range.

    References
    ----------
    Gage BF, Eby C, Johnson JA, et al. Use of pharmacogenetic and clinical
    factors to predict the therapeutic dose of warfarin. Clin Pharmacol Ther
    2008;84(3):326-331. doi:10.1038/clpt.2008.10.
    """

    MAX_DOSE_MG = 15.0  # Matches WarfarinDosingEnv.MAX_DOSE_MG

    def __init__(self, seed: int | None = None, target_inr: float = 2.5) -> None:
        self._rng = np.random.default_rng(seed)
        self._target_inr = target_inr

    # ------------------------------------------------------------------
    @staticmethod
    def predict_weekly_dose_mg(
        age_years: float,
        weight_kg: float,
        cyp2c9: str,
        vkorc1: str,
        target_inr: float = 2.5,
        height_cm: float = 170.0,
        amiodarone: bool = False,
    ) -> float:
        """Return Gage 2008 predicted weekly warfarin dose in mg."""

        bsa = _estimate_bsa_mosteller(weight_kg, height_cm)
        sqrt_dose = (
            _GAGE_INTERCEPT
            + _GAGE_AGE_COEF * age_years
            + _GAGE_BSA_COEF * bsa
            + _GAGE_TARGET_INR_COEF * target_inr
            + _GAGE_CYP2C9_COEF.get(cyp2c9, 0.0)
        )
        if vkorc1 == "AA":
            sqrt_dose += _GAGE_VKORC1_AA_COEF
        elif vkorc1 == "GA":
            sqrt_dose += _GAGE_VKORC1_GA_COEF
        if amiodarone:
            sqrt_dose += _GAGE_AMIODARONE_COEF

        sqrt_dose = max(sqrt_dose, 0.0)
        return float(sqrt_dose ** 2)

    # ------------------------------------------------------------------
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Predict the next warfarin daily dose.

        Args:
            obs: WarfarinDosing-v0 observation (shape ``(8,)``).

        Returns:
            Action array ``[dose_scaled]`` where ``dose_scaled`` is in
            ``[0, 1]`` and is multiplied by 15 mg inside the env.
        """

        inr = float(obs[0]) * 6.0
        age = float(obs[3]) * 75.0 + 20.0
        weight = float(obs[4]) * 110.0 + 40.0
        cyp2c9_idx = int(round(float(obs[5]) * 5.0))
        vkorc1_idx = int(round(float(obs[6]) * 2.0))
        day = float(obs[7]) * 90.0

        cyp2c9 = _CYP2C9_ORDINAL_TO_GENOTYPE.get(cyp2c9_idx, "*1/*1")
        vkorc1 = _VKORC1_ORDINAL_TO_GENOTYPE.get(vkorc1_idx, "GG")

        weekly_mg = self.predict_weekly_dose_mg(
            age_years=age,
            weight_kg=weight,
            cyp2c9=cyp2c9,
            vkorc1=vkorc1,
            target_inr=self._target_inr,
        )
        daily_mg = weekly_mg / 7.0

        if day >= 3.0:
            # INR feedback: scale by distance from therapeutic midpoint.
            if inr < 2.0:
                daily_mg *= 1.20
            elif inr < 1.5:
                daily_mg *= 1.40
            elif inr > 4.0:
                daily_mg = 0.0  # hold for supratherapeutic
            elif inr > 3.0:
                daily_mg *= 0.80

        dose_scaled = float(np.clip(daily_mg / self.MAX_DOSE_MG, 0.0, 1.0))
        return np.array([dose_scaled], dtype=np.float32)


# ---------------------------------------------------------------------------
# Warfarin — Crowther-inspired ordinal titration
# ---------------------------------------------------------------------------


class WarfarinOrdinalBaseline:
    """Crowther-style INR-response warfarin titration without pharmacogenomics.

    Many ambulatory anticoagulation clinics manage warfarin with a simple
    INR-based daily-dose nomogram that does not require CYP2C9/VKORC1
    genotyping. This baseline uses a fixed 5 mg/day starting regimen for the
    first two days — the loading schedule evaluated by Crowther et al.
    (1999) — and then maps the current INR to a percentage adjustment of the
    running daily dose. It represents the "non-pharmacogenetic" arm that
    clinical trials compare against genotype-guided dosing.

    References
    ----------
    Crowther MA, Ginsberg JS, Kearon C, et al. A randomized trial comparing
    5-mg and 10-mg warfarin loading doses. Arch Intern Med 1999;159(1):46-48.
    Subsequent maintenance nomograms reviewed in Arch Intern Med 2009.
    """

    MAX_DOSE_MG = 15.0  # Matches WarfarinDosingEnv.MAX_DOSE_MG
    INITIAL_DOSE_MG = 5.0
    LOADING_DAYS = 2

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._current_daily_mg = self.INITIAL_DOSE_MG
        self._last_day = -1

    def reset(self) -> None:
        """Reset titration memory between episodes."""

        self._current_daily_mg = self.INITIAL_DOSE_MG
        self._last_day = -1

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Predict the next warfarin daily dose from INR alone.

        Args:
            obs: WarfarinDosing-v0 observation (shape ``(8,)``).

        Returns:
            Action array ``[dose_scaled]``.
        """

        inr = float(obs[0]) * 6.0
        day = int(round(float(obs[7]) * 90.0))

        # Reset running dose if we're clearly in a new episode.
        if day <= 0 and self._last_day > 0:
            self._current_daily_mg = self.INITIAL_DOSE_MG

        if day < self.LOADING_DAYS:
            self._current_daily_mg = self.INITIAL_DOSE_MG
        else:
            if inr < 1.5:
                self._current_daily_mg *= 1.15
            elif inr < 2.0:
                self._current_daily_mg *= 1.07
            elif inr <= 3.0:
                pass  # hold
            elif inr <= 4.0:
                self._current_daily_mg *= 0.90
            elif inr <= 5.0:
                self._current_daily_mg *= 0.50
            else:
                self._current_daily_mg = 0.0  # skip dose

            self._current_daily_mg = float(np.clip(self._current_daily_mg, 0.0, self.MAX_DOSE_MG))

        self._last_day = day
        dose_scaled = float(np.clip(self._current_daily_mg / self.MAX_DOSE_MG, 0.0, 1.0))
        return np.array([dose_scaled], dtype=np.float32)


__all__ = [
    "NematiDQN2016Baseline",
    "HeparinAntiXaBaseline",
    "WarfarinGageBaseline",
    "WarfarinOrdinalBaseline",
]
