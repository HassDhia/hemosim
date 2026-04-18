"""Clinical Decision-Support System harness (ISC-10).

This module turns a trained hemosim policy (or a fallback clinical
baseline) into a single callable that a clinician-facing application
can use: ``dss.recommend(snapshot)``.

Design goals
------------

1. **The DSS is always callable.** If ``stable_baselines3`` / ``torch``
   are not installed, or if no policy checkpoint is present, the DSS
   falls back to the validated clinical baseline for that drug
   (Raschke 1993 for heparin, IWPC-inspired titration for warfarin).
   A deployed ICU tool cannot raise ``ImportError``.

2. **FHIR-like input.** ``PatientSnapshot`` is a dataclass of
   ``Optional`` fields so that real-world partial records
   (e.g. missing CYP2C9 genotype, stale aPTT, no platelet count yet)
   round-trip through the harness without the caller having to invent
   values.

3. **Uncertainty is first-class.** ``DosingRecommendation`` carries
   ``uncertainty_interval`` and ``confidence``. An N-policy ensemble
   widens the interval from the across-seed standard deviation; a
   single-policy recommendation uses a documented default band.

4. **Explanations are local and inspectable.** ``DSS.explain`` perturbs
   each observation feature by ``+/- 10%`` and records the resulting
   action delta, returning the ranked list of features by absolute
   impact on the recommended dose. This is a simple saliency
   approximation (occlusion / sensitivity analysis) — not a
   SHAP-grade explanation, but sufficient for a clinician to spot
   "why did the policy recommend 22 U/kg/hr?".

Clinical references
-------------------

* Raschke RA, Reilly BM, Guidry JR, et al. *The weight-based heparin
  dosing nomogram compared with a "standard care" nomogram.* Ann
  Intern Med 1993;119:874-881.
* International Warfarin Pharmacogenetics Consortium. *Estimation of
  the warfarin dose with clinical and pharmacogenetic data.* N Engl J
  Med 2009;360:753-764.
* Garcia DA, Baglin TP, Weitz JI, Samama MM. *Parenteral
  anticoagulants: Antithrombotic Therapy and Prevention of Thrombosis,
  9th ed: ACCP Evidence-Based Clinical Practice Guidelines.* Chest
  2012;141(2 Suppl):e24S-e43S. (CHEST 2012 — anti-Xa and aPTT target
  ranges.)
* Holbrook A, Schulman S, Witt DM, et al. *Evidence-based management
  of anticoagulant therapy: Antithrombotic Therapy and Prevention of
  Thrombosis, 9th ed: ACCP Evidence-Based Clinical Practice
  Guidelines.* Chest 2012;141(2 Suppl):e152S-e184S. (Warfarin INR
  management and bleeding definitions.)
* Hirsh J, Raschke R. *Heparin and low-molecular-weight heparin: the
  Seventh ACCP Conference on Antithrombotic and Thrombolytic
  Therapy.* Chest 2004;126(3 Suppl):188S-203S. (Platelet-monitoring
  thresholds and absolute-dose guardrails used in ISC-11 derive from
  general antithrombotic stewardship principles in Holbrook 2012 +
  Warkentin 2003, not from any specific partner-site protocol.)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hemosim.agents.baselines import (
    HeparinRaschkeBaseline,
    WarfarinClinicalBaseline,
)

# ---------------------------------------------------------------------------
# Optional sb3/torch
# ---------------------------------------------------------------------------

try:  # pragma: no cover - exercised only when sb3 is installed
    from stable_baselines3 import PPO as _PPO  # type: ignore[assignment]

    _SB3_AVAILABLE = True
except ImportError:  # pragma: no cover - baseline-fallback branch is tested
    _PPO = None  # type: ignore[assignment]
    _SB3_AVAILABLE = False


# ---------------------------------------------------------------------------
# Clinical constants (mirror env normalization so obs round-trips are exact)
# ---------------------------------------------------------------------------

# Heparin (HeparinInfusionEnv, 6-D obs)
_HEPARIN_APTT_MIN = 20.0
_HEPARIN_APTT_RANGE = 180.0       # 20 -> 200 s
_HEPARIN_CONC_SCALE = 1.0         # U/mL
_HEPARIN_WEIGHT_MIN = 40.0
_HEPARIN_WEIGHT_RANGE = 140.0     # 40 -> 180 kg
_HEPARIN_PLT_SCALE = 400.0        # x10^3/uL
_HEPARIN_HOURS_RANGE = 120.0      # 0 -> 120 h
_HEPARIN_MAX_INFUSION = 2500.0    # U/hr
_HEPARIN_DEFAULT_BOLUS_U_KG = 80.0  # Raschke

# Warfarin (WarfarinDosingEnv, 8-D obs)
_WARFARIN_INR_SCALE = 6.0
_WARFARIN_SWARF_SCALE = 5.0
_WARFARIN_RWARF_SCALE = 5.0
_WARFARIN_AGE_MIN = 20.0
_WARFARIN_AGE_RANGE = 75.0        # 20 -> 95 y
_WARFARIN_WEIGHT_MIN = 40.0
_WARFARIN_WEIGHT_RANGE = 110.0    # 40 -> 150 kg
_WARFARIN_EPISODE_DAYS = 90.0
_WARFARIN_MAX_DOSE_MG = 15.0

# Match env encoding dictionaries verbatim (DO NOT duplicate differently).
_CYP2C9_ENCODE: dict[str, int] = {
    "*1/*1": 0, "*1/*2": 1, "*1/*3": 2,
    "*2/*2": 3, "*2/*3": 4, "*3/*3": 5,
}
_VKORC1_ENCODE: dict[str, int] = {"GG": 0, "GA": 1, "AA": 2}


# ---------------------------------------------------------------------------
# PatientSnapshot
# ---------------------------------------------------------------------------


@dataclass
class PatientSnapshot:
    """FHIR-flavoured patient snapshot consumed by the DSS.

    All fields are ``Optional`` because clinical reality is that labs
    are late, genotype panels are not always ordered, and weight might
    be the ED triage estimate. The DSS fills missing values with
    documented physiological defaults (e.g. aPTT baseline 30 s, weight
    80 kg) so that a recommendation can still be produced.

    Fields intentionally follow FHIR Observation / MedicationRequest
    field names where practical:

    * ``aptt_seconds`` — Observation (LOINC 14979-9).
    * ``platelets_k_per_ul`` — Observation (LOINC 777-3).
    * ``weight_kg`` — Observation (LOINC 29463-7).
    * ``renal_function`` — fraction of normal CrCl in ``[0, 1]``; a
      typical mapping is ``CrCl / 120 mL/min`` clipped to ``[0, 1]``.
    * ``drug`` — the drug this snapshot is being dosed against
      (``"heparin"``, ``"warfarin"``, etc.).
    * ``current_rate_u_per_hr`` — most recent heparin infusion rate.
    * ``hours_since_last_aptt`` — staleness of the aPTT datum
      (used by POMDP-aware policies; current envs assume fresh).
    * ``cyp2c9`` / ``vkorc1`` — warfarin pharmacogenetics.
    * ``inr`` — current INR.
    * ``s_warfarin``, ``r_warfarin`` — plasma enantiomer concentrations
      if measured (rarely in practice; optional).
    * ``age_years`` — patient age.
    * ``heparin_concentration_u_per_ml`` — anti-Xa/heparin activity.
    * ``days_on_therapy`` — warfarin therapy day counter.
    * ``hours_on_therapy`` — heparin therapy hour counter.
    * ``extra`` — free-form dictionary for any other FHIR fields the
      caller wants to pass through (ignored by the DSS but preserved
      on ``to_dict()``).

    The class intentionally uses ``field(default_factory=dict)`` for
    ``extra`` to avoid the mutable-default trap.
    """

    # Heparin-facing fields
    aptt_seconds: float | None = None
    heparin_concentration_u_per_ml: float | None = None
    platelets_k_per_ul: float | None = None
    current_rate_u_per_hr: float | None = None
    hours_since_last_aptt: float | None = None
    hours_on_therapy: float | None = None
    # Warfarin-facing fields
    inr: float | None = None
    s_warfarin: float | None = None
    r_warfarin: float | None = None
    cyp2c9: str | None = None
    vkorc1: str | None = None
    days_on_therapy: float | None = None
    # Common patient fields
    weight_kg: float | None = None
    age_years: float | None = None
    renal_function: float | None = None
    drug: str | None = None
    # Any additional FHIR-flavoured fields the caller wants to pass
    # through — explicitly not part of the obs mapping but preserved by
    # ``to_dict`` / ``from_dict``.
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Round-trippable dict representation."""
        return asdict(self)

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PatientSnapshot":
        """Construct from a FHIR-like dict, preserving unknown keys in ``extra``."""
        known = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for k, v in data.items():
            if k == "extra" and isinstance(v, Mapping):
                extra.update(v)
            elif k in known:
                kwargs[k] = v
            else:
                extra[k] = v
        if extra:
            kwargs["extra"] = extra
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# DosingRecommendation
# ---------------------------------------------------------------------------


@dataclass
class DosingRecommendation:
    """Structured dosing recommendation emitted by a DSS.

    Attributes
    ----------
    action_taken
        Human-readable label (e.g. ``"heparin_infusion"``,
        ``"warfarin_oral"``). The safety layer branches on this.
    dose_or_rate
        The recommended dose or infusion rate, in the clinical unit
        that matches ``action_taken`` (U/hr for heparin infusion, mg
        for oral warfarin).
    uncertainty_interval
        ``(lower, upper)`` tuple in the same unit as ``dose_or_rate``.
        From the across-seed standard deviation when an ensemble is
        used, or a documented default band when a single policy is
        used.
    top_feature_contributions
        Ranked list of ``(feature_name, absolute_action_delta)``
        produced by :meth:`BaseDSS.explain`. Length is capped to the
        observation dimensionality.
    confidence
        Heuristic confidence in ``[0, 1]``. Derived from the width of
        the uncertainty interval relative to the clinical dose range.
    rationale
        Natural-language note identifying the policy kind
        (``"ppo"`` / ``"raschke_baseline"`` / ``"iwpc_baseline"``) and
        any flags (e.g. ``"fallback_no_sb3"``).
    bolus_given
        Heparin-only: whether the recommendation includes a bolus
        dose. Defaults to ``False``. Ignored for warfarin.
    bolus_u_per_kg
        Heparin-only: bolus dose in U/kg if ``bolus_given`` is True.
    """

    action_taken: str
    dose_or_rate: float
    uncertainty_interval: tuple[float, float]
    top_feature_contributions: list[tuple[str, float]]
    confidence: float
    rationale: str
    bolus_given: bool = False
    bolus_u_per_kg: float = 0.0


# ---------------------------------------------------------------------------
# BaseDSS
# ---------------------------------------------------------------------------


class BaseDSS:
    """Shared behavior for heparin / warfarin DSS classes.

    Subclasses must override:

    * ``_DRUG`` — canonical drug name string.
    * ``_OBS_DIM`` — observation dimensionality.
    * ``_FEATURE_NAMES`` — ordered names matching the env's obs layout.
    * ``_CLINICAL_DOSE_RANGE`` — ``(min_dose, max_dose)`` for the
      drug's clinical dose unit (used for the confidence heuristic).
    * ``_DEFAULT_BAND_FRAC`` — default uncertainty interval half-width
      as a fraction of ``_CLINICAL_DOSE_RANGE`` when no ensemble.
    * ``_snapshot_to_obs`` — map PatientSnapshot -> env obs vector.
    * ``_action_to_dose`` — map scaled env action -> clinical dose.
    * ``_build_baseline`` — construct the fallback clinical policy.
    * ``_action_taken`` — string identifier for the dosing action.
    """

    # Subclass must override all of these.
    _DRUG: str = "unknown"
    _OBS_DIM: int = 0
    _FEATURE_NAMES: tuple[str, ...] = ()
    _CLINICAL_DOSE_RANGE: tuple[float, float] = (0.0, 1.0)
    _DEFAULT_BAND_FRAC: float = 0.15
    _ACTION_TAKEN: str = "unknown"

    def __init__(
        self,
        policy_path: str | Path | None = None,
        n_ensemble: int = 1,
    ) -> None:
        if n_ensemble < 1:
            raise ValueError(f"n_ensemble must be >= 1 (got {n_ensemble})")
        self.n_ensemble = int(n_ensemble)
        self.policy_path = Path(policy_path) if policy_path is not None else None

        self._models: list[Any] = []
        self._baseline = self._build_baseline(seed=0)
        self._using_baseline = True
        self._load_policies_if_available()

    # ------------------------------------------------------------------
    # Abstract-ish hooks (subclasses override)
    # ------------------------------------------------------------------

    def _snapshot_to_obs(self, snapshot: PatientSnapshot) -> np.ndarray:
        raise NotImplementedError

    def _action_to_dose(
        self, action: np.ndarray, snapshot: PatientSnapshot
    ) -> tuple[float, bool, float]:
        """Return ``(dose_or_rate, bolus_given, bolus_u_per_kg)``."""
        raise NotImplementedError

    def _build_baseline(self, seed: int | None = None) -> Any:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Policy loading
    # ------------------------------------------------------------------

    def _load_policies_if_available(self) -> None:
        """Attempt to load N PPO checkpoints. Stay on baseline on any failure."""
        if not _SB3_AVAILABLE:
            return
        if self.policy_path is None:
            return
        paths = self._resolve_ensemble_paths()
        loaded: list[Any] = []
        for p in paths:
            if not p.exists():
                continue
            try:
                loaded.append(_PPO.load(str(p)))  # type: ignore[union-attr]
            except (ValueError, RuntimeError, OSError):
                # Incompatible checkpoint: stay on baseline but keep any
                # successfully-loaded siblings.
                continue
        if loaded:
            self._models = loaded
            self._using_baseline = False

    def _resolve_ensemble_paths(self) -> list[Path]:
        """Produce N candidate checkpoint paths.

        If ``policy_path`` points at an existing file, use it as the
        first member and look for ``<stem>_seed<i><suffix>`` for the
        remaining N-1. If it does not exist as a file, treat it as a
        prefix/template and probe ``<prefix>_seed<i>.zip``.
        """
        assert self.policy_path is not None
        base = self.policy_path
        if self.n_ensemble == 1:
            return [base]
        # Use base as member 0, then seed1..seedN-1.
        stem = base.stem
        suffix = base.suffix if base.suffix else ".zip"
        parent = base.parent
        paths = [base]
        for i in range(1, self.n_ensemble):
            paths.append(parent / f"{stem}_seed{i}{suffix}")
        return paths

    @property
    def uses_baseline(self) -> bool:
        """True when the DSS is falling back to the clinical baseline."""
        return self._using_baseline

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _predict_action(self, obs: np.ndarray) -> list[np.ndarray]:
        """Return one action per ensemble member.

        When the baseline is in use, returns ``n_ensemble`` copies of
        the deterministic baseline action. This keeps the downstream
        statistics code uniform and makes it easy for callers to ask
        "what does the ensemble think?" regardless of mode.
        """
        if self._using_baseline or not self._models:
            action = np.asarray(self._baseline.predict(obs), dtype=np.float32)
            return [action.copy() for _ in range(self.n_ensemble)]

        actions: list[np.ndarray] = []
        for model in self._models:
            # sb3 PPO.predict(obs, deterministic=False) returns
            # (action, state). We use stochastic predictions so that
            # the ensemble captures policy-level variance even from
            # identical seeds.
            act, _ = model.predict(obs, deterministic=False)
            actions.append(np.asarray(act, dtype=np.float32))
        return actions

    # ------------------------------------------------------------------
    def recommend(self, snapshot: PatientSnapshot) -> DosingRecommendation:
        """Emit a structured dosing recommendation for ``snapshot``.

        Parameters
        ----------
        snapshot
            Patient state. Fields that are ``None`` are filled with
            documented physiological defaults during observation
            encoding.

        Returns
        -------
        DosingRecommendation
            Always returned, even when the DSS is falling back to the
            clinical baseline. Check ``uses_baseline`` / the
            ``rationale`` string to distinguish.
        """
        obs = self._snapshot_to_obs(snapshot)
        actions = self._predict_action(obs)
        doses = [self._action_to_dose(a, snapshot) for a in actions]

        dose_vals = np.array([d[0] for d in doses], dtype=np.float64)
        mean_dose = float(dose_vals.mean())
        lo_dose, hi_dose = self._uncertainty_interval(dose_vals)
        # Reuse the first ensemble member's bolus fields; with a single
        # policy this is exact, with a PPO ensemble this is the member-0
        # decision (bolus remains a boolean decision, not averaged).
        _, bolus_given, bolus_u_per_kg = doses[0]

        confidence = self._confidence_from_interval(lo_dose, hi_dose)
        rationale = self._build_rationale()
        contributions = self._feature_sensitivity(obs, snapshot)

        return DosingRecommendation(
            action_taken=self._ACTION_TAKEN,
            dose_or_rate=mean_dose,
            uncertainty_interval=(lo_dose, hi_dose),
            top_feature_contributions=contributions,
            confidence=confidence,
            rationale=rationale,
            bolus_given=bool(bolus_given),
            bolus_u_per_kg=float(bolus_u_per_kg),
        )

    # ------------------------------------------------------------------
    def _uncertainty_interval(self, dose_vals: np.ndarray) -> tuple[float, float]:
        """Ensemble 1-sigma interval — with the default band as a lower bound.

        When an ensemble is used we take the across-member standard
        deviation, but we never shrink the interval below the single-
        policy default band. This ensures the property "ensemble
        interval width >= single-policy default band": adding members
        to the ensemble can only *widen* (or hold) the reported
        uncertainty, never narrow it, which matches the clinical
        intuition that more policies disagreeing is not *less*
        uncertain than a single policy's prior.
        """
        dose_min, dose_max = self._CLINICAL_DOSE_RANGE
        dose_range = max(dose_max - dose_min, 1e-9)
        mean = float(dose_vals.mean())
        default_half_band = self._DEFAULT_BAND_FRAC * dose_range

        if dose_vals.size >= 2:
            sigma = float(dose_vals.std(ddof=0))
            # Take the wider of (1 sigma) and (default band) so that
            # ensemble intervals are never narrower than the single-
            # policy prior.
            half_band = max(sigma, default_half_band)
        else:
            half_band = default_half_band

        lo = max(mean - half_band, dose_min)
        hi = min(mean + half_band, dose_max)
        return (lo, hi)

    # ------------------------------------------------------------------
    def _confidence_from_interval(self, lo: float, hi: float) -> float:
        """Map interval width to confidence in ``[0, 1]``.

        Confidence is defined as ``1 - (width / dose_range)`` clipped
        to ``[0, 1]``. A pinpoint recommendation (zero width) -> 1.0;
        a recommendation spanning the whole clinical range -> 0.0.
        """
        dose_min, dose_max = self._CLINICAL_DOSE_RANGE
        dose_range = max(dose_max - dose_min, 1e-9)
        width = max(hi - lo, 0.0)
        return float(np.clip(1.0 - width / dose_range, 0.0, 1.0))

    # ------------------------------------------------------------------
    def _build_rationale(self) -> str:
        if self._using_baseline:
            if not _SB3_AVAILABLE:
                return f"{self._DRUG}:{self._baseline_name()} (fallback_no_sb3)"
            if self.policy_path is not None:
                return (
                    f"{self._DRUG}:{self._baseline_name()} "
                    "(fallback_no_checkpoint)"
                )
            return f"{self._DRUG}:{self._baseline_name()}"
        mode = "ppo_ensemble" if self.n_ensemble > 1 else "ppo_single"
        return f"{self._DRUG}:{mode}"

    def _baseline_name(self) -> str:
        return type(self._baseline).__name__.lower()

    # ------------------------------------------------------------------
    # Saliency / explanation
    # ------------------------------------------------------------------

    def _feature_sensitivity(
        self,
        obs: np.ndarray,
        snapshot: PatientSnapshot,
        pct: float = 0.10,
    ) -> list[tuple[str, float]]:
        """Occlusion saliency: perturb each feature by ±``pct`` and record delta.

        For each feature ``i``, we compute the action at
        ``obs + pct * e_i`` and ``obs - pct * e_i`` and take the
        absolute mean action delta versus the un-perturbed action.
        The features are returned sorted by absolute impact (largest
        first). When the baseline is in use the baseline is what gets
        perturbed; when PPO is in use the first ensemble member is
        used (fast and sufficient for local sensitivity).
        """
        base_action = self._predict_action(obs)[0]
        contributions: list[tuple[str, float]] = []
        for i in range(self._OBS_DIM):
            obs_plus = obs.copy()
            obs_minus = obs.copy()
            # Additive perturbation; obs is already normalized to [0, 1]
            # so ``pct`` is an absolute normalized step.
            obs_plus[i] = float(np.clip(obs[i] + pct, 0.0, 1.0))
            obs_minus[i] = float(np.clip(obs[i] - pct, 0.0, 1.0))
            act_plus = self._predict_action(obs_plus)[0]
            act_minus = self._predict_action(obs_minus)[0]
            # Mean absolute delta on the action components.
            delta = float(
                0.5 * (np.abs(act_plus - base_action).mean()
                       + np.abs(act_minus - base_action).mean())
            )
            name = self._FEATURE_NAMES[i] if i < len(self._FEATURE_NAMES) else f"obs[{i}]"
            contributions.append((name, delta))

        contributions.sort(key=lambda kv: kv[1], reverse=True)
        return contributions

    def explain(
        self,
        snapshot: PatientSnapshot,
        recommendation: DosingRecommendation | None = None,
    ) -> list[tuple[str, float]]:
        """Public wrapper returning the ranked feature sensitivity list.

        ``recommendation`` is accepted for API convenience (callers
        often have the last recommendation in hand) but is not
        required — the method re-derives the observation from the
        snapshot deterministically.
        """
        obs = self._snapshot_to_obs(snapshot)
        return self._feature_sensitivity(obs, snapshot)


# ---------------------------------------------------------------------------
# HeparinDSS
# ---------------------------------------------------------------------------


def _coerce(value: float | None, default: float) -> float:
    """Return ``value`` if not None, else ``default``; guarantee finite."""
    if value is None or not math.isfinite(float(value)):
        return float(default)
    return float(value)


class HeparinDSS(BaseDSS):
    """Decision support for unfractionated heparin infusion titration.

    Input: :class:`PatientSnapshot` with (preferably) ``aptt_seconds``,
    ``weight_kg``, ``platelets_k_per_ul``, ``renal_function``,
    ``hours_on_therapy`` populated.

    Output: :class:`DosingRecommendation` with
    ``dose_or_rate`` = infusion rate in **U/hr** (not U/kg/hr),
    ``bolus_given`` / ``bolus_u_per_kg`` populated, and an uncertainty
    interval on the rate.

    Fallback: when no trained PPO is available, the
    :class:`hemosim.agents.baselines.HeparinRaschkeBaseline` Raschke
    1993 weight-based nomogram is used — the same policy the paper
    reports as its clinical baseline. This is the validated protocol
    used at most US academic medical centers.

    References
    ----------
    Raschke RA et al. Ann Intern Med 1993;119:874-881. Garcia DA et
    al. Chest 2012;141(2 Suppl):e24S-e43S.
    """

    _DRUG = "heparin"
    _OBS_DIM = 6
    _FEATURE_NAMES: tuple[str, ...] = (
        "aptt_norm",
        "heparin_concentration_norm",
        "weight_norm",
        "renal_function",
        "platelets_norm",
        "hours_elapsed_norm",
    )
    _CLINICAL_DOSE_RANGE = (0.0, _HEPARIN_MAX_INFUSION)
    _DEFAULT_BAND_FRAC = 0.08  # +/- 200 U/hr on a 2500 U/hr range
    _ACTION_TAKEN = "heparin_infusion"

    def _build_baseline(self, seed: int | None = None) -> Any:
        return HeparinRaschkeBaseline(seed=seed)

    # ------------------------------------------------------------------
    def _snapshot_to_obs(self, snapshot: PatientSnapshot) -> np.ndarray:
        aptt = _coerce(snapshot.aptt_seconds, 30.0)            # physiological baseline
        conc = _coerce(snapshot.heparin_concentration_u_per_ml, 0.0)
        weight = _coerce(snapshot.weight_kg, 80.0)
        renal = _coerce(snapshot.renal_function, 1.0)
        platelets = _coerce(snapshot.platelets_k_per_ul, 250.0)
        hours = _coerce(snapshot.hours_on_therapy, 0.0)

        obs = np.array([
            np.clip((aptt - _HEPARIN_APTT_MIN) / _HEPARIN_APTT_RANGE, 0.0, 1.0),
            np.clip(conc / _HEPARIN_CONC_SCALE, 0.0, 1.0),
            np.clip((weight - _HEPARIN_WEIGHT_MIN) / _HEPARIN_WEIGHT_RANGE, 0.0, 1.0),
            np.clip(renal, 0.0, 1.0),
            np.clip(platelets / _HEPARIN_PLT_SCALE, 0.0, 1.0),
            np.clip(hours / _HEPARIN_HOURS_RANGE, 0.0, 1.0),
        ], dtype=np.float32)
        return obs

    # ------------------------------------------------------------------
    def _action_to_dose(
        self, action: np.ndarray, snapshot: PatientSnapshot
    ) -> tuple[float, bool, float]:
        action = np.asarray(action, dtype=np.float32).ravel()
        infusion_scaled = float(np.clip(action[0] if action.size > 0 else 0.0, 0.0, 1.0))
        bolus_flag = float(action[1]) if action.size > 1 else 0.0
        rate_u_per_hr = infusion_scaled * _HEPARIN_MAX_INFUSION
        bolus_given = bolus_flag > 0.5
        bolus_u_per_kg = _HEPARIN_DEFAULT_BOLUS_U_KG if bolus_given else 0.0
        return rate_u_per_hr, bolus_given, bolus_u_per_kg


# ---------------------------------------------------------------------------
# WarfarinDSS
# ---------------------------------------------------------------------------


class WarfarinDSS(BaseDSS):
    """Decision support for oral warfarin titration.

    Input: :class:`PatientSnapshot` with (preferably) ``inr``,
    ``age_years``, ``weight_kg``, ``cyp2c9``, ``vkorc1``,
    ``days_on_therapy`` populated. Missing genotype is conservatively
    mapped to wild-type (``*1/*1`` / ``GG``).

    Output: :class:`DosingRecommendation` with ``dose_or_rate`` in
    **mg** of oral warfarin (daily dose).

    Fallback: when no trained PPO is available,
    :class:`hemosim.agents.baselines.WarfarinClinicalBaseline`
    (IWPC-inspired titration table) is used — same as the paper's
    clinical baseline.

    References
    ----------
    International Warfarin Pharmacogenetics Consortium. N Engl J Med
    2009;360:753-764. Holbrook A et al. Chest 2012;141(2 Suppl):e152S.
    """

    _DRUG = "warfarin"
    _OBS_DIM = 8
    _FEATURE_NAMES: tuple[str, ...] = (
        "inr_norm",
        "s_warfarin_norm",
        "r_warfarin_norm",
        "age_norm",
        "weight_norm",
        "cyp2c9_norm",
        "vkorc1_norm",
        "days_norm",
    )
    _CLINICAL_DOSE_RANGE = (0.0, _WARFARIN_MAX_DOSE_MG)
    _DEFAULT_BAND_FRAC = 0.10  # +/- 1.5 mg on a 15 mg range
    _ACTION_TAKEN = "warfarin_oral"

    def _build_baseline(self, seed: int | None = None) -> Any:
        return WarfarinClinicalBaseline(seed=seed)

    # ------------------------------------------------------------------
    def _snapshot_to_obs(self, snapshot: PatientSnapshot) -> np.ndarray:
        inr = _coerce(snapshot.inr, 1.0)
        s_warf = _coerce(snapshot.s_warfarin, 0.0)
        r_warf = _coerce(snapshot.r_warfarin, 0.0)
        age = _coerce(snapshot.age_years, 60.0)
        weight = _coerce(snapshot.weight_kg, 75.0)
        days = _coerce(snapshot.days_on_therapy, 0.0)

        cyp = snapshot.cyp2c9 if snapshot.cyp2c9 in _CYP2C9_ENCODE else "*1/*1"
        vk = snapshot.vkorc1 if snapshot.vkorc1 in _VKORC1_ENCODE else "GG"
        cyp_code = _CYP2C9_ENCODE[cyp] / 5.0
        vk_code = _VKORC1_ENCODE[vk] / 2.0

        obs = np.array([
            np.clip(inr / _WARFARIN_INR_SCALE, 0.0, 1.0),
            np.clip(s_warf / _WARFARIN_SWARF_SCALE, 0.0, 1.0),
            np.clip(r_warf / _WARFARIN_RWARF_SCALE, 0.0, 1.0),
            np.clip((age - _WARFARIN_AGE_MIN) / _WARFARIN_AGE_RANGE, 0.0, 1.0),
            np.clip((weight - _WARFARIN_WEIGHT_MIN) / _WARFARIN_WEIGHT_RANGE, 0.0, 1.0),
            cyp_code,
            vk_code,
            np.clip(days / _WARFARIN_EPISODE_DAYS, 0.0, 1.0),
        ], dtype=np.float32)
        return obs

    # ------------------------------------------------------------------
    def _action_to_dose(
        self, action: np.ndarray, snapshot: PatientSnapshot
    ) -> tuple[float, bool, float]:
        action = np.asarray(action, dtype=np.float32).ravel()
        dose_scaled = float(np.clip(action[0] if action.size > 0 else 0.0, 0.0, 1.0))
        dose_mg = dose_scaled * _WARFARIN_MAX_DOSE_MG
        return dose_mg, False, 0.0


# ---------------------------------------------------------------------------
# Re-exports for convenience
# ---------------------------------------------------------------------------


def _reexports() -> Sequence[str]:
    return (
        "BaseDSS",
        "DosingRecommendation",
        "HeparinDSS",
        "PatientSnapshot",
        "WarfarinDSS",
    )
