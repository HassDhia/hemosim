"""Clinical decision-support and safety layer for hemosim.

This package is what turns a trained hemosim policy from a "research
Gymnasium artifact" into something a clinician or clinical-AI team can
reason about deploying. It is deliberately env-agnostic on its inputs
(``PatientSnapshot`` accepts FHIR-like fields) and emits a single,
inspectable data object (``DosingRecommendation``) that a safety layer
can validate before anything reaches a patient.

The two modules are:

* :mod:`hemosim.clinical.dss` (ISC-10) — ``PatientSnapshot``,
  ``DosingRecommendation``, ``HeparinDSS``, ``WarfarinDSS``. The DSS
  loads a trained stable-baselines3 PPO policy if present and falls
  back to the validated clinical baseline (Raschke 1993 for heparin,
  IWPC-style titration for warfarin) when torch/sb3 are unavailable or
  no checkpoint is on disk. This means *the harness is always
  callable*, which is a precondition for any clinical-translation
  conversation.
* :mod:`hemosim.clinical.safety` (ISC-11) — ``SafetyBounds``,
  ``SafetyGuard``, ``SafeDSS``, ``SafetyCheckResult``. Enforces
  per-drug absolute dose bounds (UCSD antithrombotic protocol, CHEST
  2012, ACCP 2012), contraindications (e.g. HIT / platelets <50
  blocking heparin unless explicitly overridden), and
  uncertainty-aware deferral (``defer_to_clinician`` when confidence
  is below a threshold or the uncertainty interval spans more than
  half the clinical dose range).

The public surface is intentionally small — anything a downstream
caller needs should be importable from :mod:`hemosim.clinical`.
"""

from __future__ import annotations

from hemosim.clinical.dss import (
    BaseDSS,
    DosingRecommendation,
    HeparinDSS,
    PatientSnapshot,
    WarfarinDSS,
)
from hemosim.clinical.safety import (
    HEPARIN_SAFETY_BOUNDS,
    WARFARIN_SAFETY_BOUNDS,
    SafeDSS,
    SafetyBounds,
    SafetyCheckResult,
    SafetyGuard,
)

__all__ = [
    # dss
    "BaseDSS",
    "DosingRecommendation",
    "HeparinDSS",
    "PatientSnapshot",
    "WarfarinDSS",
    # safety
    "HEPARIN_SAFETY_BOUNDS",
    "WARFARIN_SAFETY_BOUNDS",
    "SafeDSS",
    "SafetyBounds",
    "SafetyCheckResult",
    "SafetyGuard",
]
