"""SR-M claim: warfarin_hamberg_pkpd

Falsification test for the Hamberg 2007 warfarin PK/PD reproduction
claim. Uses default-init WarfarinDosing-v0 gym env, drives with a
5 mg/day fixed dose for 30 simulated days on a wildtype CYP2C9*1/*1 +
VKORC1 GG patient, and asserts the steady-state INR lies in the
[2.3, 2.7] primary-source band per Hamberg 2007 Table 2.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

import hemosim  # noqa: F401  # triggers env registration

MAX_DOSE_MG = 15.0
TARGET_DOSE_MG = 5.0
DAYS = 30


def _force_wildtype(env: gym.Env) -> None:
    """Re-initialize the underlying PK/PD model for a wildtype patient.

    Public env lacks a direct "force genotype" option, so we go through
    env.unwrapped and reset the PK/PD model to the wildtype branch.
    Registry explicitly permits unwrapped access for reward-shape and
    genotype-forcing tests when primary-source target is genotype-specific.
    """
    from hemosim.models.warfarin_pkpd import WarfarinPKPD

    inner = env.unwrapped
    # Keep the generator-provided demographics, override genotype to WT.
    inner._patient["cyp2c9"] = "*1/*1"
    inner._patient["vkorc1"] = "GG"
    inner._patient["target_inr"] = 2.5
    inner._model = WarfarinPKPD(
        cyp2c9="*1/*1",
        vkorc1="GG",
        age=inner._patient["age"],
        weight=inner._patient["weight"],
    )
    inner._model.reset()


def test_srm_warfarin_hamberg_pkpd_steady_state_inr():
    """Steady-state INR under 5 mg/day is within Hamberg 2007 target [2.3, 2.7]."""
    env = gym.make("hemosim/WarfarinDosing-v0", difficulty="easy")
    obs, info = env.reset(seed=42)
    _force_wildtype(env)

    # Action scaled in [0,1] -> [0, MAX_DOSE_MG=15]; 5 mg -> 5/15.
    action = np.array([TARGET_DOSE_MG / MAX_DOSE_MG], dtype=np.float32)
    inr_trajectory: list[float] = []
    for _ in range(DAYS):
        obs, reward, terminated, truncated, info = env.step(action)
        inr_trajectory.append(float(info["inr"]))
        if terminated:
            break

    # Steady-state: last 7 days average (days 23-30) — Hamberg SS window.
    steady_inr = float(np.mean(inr_trajectory[-7:]))
    assert 2.3 <= steady_inr <= 2.7, (
        f"Steady-state INR {steady_inr:.3f} outside Hamberg 2007 primary-source "
        f"target range [2.3, 2.7] for CYP2C9*1/*1 + VKORC1 GG patient on 5 mg/day "
        f"warfarin. Trajectory (last 7 days): {inr_trajectory[-7:]}"
    )
    env.close()


def test_srm_warfarin_hamberg_pkpd_published_calibration_rmse():
    """Warfarin-fit RMSE in results/published_calibration.json ≤ 0.005."""
    import json
    from pathlib import Path

    path = Path(__file__).parent.parent / "results" / "published_calibration.json"
    if not path.is_file():
        # If the artifact isn't shipped, the abstract RMSE claim cannot be
        # assessed from tests alone — fail loudly rather than skip.
        raise AssertionError(
            f"results/published_calibration.json missing at {path}; "
            f"required for warfarin RMSE primary-source target check."
        )
    data = json.loads(path.read_text())
    fit = data.get("warfarin_fit") or data.get("fit") or {}
    rmse = float(fit.get("rmse", fit.get("normalized_rmse", 0.0)))
    assert 0.0 < rmse <= 0.005, (
        f"Published calibration RMSE = {rmse} outside primary-source target "
        f"(0, 0.005]; abstract claims normalized RMSE 0.0013."
    )
