"""SR-M claim: env_reward_function_shapes

Falsification test for the §4 reward-equation claim. Steps
WarfarinDosing-v0 and HeparinInfusion-v0 at forced on-target and
off-target INR/aPTT values via env.unwrapped (registry permits
unwrapped access for reward-shape tests) and asserts the base reward
contribution matches the paper equation to ±0.01.

Warfarin §4.1: r = -|INR - 2.5| + safety bonus
    - INR = 2.5 → base -|0| = 0.0
    - INR = 3.5 → base -|1| = -1.0
Heparin §4.2: r = -|aPTT - 75| / 30 + safety bonus
    - aPTT = 75 → base 0.0
    - aPTT = 105 → base -1.0
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

import hemosim  # noqa: F401

TOL = 0.01
SAFETY_BONUS_THERAPEUTIC = 0.5  # env adds +0.5 when in therapeutic range


def _strip_bonus(total_reward: float, in_therapeutic: bool) -> float:
    """Remove the therapeutic-range safety bonus to recover the raw distance term."""
    return total_reward - (SAFETY_BONUS_THERAPEUTIC if in_therapeutic else 0.0)


def _force_warfarin_inr(env, target_inr: float) -> None:
    """Force WarfarinPKPD internal INR via unwrapped state[7] = INR factor.

    get_inr() returns state[7] * baseline_inr_factor — we directly set the
    post-multiplication value by bypassing get_inr. Simpler path: monkey-patch
    get_inr on the model instance.
    """
    inner = env.unwrapped
    inner._model.get_inr = lambda: target_inr  # type: ignore[method-assign]


def _force_heparin_aptt(env, target_aptt: float) -> None:
    inner = env.unwrapped
    inner._model.get_aptt = lambda: target_aptt  # type: ignore[method-assign]


def test_srm_warfarin_reward_on_target_is_zero_base():
    """WarfarinDosing-v0 with forced INR=2.5 → base reward 0.0 ± 0.01."""
    env = gym.make("hemosim/WarfarinDosing-v0", difficulty="easy")
    env.reset(seed=42)
    _force_warfarin_inr(env, target_inr=2.5)
    obs, reward, term, trunc, info = env.step(np.array([0.0], dtype=np.float32))
    # Info may report a slightly different INR since step re-queries after
    # model.step; re-force and read info["inr"].
    base = _strip_bonus(reward, info["therapeutic"])
    assert -TOL <= base <= TOL, (
        f"Warfarin base reward at forced INR=2.5 = {base:.4f} (total={reward:.4f}, "
        f"therapeutic={info['therapeutic']}, info_inr={info['inr']:.4f}); "
        f"primary-source target 0.0 ± {TOL}. Registry: if coefficients drift, "
        f"§4 equation must be updated to match code (cannot drop claim)."
    )
    env.close()


def test_srm_warfarin_reward_at_inr_3p5_is_minus_one():
    """WarfarinDosing-v0 with forced INR=3.5 → base reward -1.0 ± 0.01."""
    env = gym.make("hemosim/WarfarinDosing-v0", difficulty="easy")
    env.reset(seed=42)
    _force_warfarin_inr(env, target_inr=3.5)
    obs, reward, term, trunc, info = env.step(np.array([0.0], dtype=np.float32))
    base = _strip_bonus(reward, info["therapeutic"])
    assert -1.0 - TOL <= base <= -1.0 + TOL, (
        f"Warfarin base reward at forced INR=3.5 = {base:.4f}; primary-source "
        f"target -1.0 ± {TOL}."
    )
    env.close()


def test_srm_heparin_reward_at_aptt_75_is_zero_base():
    """HeparinInfusion-v0 with forced aPTT=75 → base reward 0.0 ± 0.01."""
    env = gym.make("hemosim/HeparinInfusion-v0", difficulty="easy")
    env.reset(seed=42)
    _force_heparin_aptt(env, target_aptt=75.0)
    obs, reward, term, trunc, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    base = _strip_bonus(reward, info["therapeutic"])
    assert -TOL <= base <= TOL, (
        f"Heparin base reward at forced aPTT=75 = {base:.4f} (total={reward:.4f}, "
        f"therapeutic={info['therapeutic']}, info_aptt={info['aptt']:.4f}); "
        f"primary-source target 0.0 ± {TOL}."
    )
    env.close()


def test_srm_heparin_reward_at_aptt_105_is_minus_one():
    """HeparinInfusion-v0 with forced aPTT=105 → base reward -1.0 ± 0.01."""
    env = gym.make("hemosim/HeparinInfusion-v0", difficulty="easy")
    env.reset(seed=42)
    _force_heparin_aptt(env, target_aptt=105.0)
    obs, reward, term, trunc, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    base = _strip_bonus(reward, info["therapeutic"])
    # 105-75 = 30 → -30/30 = -1.0
    assert -1.0 - TOL <= base <= -1.0 + TOL, (
        f"Heparin base reward at forced aPTT=105 = {base:.4f}; primary-source "
        f"target -1.0 ± {TOL}."
    )
    env.close()
