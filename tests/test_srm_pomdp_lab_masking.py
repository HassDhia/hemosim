"""SR-M claim: pomdp_lab_masking

Falsification test for the POMDP env lab-masking + TAT + CV claim.
Uses HeparinInfusion-POMDP-v0 via gym.make() default init. Orders an
aPTT lab at t=0 (primed by reset), steps forward < 45 min (one env
step = 6 h, so we inspect the ordered queue BEFORE any tick returns
the sample — we use the fresh-reset observation where the sample is
pending). Then advances time past TAT and asserts the returned aPTT
differs from ground truth (CV=0.08 multiplicative noise).
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np

import hemosim  # noqa: F401
from hemosim.envs.pomdp import HEPARIN_LAB_SPECS


def test_srm_pomdp_lab_masking_spec_matches_lippi_bowen_targets():
    """HEPARIN_LAB_SPECS TAT and CV constants match primary-source targets."""
    aptt_spec = HEPARIN_LAB_SPECS["aptt"]
    antixa_spec = HEPARIN_LAB_SPECS["anti_xa"]
    plt_spec = HEPARIN_LAB_SPECS["platelets"]
    # Registry primary-source target (Lippi 2012 + Bowen 2016):
    #   aPTT CV = 0.08, TAT = 45 min
    #   anti-Xa CV = 0.12, TAT = 180 min
    #   platelets CV = 0.05, TAT = 45 min
    assert math.isclose(aptt_spec.cv, 0.08, abs_tol=1e-6), (
        f"aPTT CV = {aptt_spec.cv}; primary-source target 0.08 (Lippi 2012)."
    )
    assert math.isclose(aptt_spec.tat_minutes, 45.0, abs_tol=0.01), (
        f"aPTT TAT = {aptt_spec.tat_minutes} min; target 45 (Bowen 2016)."
    )
    assert math.isclose(antixa_spec.cv, 0.12, abs_tol=1e-6), (
        f"anti-Xa CV = {antixa_spec.cv}; target 0.12."
    )
    assert math.isclose(antixa_spec.tat_minutes, 180.0, abs_tol=0.01), (
        f"anti-Xa TAT = {antixa_spec.tat_minutes} min; target 180."
    )
    assert math.isclose(plt_spec.cv, 0.05, abs_tol=1e-6), (
        f"platelets CV = {plt_spec.cv}; target 0.05."
    )


def test_srm_pomdp_lab_masking_masks_and_returns_after_TAT():
    """Ordered aPTT is masked before TAT and returns noisy after TAT."""
    env = gym.make("hemosim/HeparinInfusion-POMDP-v0", difficulty="easy")
    obs, info = env.reset(seed=42)

    # Observation vector has >= 10 channels per registry (lab-order dims).
    assert obs.shape[0] >= 10, (
        f"POMDP heparin obs has {obs.shape[0]} channels; primary-source target "
        f">= 10 (including lab-order actions + lab age channels)."
    )

    # Immediately after reset, aPTT was ordered at t=0 and is still pending
    # (TAT = 45 min, env step = 6h). `obs[0]` is the latest-aptt slot; no
    # sample has returned yet, so sentinel -1.0 must be present.
    aptt_slot = float(obs[0])
    # After reset, no aPTT sample has been RETURNED yet (queue hasn't ticked).
    # The latest-aptt sentinel is -1.0. Assert numeric range.
    assert aptt_slot == -1.0, (
        f"aPTT observation slot = {aptt_slot} immediately after reset; "
        f"primary-source target -1.0 sentinel (masking — no returned sample)."
    )

    # Step once (6 h > 45 min TAT) — sample should return.
    inner = env.unwrapped
    ground_truth_aptt_at_order = info["ground_truth_aptt"]

    action = np.zeros(5, dtype=np.float32)  # no dose, no new lab orders
    obs, reward, terminated, truncated, info = env.step(action)

    latest = info.get("latest_aptt_sample")
    assert latest is not None, (
        "After 6h step (> 45 min TAT), aPTT sample should have returned but "
        "latest_aptt_sample is None. Masking-release claim refuted."
    )
    returned_value = float(latest.value)
    # CV=0.08 variability must produce a NON-ZERO deviation vs ground truth
    # at time of order. Tolerance 1e-6.
    deviation = abs(returned_value - ground_truth_aptt_at_order)
    assert deviation > 1e-6, (
        f"Returned aPTT = {returned_value} exactly equals ground truth "
        f"{ground_truth_aptt_at_order}; CV=0.08 multiplicative noise should "
        f"produce a non-zero deviation. Analytical-variability claim refuted."
    )
    # Upper sanity: noise within physiologic range (CV=0.08 ⇒ >99% within 3σ
    # so 24% multiplicative). Cap the allowable deviation at 50% of ground
    # truth to catch gross noise-scale bugs.
    assert deviation <= 0.5 * abs(ground_truth_aptt_at_order), (
        f"Returned aPTT deviation {deviation} > 50% of ground truth "
        f"{ground_truth_aptt_at_order}; CV=0.08 implies 3σ ≈ 24%. Noise scale off."
    )
    env.close()
