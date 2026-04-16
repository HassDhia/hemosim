"""Tests for the extended clinical and RL-derived baselines.

Covers (ISC-6):
- action-shape conformance for all four new baselines
- determinism given identical observations
- Nemati DQN proxy-policy end-to-end execution on HeparinInfusion-v0
- Gage pharmacogenetic dose prediction against the 2008 paper's regression
- backward compatibility with existing baselines (imports, package surface)
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import hemosim  # noqa: F401 - triggers env registration
from hemosim.agents.baselines_extended import (
    HeparinAntiXaBaseline,
    NematiDQN2016Baseline,
    WarfarinGageBaseline,
    WarfarinOrdinalBaseline,
)


# ---------------------------------------------------------------------------
# Observations used across tests
# ---------------------------------------------------------------------------


def _heparin_obs(
    aptt_s: float = 75.0,
    concentration: float = 0.5,
    weight_kg: float = 80.0,
    renal: float = 0.9,
    platelets: float = 240.0,
    hours: float = 0.0,
) -> np.ndarray:
    """Build a valid HeparinInfusion-v0 observation from clinical units."""

    return np.array(
        [
            np.clip((aptt_s - 20.0) / 180.0, 0.0, 1.0),
            np.clip(concentration, 0.0, 1.0),
            np.clip((weight_kg - 40.0) / 140.0, 0.0, 1.0),
            np.clip(renal, 0.0, 1.0),
            np.clip(platelets / 400.0, 0.0, 1.0),
            np.clip(hours / 120.0, 0.0, 1.0),
        ],
        dtype=np.float32,
    )


def _warfarin_obs(
    inr: float = 2.5,
    s_warf: float = 0.1,
    r_warf: float = 0.05,
    age: float = 65.0,
    weight_kg: float = 80.0,
    cyp2c9_idx: int = 0,
    vkorc1_idx: int = 0,
    days: float = 5.0,
) -> np.ndarray:
    """Build a valid WarfarinDosing-v0 observation from clinical units."""

    return np.array(
        [
            np.clip(inr / 6.0, 0.0, 1.0),
            np.clip(s_warf / 5.0, 0.0, 1.0),
            np.clip(r_warf / 5.0, 0.0, 1.0),
            np.clip((age - 20.0) / 75.0, 0.0, 1.0),
            np.clip((weight_kg - 40.0) / 110.0, 0.0, 1.0),
            cyp2c9_idx / 5.0,
            vkorc1_idx / 2.0,
            np.clip(days / 90.0, 0.0, 1.0),
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# NematiDQN2016Baseline
# ---------------------------------------------------------------------------


class TestNematiDQNBaseline:
    def test_action_shape_and_bounds(self):
        """Predicted action conforms to HeparinInfusion-v0 action space."""

        baseline = NematiDQN2016Baseline(seed=42)
        action = baseline.predict(_heparin_obs(aptt_s=50.0))
        assert action.shape == (2,)
        assert action.dtype == np.float32
        assert 0.0 <= action[0] <= 1.0
        assert action[1] in (0.0, 1.0)

    def test_initial_step_emits_bolus(self):
        """At t=0, the Raschke-style bolus should be issued."""

        baseline = NematiDQN2016Baseline(seed=42)
        action = baseline.predict(_heparin_obs(aptt_s=30.0, hours=0.0))
        assert action[1] == pytest.approx(1.0)

    def test_proxy_increases_rate_when_subtherapeutic(self):
        """Proxy policy must escalate infusion when aPTT is below range."""

        baseline = NematiDQN2016Baseline(seed=42)
        # First observation at t=0 primes the internal rate memory.
        baseline.predict(_heparin_obs(aptt_s=30.0, hours=0.0))
        low_action = baseline.predict(_heparin_obs(aptt_s=30.0, hours=6.0))
        # Reset internal state before the high-aPTT path.
        baseline.reset()
        baseline.predict(_heparin_obs(aptt_s=30.0, hours=0.0))
        high_action = baseline.predict(_heparin_obs(aptt_s=140.0, hours=6.0))
        assert low_action[0] > high_action[0]

    def test_deterministic_given_same_sequence(self):
        """Two fresh baselines receiving the same obs stream agree."""

        b1 = NematiDQN2016Baseline(seed=42)
        b2 = NematiDQN2016Baseline(seed=42)
        seq = [
            _heparin_obs(aptt_s=30.0, hours=0.0),
            _heparin_obs(aptt_s=55.0, hours=6.0),
            _heparin_obs(aptt_s=90.0, hours=12.0),
        ]
        for obs in seq:
            np.testing.assert_array_equal(b1.predict(obs), b2.predict(obs))

    def test_proxy_end_to_end_episode(self):
        """Proxy policy must complete a full HeparinInfusion-v0 episode."""

        env = gym.make("hemosim/HeparinInfusion-v0")
        try:
            baseline = NematiDQN2016Baseline(seed=42)
            obs, _ = env.reset(seed=42)
            steps = 0
            done = False
            while not done:
                action = baseline.predict(obs)
                assert env.action_space.contains(action), (
                    f"Action {action} violates HeparinInfusion-v0 action_space"
                )
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                steps += 1
            assert steps > 0
        finally:
            env.close()


# ---------------------------------------------------------------------------
# HeparinAntiXaBaseline
# ---------------------------------------------------------------------------


class TestHeparinAntiXaBaseline:
    def test_action_shape_and_bounds(self):
        baseline = HeparinAntiXaBaseline(seed=42)
        action = baseline.predict(_heparin_obs(hours=0.0))
        assert action.shape == (2,)
        assert action.dtype == np.float32
        assert 0.0 <= action[0] <= 1.0
        assert action[1] in (0.0, 1.0)

    def test_rate_increases_below_antixa_range(self):
        """Subtherapeutic anti-Xa proxy should raise the next infusion rate."""

        baseline = HeparinAntiXaBaseline(seed=42)
        baseline.predict(_heparin_obs(concentration=0.5, hours=0.0))  # prime at t=0
        low = baseline.predict(_heparin_obs(concentration=0.10, hours=6.0))
        baseline.reset()
        baseline.predict(_heparin_obs(concentration=0.5, hours=0.0))
        high = baseline.predict(_heparin_obs(concentration=0.90, hours=6.0))
        assert low[0] > high[0]

    def test_deterministic(self):
        b1 = HeparinAntiXaBaseline(seed=42)
        b2 = HeparinAntiXaBaseline(seed=42)
        for hours in (0.0, 6.0, 24.0):
            obs = _heparin_obs(concentration=0.4, hours=hours)
            np.testing.assert_array_equal(b1.predict(obs), b2.predict(obs))


# ---------------------------------------------------------------------------
# WarfarinGageBaseline
# ---------------------------------------------------------------------------


class TestWarfarinGageBaseline:
    def test_action_shape_and_bounds(self):
        baseline = WarfarinGageBaseline()
        action = baseline.predict(_warfarin_obs())
        assert action.shape == (1,)
        assert action.dtype == np.float32
        assert 0.0 <= action[0] <= 1.0

    def test_weekly_prediction_matches_gage_2008(self):
        """Gage regression should predict the expected maintenance dose range.

        For a 65-year-old, 80 kg patient with *1/*1 / GG genotype and
        target INR 2.5, the 2008 paper's regression predicts a
        therapeutic weekly dose on the order of 30-50 mg/week — well within
        the clinically reported range for wild-type patients.
        """

        weekly = WarfarinGageBaseline.predict_weekly_dose_mg(
            age_years=65,
            weight_kg=80,
            cyp2c9="*1/*1",
            vkorc1="GG",
            target_inr=2.5,
        )
        assert 25.0 <= weekly <= 55.0

    def test_poor_metabolizer_gets_lower_dose(self):
        """*3/*3 / AA patients must receive a lower predicted dose than *1/*1 / GG."""

        fast = WarfarinGageBaseline.predict_weekly_dose_mg(
            age_years=65, weight_kg=80, cyp2c9="*1/*1", vkorc1="GG", target_inr=2.5
        )
        slow = WarfarinGageBaseline.predict_weekly_dose_mg(
            age_years=65, weight_kg=80, cyp2c9="*3/*3", vkorc1="AA", target_inr=2.5
        )
        assert slow < fast

    def test_deterministic(self):
        b1 = WarfarinGageBaseline()
        b2 = WarfarinGageBaseline()
        obs = _warfarin_obs(cyp2c9_idx=2, vkorc1_idx=1, days=10.0)
        np.testing.assert_array_equal(b1.predict(obs), b2.predict(obs))


# ---------------------------------------------------------------------------
# WarfarinOrdinalBaseline
# ---------------------------------------------------------------------------


class TestWarfarinOrdinalBaseline:
    def test_action_shape_and_bounds(self):
        baseline = WarfarinOrdinalBaseline()
        action = baseline.predict(_warfarin_obs(days=0.0))
        assert action.shape == (1,)
        assert action.dtype == np.float32
        assert 0.0 <= action[0] <= 1.0

    def test_loading_dose_is_5mg(self):
        """Day 0 and day 1 should both emit the 5 mg loading dose."""

        baseline = WarfarinOrdinalBaseline()
        action_day0 = baseline.predict(_warfarin_obs(days=0.0))
        action_day1 = baseline.predict(_warfarin_obs(days=1.0))
        expected = 5.0 / 15.0
        assert action_day0[0] == pytest.approx(expected, rel=1e-4)
        assert action_day1[0] == pytest.approx(expected, rel=1e-4)

    def test_supratherapeutic_inr_holds_dose(self):
        """INR > 5 in maintenance phase should trigger a hold (dose == 0)."""

        baseline = WarfarinOrdinalBaseline()
        # Prime through loading period.
        baseline.predict(_warfarin_obs(days=0.0))
        baseline.predict(_warfarin_obs(days=1.0))
        action = baseline.predict(_warfarin_obs(inr=5.5, days=5.0))
        assert action[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Package-surface regression — baselines module must still import cleanly.
# ---------------------------------------------------------------------------


def test_extended_baselines_exported_from_agents_package():
    """Extended baselines must surface via ``hemosim.agents``."""

    from hemosim import agents

    for name in (
        "NematiDQN2016Baseline",
        "HeparinAntiXaBaseline",
        "WarfarinGageBaseline",
        "WarfarinOrdinalBaseline",
    ):
        assert hasattr(agents, name), f"hemosim.agents missing {name}"
