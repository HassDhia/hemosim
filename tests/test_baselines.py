"""Tests for clinical baseline agents."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

import hemosim  # noqa: F401
from hemosim.agents.baselines import (
    DICProtocolBaseline,
    DOACGuidelineBaseline,
    HeparinRaschkeBaseline,
    RandomBaseline,
    WarfarinClinicalBaseline,
)


class TestWarfarinBaseline:
    def test_warfarin_baseline_predict(self):
        """Warfarin baseline should return valid action."""
        baseline = WarfarinClinicalBaseline(seed=42)
        obs = np.array([0.4, 0.1, 0.05, 0.5, 0.4, 0.0, 0.0, 0.1], dtype=np.float32)
        action = baseline.predict(obs)
        assert action.shape == (1,)
        assert 0 <= action[0] <= 1

    def test_warfarin_baseline_deterministic(self):
        """Same observation should produce same action."""
        baseline = WarfarinClinicalBaseline(seed=42)
        obs = np.array([0.4, 0.1, 0.05, 0.5, 0.4, 0.0, 0.0, 0.1], dtype=np.float32)
        a1 = baseline.predict(obs)
        a2 = baseline.predict(obs)
        np.testing.assert_array_equal(a1, a2)


class TestHeparinBaseline:
    def test_heparin_baseline_predict(self):
        """Heparin baseline should return valid action."""
        baseline = HeparinRaschkeBaseline(seed=42)
        obs = np.array([0.1, 0.0, 0.4, 0.9, 0.6, 0.0], dtype=np.float32)
        action = baseline.predict(obs)
        assert action.shape == (2,)
        assert 0 <= action[0] <= 1

    def test_heparin_initial_bolus(self):
        """First step (time=0) should give bolus."""
        baseline = HeparinRaschkeBaseline(seed=42)
        obs = np.array([0.05, 0.0, 0.4, 0.9, 0.6, 0.0], dtype=np.float32)
        action = baseline.predict(obs)
        assert action[1] == 1.0  # bolus flag


class TestDOACBaseline:
    def test_doac_baseline_predict(self):
        """DOAC baseline should return valid action."""
        baseline = DOACGuidelineBaseline(seed=42)
        obs = np.array([0.1, 0.7, 0.5, 0.5, 0.3, 0.2, 0.0, 0.0], dtype=np.float32)
        action = baseline.predict(obs)
        assert len(action) == 2
        assert 0 <= action[0] <= 2  # drug choice
        assert 0 <= action[1] <= 2  # dose level


class TestDICBaseline:
    def test_dic_baseline_predict(self):
        """DIC baseline should return valid action."""
        baseline = DICProtocolBaseline(seed=42)
        obs = np.array([0.5, 0.3, 0.4, 0.4, 0.3, 0.7, 0.1, 0.2], dtype=np.float32)
        action = baseline.predict(obs)
        assert len(action) == 4
        assert 0 <= action[0] <= 3  # platelet tx
        assert 0 <= action[1] <= 3  # FFP
        assert 0 <= action[2] <= 2  # cryo
        assert 0 <= action[3] <= 2  # heparin


class TestRandomBaseline:
    def test_random_baseline_action_valid(self):
        """Random baseline should produce actions within action space."""
        env = gym.make("hemosim/WarfarinDosing-v0")
        random_agent = RandomBaseline(env.action_space, seed=42)
        obs = np.zeros(8, dtype=np.float32)
        action = random_agent.predict(obs)
        assert env.action_space.contains(action)
        env.close()


class TestBaselineEpisodeCompletion:
    def test_warfarin_baseline_episode(self):
        """Warfarin baseline should complete a full episode."""
        env = gym.make("hemosim/WarfarinDosing-v0")
        baseline = WarfarinClinicalBaseline(seed=42)
        obs, _ = env.reset(seed=42)
        done = False
        steps = 0
        while not done:
            action = baseline.predict(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        assert steps > 0
        env.close()

    def test_heparin_baseline_episode(self):
        """Heparin baseline should complete a full episode."""
        env = gym.make("hemosim/HeparinInfusion-v0")
        baseline = HeparinRaschkeBaseline(seed=42)
        obs, _ = env.reset(seed=42)
        done = False
        steps = 0
        while not done:
            action = baseline.predict(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        assert steps > 0
        env.close()

    def test_doac_baseline_episode(self):
        """DOAC baseline should complete a full episode."""
        env = gym.make("hemosim/DOACManagement-v0")
        baseline = DOACGuidelineBaseline(seed=42)
        obs, _ = env.reset(seed=42)
        done = False
        steps = 0
        while not done:
            action = baseline.predict(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        assert steps > 0
        env.close()

    def test_dic_baseline_episode(self):
        """DIC baseline should complete a full episode."""
        env = gym.make("hemosim/DICManagement-v0")
        baseline = DICProtocolBaseline(seed=42)
        obs, _ = env.reset(seed=42)
        done = False
        steps = 0
        while not done:
            action = baseline.predict(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        assert steps > 0
        env.close()
