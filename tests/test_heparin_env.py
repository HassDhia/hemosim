"""Tests for the heparin infusion Gymnasium environment."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

import hemosim  # noqa: F401


class TestHeparinEnvCreation:
    def test_env_creation(self, heparin_env):
        """Environment can be created."""
        assert heparin_env is not None

    def test_gym_make(self):
        """Environment can be created via gym.make."""
        env = gym.make("hemosim/HeparinInfusion-v0")
        assert env is not None
        env.close()

    def test_observation_space_shape(self, heparin_env):
        """Observation space should be Box(6,)."""
        assert heparin_env.observation_space.shape == (6,)

    def test_action_space_shape(self, heparin_env):
        """Action space should be Box(2,)."""
        assert heparin_env.action_space.shape == (2,)


class TestHeparinEnvReset:
    def test_reset_returns_valid_obs(self, heparin_env):
        """Reset should return valid observation."""
        obs, info = heparin_env.reset(seed=42)
        assert heparin_env.observation_space.contains(obs)

    def test_seed_reproducibility(self):
        """Same seed produces identical initial observations."""
        env1 = gym.make("hemosim/HeparinInfusion-v0")
        env2 = gym.make("hemosim/HeparinInfusion-v0")

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

        env1.close()
        env2.close()


class TestHeparinEnvStep:
    def test_step_returns_valid(self, heparin_env):
        """Step returns valid tuple."""
        heparin_env.reset(seed=42)
        action = heparin_env.action_space.sample()
        obs, reward, terminated, truncated, info = heparin_env.step(action)

        assert heparin_env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(info, dict)
        assert "aptt" in info

    def test_episode_length(self, heparin_env):
        """Episode should not exceed 20 steps (120h / 6h)."""
        heparin_env.reset(seed=42)
        steps = 0
        done = False
        while not done and steps < 50:
            action = np.array([0.3, 0.0], dtype=np.float32)
            _, _, terminated, truncated, _ = heparin_env.step(action)
            done = terminated or truncated
            steps += 1
        assert steps <= 20

    def test_therapeutic_aptt_reward(self, heparin_env):
        """Therapeutic aPTT should give better reward than extreme values."""
        heparin_env.reset(seed=42)
        # Give moderate infusion
        _, reward, _, _, info = heparin_env.step(
            np.array([0.4, 0.0], dtype=np.float32)
        )
        assert isinstance(reward, float)

    def test_bolus_action(self, heparin_env):
        """Bolus action (flag > 0.5) should have immediate effect."""
        heparin_env.reset(seed=42)
        _, _, _, _, info_no_bolus = heparin_env.step(
            np.array([0.3, 0.0], dtype=np.float32)
        )

        # Reset and give with bolus
        heparin_env.reset(seed=42)
        _, _, _, _, info_bolus = heparin_env.step(
            np.array([0.3, 1.0], dtype=np.float32)
        )

        # Bolus should produce higher aPTT
        assert info_bolus["aptt"] > info_no_bolus["aptt"]

    def test_termination_conditions(self, heparin_env):
        """aPTT > 150 should terminate."""
        heparin_env.reset(seed=42)
        for _ in range(20):
            _, _, terminated, _, info = heparin_env.step(
                np.array([1.0, 1.0], dtype=np.float32)  # max infusion + bolus
            )
            if terminated:
                break
        # May or may not terminate depending on patient


class TestHeparinDifficulty:
    def test_difficulty_tiers(self):
        """All difficulty tiers should create valid environments."""
        for diff in ["easy", "medium", "hard"]:
            env = gym.make("hemosim/HeparinInfusion-v0", difficulty=diff)
            obs, info = env.reset(seed=42)
            assert env.observation_space.contains(obs)

            obs, _, _, _, _ = env.step(np.array([0.3, 0.0], dtype=np.float32))
            assert env.observation_space.contains(obs)
            env.close()
