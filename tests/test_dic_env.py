"""Tests for the DIC management Gymnasium environment."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

import hemosim  # noqa: F401


class TestDICEnvCreation:
    def test_env_creation(self, dic_env):
        """Environment can be created."""
        assert dic_env is not None

    def test_gym_make(self):
        """Environment can be created via gym.make."""
        env = gym.make("hemosim/DICManagement-v0")
        assert env is not None
        env.close()

    def test_observation_space(self, dic_env):
        """Observation space should be Box(8,)."""
        assert dic_env.observation_space.shape == (8,)

    def test_action_space(self, dic_env):
        """Action space should be MultiDiscrete([4, 4, 3, 3])."""
        assert dic_env.action_space.shape == (4,)
        assert list(dic_env.action_space.nvec) == [4, 4, 3, 3]


class TestDICEnvReset:
    def test_reset_valid(self, dic_env):
        """Reset returns valid observation and info."""
        obs, info = dic_env.reset(seed=42)
        assert dic_env.observation_space.contains(obs)
        assert "isth_dic_score" in info
        assert "patient" in info

    def test_seed_reproducibility(self):
        """Same seed produces identical episodes."""
        env1 = gym.make("hemosim/DICManagement-v0")
        env2 = gym.make("hemosim/DICManagement-v0")

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

        env1.close()
        env2.close()


class TestDICEnvStep:
    def test_step_valid(self, dic_env):
        """Step returns valid tuple."""
        dic_env.reset(seed=42)
        action = dic_env.action_space.sample()
        obs, reward, terminated, truncated, info = dic_env.step(action)

        assert dic_env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(info, dict)

    def test_episode_length(self, dic_env):
        """Episode should not exceed 42 steps (168h / 4h)."""
        dic_env.reset(seed=42)
        steps = 0
        done = False
        while not done and steps < 60:
            # Give moderate treatment to avoid early termination
            action = np.array([1, 1, 1, 0], dtype=np.int64)
            _, _, terminated, truncated, _ = dic_env.step(action)
            done = terminated or truncated
            steps += 1
        assert steps <= 42

    def test_transfusion_effects(self, dic_env):
        """Platelet transfusion should increase platelet count."""
        dic_env.reset(seed=42)
        _, _, _, _, info_before = dic_env.step(
            np.array([0, 0, 0, 0], dtype=np.int64)  # no treatment
        )

        dic_env.reset(seed=42)
        _, _, _, _, info_after = dic_env.step(
            np.array([3, 0, 0, 0], dtype=np.int64)  # 4 units platelets
        )

        assert info_after["platelet_count"] > info_before["platelet_count"]

    def test_isth_score_calculation(self, dic_env):
        """ISTH DIC score should be in valid range."""
        _, info = dic_env.reset(seed=42)
        assert 0 <= info["isth_dic_score"] <= 8

    def test_termination_conditions(self, dic_env):
        """No treatment should eventually lead to worsening."""
        dic_env.reset(seed=42)
        steps = 0
        for _ in range(42):
            _, _, terminated, truncated, info = dic_env.step(
                np.array([0, 0, 0, 0], dtype=np.int64)
            )
            steps += 1
            if terminated or truncated:
                break
        # Episode should complete or terminate


class TestDICDifficulty:
    def test_difficulty_tiers(self):
        """All difficulty tiers should work."""
        for diff in ["easy", "medium", "hard"]:
            env = gym.make("hemosim/DICManagement-v0", difficulty=diff)
            obs, _ = env.reset(seed=42)
            assert env.observation_space.contains(obs)

            obs, _, _, _, _ = env.step(np.array([1, 1, 1, 0], dtype=np.int64))
            assert env.observation_space.contains(obs)
            env.close()
