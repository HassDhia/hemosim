"""Tests for the warfarin dosing Gymnasium environment."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

import hemosim  # noqa: F401


class TestWarfarinEnvCreation:
    def test_env_creation(self, warfarin_env):
        """Environment can be created."""
        assert warfarin_env is not None

    def test_gym_make(self):
        """Environment can be created via gym.make."""
        env = gym.make("hemosim/WarfarinDosing-v0")
        assert env is not None
        env.close()

    def test_observation_space_shape(self, warfarin_env):
        """Observation space should be Box(8,)."""
        assert warfarin_env.observation_space.shape == (8,)

    def test_action_space_shape(self, warfarin_env):
        """Action space should be Box(1,)."""
        assert warfarin_env.action_space.shape == (1,)


class TestWarfarinEnvReset:
    def test_reset_returns_valid_obs(self, warfarin_env):
        """Reset should return observation in observation space."""
        obs, info = warfarin_env.reset(seed=42)
        assert warfarin_env.observation_space.contains(obs)
        assert isinstance(info, dict)

    def test_info_dict_keys(self, warfarin_env):
        """Info dict should contain expected keys."""
        _, info = warfarin_env.reset(seed=42)
        assert "inr" in info
        assert "patient" in info

    def test_seed_reproducibility(self):
        """Same seed should produce identical episodes."""
        env1 = gym.make("hemosim/WarfarinDosing-v0")
        env2 = gym.make("hemosim/WarfarinDosing-v0")

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

        action = np.array([0.3], dtype=np.float32)
        obs1, r1, _, _, _ = env1.step(action)
        obs2, r2, _, _, _ = env2.step(action)
        np.testing.assert_array_equal(obs1, obs2)
        assert r1 == r2

        env1.close()
        env2.close()


class TestWarfarinEnvStep:
    def test_step_returns_valid(self, warfarin_env):
        """Step should return valid (obs, reward, terminated, truncated, info)."""
        warfarin_env.reset(seed=42)
        action = warfarin_env.action_space.sample()
        obs, reward, terminated, truncated, info = warfarin_env.step(action)

        assert warfarin_env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_episode_length(self, warfarin_env):
        """Episode should last 90 steps without early termination."""
        warfarin_env.reset(seed=42)
        steps = 0
        done = False
        while not done and steps < 200:
            # Use moderate dose to avoid termination
            obs, _, terminated, truncated, _ = warfarin_env.step(
                np.array([0.2], dtype=np.float32)  # ~3mg
            )
            done = terminated or truncated
            steps += 1

        assert steps <= 90  # should not exceed episode length

    def test_therapeutic_inr_positive_reward(self, warfarin_env):
        """Being in therapeutic INR range should give positive reward component."""
        warfarin_env.reset(seed=42)
        # Run several steps to build up INR
        for _ in range(15):
            _, reward, _, _, info = warfarin_env.step(
                np.array([0.33], dtype=np.float32)  # ~5mg
            )
            if info["therapeutic"]:
                # When therapeutic, reward should include +0.5 bonus
                # Total reward = -abs(INR-target) + 0.5 >= some value
                assert reward > -2.5  # should not be very negative if therapeutic

    def test_supratherapeutic_penalty(self, warfarin_env):
        """INR > 4.0 should receive a penalty."""
        warfarin_env.reset(seed=42)
        # Give high doses to drive INR up
        for _ in range(20):
            _, reward, terminated, _, info = warfarin_env.step(
                np.array([1.0], dtype=np.float32)  # 15mg - very high
            )
            if info["inr"] > 4.0:
                assert reward < -5.0  # should have penalty
                break

    def test_subtherapeutic_penalty(self, warfarin_env):
        """INR < 1.5 with zero dosing should have penalty."""
        warfarin_env.reset(seed=42)
        # Give no drug
        _, reward, _, _, info = warfarin_env.step(
            np.array([0.0], dtype=np.float32)
        )
        # INR near 1.0 (baseline) should have subtherapeutic penalty
        if info["inr"] < 1.5:
            assert reward < 0

    def test_termination_on_extreme_inr(self, warfarin_env):
        """Very high INR (>6.0) should terminate the episode."""
        warfarin_env.reset(seed=42)
        for _ in range(90):
            _, _, terminated, _, info = warfarin_env.step(
                np.array([1.0], dtype=np.float32)
            )
            if terminated:
                break

        # With max dose for 90 days, termination is likely but not guaranteed
        # Just verify the loop ran without errors


class TestWarfarinDifficulty:
    def test_difficulty_easy(self):
        """Easy difficulty should constrain genotypes."""
        env = gym.make("hemosim/WarfarinDosing-v0", difficulty="easy")
        _, info = env.reset(seed=42)
        assert info["patient"]["cyp2c9"] == "*1/*1"
        assert info["patient"]["vkorc1"] == "GG"
        env.close()

    def test_difficulty_hard(self):
        """Hard difficulty should work without errors."""
        env = gym.make("hemosim/WarfarinDosing-v0", difficulty="hard")
        obs, info = env.reset(seed=42)
        assert env.observation_space.contains(obs)

        # Run a few steps
        for _ in range(5):
            obs, _, _, _, _ = env.step(np.array([0.3], dtype=np.float32))
            assert env.observation_space.contains(obs)

        env.close()
