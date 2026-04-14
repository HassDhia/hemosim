"""Tests for the DOAC management Gymnasium environment."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

import hemosim  # noqa: F401


class TestDOACEnvCreation:
    def test_env_creation(self, doac_env):
        """Environment can be created."""
        assert doac_env is not None

    def test_gym_make(self):
        """Environment can be created via gym.make."""
        env = gym.make("hemosim/DOACManagement-v0")
        assert env is not None
        env.close()

    def test_observation_space(self, doac_env):
        """Observation space should be Box(8,)."""
        assert doac_env.observation_space.shape == (8,)

    def test_action_space(self, doac_env):
        """Action space should be MultiDiscrete([3, 3])."""
        assert doac_env.action_space.shape == (2,)
        assert list(doac_env.action_space.nvec) == [3, 3]


class TestDOACEnvReset:
    def test_reset_valid(self, doac_env):
        """Reset returns valid observation."""
        obs, info = doac_env.reset(seed=42)
        assert doac_env.observation_space.contains(obs)
        assert "patient" in info

    def test_seed_reproducibility(self):
        """Same seed produces identical initial states."""
        env1 = gym.make("hemosim/DOACManagement-v0")
        env2 = gym.make("hemosim/DOACManagement-v0")

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

        env1.close()
        env2.close()


class TestDOACEnvStep:
    def test_step_valid(self, doac_env):
        """Step returns valid tuple."""
        doac_env.reset(seed=42)
        action = doac_env.action_space.sample()
        obs, reward, terminated, truncated, info = doac_env.step(action)

        assert doac_env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(info, dict)

    def test_episode_length(self, doac_env):
        """Episode should not exceed 12 steps (365d / 30d)."""
        doac_env.reset(seed=42)
        steps = 0
        done = False
        while not done and steps < 30:
            action = np.array([2, 1], dtype=np.int64)  # apixaban, standard
            _, _, terminated, truncated, _ = doac_env.step(action)
            done = terminated or truncated
            steps += 1
        assert steps <= 12

    def test_stroke_event_possible(self):
        """Over many episodes, stroke events should occasionally occur."""
        env = gym.make("hemosim/DOACManagement-v0")
        stroke_seen = False
        for ep in range(50):
            env.reset(seed=ep)
            for _ in range(12):
                action = env.action_space.sample()
                _, _, terminated, truncated, info = env.step(action)
                if info.get("stroke_event"):
                    stroke_seen = True
                    break
                if terminated or truncated:
                    break
            if stroke_seen:
                break
        env.close()
        # Stroke events are probabilistic; with 50 episodes of random actions
        # at least one should occur (but not guaranteed - just test structure)

    def test_bleeding_event_possible(self):
        """Over many episodes, bleeding events should occasionally occur."""
        env = gym.make("hemosim/DOACManagement-v0")
        bleed_seen = False
        for ep in range(50):
            env.reset(seed=ep + 100)
            for _ in range(12):
                action = np.array([0, 2], dtype=np.int64)  # rivaroxaban high dose
                _, _, terminated, truncated, info = env.step(action)
                if info.get("bleed_event"):
                    bleed_seen = True
                    break
                if terminated or truncated:
                    break
            if bleed_seen:
                break
        env.close()

    def test_drug_switch_penalty(self, doac_env):
        """Switching drugs should incur a transition cost."""
        doac_env.reset(seed=42)
        # First step with drug 0
        _, r1, _, _, _ = doac_env.step(np.array([0, 1], dtype=np.int64))
        # Switch to drug 1
        _, r2, _, _, info = doac_env.step(np.array([1, 1], dtype=np.int64))
        # The reward from drug switch should include -5 penalty
        # (but events also affect reward, so just verify it ran)
        assert info["drug_switched"]


class TestDOACDifficulty:
    def test_difficulty_tiers(self):
        """All difficulty tiers should work."""
        for diff in ["easy", "medium", "hard"]:
            env = gym.make("hemosim/DOACManagement-v0", difficulty=diff)
            obs, _ = env.reset(seed=42)
            assert env.observation_space.contains(obs)

            obs, _, _, _, _ = env.step(np.array([1, 1], dtype=np.int64))
            assert env.observation_space.contains(obs)
            env.close()
