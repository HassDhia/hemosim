"""Integration tests for the hemosim package."""

from __future__ import annotations

import gymnasium as gym
import pytest

import hemosim


ALL_ENV_IDS = [
    "hemosim/WarfarinDosing-v0",
    "hemosim/HeparinInfusion-v0",
    "hemosim/DOACManagement-v0",
    "hemosim/DICManagement-v0",
]


class TestFullEpisodes:
    def test_full_warfarin_episode(self):
        """Run a complete warfarin episode with random actions."""
        env = gym.make("hemosim/WarfarinDosing-v0")
        obs, info = env.reset(seed=42)
        total_reward = 0
        steps = 0

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

        assert steps > 0
        assert isinstance(total_reward, float)
        env.close()

    def test_full_heparin_episode(self):
        """Run a complete heparin episode with random actions."""
        env = gym.make("hemosim/HeparinInfusion-v0")
        obs, info = env.reset(seed=42)
        total_reward = 0
        steps = 0

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

        assert steps > 0
        env.close()

    def test_full_doac_episode(self):
        """Run a complete DOAC episode with random actions."""
        env = gym.make("hemosim/DOACManagement-v0")
        obs, info = env.reset(seed=42)
        total_reward = 0
        steps = 0

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

        assert steps > 0
        env.close()

    def test_full_dic_episode(self):
        """Run a complete DIC episode with random actions."""
        env = gym.make("hemosim/DICManagement-v0")
        obs, info = env.reset(seed=42)
        total_reward = 0
        steps = 0

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

        assert steps > 0
        env.close()


class TestEnvironmentRegistration:
    def test_all_envs_registered(self):
        """All four hemosim environments should be registered."""
        registry = gym.envs.registry
        for env_id in ALL_ENV_IDS:
            assert env_id in registry, f"{env_id} not registered"

    def test_all_envs_pass_gymnasium_check(self):
        """All environments should pass the gymnasium env_checker."""
        from gymnasium.utils.env_checker import check_env

        for env_id in ALL_ENV_IDS:
            env = gym.make(env_id)
            # check_env will raise if there are issues
            try:
                check_env(env.unwrapped, skip_render_check=True)
            except Exception as e:
                pytest.fail(f"{env_id} failed gym check: {e}")
            finally:
                env.close()


class TestPackageMetadata:
    def test_version_consistency(self):
        """Package version should be 0.1.0."""
        assert hemosim.__version__ == "0.1.0"

    def test_package_importable(self):
        """All submodules should be importable."""
        from hemosim.models import (
            CoagulationCascade,
            DOACPKPD,
            HeparinPKPD,
            PatientGenerator,
            WarfarinPKPD,
        )
        from hemosim.agents.ppo import ENV_CONFIGS

        assert len(ENV_CONFIGS) == 4
        assert CoagulationCascade is not None
        assert WarfarinPKPD is not None
        assert HeparinPKPD is not None
        assert DOACPKPD is not None
        assert PatientGenerator is not None
