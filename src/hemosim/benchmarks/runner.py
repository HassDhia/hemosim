"""Benchmark runner for hemosim environments.

CLI entry point: hemosim-benchmark
Runs PPO (if trained) vs clinical baselines vs random across all environments.
Evaluates over 100 episodes per environment and saves results.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import numpy as np

import hemosim  # noqa: F401 - triggers env registration
from hemosim.agents.baselines import (
    DICProtocolBaseline,
    DOACGuidelineBaseline,
    HeparinRaschkeBaseline,
    RandomBaseline,
    WarfarinClinicalBaseline,
)
from hemosim.agents.ppo import ENV_CONFIGS

# Map env IDs to their clinical baseline classes
BASELINE_MAP = {
    "hemosim/WarfarinDosing-v0": WarfarinClinicalBaseline,
    "hemosim/HeparinInfusion-v0": HeparinRaschkeBaseline,
    "hemosim/DOACManagement-v0": DOACGuidelineBaseline,
    "hemosim/DICManagement-v0": DICProtocolBaseline,
}


def evaluate_agent(env, agent, n_episodes: int = 100, seed: int = 42) -> dict:
    """Evaluate an agent on an environment over multiple episodes.

    Returns dict with mean_reward, std_reward, episodes, and therapeutic_rate.
    """
    rewards = []
    therapeutic_steps = 0
    total_steps = 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            total_steps += 1

            # Track therapeutic rate where applicable
            if "therapeutic" in info:
                therapeutic_steps += int(info["therapeutic"])

            done = terminated or truncated

        rewards.append(episode_reward)

    therapeutic_rate = therapeutic_steps / max(total_steps, 1)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "episodes": n_episodes,
        "therapeutic_rate": float(therapeutic_rate),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
    }


def run_benchmark(
    n_episodes: int = 100,
    seed: int = 42,
    results_dir: str = "results",
    include_ppo: bool = False,
) -> dict:
    """Run full benchmark across all environments.

    Args:
        n_episodes: Episodes per agent per environment.
        seed: Random seed.
        results_dir: Directory to save results.
        include_ppo: If True, attempt to load trained PPO models.

    Returns:
        Full results dict.
    """
    results = {"environments": {}}

    for env_id in ENV_CONFIGS:
        print(f"\nBenchmarking {env_id}...")
        env = gym.make(env_id)

        env_results = {}

        # Clinical baseline
        baseline_cls = BASELINE_MAP[env_id]
        baseline = baseline_cls(seed=seed)
        print(f"  Running clinical baseline ({baseline_cls.__name__})...")
        env_results["clinical_baseline"] = evaluate_agent(
            env, baseline, n_episodes=n_episodes, seed=seed
        )

        # Random baseline
        random_agent = RandomBaseline(env.action_space, seed=seed)
        print("  Running random baseline...")
        env_results["random"] = evaluate_agent(
            env, random_agent, n_episodes=n_episodes, seed=seed
        )

        # PPO (if trained model exists)
        if include_ppo:
            env_short = env_id.split("/")[1]
            model_path = Path(results_dir) / "models" / f"{env_short}_ppo_final.zip"
            if model_path.exists():
                try:
                    from stable_baselines3 import PPO

                    ppo_model = PPO.load(str(model_path))

                    class PPOWrapper:
                        def __init__(self, model):
                            self.model = model

                        def predict(self, obs):
                            action, _ = self.model.predict(obs, deterministic=True)
                            return action

                    ppo_agent = PPOWrapper(ppo_model)
                    print("  Running PPO agent...")
                    env_results["ppo"] = evaluate_agent(
                        env, ppo_agent, n_episodes=n_episodes, seed=seed
                    )
                except ImportError:
                    print("  Skipping PPO (stable-baselines3 not installed)")
            else:
                print(f"  No trained PPO model found at {model_path}")

        env.close()
        results["environments"][env_id] = env_results

    # Add metadata
    results["training_config"] = {
        "seed": seed,
        "total_timesteps_per_env": {
            k: v["total_timesteps"] for k, v in ENV_CONFIGS.items()
        },
    }
    results["metadata"] = {
        "package_version": hemosim.__version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "test_count": n_episodes,
    }

    # Save results
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    output_file = results_path / "training_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")
    return results


def main():
    """CLI entry point for hemosim-benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark hemosim agents across all environments"
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="Episodes per agent per env"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--results-dir", type=str, default="results", help="Results directory"
    )
    parser.add_argument(
        "--ppo", action="store_true", help="Include PPO agent (requires trained models)"
    )

    args = parser.parse_args()
    run_benchmark(
        n_episodes=args.episodes,
        seed=args.seed,
        results_dir=args.results_dir,
        include_ppo=args.ppo,
    )


if __name__ == "__main__":
    main()
