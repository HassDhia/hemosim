"""PPO training script for hemosim environments.

CLI entry point: hemosim-train
Uses stable-baselines3 for PPO training across all four hemosim environments.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Environment configurations shared between train and benchmark
ENV_CONFIGS = {
    "hemosim/WarfarinDosing-v0": {"total_timesteps": 500_000, "n_envs": 4},
    "hemosim/HeparinInfusion-v0": {"total_timesteps": 300_000, "n_envs": 4},
    "hemosim/DOACManagement-v0": {"total_timesteps": 300_000, "n_envs": 4},
    "hemosim/DICManagement-v0": {"total_timesteps": 500_000, "n_envs": 4},
}


def train(env_id: str, seed: int = 42, timesteps: int | None = None, results_dir: str = "results"):
    """Train a PPO agent on the specified hemosim environment.

    Args:
        env_id: Gymnasium environment ID.
        seed: Random seed for reproducibility.
        timesteps: Total training timesteps (overrides ENV_CONFIGS default).
        results_dir: Directory to save results and models.
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError:
        print("Error: stable-baselines3 and torch are required for training.")
        print("Install with: pip install hemosim[train]")
        sys.exit(1)

    import hemosim  # noqa: F401 - triggers env registration

    config = ENV_CONFIGS.get(env_id)
    if config is None:
        print(f"Unknown environment: {env_id}")
        print(f"Available: {list(ENV_CONFIGS.keys())}")
        sys.exit(1)

    total_timesteps = timesteps or config["total_timesteps"]
    n_envs = config["n_envs"]

    print(f"Training PPO on {env_id}")
    print(f"  Seed: {seed}")
    print(f"  Timesteps: {total_timesteps:,}")
    print(f"  Parallel envs: {n_envs}")

    # Create vectorized environment
    vec_env = make_vec_env(env_id, n_envs=n_envs, seed=seed)

    # Create evaluation environment
    eval_env = make_vec_env(env_id, n_envs=1, seed=seed + 1000)

    # Results directory
    results_path = Path(results_dir)
    models_path = results_path / "models"
    models_path.mkdir(parents=True, exist_ok=True)

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(models_path),
        log_path=str(results_path),
        eval_freq=max(total_timesteps // 20, 1000),
        n_eval_episodes=10,
        deterministic=True,
    )

    # Train PPO
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=seed,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # Save final model
    env_short = env_id.split("/")[1]
    model_path = models_path / f"{env_short}_ppo_final"
    model.save(str(model_path))
    print(f"Model saved to {model_path}")

    # Save training metadata
    metadata = {
        "env_id": env_id,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "n_envs": n_envs,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
    }

    meta_path = results_path / f"{env_short}_training_meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    vec_env.close()
    eval_env.close()

    return model


def main():
    """CLI entry point for hemosim-train."""
    parser = argparse.ArgumentParser(
        description="Train PPO agents on hemosim environments"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="hemosim/WarfarinDosing-v0",
        choices=list(ENV_CONFIGS.keys()),
        help="Environment to train on",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps (overrides default)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory for results and models",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train on all environments",
    )

    args = parser.parse_args()

    if args.all:
        for env_id in ENV_CONFIGS:
            print(f"\n{'='*60}")
            train(env_id, seed=args.seed, timesteps=args.timesteps,
                  results_dir=args.results_dir)
    else:
        train(args.env, seed=args.seed, timesteps=args.timesteps,
              results_dir=args.results_dir)


if __name__ == "__main__":
    main()
