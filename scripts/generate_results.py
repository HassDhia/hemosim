"""Generate benchmark results for hemosim.

v2 — honest evaluation. Every number this script writes comes from an actual
episode rollout. PPO numbers are produced ONLY when a trained model artifact
exists under ``results/models/``; otherwise PPO entries are ``null`` so the
paper cannot accidentally quote synthetic values.

History note: a prior v0.1 version of this file fabricated PPO results via
the formula ``ppo_mean = clinical_mean * 1.25 + abs(random_mean) * 0.3``.
That fabricated generator was removed in v0.2.0. v2 policy: no formula-
synthesized results anywhere in this repo.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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
from hemosim.reproducibility import (
    HELDOUT_SEED_POOL,
    TRAIN_SEED_POOL,
    assert_held_out,
    assert_train,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
MODELS_DIR = RESULTS_DIR / "models"

BASELINE_MAP = {
    "hemosim/WarfarinDosing-v0": WarfarinClinicalBaseline,
    "hemosim/HeparinInfusion-v0": HeparinRaschkeBaseline,
    "hemosim/DOACManagement-v0": DOACGuidelineBaseline,
    "hemosim/DICManagement-v0": DICProtocolBaseline,
}


def evaluate_agent(
    env,
    agent,
    n_episodes: int = 100,
    base_seed: int = 42,
    seeds: list[int] | None = None,
) -> dict:
    """Evaluate an agent over n episodes.

    Parameters
    ----------
    env, agent, n_episodes
        Standard rollout inputs.
    base_seed
        Legacy: seed for episode ``ep`` is ``base_seed + ep``. Used when
        ``seeds`` is not supplied. Kept for backward compatibility.
    seeds
        Explicit list of per-episode seeds. If given, ``n_episodes`` is
        overridden by ``len(seeds)`` and ``base_seed`` is ignored. The
        reproducibility harness uses this to pin held-out seeds.

    Metrics:
      - mean/std/min/max episode reward
      - therapeutic_rate: fraction of episodes where > 50% of steps were in
        therapeutic range (per-episode rate, averaged across episodes — not
        a global step ratio, which conflates short and long episodes)
      - mean_therapeutic_fraction: fraction of all steps where
        info["therapeutic"] was True (legacy metric, kept for comparison)
    """
    if seeds is not None:
        episode_seeds = list(seeds)
        n_episodes = len(episode_seeds)
    else:
        episode_seeds = [base_seed + ep for ep in range(n_episodes)]

    rewards: list[float] = []
    per_episode_therapeutic_fractions: list[float] = []
    total_therapeutic_steps = 0
    total_steps = 0

    for ep_seed in episode_seeds:
        obs, info = env.reset(seed=ep_seed)
        ep_reward = 0.0
        ep_therapeutic_steps = 0
        ep_steps = 0
        done = False
        while not done:
            action = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps += 1
            total_steps += 1
            if bool(info.get("therapeutic", False)):
                ep_therapeutic_steps += 1
                total_therapeutic_steps += 1
            done = terminated or truncated
        rewards.append(ep_reward)
        per_episode_therapeutic_fractions.append(
            ep_therapeutic_steps / max(ep_steps, 1)
        )

    rewards_arr = np.asarray(rewards, dtype=np.float64)
    per_ep = np.asarray(per_episode_therapeutic_fractions, dtype=np.float64)

    return {
        "mean_reward": float(rewards_arr.mean()),
        "std_reward": float(rewards_arr.std()),
        "min_reward": float(rewards_arr.min()),
        "max_reward": float(rewards_arr.max()),
        "episodes": n_episodes,
        # Per-episode mean time-in-therapeutic-range (preferred metric)
        "therapeutic_rate": float(per_ep.mean()),
        "therapeutic_rate_std": float(per_ep.std()),
        # Fraction of episodes where majority of steps were in-range
        "episodes_majority_therapeutic": float((per_ep > 0.5).mean()),
        # Global step ratio (legacy — kept for backward compat)
        "global_therapeutic_step_ratio": float(
            total_therapeutic_steps / max(total_steps, 1)
        ),
    }


def _load_ppo_model(env_id: str):
    """Load a trained PPO model from ``results/models/`` if available.

    Returns a stable-baselines3 model or None. Uses stable-baselines3's
    .predict() under the hood — caller wraps it with an agent-like adapter.
    """
    if not MODELS_DIR.exists():
        return None

    env_short = env_id.split("/")[1]
    candidates = [
        MODELS_DIR / f"{env_short}_ppo_final.zip",
        MODELS_DIR / "best_model.zip",
    ]
    for path in candidates:
        if path.exists():
            try:
                from stable_baselines3 import PPO  # type: ignore
            except ImportError:
                print(
                    f"  [info] Found PPO model at {path} but stable-baselines3 "
                    "is not installed. Skipping PPO eval."
                )
                return None
            return PPO.load(str(path))
    return None


class _PPOAdapter:
    """Wrap stable-baselines3 model to match baseline ``.predict(obs)`` API."""

    def __init__(self, sb3_model, deterministic: bool = False) -> None:
        self._model = sb3_model
        self._deterministic = deterministic

    def predict(self, obs):
        action, _state = self._model.predict(obs, deterministic=self._deterministic)
        return action


def _resolve_eval_seeds(eval_set: str, n_episodes: int, seed: int) -> list[int]:
    """Return the per-episode seed list for the requested eval set.

    Parameters
    ----------
    eval_set
        ``"heldout"`` -> first ``n_episodes`` seeds of ``HELDOUT_SEED_POOL``.
        ``"train"`` -> seeds ``seed, seed+1, ..., seed+n_episodes-1`` from
        ``TRAIN_SEED_POOL``. Intended for sanity-checking during development
        only; never cite numbers produced this way.
    """
    if eval_set == "heldout":
        start = HELDOUT_SEED_POOL.start
        seeds = list(range(start, start + n_episodes))
        assert_held_out(seeds[0])
        assert_held_out(seeds[-1])
        return seeds
    if eval_set == "train":
        seeds = list(range(seed, seed + n_episodes))
        assert_train(seeds[0])
        assert_train(seeds[-1])
        return seeds
    raise ValueError(
        f"Unknown eval_set={eval_set!r}; expected 'heldout' or 'train'."
    )


def generate_results(
    n_episodes: int = 100,
    seed: int = 42,
    eval_set: str = "heldout",
) -> dict:
    """Run real baseline and (if available) PPO evaluations.

    Never fabricates numbers. PPO entry is ``null`` when no trained model
    is present on disk.

    Parameters
    ----------
    n_episodes
        Number of evaluation episodes.
    seed
        Seed for baseline agents' internal RNG. When ``eval_set='train'``,
        also the starting seed for per-episode rollouts.
    eval_set
        Which seed pool to draw evaluation seeds from. Defaults to
        ``'heldout'`` so the published Results table numbers come from the
        held-out pool. Set to ``'train'`` only for dev sanity checks — those
        numbers must never be cited in the paper.
    """
    episode_seeds = _resolve_eval_seeds(eval_set, n_episodes, seed)
    print(
        f"Generating benchmark results (v2 — no fabrication, "
        f"eval_set={eval_set}, seeds=[{episode_seeds[0]}..{episode_seeds[-1]}])..."
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results: dict = {"environments": {}}

    for env_id in ENV_CONFIGS:
        print(f"\n  Evaluating {env_id}...")
        env = gym.make(env_id)

        env_results: dict = {}

        # Clinical baseline
        baseline_cls = BASELINE_MAP[env_id]
        baseline = baseline_cls(seed=seed)
        print(f"    Clinical baseline ({baseline_cls.__name__})...")
        env_results["clinical_baseline"] = evaluate_agent(
            env, baseline, seeds=episode_seeds
        )

        # Random baseline
        random_agent = RandomBaseline(env.action_space, seed=seed)
        print("    Random baseline...")
        env_results["random"] = evaluate_agent(
            env, random_agent, seeds=episode_seeds
        )

        # PPO — ONLY if a real trained model exists on disk
        ppo_sb3 = _load_ppo_model(env_id)
        if ppo_sb3 is not None:
            print("    PPO (trained model found on disk — running real eval)...")
            ppo_agent = _PPOAdapter(ppo_sb3, deterministic=False)
            env_results["ppo"] = evaluate_agent(
                env, ppo_agent, seeds=episode_seeds
            )
            env_results["ppo"]["source"] = "stable_baselines3_real_eval"
        else:
            print("    PPO: no trained model on disk — leaving entry as null.")
            env_results["ppo"] = None

        env.close()
        results["environments"][env_id] = env_results

        cb = env_results["clinical_baseline"]
        rnd = env_results["random"]
        ppo_txt = (
            f"{env_results['ppo']['mean_reward']:.2f}"
            if env_results["ppo"]
            else "null (not trained)"
        )
        print(
            f"    Random: {rnd['mean_reward']:.2f}  "
            f"Clinical: {cb['mean_reward']:.2f}  PPO: {ppo_txt}"
        )

    results["training_config"] = {
        "seed": seed,
        "total_timesteps_per_env": {
            k: v["total_timesteps"] for k, v in ENV_CONFIGS.items()
        },
    }
    results["metadata"] = {
        "package_version": hemosim.__version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_eval_episodes": n_episodes,
        "eval_set": eval_set,
        "eval_seed_range": [int(episode_seeds[0]), int(episode_seeds[-1])],
        "generator": "generate_results.py v2 (honest)",
    }

    output_file = RESULTS_DIR / "training_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate honest hemosim benchmark results"
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval-set",
        choices=["heldout", "train"],
        default="heldout",
        help=(
            "Which seed pool to evaluate on. 'heldout' (default) uses the "
            "held-out pool for published numbers. 'train' uses the training "
            "pool for dev sanity checks only — never cite those numbers."
        ),
    )
    args = parser.parse_args()

    generate_results(
        n_episodes=args.episodes,
        seed=args.seed,
        eval_set=args.eval_set,
    )


if __name__ == "__main__":
    main()
