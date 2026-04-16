"""Train a Nemati 2016-style DQN on hemosim's HeparinInfusion-v0.

This script reproduces the structural setup of Nemati et al. 2016 inside the
hemosim simulator: a discrete five-action heparin infusion adjustment grid,
an aPTT-centered reward, and a DQN with a 64-64 ReLU MLP. It trains for
100,000 steps (configurable) using stable-baselines3 and saves the Q-network
weights to ``results/models/nemati_dqn.pt`` in a format loadable by
:class:`hemosim.agents.baselines_extended.NematiDQN2016Baseline`.

Requires:
    pip install hemosim[train]  # pulls in torch + stable-baselines3

Reference:
    Nemati S, Ghassemi MM, Clifford GD. Optimal medication dosing from
    suboptimal clinical examples: A deep reinforcement learning approach.
    IEEE EMBC 2016;2978-2981. doi:10.1109/EMBC.2016.7591355.

Note:
    This script is an *independent reimplementation for benchmarking*, not a
    code-transfer of the 2016 paper. The environment on which training runs
    is hemosim — not the MIMIC-II derived cohort used in Nemati 2016 — so
    the resulting policy is a hemosim-internal benchmark rather than a
    replication of the published model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def _require_training_deps():
    try:
        from stable_baselines3 import DQN  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        print("error: torch and stable-baselines3 are required for training.")
        print("install with: pip install 'hemosim[train]'")
        raise SystemExit(1) from exc


# Action grid copied from baselines_extended.NematiDQN2016Baseline for
# consistency: {-4, -2, 0, +2, +4} U/kg/hr adjustments per decision step.
_ACTION_DELTAS_UKGHR = np.array([-4.0, -2.0, 0.0, 2.0, 4.0], dtype=np.float32)
_MAX_INFUSION_U_HR = 2500.0
_HEPARIN_WEIGHT_MIN = 40.0
_HEPARIN_WEIGHT_RANGE = 140.0
_INITIAL_RATE_UKGHR = 18.0


class NematiDiscreteHeparinWrapper(gym.Wrapper):
    """Wrap HeparinInfusion-v0 with the discrete five-action Nemati grid.

    The wrapper tracks the running infusion rate (U/kg/hr), applies the
    selected delta, and emits the env's native Box action. A bolus is given
    only at ``t=0``.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.action_space = spaces.Discrete(len(_ACTION_DELTAS_UKGHR))
        self._rate_u_kg_hr = _INITIAL_RATE_UKGHR

    def reset(self, **kwargs):
        self._rate_u_kg_hr = _INITIAL_RATE_UKGHR
        return self.env.reset(**kwargs)

    def step(self, action: int):  # type: ignore[override]
        delta = float(_ACTION_DELTAS_UKGHR[int(action)])
        self._rate_u_kg_hr = float(np.clip(self._rate_u_kg_hr + delta, 0.0, 40.0))

        raw_obs = self.env.unwrapped._get_obs()  # type: ignore[attr-defined]
        weight = float(raw_obs[2]) * _HEPARIN_WEIGHT_RANGE + _HEPARIN_WEIGHT_MIN
        infusion_rate_u_hr = self._rate_u_kg_hr * max(weight, 1e-3)
        hours = float(raw_obs[5]) * 120.0

        continuous_action = np.array(
            [
                np.clip(infusion_rate_u_hr / _MAX_INFUSION_U_HR, 0.0, 1.0),
                1.0 if hours < 1.0 else 0.0,
            ],
            dtype=np.float32,
        )
        return self.env.step(continuous_action)


def train(
    total_timesteps: int = 100_000,
    seed: int = 42,
    output_path: Path = Path("results/models/nemati_dqn.pt"),
) -> Path:
    """Train the DQN policy and save the Q-network weights."""

    _require_training_deps()

    from stable_baselines3 import DQN
    import torch

    import hemosim  # noqa: F401  # triggers env registration

    def make_env() -> gym.Env:
        return NematiDiscreteHeparinWrapper(gym.make("hemosim/HeparinInfusion-v0"))

    env = make_env()

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=0.2,
        exploration_final_eps=0.02,
        policy_kwargs={"net_arch": [64, 64]},
        seed=seed,
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps)

    # Extract the underlying Q-network and save it in a format compatible
    # with baselines_extended._NematiQNetwork.
    q_net_state = model.q_net.q_net.state_dict()
    # Re-key to match _NematiQNetwork's 'net.*' prefix.
    remapped = {}
    for key, value in q_net_state.items():
        remapped[f"net.{key}"] = value

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(remapped, output_path)
    print(f"saved Nemati DQN weights to {output_path}")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/models/nemati_dqn.pt"),
    )
    args = parser.parse_args(argv)
    train(total_timesteps=args.timesteps, seed=args.seed, output_path=args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
