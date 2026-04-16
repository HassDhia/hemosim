"""Diagnose why WarfarinDosing-v0 produces identical rewards across policies.

Observed v0.1 results: clinical baseline == random == PPO == -17.0 exactly,
zero variance, 0% therapeutic rate. That means the reward function sees the
same INR trajectory regardless of dose. This script dumps the trajectory
under three fixed-dose policies to find out where the response collapses.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import gymnasium as gym
import numpy as np

import hemosim  # noqa: F401
from hemosim.agents.baselines import WarfarinClinicalBaseline


def rollout(policy_fn, seed=42, label="?"):
    env = gym.make("hemosim/WarfarinDosing-v0", difficulty="easy")
    obs, info = env.reset(seed=seed)
    inrs, doses, rewards = [info["inr"]], [0.0], []
    terminated = truncated = False
    t = 0
    while not (terminated or truncated):
        action = policy_fn(obs, t)
        obs, r, terminated, truncated, info = env.step(action)
        inrs.append(info["inr"])
        doses.append(info["dose_mg"])
        rewards.append(r)
        t += 1
    env.close()
    print(f"\n=== {label} (seed={seed}) ===")
    print(f"  steps: {t}  terminated={terminated}  truncated={truncated}")
    print(f"  total reward: {sum(rewards):.3f}")
    print(f"  INR min/mean/max: {min(inrs):.2f} / {np.mean(inrs):.2f} / {max(inrs):.2f}")
    print(f"  Dose mg min/mean/max: "
          f"{min(doses):.2f} / {np.mean(doses):.2f} / {max(doses):.2f}")
    print(f"  INR trajectory (every 10 days):")
    for i in range(0, len(inrs), 10):
        print(f"    day {i:2d}: INR={inrs[i]:.3f}  dose={doses[i]:.2f}mg")
    return inrs, doses, rewards


def fixed_dose_policy(mg):
    def p(obs, t):
        return np.array([mg / 15.0], dtype=np.float32)
    return p


def zero_dose_policy(obs, t):
    return np.array([0.0], dtype=np.float32)


def main():
    baseline = WarfarinClinicalBaseline(seed=42)
    def clinical_policy(obs, t):
        return baseline.predict(obs)

    rollout(zero_dose_policy, seed=42, label="ZERO DOSE (no drug)")
    rollout(fixed_dose_policy(5.0), seed=42, label="FIXED 5mg daily")
    rollout(fixed_dose_policy(10.0), seed=42, label="FIXED 10mg daily")
    rollout(fixed_dose_policy(15.0), seed=42, label="FIXED 15mg (max)")
    rollout(clinical_policy, seed=42, label="IWPC CLINICAL BASELINE")


if __name__ == "__main__":
    main()
