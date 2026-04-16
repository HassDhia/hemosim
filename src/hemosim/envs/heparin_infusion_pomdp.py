"""POMDP-formulated heparin infusion environment (ISC-5).

Wraps HeparinInfusionEnv with explicit lab ordering and measurement
delay/noise. Directly addresses the critique that v0.1 envs expose
perfect oracle lab values every timestep.

Semantics
---------
- Internal state (aPTT, platelets, heparin concentration) is hidden.
- Observation is a summary of the most recent RETURNED lab samples plus
  time since last order, plus patient covariates.
- Action space extended: a 3-bit lab-order mask for {aPTT, anti-Xa,
  platelets}. Lab orders carry small reward penalties.
- Each step: returned labs are noisy copies of the ground truth AT THE
  TIME THE LAB WAS ORDERED (realistic — the lab draw happened earlier,
  the patient state has moved on by the time the result comes back).
- Measurement noise and TAT values cited to clinical literature in
  ``hemosim.envs.pomdp.HEPARIN_LAB_SPECS``.

Reference: the pharmacist critique from the Nemati v0.1 review — "does
not incorporate feedback from real clinical variables or measured
concentrations." This env is the structural answer.
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hemosim.envs.heparin_infusion import (
    EPISODE_HOURS,
    HeparinInfusionEnv,
    MAX_INFUSION_RATE,
    STEP_HOURS,
)
from hemosim.envs.pomdp import HEPARIN_LAB_SPECS, LabOrderQueue


# Observation layout (10 dims):
#   0: latest returned aPTT (normalized, NaN filled with -1.0 sentinel)
#   1: hours since last aPTT sample returned (normalized to 0-1 over 12h)
#   2: latest returned anti-Xa (normalized, sentinel -1.0)
#   3: hours since last anti-Xa returned (normalized over 12h)
#   4: latest returned platelets (normalized, sentinel -1.0)
#   5: hours since last platelets returned (normalized over 12h)
#   6: pending aPTT orders flag (0 or 1)
#   7: patient weight (normalized)
#   8: patient renal function (0-1)
#   9: hours elapsed in episode (normalized)
OBS_DIM = 10

# Action layout (5 dims — 2 dose + 3 lab-order bits)
#   0: infusion rate scaled [0, 1] -> [0, MAX_INFUSION_RATE]
#   1: bolus flag (> 0.5 triggers 80 U/kg bolus)
#   2: order aPTT? (> 0.5)
#   3: order anti-Xa? (> 0.5)
#   4: order platelets? (> 0.5)
ACT_DIM = 5

LAB_AGE_NORM_HOURS = 12.0  # for "hours since last lab" normalization


class HeparinInfusionPOMDPEnv(gym.Env):
    """POMDP wrapper over HeparinInfusionEnv with lab-order action semantics."""

    metadata = {"render_modes": []}

    def __init__(self, difficulty: str = "medium", **kwargs) -> None:
        super().__init__()
        self._inner = HeparinInfusionEnv(difficulty=difficulty, **kwargs)

        self.observation_space = spaces.Box(
            low=np.array(
                [-1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            high=np.array(
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.zeros(ACT_DIM, dtype=np.float32),
            high=np.ones(ACT_DIM, dtype=np.float32),
            dtype=np.float32,
        )

        self._lab_queue: LabOrderQueue | None = None
        self._hours_elapsed = 0.0
        self._patient: dict | None = None

    # ---- Gymnasium API -------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        inner_obs, inner_info = self._inner.reset(seed=seed, options=options)
        self._patient = inner_info["patient"]
        self._hours_elapsed = 0.0

        # Re-derive an rng from inner's np_random so lab-order noise is
        # reproducible under the same env seed.
        rng = self._inner._np_random or np.random.default_rng(seed)
        self._lab_queue = LabOrderQueue(
            specs=HEPARIN_LAB_SPECS,
            rng=rng,
            history_maxlen=32,
        )

        # Prime the queue with a t=0 aPTT draw, as per clinical norms.
        self._lab_queue.order(
            "aptt", current_hours=0.0, ground_truth_value=inner_info["aptt"]
        )
        self._lab_queue.order(
            "platelets",
            current_hours=0.0,
            ground_truth_value=inner_info["platelet_count"],
        )

        obs = self._build_observation()
        info = self._build_info(inner_info, lab_cost=0.0, orders_placed=[])
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)

        # 1) Dispatch dose action to inner env.
        inner_action = np.array(
            [float(np.clip(action[0], 0.0, 1.0)), float(action[1])],
            dtype=np.float32,
        )
        inner_obs, inner_reward, terminated, truncated, inner_info = \
            self._inner.step(inner_action)

        self._hours_elapsed += STEP_HOURS

        # 2) Place any lab orders requested this step.
        lab_cost = 0.0
        orders_placed: list[str] = []
        assert self._lab_queue is not None
        if float(action[2]) > 0.5:
            lab_cost += self._lab_queue.order(
                "aptt",
                current_hours=self._hours_elapsed,
                ground_truth_value=inner_info["aptt"],
            )
            orders_placed.append("aptt")
        if float(action[3]) > 0.5:
            lab_cost += self._lab_queue.order(
                "anti_xa",
                current_hours=self._hours_elapsed,
                ground_truth_value=self._inner._model.get_concentration(),
            )
            orders_placed.append("anti_xa")
        if float(action[4]) > 0.5:
            lab_cost += self._lab_queue.order(
                "platelets",
                current_hours=self._hours_elapsed,
                ground_truth_value=inner_info["platelet_count"],
            )
            orders_placed.append("platelets")

        # 3) Tick the queue — some orders may now be returning.
        self._lab_queue.tick(self._hours_elapsed)

        # 4) Combine rewards.
        reward = float(inner_reward) + float(lab_cost)

        obs = self._build_observation()
        info = self._build_info(inner_info, lab_cost, orders_placed)
        return obs, reward, terminated, truncated, info

    def close(self) -> None:  # pragma: no cover
        self._inner.close()

    # ---- Observation and info construction -----------------------------

    def _build_observation(self) -> np.ndarray:
        assert self._lab_queue is not None
        assert self._patient is not None

        def lab_slot(name: str, norm_range: tuple[float, float]) -> tuple[float, float]:
            latest = self._lab_queue.latest(name)
            if latest is None:
                return (-1.0, 1.0)  # sentinel value, age saturated
            lo, hi = norm_range
            val_norm = float(np.clip((latest.value - lo) / (hi - lo), 0.0, 1.0))
            age = max(self._hours_elapsed - latest.returned_at_hours, 0.0)
            age_norm = float(min(age / LAB_AGE_NORM_HOURS, 1.0))
            return val_norm, age_norm

        aptt_val, aptt_age = lab_slot("aptt", (20.0, 200.0))
        antixa_val, antixa_age = lab_slot("anti_xa", (0.0, 1.0))
        plt_val, plt_age = lab_slot("platelets", (0.0, 400.0))

        pending_aptt = 1.0 if self._lab_queue.num_pending("aptt") > 0 else 0.0

        obs = np.array(
            [
                aptt_val,
                aptt_age,
                antixa_val,
                antixa_age,
                plt_val,
                plt_age,
                pending_aptt,
                float(np.clip((self._patient["weight"] - 40) / 140, 0.0, 1.0)),
                float(np.clip(self._patient["renal_function"], 0.0, 1.0)),
                float(np.clip(self._hours_elapsed / EPISODE_HOURS, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )
        return obs

    def _build_info(
        self,
        inner_info: dict,
        lab_cost: float,
        orders_placed: list[str],
    ) -> dict:
        assert self._lab_queue is not None
        return {
            "pomdp_mode": True,
            "ground_truth_aptt": inner_info["aptt"],
            "ground_truth_platelet_count": inner_info["platelet_count"],
            "therapeutic": inner_info["therapeutic"],
            "hours_elapsed": self._hours_elapsed,
            "patient": self._patient,
            "labs_placed_this_step": list(orders_placed),
            "lab_cost_this_step": lab_cost,
            "num_pending_labs": self._lab_queue.num_pending(),
            "latest_aptt_sample": self._lab_queue.latest("aptt"),
            "latest_anti_xa_sample": self._lab_queue.latest("anti_xa"),
            "latest_platelets_sample": self._lab_queue.latest("platelets"),
        }
