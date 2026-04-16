"""Tests for DIC env CoagulationCascade integration (ISC-3).

Verifies that:
- Default mode (coag_cascade_mode=False) is unchanged from v0.1 behavior.
- Cascade mode (coag_cascade_mode=True) runs end-to-end without errors.
- Cascade state is exposed in info when enabled.
- Cascade-derived fibrinogen differs from flat-only fibrinogen over an
  episode (sanity check that the ODE is actually integrated).
- Severity tiers produce differentiated cascade trajectories.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

import hemosim  # noqa: F401 — triggers env registration


def _rollout(env, n_steps: int = 10, seed: int = 100000):
    obs, info = env.reset(seed=seed)
    infos = [info]
    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        infos.append(info)
        if term or trunc:
            break
    return infos


def test_default_mode_no_cascade_in_info():
    env = gym.make("hemosim/DICManagement-v0")
    infos = _rollout(env, n_steps=3)
    for info in infos:
        assert info.get("coag_cascade_mode") is False
        assert "cascade_state" not in info


def test_cascade_mode_runs_end_to_end():
    env = gym.make("hemosim/DICManagement-v0", coag_cascade_mode=True)
    infos = _rollout(env, n_steps=5)
    assert len(infos) >= 2  # reset + at least one step
    for info in infos:
        assert info["coag_cascade_mode"] is True
        assert "cascade_state" in info
        cs = info["cascade_state"]
        for key in ("tf_viia", "xa", "va", "thrombin", "fibrinogen_cascade",
                    "fibrin", "at3_bound", "platelet_act_frac"):
            assert key in cs
            assert np.isfinite(cs[key])


def test_cascade_state_evolves():
    """Cascade state must move between steps, not sit at init."""
    env = gym.make("hemosim/DICManagement-v0", coag_cascade_mode=True)
    infos = _rollout(env, n_steps=5, seed=100001)
    cascade_states = [info["cascade_state"] for info in infos]
    # At least some cascade variables must change across steps
    thrombin_trajectory = [cs["thrombin"] for cs in cascade_states]
    assert len(set(thrombin_trajectory)) > 1, (
        "thrombin is constant across 5 steps — cascade isn't integrating"
    )


def test_cascade_blends_into_fibrinogen():
    """With cascade mode on, fibrinogen trajectory should differ from
    flat-only mode given identical seeds/actions (deterministic PatientGen
    + same action stream)."""
    # Fixed action sequence
    actions = [np.array([1, 1, 1, 0], dtype=np.int64) for _ in range(5)]

    env_flat = gym.make("hemosim/DICManagement-v0", coag_cascade_mode=False)
    env_cas = gym.make("hemosim/DICManagement-v0", coag_cascade_mode=True)

    obs_f, info_f = env_flat.reset(seed=100002)
    obs_c, info_c = env_cas.reset(seed=100002)
    fibs_f = [info_f["fibrinogen"]]
    fibs_c = [info_c["fibrinogen"]]
    for a in actions:
        obs_f, _, tf, trf, info_f = env_flat.step(a)
        obs_c, _, tc, trc, info_c = env_cas.step(a)
        fibs_f.append(info_f["fibrinogen"])
        fibs_c.append(info_c["fibrinogen"])
        if (tf or trf) or (tc or trc):
            break

    # The blended fibrinogen must not be bit-identical to flat-only across
    # the whole trajectory — the cascade is contributing something.
    diffs = [abs(f - c) for f, c in zip(fibs_f, fibs_c)]
    assert max(diffs) > 0.01, (
        f"cascade-blended fibrinogen == flat-only trajectory "
        f"(max |diff| = {max(diffs):.4f}); ODE not wired"
    )


def test_severity_differentiates_initial_cascade():
    """Different severity patients should seed different cascade states."""
    mild_states = []
    severe_states = []
    for seed_offset in range(10):
        for difficulty, target in [("easy", mild_states), ("hard", severe_states)]:
            env = gym.make(
                "hemosim/DICManagement-v0",
                difficulty=difficulty,
                coag_cascade_mode=True,
            )
            _, info = env.reset(seed=100100 + seed_offset)
            target.append(info["cascade_state"]["thrombin"])

    mean_mild = float(np.mean(mild_states))
    mean_severe = float(np.mean(severe_states))
    assert mean_severe > mean_mild, (
        f"hard difficulty should produce higher initial thrombin, "
        f"got mild={mean_mild:.2f}, severe={mean_severe:.2f}"
    )


def test_cascade_mode_does_not_break_reward_shape():
    """Reward signature unchanged by cascade mode (float, bounded)."""
    env = gym.make("hemosim/DICManagement-v0", coag_cascade_mode=True)
    env.reset(seed=100003)
    for _ in range(5):
        _, reward, term, trunc, _ = env.step(env.action_space.sample())
        assert isinstance(reward, float)
        assert np.isfinite(reward)
        if term or trunc:
            break
