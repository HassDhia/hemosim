"""Reproducibility seed pools for hemosim (ISC-13).

Train/held-out seed separation prevents accidental train-test contamination
in the ``PatientGenerator``. Every downstream consumer that generates patients
derives its RNG state from a single integer seed; if training and evaluation
draw from the *same* seed pool, the same virtual patients can appear in both —
silently inflating reported performance.

Convention
----------

- **Training** uses seeds in ``TRAIN_SEED_POOL`` (``range(0, 10000)``). These
  are the seeds passed to ``env.reset(seed=...)`` and ``PatientGenerator(seed=...)``
  during learning, tuning, and development.
- **Evaluation** uses seeds in ``HELDOUT_SEED_POOL`` (``range(100000, 101000)``).
  These are the seeds used for the published numbers in the paper's Results
  table. They must never be used during training.
- **Published-result seeds** are the ``HELDOUT_PUBLISHED_SEEDS`` slice —
  the exact seeds used to produce ``results/EXPECTED_RESULTS.json``.

The two pools are disjoint by construction (a 90k-wide buffer between them),
so any accidental overlap is a bug in this file, not a silent contamination.

See ``HELDOUT_SEEDS.md`` at the repo root for policy and rationale.
"""

from __future__ import annotations

from typing import Iterable

# --- Seed pools --------------------------------------------------------------

# Training pool: used during PPO training, baseline tuning, dev work.
TRAIN_SEED_POOL: range = range(0, 10000)

# Held-out evaluation pool: used for published numbers. Never train on these.
HELDOUT_SEED_POOL: range = range(100000, 101000)

# Exact seeds used to generate ``results/EXPECTED_RESULTS.json``.
# The reproducibility harness uses the first 100 held-out seeds.
HELDOUT_PUBLISHED_SEEDS: tuple[int, ...] = tuple(range(100000, 100100))


# --- Assertions --------------------------------------------------------------


def assert_held_out(seed: int) -> None:
    """Raise ``ValueError`` if ``seed`` is not in ``HELDOUT_SEED_POOL``.

    Use this as a guard at the top of any evaluation script that is expected
    to produce published numbers.
    """
    if not isinstance(seed, (int,)) or isinstance(seed, bool):
        raise TypeError(
            f"seed must be an int, got {type(seed).__name__}: {seed!r}"
        )
    if seed not in HELDOUT_SEED_POOL:
        raise ValueError(
            f"seed={seed} is not in HELDOUT_SEED_POOL "
            f"({HELDOUT_SEED_POOL.start}..{HELDOUT_SEED_POOL.stop - 1}). "
            "Held-out evaluation must use seeds from the held-out pool only. "
            "See HELDOUT_SEEDS.md."
        )


def assert_train(seed: int) -> None:
    """Raise ``ValueError`` if ``seed`` is not in ``TRAIN_SEED_POOL``.

    Use this as a guard at the top of any training script to prevent
    accidental use of held-out seeds during training.
    """
    if not isinstance(seed, (int,)) or isinstance(seed, bool):
        raise TypeError(
            f"seed must be an int, got {type(seed).__name__}: {seed!r}"
        )
    if seed not in TRAIN_SEED_POOL:
        raise ValueError(
            f"seed={seed} is not in TRAIN_SEED_POOL "
            f"({TRAIN_SEED_POOL.start}..{TRAIN_SEED_POOL.stop - 1}). "
            "Training must use seeds from the training pool only. "
            "Using held-out seeds during training contaminates evaluation. "
            "See HELDOUT_SEEDS.md."
        )


# --- Convenience helpers -----------------------------------------------------


def assert_seeds_held_out(seeds: Iterable[int]) -> None:
    """Apply ``assert_held_out`` to every seed in ``seeds``."""
    for s in seeds:
        assert_held_out(s)


def assert_seeds_train(seeds: Iterable[int]) -> None:
    """Apply ``assert_train`` to every seed in ``seeds``."""
    for s in seeds:
        assert_train(s)


def heldout_seeds(n: int, offset: int = 0) -> list[int]:
    """Return the first ``n`` held-out seeds starting at ``offset``.

    Parameters
    ----------
    n
        Number of seeds to return.
    offset
        Starting index within ``HELDOUT_SEED_POOL`` (0-based).

    Raises
    ------
    ValueError
        If the requested slice would exceed ``HELDOUT_SEED_POOL``.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if offset < 0:
        raise ValueError(f"offset must be non-negative, got {offset}")
    end = offset + n
    if end > len(HELDOUT_SEED_POOL):
        raise ValueError(
            f"Requested {n} seeds at offset {offset} exceeds "
            f"HELDOUT_SEED_POOL size ({len(HELDOUT_SEED_POOL)})."
        )
    start = HELDOUT_SEED_POOL.start + offset
    return list(range(start, start + n))


def train_seeds(n: int, offset: int = 0) -> list[int]:
    """Return the first ``n`` training seeds starting at ``offset``."""
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if offset < 0:
        raise ValueError(f"offset must be non-negative, got {offset}")
    end = offset + n
    if end > len(TRAIN_SEED_POOL):
        raise ValueError(
            f"Requested {n} seeds at offset {offset} exceeds "
            f"TRAIN_SEED_POOL size ({len(TRAIN_SEED_POOL)})."
        )
    start = TRAIN_SEED_POOL.start + offset
    return list(range(start, start + n))


__all__ = [
    "TRAIN_SEED_POOL",
    "HELDOUT_SEED_POOL",
    "HELDOUT_PUBLISHED_SEEDS",
    "assert_held_out",
    "assert_train",
    "assert_seeds_held_out",
    "assert_seeds_train",
    "heldout_seeds",
    "train_seeds",
]
