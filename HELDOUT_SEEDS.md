# Held-out seed policy (ISC-13)

This document defines the seed-pool discipline that prevents accidental
train-test contamination in hemosim.

## TL;DR

| Pool | Range | Size | Purpose |
|---|---|---:|---|
| **Training** | `range(0, 10000)` | 10,000 | Training, tuning, dev work |
| **Held-out evaluation** | `range(100000, 101000)` | 1,000 | Published Results table |
| **Published-result seeds** | `range(100000, 100100)` | 100 | Exact seeds in `results/EXPECTED_RESULTS.json` |

The pools are **disjoint** — a 90,000-wide buffer sits between seed `9999`
(last training seed) and seed `100000` (first held-out seed). The gap is
deliberate so that "off-by-a-few" mistakes (e.g. `base_seed + n_episodes`
running a few seeds past the intended range) cannot accidentally cross the
boundary.

## Why this matters for hemosim

Every hemosim environment derives its virtual patient from a seed:

```python
# src/hemosim/envs/warfarin_dosing.py (and the three others)
super().reset(seed=seed)
patient_seed = int(self.np_random.integers(0, 2**31))
gen = PatientGenerator(seed=patient_seed)
patient = gen.generate_warfarin_patient()
```

Two calls to `env.reset(seed=42)` produce **the same** virtual patient
(same age, weight, CYP2C9 genotype, VKORC1 genotype, etc.). That is exactly
what we want for reproducibility — but it is *also* how train-test
contamination happens if training and evaluation happen to pick the same seed.

Concretely, if a PPO run trains on `env.reset(seed=0..9999)` and we later
evaluate on `env.reset(seed=42)`, the agent has already seen that specific
patient dozens of times during training. The reported reward is not
generalization — it is memorization.

The two disjoint pools solve this at the seed layer:

- **Training uses `TRAIN_SEED_POOL`.** Enforced by `assert_train(seed)`.
- **Evaluation uses `HELDOUT_SEED_POOL`.** Enforced by `assert_held_out(seed)`.
- **The pools never overlap.** Enforced by `test_reproducibility.py` in CI.

## Policy

1. **Training code** (anything under `hemosim-train`, any script that
   optimizes policy weights) MUST draw seeds from `TRAIN_SEED_POOL` only.
   It MUST NEVER use a seed `>= 100000`.

2. **Evaluation code** (anything that produces a number for the paper or
   for `results/EXPECTED_RESULTS.json`) MUST draw seeds from
   `HELDOUT_SEED_POOL` only. Use `assert_held_out(seed)` as a guard.

3. **Published numbers** are generated from the first 100 held-out seeds
   (`range(100000, 100100)`) via `scripts/generate_results.py --eval-set heldout`
   (the default). Any rerun that aims to match `results/EXPECTED_RESULTS.json`
   must use this exact set.

4. **`scripts/reproduce.sh`** is the canonical one-command repro path:
   fresh venv → install → tests → honest baseline eval → compare to
   `results/EXPECTED_RESULTS.json` with tolerance.

## Non-contamination argument (informal proof)

- The virtual patient space generated at seed `s` is a pure function of `s`
  (numpy `default_rng(s)` is deterministic).
- Gymnasium's `env.reset(seed=s)` derives `patient_seed` from `np_random`,
  which is itself seeded from `s`. The derivation is a pure function of `s`.
- Therefore, the set of patients reachable from `TRAIN_SEED_POOL` is the
  image of the injective map `s -> patient(s)` over `[0, 10000)`, and the
  set reachable from `HELDOUT_SEED_POOL` is the image over `[100000, 101000)`.
- These two input sets are disjoint, and the map is injective per
  `numpy.random`'s documented determinism, so the output sets are disjoint.
- Training on `TRAIN_SEED_POOL` therefore cannot reveal any patient that
  evaluation on `HELDOUT_SEED_POOL` will ever present.

(The "injective" step is the honest hand-wave: numpy's hashing from seed to
state is not rigorously injective in pathological cases, but the collision
probability at these pool sizes is negligible and matches standard ML
evaluation practice.)

## How to extend

Need more held-out seeds? Add them to `HELDOUT_SEED_POOL` in
`src/hemosim/reproducibility.py` and document the change here. Never shrink
the gap between the pools.

Need a separate **validation** pool for hyperparameter tuning? Carve one
out of the training pool (e.g. `range(8000, 10000)`) rather than expanding
into the held-out range. The held-out pool stays untouched for final
reporting only.

## See also

- `src/hemosim/reproducibility.py` — source of truth for the pools.
- `tests/test_reproducibility.py` — CI check that the pools stay disjoint.
- `scripts/reproduce.sh` — one-command reproduction harness.
- `results/EXPECTED_RESULTS.json` — reference numbers for the tolerance check.
