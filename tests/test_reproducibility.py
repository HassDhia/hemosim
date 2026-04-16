"""Tests for the reproducibility harness (ISC-13).

Covers:
- Pool boundaries and disjointness
- ``assert_held_out`` and ``assert_train`` behavior
- Helper functions (``heldout_seeds``, ``train_seeds``)
- ``scripts/reproduce.sh`` existence, executability, and shell syntax
- ``results/EXPECTED_RESULTS.json`` existence and schema

These tests must never modify the checked-in expected results or the
scripts under test. They only verify static properties.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from hemosim.reproducibility import (
    HELDOUT_PUBLISHED_SEEDS,
    HELDOUT_SEED_POOL,
    TRAIN_SEED_POOL,
    assert_held_out,
    assert_seeds_held_out,
    assert_seeds_train,
    assert_train,
    heldout_seeds,
    train_seeds,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REPRODUCE_SH = REPO_ROOT / "scripts" / "reproduce.sh"
EXPECTED_RESULTS = REPO_ROOT / "results" / "EXPECTED_RESULTS.json"
HELDOUT_DOC = REPO_ROOT / "HELDOUT_SEEDS.md"


# ---------------------------------------------------------------------------
# Pool boundaries and disjointness
# ---------------------------------------------------------------------------


class TestSeedPoolBoundaries:
    def test_train_pool_is_0_to_9999(self):
        assert TRAIN_SEED_POOL.start == 0
        assert TRAIN_SEED_POOL.stop == 10000
        assert len(TRAIN_SEED_POOL) == 10000

    def test_heldout_pool_is_100000_to_100999(self):
        assert HELDOUT_SEED_POOL.start == 100000
        assert HELDOUT_SEED_POOL.stop == 101000
        assert len(HELDOUT_SEED_POOL) == 1000

    def test_pools_are_disjoint(self):
        train_set = set(TRAIN_SEED_POOL)
        heldout_set = set(HELDOUT_SEED_POOL)
        assert train_set.isdisjoint(heldout_set), (
            "TRAIN_SEED_POOL and HELDOUT_SEED_POOL must not overlap. "
            f"Intersection: {sorted(train_set & heldout_set)[:10]}..."
        )

    def test_buffer_between_pools(self):
        """The 90k-wide buffer is deliberate. If this ever shrinks, update
        HELDOUT_SEEDS.md and justify the change."""
        gap = HELDOUT_SEED_POOL.start - TRAIN_SEED_POOL.stop
        assert gap >= 90_000, (
            f"Expected >=90k buffer between training and held-out pools, "
            f"got {gap}. Shrinking the buffer increases the risk that "
            f"'base_seed + n_episodes' off-by-many mistakes cross the boundary."
        )

    def test_published_seeds_are_subset_of_heldout(self):
        for s in HELDOUT_PUBLISHED_SEEDS:
            assert s in HELDOUT_SEED_POOL

    def test_published_seeds_are_100_seeds(self):
        assert len(HELDOUT_PUBLISHED_SEEDS) == 100


# ---------------------------------------------------------------------------
# assert_held_out / assert_train
# ---------------------------------------------------------------------------


class TestAssertHeldOut:
    def test_accepts_first_heldout_seed(self):
        assert_held_out(100000)

    def test_accepts_last_heldout_seed(self):
        assert_held_out(100999)

    def test_accepts_midrange_heldout_seed(self):
        assert_held_out(100500)

    def test_rejects_training_seeds(self):
        with pytest.raises(ValueError, match="HELDOUT_SEED_POOL"):
            assert_held_out(0)
        with pytest.raises(ValueError, match="HELDOUT_SEED_POOL"):
            assert_held_out(42)
        with pytest.raises(ValueError, match="HELDOUT_SEED_POOL"):
            assert_held_out(9999)

    def test_rejects_out_of_range_high(self):
        with pytest.raises(ValueError, match="HELDOUT_SEED_POOL"):
            assert_held_out(101000)
        with pytest.raises(ValueError, match="HELDOUT_SEED_POOL"):
            assert_held_out(999999)

    def test_rejects_negative_seeds(self):
        with pytest.raises(ValueError, match="HELDOUT_SEED_POOL"):
            assert_held_out(-1)

    def test_rejects_non_int(self):
        with pytest.raises(TypeError):
            assert_held_out(100000.0)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            assert_held_out("100000")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            assert_held_out(True)  # type: ignore[arg-type]  # bool guard


class TestAssertTrain:
    def test_accepts_first_training_seed(self):
        assert_train(0)

    def test_accepts_last_training_seed(self):
        assert_train(9999)

    def test_accepts_midrange_training_seed(self):
        assert_train(42)

    def test_rejects_heldout_seeds(self):
        with pytest.raises(ValueError, match="TRAIN_SEED_POOL"):
            assert_train(100000)
        with pytest.raises(ValueError, match="TRAIN_SEED_POOL"):
            assert_train(100500)
        with pytest.raises(ValueError, match="TRAIN_SEED_POOL"):
            assert_train(100999)

    def test_rejects_out_of_range_high(self):
        with pytest.raises(ValueError, match="TRAIN_SEED_POOL"):
            assert_train(10000)
        with pytest.raises(ValueError, match="TRAIN_SEED_POOL"):
            assert_train(50000)

    def test_rejects_negative_seeds(self):
        with pytest.raises(ValueError, match="TRAIN_SEED_POOL"):
            assert_train(-1)

    def test_rejects_non_int(self):
        with pytest.raises(TypeError):
            assert_train(42.0)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            assert_train("42")  # type: ignore[arg-type]


class TestBulkAssertions:
    def test_assert_seeds_held_out_passes_for_heldout(self):
        assert_seeds_held_out([100000, 100001, 100002])

    def test_assert_seeds_held_out_fails_on_first_contaminated(self):
        with pytest.raises(ValueError):
            assert_seeds_held_out([100000, 42, 100002])

    def test_assert_seeds_train_passes_for_train(self):
        assert_seeds_train([0, 1, 2, 9999])

    def test_assert_seeds_train_fails_on_first_contaminated(self):
        with pytest.raises(ValueError):
            assert_seeds_train([0, 100000, 2])


# ---------------------------------------------------------------------------
# Helper slicers
# ---------------------------------------------------------------------------


class TestHeldoutSeeds:
    def test_default_offset_returns_first_n(self):
        assert heldout_seeds(5) == [100000, 100001, 100002, 100003, 100004]

    def test_offset_returns_shifted_range(self):
        assert heldout_seeds(3, offset=10) == [100010, 100011, 100012]

    def test_zero_returns_empty(self):
        assert heldout_seeds(0) == []

    def test_exact_pool_size_is_allowed(self):
        result = heldout_seeds(len(HELDOUT_SEED_POOL))
        assert len(result) == len(HELDOUT_SEED_POOL)
        assert result[0] == HELDOUT_SEED_POOL.start
        assert result[-1] == HELDOUT_SEED_POOL.stop - 1

    def test_rejects_overflow(self):
        with pytest.raises(ValueError, match="HELDOUT_SEED_POOL"):
            heldout_seeds(len(HELDOUT_SEED_POOL) + 1)

    def test_rejects_negative_n(self):
        with pytest.raises(ValueError):
            heldout_seeds(-1)

    def test_rejects_negative_offset(self):
        with pytest.raises(ValueError):
            heldout_seeds(5, offset=-1)

    def test_all_returned_seeds_pass_assert_held_out(self):
        for s in heldout_seeds(100):
            assert_held_out(s)


class TestTrainSeeds:
    def test_default_offset_returns_first_n(self):
        assert train_seeds(5) == [0, 1, 2, 3, 4]

    def test_offset_returns_shifted_range(self):
        assert train_seeds(3, offset=10) == [10, 11, 12]

    def test_rejects_overflow(self):
        with pytest.raises(ValueError, match="TRAIN_SEED_POOL"):
            train_seeds(len(TRAIN_SEED_POOL) + 1)

    def test_all_returned_seeds_pass_assert_train(self):
        for s in train_seeds(100):
            assert_train(s)


# ---------------------------------------------------------------------------
# scripts/reproduce.sh harness
# ---------------------------------------------------------------------------


class TestReproduceScript:
    def test_script_exists(self):
        assert REPRODUCE_SH.exists(), (
            f"Expected reproducibility harness at {REPRODUCE_SH}. "
            "See ISC-13."
        )

    def test_script_is_executable(self):
        import os

        assert os.access(REPRODUCE_SH, os.X_OK), (
            f"{REPRODUCE_SH} is not executable. Run: chmod +x {REPRODUCE_SH}"
        )

    def test_script_is_bash(self):
        first_line = REPRODUCE_SH.read_text(encoding="utf-8").splitlines()[0]
        assert first_line.startswith("#!"), "reproduce.sh must have a shebang"
        assert "bash" in first_line, (
            f"Expected bash shebang, got {first_line!r}"
        )

    def test_script_has_set_errexit(self):
        """Ensure the script fails fast on error, so 'PASS' output can't
        be produced after a silent failure upstream."""
        text = REPRODUCE_SH.read_text(encoding="utf-8")
        # Accept either `set -e` (alone or combined) or `set -euo pipefail`.
        assert ("set -e" in text) or ("set -euo" in text), (
            "reproduce.sh must use `set -e` (or stricter) to fail fast."
        )

    def test_script_syntax_is_valid(self):
        """Parse the script with ``bash -n`` (no execute) to catch syntax
        errors without running anything."""
        bash = shutil.which("bash")
        if bash is None:  # pragma: no cover
            pytest.skip("bash not available on PATH")
        proc = subprocess.run(
            [bash, "-n", str(REPRODUCE_SH)],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, (
            f"bash -n failed for {REPRODUCE_SH}:\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )

    def test_script_references_expected_results(self):
        """Catch accidental renames of the reference artifact."""
        text = REPRODUCE_SH.read_text(encoding="utf-8")
        assert "EXPECTED_RESULTS.json" in text, (
            "reproduce.sh must compare against results/EXPECTED_RESULTS.json"
        )

    def test_script_uses_heldout_eval_set(self):
        """Reproduction must run against the held-out pool, not training."""
        text = REPRODUCE_SH.read_text(encoding="utf-8")
        assert "--eval-set heldout" in text or "--eval-set=heldout" in text, (
            "reproduce.sh must pass --eval-set heldout to generate_results.py"
        )


# ---------------------------------------------------------------------------
# results/EXPECTED_RESULTS.json reference artifact
# ---------------------------------------------------------------------------


class TestExpectedResults:
    def test_file_exists(self):
        assert EXPECTED_RESULTS.exists(), (
            f"Expected reference results at {EXPECTED_RESULTS}. "
            "Generated from results/training_results.json at ISC-13 commit."
        )

    def test_is_valid_json(self):
        with open(EXPECTED_RESULTS) as f:
            json.load(f)

    def test_has_expected_envs(self):
        with open(EXPECTED_RESULTS) as f:
            data = json.load(f)
        assert "environments" in data
        envs = data["environments"]
        for env_id in (
            "hemosim/WarfarinDosing-v0",
            "hemosim/HeparinInfusion-v0",
            "hemosim/DOACManagement-v0",
            "hemosim/DICManagement-v0",
        ):
            assert env_id in envs, f"Missing env {env_id} in expected results"

    def test_every_baseline_has_mean_reward(self):
        with open(EXPECTED_RESULTS) as f:
            data = json.load(f)
        for env_id, env_data in data["environments"].items():
            for baseline_name in ("clinical_baseline", "random"):
                baseline = env_data.get(baseline_name)
                assert baseline is not None, (
                    f"{env_id}/{baseline_name} missing from expected results"
                )
                assert "mean_reward" in baseline, (
                    f"{env_id}/{baseline_name} missing 'mean_reward'"
                )


# ---------------------------------------------------------------------------
# HELDOUT_SEEDS.md
# ---------------------------------------------------------------------------


class TestHeldoutSeedsDoc:
    def test_doc_exists(self):
        assert HELDOUT_DOC.exists(), (
            f"Expected {HELDOUT_DOC} documenting the held-out policy."
        )

    def test_doc_mentions_both_pools(self):
        text = HELDOUT_DOC.read_text(encoding="utf-8")
        assert "0, 10000" in text or "range(0, 10000)" in text
        assert "100000, 101000" in text or "range(100000, 101000)" in text

    def test_doc_mentions_contamination_concept(self):
        text = HELDOUT_DOC.read_text(encoding="utf-8").lower()
        assert "contamination" in text or "train-test" in text
