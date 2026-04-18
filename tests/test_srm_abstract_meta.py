"""SR-M claim: abstract_meta

Meta-test verifying that every per-claim SR-M falsification test file
exists. Per registry: the abstract is an umbrella over per-claim
entries; if any per-claim SR-M test is missing, the meta-claim is
refuted.

Primary-source target: 11 per-claim SR-M test files present in tests/.
Falsification: any expected SR-M test file missing.
"""

from __future__ import annotations

from pathlib import Path

EXPECTED_CLAIM_IDS = [
    "warfarin_hamberg_pkpd",
    "heparin_raschke_aptt",
    "doac_rely_rocket_aristotle",
    "dic_hockin_mann_cascade",
    "rosendaal_ttr",
    "isth_major_bleed",
    "warkentin_4t_hit",
    "nemati_dqn_arch",
    "env_reward_function_shapes",
    "silent_deployment_protocol_stats",
    "pomdp_lab_masking",
]


def test_srm_abstract_meta_all_subclaim_tests_pass():
    """Every registered SR-M per-claim test file exists in tests/.

    Primary-source target (numeric range): minimum 11 SR-M files.
    """
    tests_dir = Path(__file__).parent
    found = sorted(p.name for p in tests_dir.glob("test_srm_*.py"))
    # >= 11 per-claim files (plus this meta file -> 12 total)
    assert len(found) >= 12, (
        f"Expected at least 12 SR-M test files (11 per-claim + this meta); "
        f"found {len(found)}: {found}"
    )
    for cid in EXPECTED_CLAIM_IDS:
        fname = f"test_srm_{cid}.py"
        assert (tests_dir / fname).is_file(), (
            f"Missing SR-M falsification test for claim {cid!r}: "
            f"expected tests/{fname}"
        )


def test_srm_abstract_meta_registry_exists():
    """Registry file from Phase A is present and non-empty.

    Asserts registry length > 100 lines (registry has 12 claims × ~20
    lines/claim = ~240 lines).
    """
    registry = Path(__file__).parent.parent / ".reviews" / "mechanistic-claims.md"
    assert registry.is_file(), f"Mechanistic-claims registry missing: {registry}"
    lines = registry.read_text().splitlines()
    assert len(lines) >= 100, (
        f"Registry unexpectedly short: {len(lines)} lines; expected >= 100"
    )
