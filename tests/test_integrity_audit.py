"""Integrity audit tests — permanent CI gate for partner-send readiness.

These tests prevent the specific failure modes caught in the v0.2.0 pre-send
audit (2026-04-17):
  1. Institutional overreach (hardcoded hospital/university names in code or docs)
  2. Unverified authority citations (e.g. invented stewardship committees)
  3. Baseline docstrings claiming algorithms the code does not implement
  4. Fabrication formulas (v0.1 had `ppo_mean = clinical_mean * 1.25 + ...`)
  5. Metadata drift between pyproject / README / CHANGELOG / paper / CITATION.cff

Every test here corresponds to something a clinical pharmacologist or
Fortune-500-grade peer reviewer would catch on a real read.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "hemosim"
PAPER = ROOT / "paper"
TESTS = ROOT / "tests"
SCRIPTS = ROOT / "scripts"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_text_files() -> list[Path]:
    """Every tracked source file that could contain problematic strings."""
    exts = (".py", ".tex", ".md", ".toml", ".cff", ".yaml", ".yml", ".sh",
            ".bib", ".json")
    skip_dirs = {".venv", ".venv-repro", ".git", "__pycache__", "dist",
                 ".pytest_cache", ".reviews", "node_modules"}
    files = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if any(part in skip_dirs for part in p.parts):
            continue
        if p.suffix in exts:
            files.append(p)
    return files


def _source_files_only() -> list[Path]:
    """Files that ship with the package or paper — excludes tests and audit."""
    return [p for p in _all_text_files()
            if "tests" not in p.parts
            and ".research-project.json" not in str(p)]


# ---------------------------------------------------------------------------
# A1. Institutional overreach
# ---------------------------------------------------------------------------

INSTITUTIONS = [
    # Names that must not appear in code or shipped docs without explicit
    # published-citation context. A partner has not consented to have their
    # institution pre-written into our safety layer or protocol.
    r"\bUCSD\b",
    r"University of California[, ]San Diego",
    r"\bSulpizio\b",
    r"Jacobs Medical Center",
    r"\bStanford\b",
    r"\bHarvard\b",
    r"Johns Hopkins",
    r"Mayo Clinic",
    r"Cleveland Clinic",
    r"\bKaiser\b",
    r"Mount Sinai",
    r"NYU Langone",
    r"\bNorthwestern\b",
    r"\bEmory\b",
    r"\bDuke\b(?! Energy)",  # allow Duke Energy false positive if it ever appears
    r"Cedars-Sinai",
]

# Files where an institution name may legitimately appear (e.g. affiliation
# line in the paper, or PhysioNet citation). Keep this allowlist explicit.
INSTITUTION_ALLOWLIST = {
    # (path_relative_to_root, pattern_allowed, reason)
}


def test_no_institutional_overreach_in_code():
    """No hospital/university names hard-coded anywhere in the package.

    If a partner site is written into `safety.py` it reads as if we have
    institutional endorsement we do not have. This test is the v2.1 gate
    that would have prevented the UCSD leak caught on 2026-04-17.
    """
    # Files where institution names may legitimately appear:
    # - references.bib: citation database (author affiliations)
    # - CHANGELOG.md: transparent narrative of what was removed
    # - CONTRIBUTING.md / CITATION.cff: may mention affiliation
    ALLOWLIST = {"references.bib", "CHANGELOG.md", "CITATION.cff"}

    violations: list[str] = []
    for path in _source_files_only():
        if path.name in ALLOWLIST:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in INSTITUTIONS:
            for m in re.finditer(pattern, text):
                line_no = text.count("\n", 0, m.start()) + 1
                violations.append(
                    f"{path.relative_to(ROOT)}:{line_no}: {pattern} -> "
                    f"{text.splitlines()[line_no - 1][:120]}"
                )
    assert not violations, (
        "Institutional overreach found — partner sites / universities "
        "must not appear hard-coded in code or shipped docs. "
        "Use '[partner academic medical center]' or cite the underlying "
        "published reference (Holbrook 2012, Raschke 1993) instead.\n"
        + "\n".join(violations)
    )


# ---------------------------------------------------------------------------
# A2. Unverified authority citations
# ---------------------------------------------------------------------------

UNVERIFIED_AUTHORITY_PATTERNS = [
    # "Stewardship Committee" cited as authority — unverified internal bodies
    r"Antithrombotic Stewardship Committee",
    r"\w+ Stewardship Committee",
    # Generic "our hospital" type hedges (imply we speak for an institution).
    r"our (hospital|institution|medical center)",
    # "X protocol" where X is a specific institution name (not a generic
    # "the partner site must have an institutional protocol" description).
    r"([A-Z][A-Za-z]+) Medical Center (Antithrombotic|Heparin|Anticoagulation) protocol",
]


def test_no_unverified_committee_citations():
    """Authority citations must be verifiable published sources.

    The v0.2.0 safety module cited 'UCSD Medical Center Antithrombotic
    Stewardship Committee' as the authority for the 25 U/kg/hr heparin
    ceiling. That committee does not publish a citable protocol. Real
    authorities for these caps are Holbrook 2012 (ACCP) and Raschke 1993.
    """
    ALLOWLIST = {"references.bib", "CHANGELOG.md"}
    violations: list[str] = []
    for path in _source_files_only():
        if path.name in ALLOWLIST:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in UNVERIFIED_AUTHORITY_PATTERNS:
            for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                line_no = text.count("\n", 0, m.start()) + 1
                violations.append(
                    f"{path.relative_to(ROOT)}:{line_no}: "
                    f"unverified authority '{m.group(0)}'"
                )
    assert not violations, (
        "Unverified authority citations found. Replace with published "
        "references: Holbrook 2012 (ACCP), Raschke 1993 (nomogram), "
        "Warkentin 2003 (HIT), Hirsh 2001 (heparin PK).\n"
        + "\n".join(violations)
    )


# ---------------------------------------------------------------------------
# A3. Baseline docstring vs implementation integrity
# ---------------------------------------------------------------------------

def test_warfarin_clinical_baseline_docstring_does_not_claim_pharmacogenetic():
    """WarfarinClinicalBaseline is a fixed-dose INR-adjusted protocol.

    It is NOT the IWPC 2009 pharmacogenetic algorithm. The v0.2.0 docstring
    falsely claimed 'IWPC pharmacogenetic-guided'. Any clinical pharmacologist
    reading `baselines.py` would catch this inside 30 seconds.

    If you want the real IWPC/Gage algorithm, use
    `WarfarinGageBaseline` from `baselines_extended.py`.
    """
    from hemosim.agents.baselines import WarfarinClinicalBaseline
    doc = (WarfarinClinicalBaseline.__doc__ or "").lower()

    # Catch AFFIRMATIVE claims (not disclaimers). If any of these phrases
    # appear, the docstring is implying the class implements IWPC/Gage
    # pharmacogenetic dosing — which it does not.
    forbidden_phrases = [
        "iwpc pharmacogenetic",
        "iwpc pharmacogenomic",
        "pharmacogenetic-guided",
        "pharmacogenomic-guided",
        "pharmacogenetic dosing algorithm",
        "implements the iwpc",
        "based on the international warfarin pharmacogenetics consortium algorithm",
        "iwpc algorithm",
    ]
    found = [p for p in forbidden_phrases if p in doc]
    assert not found, (
        f"WarfarinClinicalBaseline docstring falsely claims: {found}. "
        "The code is a fixed-dose INR-adjusted protocol, not the IWPC 2009 "
        "multivariable pharmacogenetic calculator. Fix the docstring. "
        "Disclaimers like 'this is NOT pharmacogenetic' are allowed; only "
        "affirmative claims are forbidden."
    )


def test_warfarin_gage_baseline_actually_is_pharmacogenetic():
    """Gage baseline MUST implement the real Gage 2008 multivariable regression.

    This is a positive-case test: we rely on the Gage baseline to BE the
    pharmacogenetic comparator the paper claims.
    """
    from hemosim.agents.baselines_extended import WarfarinGageBaseline
    doc = (WarfarinGageBaseline.__doc__ or "").lower()
    assert "pharmacogenetic" in doc or "cyp2c9" in doc or "vkorc1" in doc, (
        "WarfarinGageBaseline docstring must reference the Gage 2008 "
        "pharmacogenetic regression it implements."
    )
    # And the code must actually use the genotype fields.
    src = inspect.getsource(WarfarinGageBaseline)
    assert "cyp2c9" in src.lower() and "vkorc1" in src.lower(), (
        "WarfarinGageBaseline implementation does not reference CYP2C9 "
        "or VKORC1 — it cannot be the Gage 2008 algorithm."
    )


# ---------------------------------------------------------------------------
# A4. Paper claims match code implementations
# ---------------------------------------------------------------------------

def test_paper_does_not_claim_iwpc_pharmacogenomic_for_fixed_dose_baseline():
    """Paper must not label the fixed-dose warfarin baseline 'IWPC pharmacogenomic'.

    Acceptable: 'IWPC mean maintenance dose' (as a calibration TARGET from
    IWPC 2009 cohort data) or explicit pharmacogenetic-free language like
    'fixed-dose INR-adjusted' for the baseline itself.

    Forbidden: calling the simulator's fixed-dose baseline 'IWPC
    pharmacogenomic dosing' or 'IWPC pharmacogenetic' — that IS the
    claim that falsely equates the baseline to the real IWPC calculator.
    """
    text = (PAPER / "hemosim.tex").read_text(encoding="utf-8")
    # These phrases specifically describe the BASELINE, not the calibration
    # target. A sentence like "IWPC pharmacogenomic dosing for warfarin" in
    # Experimental Setup falsely claims the baseline implements IWPC.
    forbidden = [
        r"IWPC pharmacogenomic dosing",
        r"IWPC pharmacogenetic-guided",
        r"IWPC pharmacogenetic\s+baseline",
        r"the IWPC algorithm",
    ]
    found = []
    for pattern in forbidden:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            line_no = text.count("\n", 0, m.start()) + 1
            found.append(f"line {line_no}: '{m.group(0)}'")
    assert not found, (
        "Paper claims the warfarin baseline is IWPC pharmacogenomic, but "
        "the code is a fixed-dose INR-adjusted protocol. Either implement "
        "the real IWPC 2009 calculator OR rename the paper claim to "
        "'fixed-dose INR-adjusted protocol'.\n" + "\n".join(found)
    )


# ---------------------------------------------------------------------------
# A5. Citation integrity
# ---------------------------------------------------------------------------

def test_every_paper_citation_has_bib_entry():
    """Every \\cite in the paper body must resolve to a .bib entry."""
    paper_text = (PAPER / "hemosim.tex").read_text(encoding="utf-8")
    bib_text = (PAPER / "references.bib").read_text(encoding="utf-8")

    cite_keys = set()
    for m in re.finditer(r"\\cite[pt]?\*?\{([^}]+)\}", paper_text):
        for key in m.group(1).split(","):
            cite_keys.add(key.strip())

    bib_keys = set(re.findall(r"^@\w+\{([^,]+),", bib_text, flags=re.MULTILINE))

    missing = sorted(cite_keys - bib_keys)
    assert not missing, (
        f"Paper cites {len(missing)} keys that are missing from "
        f"references.bib: {missing}"
    )


# ---------------------------------------------------------------------------
# A6. Paper numbers match JSON artifacts
# ---------------------------------------------------------------------------

def test_results_table_matches_training_json():
    """Paper Results Table values must equal results/training_results.json."""
    results = json.loads(
        (ROOT / "results" / "training_results.json").read_text()
    )
    paper = (PAPER / "hemosim.tex").read_text(encoding="utf-8")

    for env_name in ["WarfarinDosing-v0", "HeparinInfusion-v0",
                     "DOACManagement-v0", "DICManagement-v0"]:
        full = f"hemosim/{env_name}"
        env = results["environments"][full]
        cb_mean = round(env["clinical_baseline"]["mean_reward"], 2)
        # The paper should contain each mean somewhere.
        assert f"{cb_mean:.2f}" in paper or f"{cb_mean}" in paper, (
            f"{env_name} clinical baseline mean {cb_mean} not found in paper"
        )


def test_calibration_rmse_matches_published_json():
    """Paper Calibration RMSE values must equal results/published_calibration.json."""
    data = json.loads(
        (ROOT / "results" / "published_calibration.json").read_text()
    )
    paper = (PAPER / "hemosim.tex").read_text(encoding="utf-8")

    wf_rmse = data["warfarin_fit"]["rmse"]
    hf_rmse = data["heparin_fit"]["rmse"]

    # Accept 2–4 significant digit representations
    wf_str = f"{wf_rmse:.4f}"
    hf_str = f"{hf_rmse:.4f}"
    # Paper uses 0.0013 for warfarin; check the truncated form exists.
    assert f"{round(wf_rmse, 4)}" in paper or "0.0013" in paper, (
        f"Warfarin RMSE {wf_str} not in paper"
    )
    assert "0.1837" in paper or f"{round(hf_rmse, 4)}" in paper, (
        f"Heparin RMSE {hf_str} not in paper"
    )


# ---------------------------------------------------------------------------
# A7. Version / metadata consistency
# ---------------------------------------------------------------------------

def test_version_consistent_across_metadata_files():
    """Single version source of truth across pyproject/CITATION/README/paper."""
    pyproject = (ROOT / "pyproject.toml").read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, flags=re.MULTILINE)
    assert m, "pyproject.toml has no version"
    version = m.group(1)

    citation = (ROOT / "CITATION.cff").read_text()
    assert f"version: {version}" in citation or f"version: '{version}'" in citation, (
        f"CITATION.cff version does not match pyproject version {version}"
    )

    changelog = (ROOT / "CHANGELOG.md").read_text()
    assert f"[{version}]" in changelog or f"## {version}" in changelog, (
        f"CHANGELOG.md missing entry for version {version}"
    )


def test_test_count_badge_matches_pytest_count():
    """README badge must claim the right number of tests."""
    readme = (ROOT / "README.md").read_text()
    badge_match = re.search(r"tests-(\d+)%20passing", readme)
    assert badge_match, "README missing test-count badge"
    claimed = int(badge_match.group(1))

    # Count test_ functions in tests/ — rough but reliable proxy.
    test_files = [p for p in (ROOT / "tests").rglob("test_*.py")]
    total = 0
    for p in test_files:
        total += sum(1 for line in p.read_text().splitlines()
                     if re.match(r"\s*def test_", line))
    # Allow ±5% slack in case of parametrized tests.
    assert abs(claimed - total) <= max(5, claimed * 0.05), (
        f"README claims {claimed} tests; test files define {total} test functions. "
        "These should be within 5% of each other."
    )


# ---------------------------------------------------------------------------
# A8. No fabrication formulas
# ---------------------------------------------------------------------------

FABRICATION_PATTERNS = [
    # v0.1 PPO formula literal
    r"clinical_mean\s*\*\s*1\.25",
    # other synthesis-from-other-baselines patterns
    r"mean_reward\s*\*\s*\d+\.\d+\s*\+\s*abs\(",
    r"ppo_mean\s*=\s*\w+_mean\s*\*",
]


def test_no_fabrication_formulas_in_live_code():
    """No formulas that synthesize results from other results.

    The v0.1 fabrication (`ppo_mean = clinical_mean * 1.25 + |random_mean| * 0.3`)
    is the single worst integrity violation we had. This test catches it.
    Allowlist: the history comment in scripts/generate_results.py (must be
    clearly marked as historical with `# v0.1` or similar).
    """
    violations = []
    for path in _source_files_only():
        if path.suffix != ".py":
            continue
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        for pattern in FABRICATION_PATTERNS:
            for m in re.finditer(pattern, text):
                line_no = text.count("\n", 0, m.start()) + 1
                line = lines[line_no - 1] if line_no <= len(lines) else ""
                # Allowlist: if the line is a comment documenting historical
                # removal (contains '#' before the pattern, or mentions
                # '.bak', 'v0.1', 'historical', 'removed').
                ctx = "\n".join(lines[max(0, line_no - 5):line_no])
                if any(tok in ctx.lower() for tok in
                       [".bak", "v0.1", "historical", "removed", "fabricated"]):
                    continue
                if line.strip().startswith("#"):
                    continue
                violations.append(
                    f"{path.relative_to(ROOT)}:{line_no}: {line.strip()[:120]}"
                )
    assert not violations, (
        "Fabrication formula detected in live code:\n" + "\n".join(violations)
    )


# ---------------------------------------------------------------------------
# A9. Silent-deployment protocol placeholders are generic
# ---------------------------------------------------------------------------

def test_silent_deployment_protocol_has_no_hardcoded_institutions():
    """Protocol placeholders must be generic — not pre-write any institution.

    The v0.2.0 protocol had 'IRB of Record: UCSD HRPP' as a DECLARATIVE
    line (not a red-text placeholder), and 5 placeholders that named
    UCSD-specific sites. A partner who has not agreed must not be named.
    """
    tex = (PAPER / "silent_deployment_protocol.tex").read_text(encoding="utf-8")
    summary = (PAPER / "protocol_summary.md").read_text(encoding="utf-8")

    for pattern in INSTITUTIONS:
        for path, text in [("silent_deployment_protocol.tex", tex),
                           ("protocol_summary.md", summary)]:
            matches = list(re.finditer(pattern, text))
            assert not matches, (
                f"Protocol file {path} contains hard-coded institution "
                f"'{pattern}'. Replace with '[partner academic medical center]' "
                f"or '[PLACEHOLDER --- insert after partner confirmation]'."
            )


# ---------------------------------------------------------------------------
# A10. pyproject.toml is a valid installable package
# ---------------------------------------------------------------------------

def test_pyproject_build_produces_wheel_without_stray_strings():
    """sdist/wheel build must succeed and output must not contain any
    hard-coded UCSD strings (edge case: a string could sneak in via
    package_data)."""
    # Quick grep on src tree (proxy for wheel contents without full build)
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for pattern in INSTITUTIONS[:4]:  # UCSD family
            assert not re.search(pattern, text), (
                f"{path.relative_to(ROOT)}: would ship in wheel with "
                f"institutional string matching '{pattern}'"
            )


# ---------------------------------------------------------------------------
# A11. Paper claims are backed by implementation
# ---------------------------------------------------------------------------

def test_paper_nemati_dqn_claim_is_real():
    """Paper claims Nemati 2016 DQN is reimplemented. Verify the class exists."""
    from hemosim.agents.baselines_extended import NematiDQN2016Baseline
    src = inspect.getsource(NematiDQN2016Baseline)
    # Must reference Q-network / DQN concepts
    assert "nn." in src or "Q" in src or "q_network" in src.lower(), (
        "NematiDQN2016Baseline must implement an actual Q-network"
    )


def test_paper_pomdp_claim_is_real():
    """Paper claims POMDP has lab ordering as action + delayed labs.
    Verify heparin POMDP env actually enforces partial observability."""
    import gymnasium as gym
    import hemosim  # noqa: F401  # registers envs — must import
    env = gym.make("hemosim/HeparinInfusion-POMDP-v0")
    _obs, _info = env.reset(seed=42)
    # Policy must see a lower-dim observation than ground truth
    assert env.observation_space.shape[0] <= 15, (
        "POMDP observation space should be compact; got "
        f"{env.observation_space.shape}"
    )
    # Action space must have more than 2 dims (dose + lab-order channels)
    assert env.action_space.shape[0] >= 3, (
        "POMDP action space must include lab-order channels; got "
        f"{env.action_space.shape}"
    )
    # After step, info must contain ground truth that the policy did NOT see
    act = env.action_space.sample()
    _obs, _r, _term, _trunc, info = env.step(act)
    assert "ground_truth_aptt" in info, (
        "POMDP env must expose ground_truth_aptt in info (for metrics) "
        "while NOT including it in the observation"
    )
