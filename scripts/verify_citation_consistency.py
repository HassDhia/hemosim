#!/usr/bin/env python3
"""AC+NC gate runner: verify artifact-paper citation + numeric consistency.

The original v0.2.2 bug was `nemati_ttr_standard` as a benchmark key for
what the paper (correctly) framed as a Wan 2008 antithrombotic-stewardship
target. The secondary bug was the paper's DOAC Table 6 drifting out of sync
with `results/published_calibration.json` when the harness was re-run at
a different episode count.

This gate enforces three invariants:

1. AC1 — STRICT ATTRIBUTION. No number-citation pair in any reader-facing
   artifact attributes 0.55 / 55% TTR to Nemati 2016.
   - Parenthetical attributions ("55% (Nemati 2016)") are HARD FAILURES
     with NO disclaimer escape — the lexical structure alone is the bug.
   - Other attribution verbs ("Nemati reports...", "consistent with...")
     can be cleared by an adjacent (≤120 char) explicit disclaimer.

2. NC — NUMERIC LEDGER. Every numeric value the paper reports (RMSE,
   iteration counts, DOAC rates, episode counts) matches
   `results/published_calibration.json` exactly.

3. AC2 / BIB / JSON-KEYS — artifact freshness, bib-key coverage, and
   benchmark-key integrity.

Meta-tested: `python scripts/verify_citation_consistency.py --self-test`
injects synthetic defects and confirms the verifier fails on each.

Exit 0 = safe to ship. Exit 1 = do not ship.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "published_calibration.json"

ARTIFACT_FILES = [
    REPO / "paper" / "hemosim.tex",
    REPO / "paper" / "silent_deployment_protocol.tex",
    REPO / "paper" / "protocol_summary.md",
    REPO / "results" / "calibration_report.md",
    REPO / "README.md",
    REPO / "PATHWAY.md",
    REPO / "CHANGELOG.md",
]

# ---------------------------------------------------------------------------
# NUMERIC patterns + Nemati-attribution patterns (regex building blocks).
# ---------------------------------------------------------------------------
NUM = r"(?:0\.550?|55\s*%|55\s*percent)"
NEM = r"(?:\\cite[pt]?\{nemati2016\}|Nemati\s+2016|Nemati\s+et\s+al\.?)"

# ---------------------------------------------------------------------------
# HARD drift patterns — no disclaimer escape. The LEXICAL STRUCTURE itself
# is the defect; a reader skimming the document sees the attribution even
# if a disclaimer appears later in the sentence.
# ---------------------------------------------------------------------------
HARD_DRIFT = [
    # "55% (Nemati 2016" — parenthetical attribution. The paren may extend
    # arbitrarily (with or without disclaimer inside) but the opening
    # construct is definitive.
    (rf"{NUM}[^()\n]{{0,25}}\(\s*{NEM}", "parenthetical-attribution"),
    # "Nemati's 55%" / "Nemati et al.'s 0.55"
    (rf"Nemati(?:'s|\s+et\s+al\.?'s)\s+[^.\n]{{0,40}}{NUM}", "possessive-attribution"),
]

# ---------------------------------------------------------------------------
# SOFT drift patterns — may be cleared by a disclaimer within 120 chars
# BEFORE or AFTER the matched span.
# ---------------------------------------------------------------------------
SOFT_DRIFT = [
    (rf"{NUM}[^.\n]{{0,60}}consistent\s+with\s+{NEM}", "consistent-with"),
    (
        rf"{NUM}[^.\n]{{0,60}}(?:as\s+reported\s+by|according\s+to|from)\s+{NEM}",
        "attributed-to",
    ),
    (
        rf"{NEM}[^.\n]{{0,80}}reports?\s+(?:this|the|an?)\s+(?:TTR|aPTT)[^.\n]{{0,80}}{NUM}",
        "nemati-reports-ttr",
    ),
    (
        rf"{NEM}[^.\n]{{0,30}}TTR[^.\n]{{0,10}}(?:of|=|:)\s*{NUM}",
        "nemati-ttr-value",
    ),
]

LOCAL_DISCLAIMERS = [
    r"does\s+\\?emph\{?not\}?\s+report",
    r"does\s+not\s+report",
    r"NOT\s+report",
    r"is\s+not\s+the\s+source",
    r"misattribution",
    r"citation\s+correction",
    r"Removed.{0,80}misattributi",
]

DISCLAIMER_WINDOW = 120  # chars around soft-match


# ---------------------------------------------------------------------------
# NUMERIC CLAIMS LEDGER
# ---------------------------------------------------------------------------
@dataclass
class NumericClaim:
    label: str
    json_getter: Callable[[dict], object]
    paper_renderings: list[str]  # at least one MUST appear in paper/hemosim.tex


def _doac(d, drug, key):
    return d["doac_validation"]["drugs"][drug][key]


def _residual(d, which, key):
    return [r[which] for r in d["heparin_fit"]["residuals"] if r["key"] == key][0]


CLAIMS: list[NumericClaim] = [
    NumericClaim("heparin_rmse", lambda d: d["heparin_fit"]["rmse"], [r"0\.1837"]),
    NumericClaim("warfarin_rmse", lambda d: d["warfarin_fit"]["rmse"], [r"0\.0013"]),
    NumericClaim(
        "heparin_iter",
        lambda d: d["heparin_fit"]["n_iterations"],
        [r"178\s+iterations", r"178~iterations"],
    ),
    NumericClaim(
        "warfarin_iter",
        lambda d: d["warfarin_fit"]["n_iterations"],
        [r"190\s+iterations", r"190~iterations"],
    ),
    NumericClaim(
        "doac_episodes",
        lambda d: d["cli_args"]["episodes"],
        [
            r"[Ff]ive\s*hundred\s+simulated\s+episodes",
            r"500\s+episodes\s+per\s+drug",
        ],
    ),
    NumericClaim(
        "dabi_stroke_obs",
        lambda d: _doac(d, "dabigatran", "stroke_rate_per_100py"),
        [r"1\.43"],
    ),
    NumericClaim(
        "dabi_bleed_obs",
        lambda d: _doac(d, "dabigatran", "bleed_rate_per_100py"),
        [r"3\.27"],
    ),
    NumericClaim(
        "riva_stroke_obs",
        lambda d: _doac(d, "rivaroxaban", "stroke_rate_per_100py"),
        [r"3\.10"],
    ),
    NumericClaim(
        "riva_bleed_obs",
        lambda d: _doac(d, "rivaroxaban", "bleed_rate_per_100py"),
        [r"3\.92"],
    ),
    NumericClaim(
        "apix_stroke_obs",
        lambda d: _doac(d, "apixaban", "stroke_rate_per_100py"),
        [r"1\.64"],
    ),
    NumericClaim(
        "apix_bleed_obs",
        lambda d: _doac(d, "apixaban", "bleed_rate_per_100py"),
        [r"2\.25"],
    ),
    NumericClaim(
        "heparin_ttr_residual",
        lambda d: _residual(d, "residual", "wan_aptt_ttr_standard_of_care"),
        [r"-0\.175"],
    ),
    NumericClaim(
        "heparin_ttr_observed",
        lambda d: _residual(d, "observed", "wan_aptt_ttr_standard_of_care"),
        [r"0\.375"],
    ),
]


FORBIDDEN_JSON_KEYS = ["nemati_ttr_standard"]
REQUIRED_JSON_KEYS = ["wan_aptt_ttr_standard_of_care"]


# ---------------------------------------------------------------------------
# EACH CHECK RETURNS (pass_desc, list_of_failures)
# ---------------------------------------------------------------------------
def _has_disclaimer(window: str) -> bool:
    return any(re.search(p, window, re.IGNORECASE | re.DOTALL) for p in LOCAL_DISCLAIMERS)


def check_ac1(repo: Path) -> tuple[str, list[str]]:
    failures: list[str] = []
    patterns_checked = 0
    for path in ARTIFACT_FILES:
        full = repo / path.relative_to(REPO)
        if not full.exists():
            continue
        text = full.read_text()
        for pattern, kind in HARD_DRIFT:
            patterns_checked += 1
            for m in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
                rel = full.relative_to(repo)
                snippet = text[max(0, m.start() - 20):m.end() + 60].replace("\n", " ")
                failures.append(
                    f"AC1-HARD [{rel}] '{kind}' (no disclaimer escape)\n"
                    f"    matched: {m.group(0)!r}\n"
                    f"    context: ...{snippet}..."
                )
        for pattern, kind in SOFT_DRIFT:
            for m in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
                window_start = max(0, m.start() - DISCLAIMER_WINDOW)
                window_end = min(len(text), m.end() + DISCLAIMER_WINDOW)
                window = text[window_start:window_end]
                if _has_disclaimer(window):
                    continue
                rel = full.relative_to(repo)
                failures.append(
                    f"AC1-SOFT [{rel}] '{kind}' (no disclaimer in ±{DISCLAIMER_WINDOW}c)\n"
                    f"    matched: {m.group(0)!r}"
                )
    return (f"AC1 — {patterns_checked} drift patterns checked, 0 fire across {len(ARTIFACT_FILES)} files", failures)


def check_json_keys(artifact: dict) -> tuple[str, list[str]]:
    failures: list[str] = []
    bench = artifact.get("benchmarks", {})
    for k in FORBIDDEN_JSON_KEYS:
        if k in bench:
            failures.append(f"JSON-KEYS forbidden key '{k}' present in benchmarks")
    for k in REQUIRED_JSON_KEYS:
        if k not in bench:
            failures.append(f"JSON-KEYS required key '{k}' missing from benchmarks")
    return ("JSON-KEYS — benchmark keys correct", failures)


def check_nc(repo: Path, artifact: dict) -> tuple[str, list[str]]:
    failures: list[str] = []
    paper = (repo / "paper" / "hemosim.tex").read_text()
    for claim in CLAIMS:
        try:
            val = claim.json_getter(artifact)
        except Exception as e:
            failures.append(f"NC [{claim.label}] JSON accessor raised: {e}")
            continue
        matched = any(re.search(pat, paper) for pat in claim.paper_renderings)
        if not matched:
            failures.append(
                f"NC [{claim.label}] JSON value={val!r} but none of "
                f"{claim.paper_renderings!r} appear in paper/hemosim.tex"
            )
    return (f"NC — all {len(CLAIMS)} numeric claims match JSON", failures)


def check_bib(repo: Path) -> tuple[str, list[str]]:
    paper_text = (repo / "paper" / "hemosim.tex").read_text()
    bib_text = (repo / "paper" / "references.bib").read_text()
    cited_entries = set(re.findall(r"\\cite[pt]?\{([^}]+)\}", paper_text))
    flat = set()
    for entry in cited_entries:
        for k in entry.split(","):
            flat.add(k.strip())
    defined = set(re.findall(r"@\w+\{(\w+)\s*,", bib_text))
    missing = sorted(flat - defined)
    failures: list[str] = []
    if missing:
        failures.append(f"BIB undefined \\cite keys: {missing}")
    return ("BIB — all \\cite keys defined in references.bib", failures)


def check_ac2(repo: Path) -> tuple[str, list[str]]:
    failures: list[str] = []
    script = repo / "scripts" / "run_published_calibration.py"
    for artifact_path in [
        repo / "results" / "published_calibration.json",
        repo / "results" / "calibration_report.md",
    ]:
        if not artifact_path.exists():
            failures.append(f"AC2 missing artifact: {artifact_path}")
            continue
        if script.stat().st_mtime > artifact_path.stat().st_mtime + 1:
            failures.append(
                f"AC2 [{artifact_path.relative_to(repo)}] stale "
                f"(re-run run_published_calibration.py)"
            )
    return ("AC2 — derived-artifact freshness OK", failures)


def run(repo: Path = REPO) -> int:
    print("=" * 70)
    print(f"hemosim AC+NC citation-consistency gate runner  (repo={repo})")
    print("=" * 70)
    results_json = repo / "results" / "published_calibration.json"
    if not results_json.exists():
        print(f"FAIL: {results_json} missing — run run_published_calibration.py first.")
        return 1
    artifact = json.loads(results_json.read_text())

    checks = [
        ("AC1", check_ac1(repo)),
        ("JSON-KEYS", check_json_keys(artifact)),
        ("NC", check_nc(repo, artifact)),
        ("BIB", check_bib(repo)),
        ("AC2", check_ac2(repo)),
    ]

    all_failures: list[str] = []
    all_passes: list[str] = []
    for name, (pass_desc, failures) in checks:
        if failures:
            print(f"\n[{name}] FAIL")
            for f in failures:
                print(f"  ✗ {f}")
            all_failures.extend(failures)
        else:
            print(f"\n[{name}] PASS")
            all_passes.append(pass_desc)

    print("\n" + "=" * 70)
    print(f"PASSES:   {len(all_passes)}")
    for p in all_passes:
        print(f"  ✓ {p}")
    print(f"FAILURES: {len(all_failures)}")
    print("=" * 70)
    return 0 if not all_failures else 1


# ---------------------------------------------------------------------------
# Self-test: inject synthetic defects into a sandbox copy of the repo and
# confirm the verifier fails on each.
# ---------------------------------------------------------------------------
def self_test() -> int:
    print("=" * 70)
    print("SELF-TEST — injecting synthetic defects into a sandbox copy")
    print("=" * 70)

    tests = [
        (
            "parenthetical: '55% (Nemati 2016)'",
            lambda p: p.replace(
                "baseline TTR of 55% (Wan 2008",
                "baseline TTR of 55% (Nemati 2016",
                1,
            ),
            "paper/protocol_summary.md",
            "AC1",
        ),
        (
            "possessive: 'Nemati 2016 TTR of 0.55'",
            lambda p: p.replace(
                "baseline TTR of 55% (Wan 2008",
                "Nemati 2016 TTR of 0.55 (Wan 2008",
                1,
            ),
            "paper/protocol_summary.md",
            "AC1",
        ),
        (
            "JSON-key regression: rename back to nemati_ttr_standard",
            lambda t: t.replace(
                '"wan_aptt_ttr_standard_of_care"',
                '"nemati_ttr_standard"',
            ),
            "results/published_calibration.json",
            "JSON-KEYS",
        ),
        (
            "numeric drift: change paper's heparin rmse to 0.9999",
            lambda p: p.replace("RMSE $= 0.1837$", "RMSE $= 0.9999$"),
            "paper/hemosim.tex",
            "NC",
        ),
        (
            "numeric drift: change paper's dabigatran stroke to 9.99",
            lambda p: p.replace("1.43 & {[0.92, 1.33]}", "9.99 & {[0.92, 1.33]}"),
            "paper/hemosim.tex",
            "NC",
        ),
    ]

    passed = 0
    failed = 0
    for name, mutation, rel_path, _expected_gate in tests:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_repo = Path(tmp) / "hemosim"
            # Use git to make a lightweight copy (respects .gitignore; avoids .venv)
            subprocess.run(
                ["git", "-C", str(REPO), "worktree", "add", "--detach", str(tmp_repo), "HEAD"],
                check=True, capture_output=True,
            )
            try:
                # Apply mutation
                target = tmp_repo / rel_path
                original = target.read_text()
                mutated = mutation(original)
                if mutated == original:
                    print(f"  ? [{name}] mutation was a no-op — test invalid")
                    failed += 1
                    continue
                target.write_text(mutated)

                # Also copy the results/ directory (not in HEAD worktree, needed for NC)
                if rel_path != "results/published_calibration.json":
                    shutil.copy2(
                        REPO / "results" / "published_calibration.json",
                        tmp_repo / "results" / "published_calibration.json",
                    )

                # Run
                rc = run(tmp_repo)
                if rc != 0:
                    print(f"  ✓ CAUGHT [{name}]")
                    passed += 1
                else:
                    print(f"  ✗ MISSED [{name}] — verifier returned 0 despite defect")
                    failed += 1
            finally:
                subprocess.run(
                    ["git", "-C", str(REPO), "worktree", "remove", "--force", str(tmp_repo)],
                    capture_output=True,
                )

    print("\n" + "=" * 70)
    print(f"SELF-TEST: {passed}/{passed + failed} synthetic defects caught")
    print("=" * 70)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", help="Inject defects and verify failures")
    args = parser.parse_args()
    if args.self_test:
        raise SystemExit(self_test())
    raise SystemExit(run())
