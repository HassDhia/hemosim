"""Gate test: AC+NC citation-consistency verifier.

Runs `scripts/verify_citation_consistency.py` and asserts exit 0 on the
current repo, plus confirms `--self-test` passes (meta-test that catches
known synthetic defect shapes).

If this test fails, the paper or an artifact has drifted from the JSON
source of truth — fix before shipping.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "verify_citation_consistency.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO,
    )


def test_verifier_passes_on_clean_repo():
    """No citation/numeric drift on current HEAD."""
    r = _run()
    assert r.returncode == 0, (
        f"verify_citation_consistency.py returned {r.returncode}\n"
        f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )
    assert "FAILURES: 0" in r.stdout


def test_verifier_self_test_catches_synthetic_defects():
    """Meta-test: injecting known defect shapes into a sandboxed worktree
    must cause the verifier to fail. Proves the verifier is not a rubber
    stamp."""
    r = _run("--self-test")
    assert r.returncode == 0, (
        f"self-test returned {r.returncode}\n"
        f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )
    # Must have caught every injected defect.
    assert "MISSED" not in r.stdout, r.stdout
    assert "SELF-TEST: 5/5 synthetic defects caught" in r.stdout
