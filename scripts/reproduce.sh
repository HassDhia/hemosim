#!/usr/bin/env bash
# hemosim one-command reproducibility harness (ISC-13).
#
# Goal: from a fresh clone, verify the published baseline numbers in
# results/EXPECTED_RESULTS.json are reproducible. End-to-end under 30 min.
#
# Usage:
#   ./scripts/reproduce.sh           # uses a fresh .venv-repro
#   HEMOSIM_REPRO_USE_CURRENT_VENV=1 ./scripts/reproduce.sh
#                                    # reuse the existing .venv (faster; for dev)
#
# Exit codes:
#   0 = PASS (results match EXPECTED_RESULTS.json within tolerance)
#   1 = FAIL (divergence, missing files, or non-zero pytest)
#
# Configuration via environment:
#   PYTHON_BIN                      python interpreter to bootstrap the venv
#                                   (default: python3)
#   HEMOSIM_REPRO_EPISODES          episodes per baseline (default: 100)
#   HEMOSIM_REPRO_TOLERANCE         relative tolerance on mean_reward
#                                   (default: 0.05 = 5%)
#   HEMOSIM_REPRO_USE_CURRENT_VENV  if "1", reuse existing .venv instead of
#                                   creating a fresh .venv-repro

set -euo pipefail

# --- Configuration -----------------------------------------------------------

HEMOSIM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HEMOSIM_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
N_EPISODES="${HEMOSIM_REPRO_EPISODES:-100}"
TOLERANCE="${HEMOSIM_REPRO_TOLERANCE:-0.05}"
USE_CURRENT_VENV="${HEMOSIM_REPRO_USE_CURRENT_VENV:-0}"

if [[ "${USE_CURRENT_VENV}" == "1" && -x ".venv/bin/python" ]]; then
    VENV_DIR="${HEMOSIM_ROOT}/.venv"
    VENV_MODE="reuse"
else
    VENV_DIR="${HEMOSIM_ROOT}/.venv-repro"
    VENV_MODE="fresh"
fi

if [[ -t 1 ]]; then
    RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[0;33m'
    BOLD=$'\033[1m'; RESET=$'\033[0m'
else
    RED=""; GREEN=""; YELLOW=""; BOLD=""; RESET=""
fi

log()  { printf "%s[repro]%s %s\n"  "${BOLD}"   "${RESET}" "$*"; }
ok()   { printf "%s[OK]%s    %s\n"  "${GREEN}"  "${RESET}" "$*"; }
warn() { printf "%s[warn]%s  %s\n"  "${YELLOW}" "${RESET}" "$*"; }
die()  { printf "%s[FAIL]%s  %s\n"  "${RED}"    "${RESET}" "$*" >&2; exit 1; }

T_START="$(date +%s)"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then die "harness exited with rc=$rc"; fi' EXIT

# --- Banner -----------------------------------------------------------------

log "hemosim reproducibility harness (ISC-13)"
log "repo:      ${HEMOSIM_ROOT}"
log "branch:    $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
log "commit:    $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
log "python:    ${PYTHON_BIN} ($(${PYTHON_BIN} --version 2>&1))"
log "venv:      ${VENV_DIR} (${VENV_MODE})"
log "episodes:  ${N_EPISODES}"
log "tolerance: ${TOLERANCE} (relative)"

# --- Step 1: venv ------------------------------------------------------------

if [[ "${VENV_MODE}" == "fresh" ]]; then
    if [[ -d "${VENV_DIR}" ]]; then
        log "removing stale venv at ${VENV_DIR}"
        rm -rf "${VENV_DIR}"
    fi
    log "creating fresh venv"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

if [[ "${VENV_MODE}" == "fresh" ]]; then
    log "upgrading pip"
    python -m pip install --quiet --upgrade pip
    log "installing hemosim with [dev] extras"
    python -m pip install --quiet -e ".[dev]"
fi

# --- Step 2: test suite ------------------------------------------------------

log "running test suite"
PYTEST_ARGS=(-q --no-header)
# ISC-4 (clinical metrics) is a parallel workstream. If its test file is
# present but the module hasn't been built yet, skip that one file rather
# than blocking the whole harness. Once ISC-4 lands this branch becomes a
# no-op.
if [[ -f "tests/test_clinical_metrics.py" ]] && \
   ! python -c "from hemosim.metrics import clinical" 2>/dev/null; then
    warn "tests/test_clinical_metrics.py present but hemosim.metrics.clinical not yet built — skipping (ISC-4 in progress)"
    PYTEST_ARGS+=(--ignore=tests/test_clinical_metrics.py)
fi
python -m pytest "${PYTEST_ARGS[@]}" tests/

# --- Step 3: ISC-12 self-test (if present) -----------------------------------

ISC12_SELF_TEST=""
for candidate in scripts/self_test.py scripts/isc12_self_test.py scripts/run_self_test.py; do
    if [[ -f "${candidate}" ]]; then
        ISC12_SELF_TEST="${candidate}"
        break
    fi
done
if [[ -n "${ISC12_SELF_TEST}" ]]; then
    log "running ISC-12 self-test (${ISC12_SELF_TEST})"
    python "${ISC12_SELF_TEST}"
else
    warn "no ISC-12 self-test script found — skipping"
fi

# --- Step 4: held-out baseline evaluation ------------------------------------

log "running honest baseline evaluation (${N_EPISODES} episodes, held-out seeds)"
python scripts/generate_results.py \
    --episodes "${N_EPISODES}" \
    --eval-set heldout

# --- Step 5: compare to EXPECTED_RESULTS.json --------------------------------

EXPECTED="${HEMOSIM_ROOT}/results/EXPECTED_RESULTS.json"
ACTUAL="${HEMOSIM_ROOT}/results/training_results.json"

if [[ ! -f "${EXPECTED}" ]]; then
    die "missing ${EXPECTED} — cannot verify reproducibility"
fi

log "comparing $(basename "${ACTUAL}") to $(basename "${EXPECTED}") (rel tol=${TOLERANCE})"

set +e
python - "${EXPECTED}" "${ACTUAL}" "${TOLERANCE}" <<'PYEOF'
"""Inline comparator — kept in-script to avoid a fragile import path on
partial installs. Compares mean_reward for every baseline in
EXPECTED_RESULTS.json using relative tolerance. PPO entries are compared
only when BOTH sides have a non-null value — if EXPECTED has PPO=null
(no trained model committed) the comparator silently skips it."""
import json
import sys

expected_path, actual_path, tol_s = sys.argv[1], sys.argv[2], sys.argv[3]
tol = float(tol_s)

with open(expected_path) as f:
    expected = json.load(f)
with open(actual_path) as f:
    actual = json.load(f)

BASELINES = ("clinical_baseline", "random")
failures = []
rows = []

for env_id, env_exp in expected["environments"].items():
    env_act = actual.get("environments", {}).get(env_id)
    if env_act is None:
        failures.append(f"{env_id}: missing from actual results")
        continue
    for baseline in BASELINES:
        exp = env_exp.get(baseline)
        act = env_act.get(baseline)
        if exp is None:
            continue
        if act is None:
            failures.append(f"{env_id}/{baseline}: missing mean_reward in actual")
            continue
        e = float(exp["mean_reward"])
        a = float(act["mean_reward"])
        denom = max(abs(e), 1.0)
        rel = abs(a - e) / denom
        status = "OK" if rel <= tol else "FAIL"
        rows.append((env_id, baseline, e, a, rel, status))
        if rel > tol:
            failures.append(
                f"{env_id}/{baseline}: mean_reward {a:.4f} vs expected {e:.4f} "
                f"(rel err {rel:.4f} > tol {tol})"
            )
    # PPO: compare only when both sides present
    exp_ppo = env_exp.get("ppo")
    act_ppo = env_act.get("ppo")
    if exp_ppo is not None and act_ppo is not None:
        e = float(exp_ppo["mean_reward"])
        a = float(act_ppo["mean_reward"])
        denom = max(abs(e), 1.0)
        rel = abs(a - e) / denom
        status = "OK" if rel <= tol else "FAIL"
        rows.append((env_id, "ppo", e, a, rel, status))
        if rel > tol:
            failures.append(
                f"{env_id}/ppo: mean_reward {a:.4f} vs expected {e:.4f} "
                f"(rel err {rel:.4f} > tol {tol})"
            )

print()
print(f"{'env':34s} {'baseline':18s} {'expected':>12s} {'actual':>12s} {'rel_err':>10s} {'status':>6s}")
print("-" * 100)
for env_id, baseline, e, a, rel, status in rows:
    print(f"{env_id:34s} {baseline:18s} {e:12.4f} {a:12.4f} {rel:10.4f} {status:>6s}")
print()

if failures:
    print("FAILURES:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
else:
    print("All metrics within tolerance.")
    sys.exit(0)
PYEOF
COMPARE_RC=$?
set -e

# --- Step 6: verdict ---------------------------------------------------------

T_END="$(date +%s)"
ELAPSED=$(( T_END - T_START ))

trap - EXIT

if [[ ${COMPARE_RC} -eq 0 ]]; then
    ok "reproducibility check PASSED in ${ELAPSED}s"
    echo
    printf "%s%sPASS%s  hemosim reproduced EXPECTED_RESULTS.json within tol=%s\n" \
        "${GREEN}" "${BOLD}" "${RESET}" "${TOLERANCE}"
    exit 0
else
    printf "%s%sFAIL%s  hemosim diverged from EXPECTED_RESULTS.json (see diff above)\n" \
        "${RED}" "${BOLD}" "${RESET}"
    exit 1
fi
