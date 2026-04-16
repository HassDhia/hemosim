#!/usr/bin/env python3
"""Run the published-data PK/PD calibration harness (ISC-12).

Usage:
    python scripts/run_published_calibration.py            # default seed 42
    python scripts/run_published_calibration.py --seed 7
    python scripts/run_published_calibration.py --episodes 2000  # DOAC n

Outputs:
    results/published_calibration.json
    results/calibration_report.md

The JSON artifact contains full fitted parameters, residuals, and DOAC
validation rates. The Markdown report is the human-readable view that
gets pasted into the paper's Calibration section.

This script is the single source of truth for those two files — it must
be runnable by any reviewer from a clean checkout. No post-hoc hand-edits
of the artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python scripts/run_published_calibration.py` from the
# repo root without installing the package.
_HERE = Path(__file__).resolve()
_SRC = _HERE.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hemosim.validation.published_calibration import (  # noqa: E402
    BENCHMARKS,
    fit_heparin_pkpd,
    fit_warfarin_pkpd,
    full_report_markdown,
    validate_doac_rates,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Calibrate HeparinPKPD and WarfarinPKPD against published "
            "clinical-trial summary statistics; validate DOAC event "
            "rates against RE-LY/ROCKET-AF/ARISTOTLE."
        )
    )
    p.add_argument("--seed", type=int, default=42, help="Global seed (default 42)")
    p.add_argument(
        "--heparin-iter",
        type=int,
        default=600,
        help="Nelder-Mead iterations for heparin fit (default 600)",
    )
    p.add_argument(
        "--warfarin-iter",
        type=int,
        default=800,
        help="Nelder-Mead iterations for warfarin fit (default 800)",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Episodes per drug for DOAC rate validation (default 1000)",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results",
        help="Output directory (default <repo>/results)",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[ISC-12] Running published calibration — "
        f"seed={args.seed}, n_doac={args.episodes}",
        flush=True,
    )

    print("[1/3] Fitting HeparinPKPD against Raschke/Hirsh/Nemati ...", flush=True)
    heparin = fit_heparin_pkpd(
        benchmarks=BENCHMARKS, seed=args.seed, max_iter=args.heparin_iter
    )
    print(
        f"    converged={heparin.converged} rmse={heparin.rmse:.4f} "
        f"nit={heparin.n_iterations}",
        flush=True,
    )

    print("[2/3] Fitting WarfarinPKPD against IWPC/Hamberg ...", flush=True)
    warfarin = fit_warfarin_pkpd(
        benchmarks=BENCHMARKS, seed=args.seed, max_iter=args.warfarin_iter
    )
    print(
        f"    converged={warfarin.converged} rmse={warfarin.rmse:.4f} "
        f"nit={warfarin.n_iterations}",
        flush=True,
    )

    print(
        f"[3/3] Validating DOAC event rates (n={args.episodes}/drug) ...",
        flush=True,
    )
    doac = validate_doac_rates(
        benchmarks=BENCHMARKS, n_episodes=args.episodes, seed=args.seed
    )
    for drug, d in doac["drugs"].items():
        print(
            f"    {drug}: stroke {d['stroke_rate_per_100py']:.2f}/yr "
            f"(trial {d['expected_stroke']:.2f}), "
            f"bleed {d['bleed_rate_per_100py']:.2f}/yr "
            f"(trial {d['expected_bleed']:.2f})",
            flush=True,
        )

    # Write JSON (source of truth — gets diffed on PRs).
    json_payload = {
        "schema_version": "1",
        "benchmarks": {k: b.to_dict() for k, b in BENCHMARKS.items()},
        "heparin_fit": heparin.to_dict(),
        "warfarin_fit": warfarin.to_dict(),
        "doac_validation": doac,
        "cli_args": {
            "seed": args.seed,
            "heparin_iter": args.heparin_iter,
            "warfarin_iter": args.warfarin_iter,
            "episodes": args.episodes,
        },
    }
    json_path = results_dir / "published_calibration.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, sort_keys=True)
    print(f"[OK] Wrote {json_path}", flush=True)

    # Write Markdown report (for paper).
    md = full_report_markdown(heparin, warfarin, doac)
    md_path = results_dir / "calibration_report.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"[OK] Wrote {md_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
