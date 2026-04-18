# Contributing to hemosim

Thank you for your interest in contributing to hemosim. This is an open-source clinical-translation simulation platform; the correctness bar is high. A few rules keep it that way.

## Non-negotiables

1. **No synthetic or formula-generated numbers.** Every reported number — in the paper, README, or `results/` — must trace back to a real script execution. If you add a metric, add a test that runs it against real simulation output. The v0.1 fabricated-PPO incident is in the commit history as a cautionary tale.
2. **No mock labs in calibration.** Integration tests that exercise the PK/PD pipeline must run the real ODE integrator against real cohort data. Mocked calibration tests are not accepted.
3. **Every paper claim is code-verifiable.** If the paper says "X terminates when INR<0.5," the code must enforce exactly INR<0.5. If the two disagree, the paper and code must be reconciled in the same PR.
4. **Seed discipline.** Training uses seeds in `range(0, 10000)`; held-out evaluation uses `range(100000, 101000)`. Never mix. `assert_train(seed)` and `assert_held_out(seed)` guard the two at runtime.

## Before opening a PR

1. **Run the full test suite.**
   ```bash
   .venv/bin/python -m pytest tests/ -q
   ```
   All 383+ tests must pass.

2. **Run the reproducibility harness.**
   ```bash
   ./scripts/reproduce.sh
   ```
   Must exit 0 (5% relative-tolerance match to `results/EXPECTED_RESULTS.json`).

3. **If you changed PK/PD parameters, env dynamics, or baselines**, regenerate `results/EXPECTED_RESULTS.json` in the same PR. Explain the delta in the PR description.

4. **Lint and type-check.**
   ```bash
   .venv/bin/ruff check .
   .venv/bin/mypy src/hemosim
   ```

## Pull-request discipline

- One logical change per PR. Separate "add new metric," "re-calibrate PK/PD," and "paper update" into distinct PRs if they can stand alone.
- Include the clinical or methodological source for any new parameter. "This matches the IWPC 2009 algorithm Table 2" is a valid justification; "I tuned this by eye" is not.
- Changes to `src/hemosim/clinical/safety.py` (contraindications, dose caps, deferral thresholds) require a citation to the FDA label, ACCP guideline, or equivalent. Safety-layer changes without clinical justification will be rejected.

## Reporting issues

- **Numerical regression**: open an issue with the seed, the env, the script command, and the diff against `EXPECTED_RESULTS.json`.
- **Clinical correctness**: open an issue with the guideline/source citation and the specific line of code or paper text. We treat these as high-priority.
- **Install / packaging**: platform, Python version, exact `pip install` command and full traceback.

## Security

If you find a safety issue in the CDS harness or safety layer that would affect a downstream clinical deployment, please email partners@smarttechinvest.com directly rather than opening a public issue.

## License

By contributing, you agree that your contributions will be licensed under the MIT License (see `LICENSE`).
