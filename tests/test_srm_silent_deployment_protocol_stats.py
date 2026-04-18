"""SR-M claim: silent_deployment_protocol_stats

Falsification test for the SPIRIT 2013 / CONSORT 2010 sample-size
formula claim. This is a pure-Python formula check (protocol is spec,
not code). Computes n = 2σ²(z₁₋α + z₁₋β)² / Δ² for α=0.05, β=0.20,
σ=0.15, Δ=0.05 using scipy.stats.norm.ppf for z-scores and asserts
n ∈ [135, 150] per registry.
"""

from __future__ import annotations

import math


def test_srm_sample_size_formula_canonical_case():
    """Canonical sample-size formula yields n ∈ [135, 150] per arm."""
    try:
        from scipy.stats import norm  # type: ignore[import-not-found]
    except ImportError as e:
        raise AssertionError(
            f"scipy required for z-score computation (sample-size formula); "
            f"install scipy to run SR-M silent_deployment_protocol_stats test. {e}"
        )

    alpha = 0.05
    beta = 0.20
    sigma = 0.15
    delta = 0.05

    z_1_minus_alpha = float(norm.ppf(1.0 - alpha))    # one-sided ~1.645 (or /2 for two-sided)
    z_1_minus_beta = float(norm.ppf(1.0 - beta))      # ~0.8416

    # Two-sided test — use alpha/2.
    z_two_sided = float(norm.ppf(1.0 - alpha / 2.0))  # ~1.960

    n_per_arm = 2.0 * (sigma ** 2) * ((z_two_sided + z_1_minus_beta) ** 2) / (delta ** 2)
    n_rounded = math.ceil(n_per_arm)

    # Primary-source target: n ∈ [135, 150] (tolerance accounts for z-table rounding).
    assert 135 <= n_rounded <= 150, (
        f"Canonical sample size n = {n_rounded} (raw {n_per_arm:.3f}) outside "
        f"primary-source target [135, 150]. Formula: 2σ²(z₁₋α + z₁₋β)² / Δ² "
        f"with α=0.05, β=0.20, σ=0.15, Δ=0.05 should yield ~142 per arm. "
        f"Registry: not rescopable — any deviation is an error in the protocol .tex."
    )


def test_srm_sample_size_formula_present_in_protocol_tex():
    """The protocol .tex contains the canonical formula expression."""
    from pathlib import Path

    tex = Path(__file__).parent.parent / "paper" / "silent_deployment_protocol.tex"
    if not tex.is_file():
        raise AssertionError(
            f"Protocol .tex missing at {tex}; SR-M1 registry cites this file."
        )
    text = tex.read_text()
    # Look for hallmarks of the canonical formula. Accept LaTeX variants.
    # Primary-source target (literal): n = 2σ²(z + z)² / Δ² form.
    hallmarks_required = ["\\sigma", "\\Delta", "z_{1"]
    missing = [h for h in hallmarks_required if h not in text]
    assert not missing, (
        f"Protocol .tex lacks canonical sample-size formula hallmarks: {missing}. "
        f"Primary-source target: SPIRIT 2013 / CONSORT 2010 canonical form "
        f"n = 2σ²(z₁₋α + z₁₋β)² / Δ²."
    )
