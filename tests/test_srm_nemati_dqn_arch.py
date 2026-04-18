"""SR-M claim: nemati_dqn_arch

Falsification test for the Nemati 2016 DQN reimplementation claim.
Instantiates NematiDQN2016Baseline, asserts the action grid is exactly
{-4, -2, 0, +2, +4} U/kg/hr (5 actions) and — if torch is available —
that the Q-network is a 2-layer MLP with 64 hidden units and ReLU.
Also asserts the honesty caveat is present in the docstring.
"""

from __future__ import annotations

import pytest

from hemosim.agents.baselines_extended import (
    NematiDQN2016Baseline,
    _NEMATI_ACTION_DELTAS_UKGHR,
    _NEMATI_N_ACTIONS,
)


def test_srm_nemati_dqn_arch_structure_matches_description():
    """Action grid is exactly 5 actions {-4, -2, 0, +2, +4} U/kg/hr."""
    # Primary-source target: 5 actions, exact set {-4, -2, 0, +2, +4}.
    assert _NEMATI_N_ACTIONS == 5, (
        f"Nemati action grid has {_NEMATI_N_ACTIONS} actions; Nemati 2016 §II-C "
        f"specifies exactly 5. Claim cannot survive as 'faithful reimplementation'."
    )
    expected = {-4.0, -2.0, 0.0, 2.0, 4.0}
    actual = set(float(x) for x in _NEMATI_ACTION_DELTAS_UKGHR.tolist())
    assert actual == expected, (
        f"Nemati action deltas {actual} do not match primary-source target "
        f"{expected}. Registry: drop 'faithful' qualifier if this fails."
    )


def test_srm_nemati_dqn_arch_network_is_2layer_mlp_with_relu():
    """Q-network is a 2-layer MLP with ReLU activations (Nemati 2016 description)."""
    try:
        import torch  # noqa: F401
        from hemosim.agents.baselines_extended import _NematiQNetwork
    except ImportError:
        pytest.skip("torch not installed — architectural check requires torch")

    net = _NematiQNetwork()
    # The net.net Sequential should contain exactly 2 hidden Linear→ReLU pairs
    # plus an output Linear = 5 sub-modules: Linear, ReLU, Linear, ReLU, Linear.
    children = list(net.net.children())
    linear_layers = [m for m in children if m.__class__.__name__ == "Linear"]
    relu_layers = [m for m in children if m.__class__.__name__ == "ReLU"]
    # "2-layer MLP" = 2 hidden layers → 3 Linears total (2 hidden + 1 output).
    assert len(linear_layers) == 3, (
        f"Network has {len(linear_layers)} Linear layers; expected 3 (2 hidden "
        f"+ 1 output) for a 2-layer MLP per Nemati 2016."
    )
    assert len(relu_layers) == 2, (
        f"Network has {len(relu_layers)} ReLU activations; expected 2 for a "
        f"2-layer MLP with ReLU per Nemati 2016."
    )
    # Hidden width — registry primary-source target 64.
    assert linear_layers[0].out_features == 64, (
        f"First hidden layer width = {linear_layers[0].out_features}; "
        f"expected 64 (Nemati 2016 description-inferred, honesty caveat in docstring)."
    )
    assert linear_layers[-1].out_features == 5, (
        f"Output layer width = {linear_layers[-1].out_features}; expected 5 "
        f"to match the discrete action set."
    )


def test_srm_nemati_dqn_arch_honesty_caveat_in_docstring():
    """Registry requires the 'hidden width inferred' honesty caveat in docstring."""
    doc = NematiDQN2016Baseline.__doc__ or ""
    # Per registry: docstring must explicitly flag that hidden width and/or
    # activation are inferred from Nemati's written description. Tolerance
    # on exact wording — look for signal words.
    inferred_signals = ("inferred", "reproduced from the paper's written",
                        "does not specify", "approximated")
    assert any(s in doc.lower() for s in (x.lower() for x in inferred_signals)), (
        f"NematiDQN2016Baseline docstring lacks the honesty caveat about "
        f"inferred hidden-width / activation. Per registry this is a hard "
        f"requirement for the 'faithful reimplementation' claim to survive. "
        f"Doc excerpt: {doc[:400]!r}"
    )
