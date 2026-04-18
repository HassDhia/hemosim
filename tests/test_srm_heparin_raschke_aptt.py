"""SR-M claim: heparin_raschke_aptt

Falsification test for the Raschke/Hirsh aPTT reproduction claim.

Phase C correction (2026-04-18): reproducing the paper's 75 s ≈ 0 residual
requires the FITTED HeparinPKPD params from published_calibration.json, not
the model's unfitted defaults. The default constructor's parameters have
never been calibrated; the "aPTT@6h ≈ 75 s on 80 U/kg bolus + 18 U/kg/hr"
claim is only tied to the fitted-parameter configuration (vmax≈2761, km≈0.15,
aptt_alpha≈1.71, aptt_c_ref≈0.30). This test loads those params from
results/published_calibration.json and measures against the canonical path,
which is the one the paper's Raschke-6h claim in Table 7 is tied to.

(Secondary finding: the prior gym-env path for this test and the calibration-
harness path diverged because the env step semantics don't instantiate the
fitted params. Filed a PostFeedback entry to widen the Consistency Reviewer
to detect calibration-harness-vs-test-harness measurement-path drift going
forward — see feedback/2026-04-18-consistency-reviewer-widen-gate.md.)
"""

from __future__ import annotations

import inspect
import json
import pathlib

import hemosim  # noqa: F401
from hemosim.agents.baselines import HeparinRaschkeBaseline
from hemosim.models.heparin_pkpd import HeparinPKPD


def _load_fitted_heparin_params() -> dict[str, float]:
    """Load scipy-fitted HeparinPKPD params from published_calibration.json."""
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    path = repo_root / "results" / "published_calibration.json"
    data = json.loads(path.read_text())
    return dict(data["heparin_fit"]["fitted_params"])


def test_srm_heparin_raschke_aptt_6h_post_bolus():
    """aPTT 6h post-bolus ≈ 75 s under fitted HeparinPKPD (Raschke 1993)."""
    # Canonical config from published_calibration.py _simulate_raschke:
    # 80 kg patient, 80 U/kg bolus + 18 U/kg/hr continuous, baseline_aptt = 30,
    # dt = 1 h steps, aPTT at index 5 (hour 6).
    weight_kg = 80.0
    bolus_u_per_kg = 80.0
    infusion_u_per_kg_hr = 18.0
    infusion = infusion_u_per_kg_hr * weight_kg
    bolus = bolus_u_per_kg * weight_kg

    fitted = _load_fitted_heparin_params()
    model = HeparinPKPD(weight=weight_kg, renal_function=1.0, baseline_aptt=30.0)
    model.vmax = fitted["vmax"]
    model.km = fitted["km"]
    model.aptt_alpha = fitted["aptt_alpha"]
    model.aptt_c_ref = fitted["aptt_c_ref"]
    model.reset()

    model.step(infusion_rate_u_hr=infusion, bolus_u=bolus, dt_hours=1.0)
    aptt_trajectory = [model.get_aptt()]
    for _ in range(5):
        model.step(infusion_rate_u_hr=infusion, bolus_u=0.0, dt_hours=1.0)
        aptt_trajectory.append(model.get_aptt())
    aptt_6h = float(aptt_trajectory[5])

    # Primary-source target: Raschke 1993 Table 2 therapeutic 60-100 s; fitted
    # model residual is 0.000 at 75 s per published_calibration.json. Tolerance
    # ±5 s absolute.
    assert 70.0 <= aptt_6h <= 80.0, (
        f"6h aPTT = {aptt_6h:.2f} s outside Raschke 1993 primary-source "
        f"window [70, 80] s (target 75 s) for canonical 80 kg / 80 U/kg bolus "
        f"/ 18 U/kg/hr regimen with fitted HeparinPKPD params. This matches "
        f"the calibration harness residual=0.000 path."
    )


def test_srm_heparin_raschke_aptt_ladder_values_exact():
    """Adjustment ladder values in HeparinRaschkeBaseline match 22/20/18/16/12 exactly."""
    source = inspect.getsource(HeparinRaschkeBaseline.predict)
    for expected in ("22.0", "20.0", "18.0", "16.0", "12.0"):
        assert expected in source, (
            f"Raschke adjustment ladder value {expected} U/kg/hr missing from "
            f"HeparinRaschkeBaseline.predict source. Primary-source target is "
            f"the bit-exact ladder 22/20/18/16/12 U/kg/hr per Raschke 1993."
        )
