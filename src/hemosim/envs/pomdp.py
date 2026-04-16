"""POMDP infrastructure for hemosim (ISC-5).

Implements the clinical-fidelity critique raised by Dr. Nemati's pharmacist
reviewer: v0.1 envs expose perfect oracle lab values every timestep. Real
clinicians must ORDER labs, wait for the TAT, and interpret noisy readings.
This module provides the partial-observability substrate that the paper's
POMDP-formulated envs use.

Design
------

- ``LabSpec`` — static spec of a lab (name, TAT in minutes, CV%, reward cost).
- ``LabSample`` — a returned lab reading (value, ordered-at, returned-at).
- ``LabOrderQueue`` — tracks pending orders, applies noise on return, keeps
  history buffer. Ticks on the env's simulated clock.
- ``HEPARIN_LAB_SPECS`` / ``WARFARIN_LAB_SPECS`` / ``DOAC_LAB_SPECS`` /
  ``DIC_LAB_SPECS`` — realistic TAT and CV values cited to clinical sources.

Clinical references for TAT and CV defaults:
    Lippi G, Salvagno GL, Montagnana M, et al. Quality standards for sample
    processing, transportation and storage in hemostasis testing. Sem Thromb
    Hemost. 2012;38(6):565-575.
    Bowen RAR et al. Clin Chim Acta 2016 — hospital TAT distributions.
    Whitehead SJ et al. J Clin Pathol 2020 — POCT vs central-lab CV comparison.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class LabSpec:
    """Static specification of a lab test.

    Attributes
    ----------
    name
        Canonical lab key (e.g. ``"aptt"``).
    tat_minutes
        Mean turnaround time (order → result) in minutes.
    cv
        Analytical coefficient of variation (multiplicative noise applied
        at the moment of result return). 0.08 = 8%.
    cost_reward
        Per-order reward penalty. Small-magnitude negative number representing
        the cost-of-care of a lab draw (~$15-30 clinical proxy, scaled to
        hemosim reward units).
    """

    name: str
    tat_minutes: float
    cv: float
    cost_reward: float


@dataclass
class LabSample:
    """A returned lab reading held in the observation history buffer."""

    lab_name: str
    value: float
    ordered_at_hours: float
    returned_at_hours: float


@dataclass
class PendingLabOrder:
    """A lab order in flight, waiting for TAT to elapse."""

    lab_name: str
    ordered_at_hours: float
    returns_at_hours: float
    ground_truth_at_order: float


class LabOrderQueue:
    """Manages pending lab orders and delivers noisy results after TAT.

    The queue is ticked on the env clock; any orders whose ``returns_at_hours``
    have elapsed are sampled (ground truth at order time × (1 + N(0, cv))),
    moved from pending to history, and returned to the caller.

    History buffer is a bounded deque (default 32 samples) so observation
    tensors have a fixed upper bound.
    """

    def __init__(
        self,
        specs: dict[str, LabSpec],
        rng: np.random.Generator,
        history_maxlen: int = 32,
    ) -> None:
        if not specs:
            raise ValueError("specs must not be empty")
        self.specs = dict(specs)
        self._rng = rng
        self.pending: list[PendingLabOrder] = []
        self.history: deque[LabSample] = deque(maxlen=history_maxlen)

    def order(
        self,
        lab_name: str,
        current_hours: float,
        ground_truth_value: float,
    ) -> float:
        """Queue an order. Returns the reward cost (negative)."""
        if lab_name not in self.specs:
            raise KeyError(f"unknown lab {lab_name!r}; specs: {list(self.specs)}")
        spec = self.specs[lab_name]
        returns_at = current_hours + (spec.tat_minutes / 60.0)
        self.pending.append(
            PendingLabOrder(
                lab_name=lab_name,
                ordered_at_hours=float(current_hours),
                returns_at_hours=float(returns_at),
                ground_truth_at_order=float(ground_truth_value),
            )
        )
        return float(spec.cost_reward)

    def tick(self, current_hours: float) -> list[LabSample]:
        """Advance clock. Return any labs that just completed."""
        returned: list[LabSample] = []
        still_pending: list[PendingLabOrder] = []
        for order in self.pending:
            if order.returns_at_hours <= current_hours:
                spec = self.specs[order.lab_name]
                # Multiplicative Gaussian noise with clipping for positivity.
                noise_factor = max(0.1, 1.0 + float(self._rng.normal(0.0, spec.cv)))
                noisy_value = order.ground_truth_at_order * noise_factor
                sample = LabSample(
                    lab_name=order.lab_name,
                    value=float(noisy_value),
                    ordered_at_hours=order.ordered_at_hours,
                    returned_at_hours=float(current_hours),
                )
                returned.append(sample)
                self.history.append(sample)
            else:
                still_pending.append(order)
        self.pending = still_pending
        return returned

    def latest(self, lab_name: str) -> Optional[LabSample]:
        """Most recent returned sample for this lab, or None."""
        for sample in reversed(self.history):
            if sample.lab_name == lab_name:
                return sample
        return None

    def num_pending(self, lab_name: Optional[str] = None) -> int:
        """Count pending orders (optionally filtered by lab)."""
        if lab_name is None:
            return len(self.pending)
        return sum(1 for o in self.pending if o.lab_name == lab_name)

    def reset(self) -> None:
        """Clear all pending orders and history."""
        self.pending.clear()
        self.history.clear()


# ---------------------------------------------------------------------------
# Per-environment lab specs
# ---------------------------------------------------------------------------
#
# TAT and CV values are contemporary central-lab defaults in a tertiary ICU.
# For each value, the citation in the block comment should be traceable to
# the paper's Related Work / Methods section.

# ---- Heparin (unfractionated) -------------------------------------------
#
# aPTT: central-lab TAT 30-60 min; analytical CV 5-10% (median ~8%).
#   Bowen RAR Clin Chim Acta 2016.
# Anti-Xa: TAT 2-4 h (batched in many labs); CV 10-15%.
#   Greinacher A et al. Semin Thromb Hemost 2019.
# Platelets (CBC): TAT 30-60 min; CV 3-5%.
#   Whitehead SJ J Clin Pathol 2020.
HEPARIN_LAB_SPECS: dict[str, LabSpec] = {
    "aptt":     LabSpec("aptt",     tat_minutes=45.0,  cv=0.08, cost_reward=-0.10),
    "anti_xa":  LabSpec("anti_xa",  tat_minutes=180.0, cv=0.12, cost_reward=-0.20),
    "platelets":LabSpec("platelets",tat_minutes=45.0,  cv=0.05, cost_reward=-0.10),
}

# ---- Warfarin ------------------------------------------------------------
#
# INR / PT: TAT 30-60 min, CV 4-8%.
#   Van den Besselaar Thromb Res 2010; Lippi G Sem Thromb Hemost 2012.
WARFARIN_LAB_SPECS: dict[str, LabSpec] = {
    "inr":LabSpec("inr",tat_minutes=45.0,cv=0.06,cost_reward=-0.10),
    "pt": LabSpec("pt", tat_minutes=45.0,cv=0.06,cost_reward=-0.10),
}

# ---- DOAC ---------------------------------------------------------------
#
# CrCl requires creatinine + demographics; central-lab TAT 30-90 min; CV 4-8%.
# DOAC-calibrated anti-Xa: TAT 120-240 min, CV 10-15%.
DOAC_LAB_SPECS: dict[str, LabSpec] = {
    "crcl":       LabSpec("crcl",      tat_minutes=60.0,  cv=0.07, cost_reward=-0.10),
    "doac_anti_xa":LabSpec("doac_anti_xa", tat_minutes=180.0, cv=0.12, cost_reward=-0.25),
}

# ---- DIC ----------------------------------------------------------------
#
# Full ICU DIC panel: platelets (CBC), fibrinogen, PT, D-dimer.
#   Platelets: TAT 30-60 min, CV 3-5%.
#   Fibrinogen (Clauss): TAT 45-90 min, CV 5-10%.
#   PT: TAT 30-60 min, CV 4-8%.
#   D-dimer: TAT 45-120 min, CV 10-20% (more assay-dependent).
DIC_LAB_SPECS: dict[str, LabSpec] = {
    "platelets": LabSpec("platelets", tat_minutes=45.0, cv=0.05, cost_reward=-0.10),
    "fibrinogen":LabSpec("fibrinogen",tat_minutes=60.0, cv=0.08, cost_reward=-0.15),
    "pt":        LabSpec("pt",        tat_minutes=45.0, cv=0.06, cost_reward=-0.10),
    "d_dimer":   LabSpec("d_dimer",   tat_minutes=90.0, cv=0.15, cost_reward=-0.20),
}


__all__ = [
    "LabSpec",
    "LabSample",
    "PendingLabOrder",
    "LabOrderQueue",
    "HEPARIN_LAB_SPECS",
    "WARFARIN_LAB_SPECS",
    "DOAC_LAB_SPECS",
    "DIC_LAB_SPECS",
]
