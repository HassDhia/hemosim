"""Clinical outcomes metrics for hemosim environments.

This package exposes pure, env-independent functions that score a patient
episode against standard anticoagulation-therapy clinical endpoints
(time-in-therapeutic-range, ISTH major bleeding, thromboembolic events,
HIT 4T score, mortality proxy, and a per-episode summary).

The public surface is intentionally functional — envs, evaluation scripts,
and analysis notebooks all import these functions directly. Wiring into
``env.step(...) -> info`` happens in a later workstream (ISC-5 and beyond);
this package only defines the vocabulary.
"""

from hemosim.metrics.clinical import (
    hit_4t_score,
    isth_major_bleeding,
    mortality_proxy,
    patient_outcome_summary,
    thromboembolic_events,
    time_in_therapeutic_range,
)

__all__ = [
    "hit_4t_score",
    "isth_major_bleeding",
    "mortality_proxy",
    "patient_outcome_summary",
    "thromboembolic_events",
    "time_in_therapeutic_range",
]
