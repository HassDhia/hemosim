"""Clinical outcomes metrics for anticoagulation-therapy simulations.

This module implements six pure, env-independent functions that score
a simulated patient episode against standard clinical endpoints used
in the anticoagulation literature:

* :func:`time_in_therapeutic_range` — Rosendaal linear-interpolation TTR.
* :func:`isth_major_bleeding` — ISTH 2005 major bleeding definition.
* :func:`thromboembolic_events` — aggregator for stroke/DVT/PE/systemic embolism.
* :func:`hit_4t_score` — Warkentin 4T pretest score for heparin-induced
  thrombocytopenia.
* :func:`mortality_proxy` — composite logistic combining organ dysfunction,
  hemorrhage severity, active thrombosis, and exposure duration. Flagged
  as a *proxy* — calibration against real ICU mortality (e.g. APACHE-II or
  MIMIC-IV survival) is explicit future work.
* :func:`patient_outcome_summary` — convenience wrapper that takes a
  sequence of per-step ``info`` dicts from an env rollout and returns
  all of the above where applicable.

The functions are intentionally decoupled from any gymnasium env. Wiring
into ``step(...) -> info`` happens in a later ISC; this module is the
vocabulary, and evaluation scripts / baselines / RL agents can call it
directly on a rollout.

References
----------
* Rosendaal FR, Cannegieter SC, van der Meer FJ, Briët E. A method to
  determine the optimal intensity of oral anticoagulant therapy.
  Thromb Haemost. 1993;69(3):236–239.
* Schulman S, Kearon C; Subcommittee on Control of Anticoagulation,
  SSC of the ISTH. Definition of major bleeding in clinical investigations
  of antihemostatic medicinal products in non-surgical patients.
  J Thromb Haemost. 2005;3(4):692–694.
* Warkentin TE. Heparin-induced thrombocytopenia: pathogenesis and
  management. Br J Haematol. 2003;121(4):535–555. (4T scoring system
  formalized in Lo GK, Juhl D, Warkentin TE, et al. Evaluation of pretest
  clinical score (4 T's) for the diagnosis of heparin-induced
  thrombocytopenia in two clinical settings. J Thromb Haemost. 2006;4:759–765.)
* Knaus WA, Draper EA, Wagner DP, Zimmerman JE. APACHE II: a severity of
  disease classification system. Crit Care Med. 1985;13(10):818–829.
  (Conceptual anchor for :func:`mortality_proxy`; hemosim's proxy is
  not calibrated against APACHE-II — see docstring.)
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

# Hours to years for rate-per-patient-years conversions.
_HOURS_PER_YEAR = 24.0 * 365.25


# ---------------------------------------------------------------------------
# Rosendaal time-in-therapeutic-range
# ---------------------------------------------------------------------------


def time_in_therapeutic_range(
    values: Sequence[float],
    times_hours: Sequence[float],
    low: float,
    high: float,
) -> float:
    """Rosendaal linear-interpolation time-in-therapeutic-range.

    Between consecutive measurements the true assay trajectory is assumed
    piecewise-linear. The fraction of elapsed time during which the
    interpolated curve lies in ``[low, high]`` inclusive is returned.

    This is the standard warfarin TTR metric used throughout the
    long-term-anticoagulation literature.

    Parameters
    ----------
    values
        Measured assay values (e.g. INR). Must be the same length as
        ``times_hours``.
    times_hours
        Time of each measurement in hours from the start of the episode.
        Must be monotonically non-decreasing.
    low, high
        Therapeutic-range bounds (same units as ``values``). Must satisfy
        ``low < high``.

    Returns
    -------
    float
        TTR in ``[0.0, 1.0]``. Returns 0.0 for an empty input.

    Raises
    ------
    ValueError
        If ``values`` and ``times_hours`` have different lengths or if
        ``low >= high``.

    References
    ----------
    Rosendaal FR et al. Thromb Haemost 1993;69:236–239.

    Examples
    --------
    >>> time_in_therapeutic_range([2.0, 2.5, 2.8], [0.0, 24.0, 48.0], 2.0, 3.0)
    1.0
    >>> time_in_therapeutic_range([1.0, 3.0], [0.0, 24.0], 2.0, 3.0)
    0.5
    """
    if len(values) != len(times_hours):
        raise ValueError(
            f"values and times_hours must have equal length "
            f"(got {len(values)} vs {len(times_hours)})"
        )
    if low >= high:
        raise ValueError(f"low must be < high (got low={low}, high={high})")

    n = len(values)
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0 if low <= values[0] <= high else 0.0

    total_time = 0.0
    in_range_time = 0.0
    for i in range(n - 1):
        t1, t2 = float(times_hours[i]), float(times_hours[i + 1])
        v1, v2 = float(values[i]), float(values[i + 1])
        dt = t2 - t1
        if dt < 0:
            raise ValueError("times_hours must be monotonically non-decreasing")
        if dt == 0.0:
            continue
        total_time += dt
        in_range_time += _segment_in_range(v1, v2, low, high) * dt

    if total_time == 0.0:
        # All samples share a single timestamp; fall back to indicator.
        return 1.0 if low <= float(values[-1]) <= high else 0.0

    return in_range_time / total_time


def _segment_in_range(v1: float, v2: float, low: float, high: float) -> float:
    """Fraction of a unit-length segment whose linear interpolation is in [low, high].

    The segment is parameterised by ``s in [0, 1]`` with
    ``f(s) = v1 + (v2 - v1) * s``. Returns ``|{s in [0,1] : low <= f(s) <= high}|``.
    """
    # If the segment is constant, it is in range iff its single value is.
    if v1 == v2:
        return 1.0 if low <= v1 <= high else 0.0

    # Find the fraction of [0, 1] where v1 + (v2 - v1) * s is in [low, high].
    # This is the intersection of two half-planes: f(s) >= low and f(s) <= high.
    dv = v2 - v1

    def _s_at(target: float) -> float:
        return (target - v1) / dv

    # s where f(s) = low, f(s) = high
    s_low = _s_at(low)
    s_high = _s_at(high)

    # The set f(s) in [low, high] corresponds to s between min(s_low, s_high)
    # and max(s_low, s_high), intersected with [0, 1].
    s_enter = min(s_low, s_high)
    s_exit = max(s_low, s_high)
    s_enter = max(s_enter, 0.0)
    s_exit = min(s_exit, 1.0)
    if s_exit <= s_enter:
        return 0.0
    return s_exit - s_enter


# ---------------------------------------------------------------------------
# ISTH 2005 major bleeding
# ---------------------------------------------------------------------------


_ISTH_HB_DROP_G_DL_THRESHOLD = 2.0  # equivalent to 20 g/L
_ISTH_UNITS_TRANSFUSED_THRESHOLD = 2


def isth_major_bleeding(
    events: Sequence[Mapping[str, Any]],
    patient_years: float | None = None,
) -> dict:
    """Classify bleeding events against the ISTH 2005 major-bleeding definition.

    An event qualifies as major bleeding if **any** of the following hold:

    * ``fatal`` is True, or
    * ``critical_site`` is True (intracranial, intraspinal, intraocular,
      retroperitoneal, intra-articular, pericardial, or intramuscular with
      compartment syndrome), or
    * ``hb_drop_g_dl`` >= 2.0 g/dL  (equivalent to the ISTH threshold of 20 g/L), or
    * ``units_transfused`` >= 2 (whole blood or packed red cells).

    Parameters
    ----------
    events
        Sequence of event dicts with keys
        ``{type, hb_drop_g_dl, units_transfused, critical_site, fatal}``.
        Missing keys default to a non-qualifying value (0 / False).
    patient_years
        Cumulative patient-years of exposure used to compute the rate.
        If ``None`` or ``<= 0``, ``rate_per_100_patient_years`` is 0.0
        when ``count`` is 0 and ``float('inf')`` when ``count`` > 0
        (undefined rate over zero exposure).

    Returns
    -------
    dict
        ``{"count": int, "rate_per_100_patient_years": float,
           "events": list[dict]}`` where ``events`` contains only the
        qualifying events, each annotated with ``"criteria_met"``.

    References
    ----------
    Schulman S, Kearon C. J Thromb Haemost 2005;3:692–694.
    """
    qualifying: list[dict] = []
    for ev in events:
        criteria: list[str] = []
        if ev.get("fatal", False):
            criteria.append("fatal")
        if ev.get("critical_site", False):
            criteria.append("critical_site")
        if float(ev.get("hb_drop_g_dl", 0.0)) >= _ISTH_HB_DROP_G_DL_THRESHOLD:
            criteria.append("hb_drop_>=2_g_dl")
        if int(ev.get("units_transfused", 0)) >= _ISTH_UNITS_TRANSFUSED_THRESHOLD:
            criteria.append("transfusion_>=2_units")

        if criteria:
            annotated = dict(ev)
            annotated["criteria_met"] = criteria
            qualifying.append(annotated)

    count = len(qualifying)
    rate = _rate_per_100_patient_years(count, patient_years)
    return {
        "count": count,
        "rate_per_100_patient_years": rate,
        "events": qualifying,
    }


# ---------------------------------------------------------------------------
# Thromboembolic events aggregator
# ---------------------------------------------------------------------------


def thromboembolic_events(
    events: Sequence[Mapping[str, Any]],
    patient_years: float | None = None,
) -> dict:
    """Aggregate thromboembolic events and compute a per-100-patient-years rate.

    This function treats every event whose ``type`` appears in the input
    as a thromboembolic endpoint. Typical clinical categories are
    ``stroke``, ``TIA``, ``DVT``, ``PE``, and ``systemic_embolism``, but
    the function is type-agnostic — it simply tallies by the ``type``
    field so that env-specific categories can be added without changing
    the implementation.

    Parameters
    ----------
    events
        Sequence of event dicts. The only required key is ``"type"``
        (string). Other keys are ignored.
    patient_years
        Cumulative patient-years of exposure. Same semantics as in
        :func:`isth_major_bleeding`.

    Returns
    -------
    dict
        ``{"count": int, "rate_per_100_patient_years": float,
           "by_type": dict[str, int]}``.
    """
    by_type: Counter[str] = Counter()
    for ev in events:
        event_type = str(ev.get("type", "unknown"))
        by_type[event_type] += 1
    count = int(sum(by_type.values()))
    rate = _rate_per_100_patient_years(count, patient_years)
    return {
        "count": count,
        "rate_per_100_patient_years": rate,
        "by_type": dict(by_type),
    }


def _rate_per_100_patient_years(count: int, patient_years: float | None) -> float:
    """Helper: events per 100 patient-years with defined zero-exposure handling."""
    if patient_years is None or patient_years <= 0:
        if count == 0:
            return 0.0
        return float("inf")
    return (count / patient_years) * 100.0


# ---------------------------------------------------------------------------
# Warkentin 4T HIT score
# ---------------------------------------------------------------------------


# Valid categorical inputs for the "other causes" component.
_OTHER_CAUSE_POINTS = {
    "none": 2,       # "no other obvious cause" (Warkentin original wording)
    "possible": 1,
    "definite": 0,
}


def hit_4t_score(
    platelet_trajectory: Sequence[float],
    timing_days_from_heparin: float,
    other_cause: str,
    thrombosis: bool,
) -> dict:
    """Warkentin 4T pretest probability score for heparin-induced thrombocytopenia.

    The 4T score sums four categorical components, each scoring 0, 1, or 2
    points:

    1. **Thrombocytopenia** (platelet fall):

       * >50% fall **and** nadir >= 20 x10^9/L  -> 2 pts
       * 30–50% fall **or**  nadir 10–19 x10^9/L -> 1 pt
       * <30% fall **or** nadir <10             -> 0 pts

    2. **Timing of platelet fall** (days since heparin exposure):

       * Clear onset day 5–10 (or <=1 day with heparin exposure within the
         past 30 days) -> 2 pts
       * Consistent with day 5–10 but unclear, onset after day 10, or fall
         <=1 day with prior heparin exposure 30–100 days ago -> 1 pt
       * Platelet fall within 4 days without recent exposure -> 0 pts

       hemosim exposes ``timing_days_from_heparin`` as a single scalar; the
       mapping used here is 5 <= t <= 10 -> 2 pts, 10 < t <= 14 or equivalent
       consistent range -> 1 pt, otherwise 0 pts. "Prior exposure" is not
       modelled — extend ``timing_days_from_heparin`` with a composite if
       that feature becomes relevant.

    3. **Thrombosis or other sequelae**:

       * ``thrombosis == True``  -> 2 pts (new confirmed thrombosis,
         skin necrosis, or acute systemic reaction post-bolus)
       * ``thrombosis == False`` -> 0 pts (progressive / silent thrombosis
         is not modelled at this granularity)

    4. **Other causes of thrombocytopenia** (mapped from ``other_cause``):

       * ``"none"``     -> 2 pts
       * ``"possible"`` -> 1 pt
       * ``"definite"`` -> 0 pts

    Risk categories (from Lo GK et al. 2006):

    * 0–3 pts: **low**
    * 4–5 pts: **intermediate**
    * 6–8 pts: **high**

    Parameters
    ----------
    platelet_trajectory
        Platelet counts over the HIT evaluation window, in units of
        10^9/L (i.e. typical values are 100–300). The first element is
        treated as baseline. An empty sequence scores 0 for the
        thrombocytopenia component.
    timing_days_from_heparin
        Days from heparin initiation to the first platelet fall.
    other_cause
        One of ``"none"``, ``"possible"``, ``"definite"``.
    thrombosis
        Whether a thrombotic complication (confirmed new thrombosis,
        skin necrosis, or acute systemic reaction) occurred.

    Returns
    -------
    dict
        ``{"total_score": int, "risk_category": str, "components": dict}``
        where ``components`` reports the per-category scores.

    References
    ----------
    Warkentin TE. Br J Haematol 2003;121:535–555. Lo GK, Juhl D,
    Warkentin TE et al. J Thromb Haemost 2006;4:759–765.
    """
    if other_cause not in _OTHER_CAUSE_POINTS:
        raise ValueError(
            f"other_cause must be one of {sorted(_OTHER_CAUSE_POINTS)}, "
            f"got {other_cause!r}"
        )

    thrombocytopenia_pts = _score_thrombocytopenia(platelet_trajectory)
    timing_pts = _score_timing(timing_days_from_heparin)
    thrombosis_pts = 2 if thrombosis else 0
    other_pts = _OTHER_CAUSE_POINTS[other_cause]

    total = thrombocytopenia_pts + timing_pts + thrombosis_pts + other_pts
    if total <= 3:
        category = "low"
    elif total <= 5:
        category = "intermediate"
    else:
        category = "high"

    return {
        "total_score": total,
        "risk_category": category,
        "components": {
            "thrombocytopenia": thrombocytopenia_pts,
            "timing": timing_pts,
            "thrombosis": thrombosis_pts,
            "other_causes": other_pts,
        },
    }


def _score_thrombocytopenia(trajectory: Sequence[float]) -> int:
    """Return 0/1/2 for the thrombocytopenia 4T component."""
    if len(trajectory) == 0:
        return 0
    baseline = float(trajectory[0])
    nadir = float(min(trajectory))
    if baseline <= 0:
        # Degenerate baseline; fall back to nadir-only scoring.
        if nadir < 10:
            return 0
        if nadir < 20:
            return 1
        return 0
    pct_fall = (baseline - nadir) / baseline * 100.0

    # "2 pts" requires both a >50% fall and nadir >= 20.
    if pct_fall > 50.0 and nadir >= 20.0:
        return 2
    # "1 pt" is satisfied by either 30-50% fall OR nadir in [10, 20).
    if (30.0 <= pct_fall <= 50.0) or (10.0 <= nadir < 20.0):
        return 1
    return 0


def _score_timing(days: float) -> int:
    """Return 0/1/2 for the timing 4T component (no-prior-exposure mapping)."""
    # 5-10 day window is the classic HIT signal.
    if 5.0 <= days <= 10.0:
        return 2
    # After day 10 the picture is "consistent with but unclear" — 1 pt.
    if 10.0 < days <= 14.0:
        return 1
    # <4 days without prior exposure is the 0-pt bucket.
    return 0


# ---------------------------------------------------------------------------
# Mortality proxy
# ---------------------------------------------------------------------------


# Logistic coefficients for the mortality proxy.
#
# NOTE: These coefficients are *not* fitted to real ICU outcomes; they are
# hand-set so that (a) healthy states (organ_function=1, no hemorrhage, no
# thrombosis) map to mortality <5%, (b) severe states (organ_function~0.1,
# hemorrhage ~1.0, thrombosis, week-long exposure) map to mortality >95%,
# and (c) the three clinical inputs each monotonically increase risk.
# Calibration against APACHE-II and/or MIMIC-IV survival is explicit future
# work (ISC-7, MIMIC-IV calibration scaffold).
_MORTALITY_INTERCEPT = -4.0
_MORTALITY_BETA_ORGAN = 3.0         # coefficient on (1 - organ_function)
_MORTALITY_BETA_HEMORRHAGE = 4.0    # coefficient on hemorrhage_severity
_MORTALITY_BETA_THROMBOSIS = 1.5    # coefficient on thrombosis indicator
_MORTALITY_BETA_DURATION = 0.003    # coefficient on duration_hours


def mortality_proxy(
    organ_function: float,
    hemorrhage_severity: float,
    thrombosis: bool,
    duration_hours: float,
) -> float:
    """Composite mortality-probability proxy for a simulated episode.

    **This is a proxy, not a validated mortality model.** The coefficients
    below are hand-set to produce physiologically-plausible probabilities
    (healthy ICU patient ~2%, severe DIC-class patient ~99%) and are *not*
    fitted to real outcome data. Calibration against a published ICU
    mortality model such as APACHE-II (Knaus et al., Crit Care Med 1985)
    or a real MIMIC-IV survival cohort is explicit future work (see
    ISC-7, ``src/hemosim/validation/mimic_calibration.py``).

    The proxy is a logistic of four inputs::

        logit(p) = -4.0
                 + 3.0 * (1 - organ_function)
                 + 4.0 * hemorrhage_severity
                 + 1.5 * thrombosis_indicator
                 + 0.003 * duration_hours

    Parameters
    ----------
    organ_function
        Aggregate organ-function score in ``[0, 1]`` where 1.0 is normal
        and 0.0 is total multi-organ failure. Clipped to ``[0, 1]``.
    hemorrhage_severity
        Bleeding severity in ``[0, 1]`` where 0 is no bleeding and 1 is
        life-threatening hemorrhage. Clipped to ``[0, 1]``.
    thrombosis
        True if an active thrombotic event occurred during the episode.
    duration_hours
        Duration of the episode in hours. Longer exposure to a high-risk
        state increases modelled mortality very slightly.

    Returns
    -------
    float
        Mortality probability in ``[0, 1]``.

    References
    ----------
    Knaus WA, Draper EA, Wagner DP, Zimmerman JE. APACHE II: a severity of
    disease classification system. Crit Care Med. 1985;13(10):818–829.
    (Conceptual anchor only — no coefficients borrowed.)
    """
    organ = _clip(float(organ_function), 0.0, 1.0)
    hemorrhage = _clip(float(hemorrhage_severity), 0.0, 1.0)
    duration = max(float(duration_hours), 0.0)

    logit = (
        _MORTALITY_INTERCEPT
        + _MORTALITY_BETA_ORGAN * (1.0 - organ)
        + _MORTALITY_BETA_HEMORRHAGE * hemorrhage
        + _MORTALITY_BETA_THROMBOSIS * (1.0 if thrombosis else 0.0)
        + _MORTALITY_BETA_DURATION * duration
    )
    return _sigmoid(logit)


def _sigmoid(x: float) -> float:
    """Numerically-stable logistic sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _clip(x: float, low: float, high: float) -> float:
    if x < low:
        return low
    if x > high:
        return high
    return x


# ---------------------------------------------------------------------------
# Episode-level summary
# ---------------------------------------------------------------------------


def patient_outcome_summary(info_trajectory: Sequence[Mapping[str, Any]]) -> dict:
    """Compute all clinical outcome metrics from a sequence of per-step info dicts.

    This is the function that envs (or evaluation scripts) will eventually
    call on the list of ``info`` dicts returned by a rollout of
    ``env.step(...)``. Only keys that are present are used; the function
    tolerates partial instrumentation so that each env can adopt the
    clinical vocabulary at its own pace.

    Recognised keys per step (all optional):

    * ``time_hours``: elapsed time since episode start (hours)
    * ``inr``, ``inr_low``, ``inr_high``: INR value and therapeutic bounds
    * ``aptt``, ``aptt_low``, ``aptt_high``: aPTT value and therapeutic bounds
    * ``bleeding_events``: list of ISTH-style event dicts for this step
    * ``thromboembolic_events``: list of thromboembolic event dicts
    * ``platelets``: platelet count for 4T-style tracking
    * ``organ_function``, ``hemorrhage_severity``, ``active_thrombosis``:
      final-state inputs for :func:`mortality_proxy` (taken from the last
      step).

    Parameters
    ----------
    info_trajectory
        Sequence of ``info`` dicts, one per env step.

    Returns
    -------
    dict
        ``{"ttr": dict, "major_bleeding": dict, "thromboembolic": dict,
           "mortality_proxy": float, "duration_hours": float}``
    """
    if len(info_trajectory) == 0:
        return {
            "ttr": {},
            "major_bleeding": {
                "count": 0, "rate_per_100_patient_years": 0.0, "events": [],
            },
            "thromboembolic": {
                "count": 0, "rate_per_100_patient_years": 0.0, "by_type": {},
            },
            "mortality_proxy": 0.0,
            "duration_hours": 0.0,
        }

    times = [float(step.get("time_hours", i)) for i, step in enumerate(info_trajectory)]
    duration_hours = max(times[-1] - times[0], 0.0)
    patient_years = duration_hours / _HOURS_PER_YEAR if duration_hours > 0 else None

    ttr_results: dict[str, float] = {}
    ttr_results.update(
        _ttr_for_key(info_trajectory, times, "inr", "inr_low", "inr_high")
    )
    ttr_results.update(
        _ttr_for_key(info_trajectory, times, "aptt", "aptt_low", "aptt_high")
    )

    bleeding_events = _flatten(info_trajectory, "bleeding_events")
    major = isth_major_bleeding(bleeding_events, patient_years=patient_years)

    te_events = _flatten(info_trajectory, "thromboembolic_events")
    thromboembolic = thromboembolic_events(te_events, patient_years=patient_years)

    last = info_trajectory[-1]
    mortality = mortality_proxy(
        organ_function=float(last.get("organ_function", 1.0)),
        hemorrhage_severity=float(last.get("hemorrhage_severity", 0.0)),
        thrombosis=bool(last.get("active_thrombosis", False)),
        duration_hours=duration_hours,
    )

    return {
        "ttr": ttr_results,
        "major_bleeding": major,
        "thromboembolic": thromboembolic,
        "mortality_proxy": mortality,
        "duration_hours": duration_hours,
    }


def _ttr_for_key(
    trajectory: Sequence[Mapping[str, Any]],
    times: Sequence[float],
    value_key: str,
    low_key: str,
    high_key: str,
) -> dict[str, float]:
    """Compute TTR for an assay if every step exposes a value + bounds."""
    values: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for step in trajectory:
        if value_key not in step or low_key not in step or high_key not in step:
            return {}
        values.append(float(step[value_key]))
        lows.append(float(step[low_key]))
        highs.append(float(step[high_key]))
    if len(values) == 0:
        return {}
    # Use the most common range if bounds drift; clinical ranges are
    # typically constant per-patient.
    low = lows[0]
    high = highs[0]
    return {value_key: time_in_therapeutic_range(values, times, low, high)}


def _flatten(
    trajectory: Sequence[Mapping[str, Any]],
    key: str,
) -> list[Mapping[str, Any]]:
    """Concatenate per-step event lists stored under ``key``."""
    out: list[Mapping[str, Any]] = []
    for step in trajectory:
        events: Iterable[Mapping[str, Any]] = step.get(key, []) or []
        for ev in events:
            out.append(ev)
    return out
