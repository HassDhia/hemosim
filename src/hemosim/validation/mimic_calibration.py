"""Individual-patient-level PK/PD calibration against MIMIC-IV heparin cohorts (ISC-7).

This module is the Phase-2-collaboration scaffold for hemosim. Its purpose
is to make the pipeline *exist end-to-end* before we have real MIMIC-IV
data in hand:

1. A `MIMICHeparinCohort` dataclass that carries exactly the per-patient
   time series a credentialed MIMIC-IV extract provides (weight, renal
   function, aPTT observations, heparin rate/bolus history).
2. A `synthetic_cohort(...)` generator that produces a **clearly labeled**
   synthetic cohort with MIMIC-IV-shaped statistics so the harness is
   unit-testable and exercisable in `reproduce.sh` without credentialed
   access.
3. `calibrate_heparin_pkpd(...)` — a `scipy.optimize`-based fit loop over
   the tunable PK/PD parameters (`vmax`, `km`, `aptt_alpha`,
   `aptt_c_ref`) minimizing the sum-of-squared-residuals between
   simulated aPTT and observed aPTT per patient.
4. A `CalibrationResult` dataclass that serializes cleanly to JSON and
   renders a publication-ready Markdown report.

**Honest scope statement.** The *real* MIMIC-IV fit requires PhysioNet
credentialed access and a Data Use Agreement. What ships in this module
is the scaffold. Running `--synthetic` produces synthetic numbers that
are useful for verifying the pipeline shape but must **not** be quoted
in the paper as if they came from real MIMIC-IV. All reports written by
this module stamp `data_source` so that provenance is unambiguous.

Dataset citation (when real data are used):

    Johnson AEW, Bulgarelli L, Shen L, Gayles A, Shammout A, Horng S,
    Pollard TJ, Hao S, Moody B, Gow B, Lehman L-H, Celi LA, Mark RG.
    MIMIC-IV, a freely accessible electronic health record dataset.
    Scientific Data 10, 1 (2023). doi:10.1038/s41597-022-01899-x

See `mimic_schema.md` (colocated) for the exact cohort-extraction query.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from hemosim.models.heparin_pkpd import HeparinPKPD

# ---------------------------------------------------------------------------
# MIMICHeparinCohort
# ---------------------------------------------------------------------------

# Columns the CSV contract exposes (see mimic_schema.md §4).
_CSV_COLUMNS: tuple[str, ...] = (
    "subject_id",
    "time_hours",
    "aptt",
    "current_rate_u_per_hr",
    "bolus_in_last_hour_u",
    "weight_kg",
    "crcl_recent",
    "baseline_aptt",
)

# Nominal healthy CrCl used when the extract reports missing/empty.
_DEFAULT_CRCL = 90.0


@dataclass
class MIMICHeparinCohort:
    """One patient's heparin trajectory, aligned for calibration.

    All time series are **parallel** — index `i` across `time_hours`,
    `aptt_observations`, `heparin_rate_u_per_hr`, and `heparin_bolus_u`
    refers to the same observation event, exactly as prepared by the
    MIMIC export described in `mimic_schema.md`.

    Attributes:
        patient_id: Opaque subject identifier (the MIMIC `subject_id`
            for real data, an arbitrary string for synthetic rows).
        weight_kg: Admission weight in kilograms.
        renal_function: Scalar fraction in (0, 1.5] — 1.0 is normal.
            Derived from Cockcroft-Gault CrCl / 90 mL/min, clipped to
            the HeparinPKPD PK model's valid range.
        baseline_aptt: Pre-heparin aPTT in seconds (or the first
            observed aPTT if no pre-drug draw exists).
        time_hours: Monotonically ascending times (hours since cohort
            `t0`) at which aPTT observations are available.
        aptt_observations: Observed aPTT values in seconds, one per
            entry in `time_hours`.
        heparin_rate_u_per_hr: Active continuous infusion rate (U/hr)
            at each observation time; `0.0` if no infusion active.
        heparin_bolus_u: Heparin boluses (Units) delivered in the hour
            leading up to each observation time; `0.0` if none.
    """

    patient_id: str
    weight_kg: float
    renal_function: float
    baseline_aptt: float
    time_hours: list[float]
    aptt_observations: list[float]
    heparin_rate_u_per_hr: list[float]
    heparin_bolus_u: list[float]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        n = len(self.time_hours)
        if n == 0:
            raise ValueError(
                f"MIMICHeparinCohort({self.patient_id!r}): need at least one "
                "observation; received an empty trajectory."
            )
        for name, series in (
            ("aptt_observations", self.aptt_observations),
            ("heparin_rate_u_per_hr", self.heparin_rate_u_per_hr),
            ("heparin_bolus_u", self.heparin_bolus_u),
        ):
            if len(series) != n:
                raise ValueError(
                    f"MIMICHeparinCohort({self.patient_id!r}): {name} has "
                    f"length {len(series)} but time_hours has length {n}."
                )
        # Monotone time axis — same assumption the optimizer makes so
        # that forward-simulating in step order gives a meaningful aPTT.
        times = np.asarray(self.time_hours, dtype=float)
        if np.any(np.diff(times) < 0):
            raise ValueError(
                f"MIMICHeparinCohort({self.patient_id!r}): time_hours must "
                "be monotonically non-decreasing."
            )
        if self.weight_kg <= 0:
            raise ValueError(
                f"MIMICHeparinCohort({self.patient_id!r}): weight_kg must "
                f"be positive, got {self.weight_kg!r}."
            )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(cls, path: str | Path) -> list["MIMICHeparinCohort"]:
        """Load a list of cohort objects from a MIMIC-IV export CSV.

        The CSV shape is documented in `mimic_schema.md` §4. This
        classmethod groups rows by `subject_id` (stable encounter order
        preserved), validates required columns, and coerces types.

        Empty `crcl_recent` values are interpreted as "unknown" and
        mapped to normal renal function (1.0). Empty `baseline_aptt`
        falls back to the first observed aPTT.

        Parameters
        ----------
        path : str or pathlib.Path
            CSV file on disk. UTF-8 with header row.

        Returns
        -------
        list[MIMICHeparinCohort]
            One entry per unique subject, ordered by first appearance.
        """
        csv_path = Path(path)
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        return cls._from_row_dicts(rows)

    @classmethod
    def _from_row_dicts(
        cls, rows: list[dict[str, str]]
    ) -> list["MIMICHeparinCohort"]:
        """Internal shared path between `from_csv` and in-memory tests."""
        if not rows:
            return []
        missing = [c for c in _CSV_COLUMNS if c not in rows[0]]
        if missing:
            raise KeyError(
                f"MIMIC CSV missing required columns: {missing!r}. "
                f"Required columns: {list(_CSV_COLUMNS)!r}."
            )

        # Preserve first-appearance order of subjects.
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            grouped.setdefault(row["subject_id"], []).append(row)

        cohorts: list[MIMICHeparinCohort] = []
        for subject_id, subject_rows in grouped.items():
            subject_rows.sort(key=lambda r: float(r["time_hours"]))
            weight_kg = float(subject_rows[0]["weight_kg"])
            # Parse a maybe-empty float.
            crcl_raw = subject_rows[0].get("crcl_recent", "").strip()
            if crcl_raw:
                crcl = float(crcl_raw)
                renal_function = float(np.clip(crcl / _DEFAULT_CRCL, 0.1, 1.5))
            else:
                renal_function = 1.0

            baseline_raw = subject_rows[0].get("baseline_aptt", "").strip()
            if baseline_raw:
                baseline_aptt = float(baseline_raw)
            else:
                baseline_aptt = float(subject_rows[0]["aptt"])

            times: list[float] = []
            aptts: list[float] = []
            rates: list[float] = []
            boluses: list[float] = []
            for r in subject_rows:
                times.append(float(r["time_hours"]))
                aptts.append(float(r["aptt"]))
                rates.append(float(r["current_rate_u_per_hr"] or 0.0))
                boluses.append(float(r["bolus_in_last_hour_u"] or 0.0))

            cohorts.append(
                cls(
                    patient_id=str(subject_id),
                    weight_kg=weight_kg,
                    renal_function=renal_function,
                    baseline_aptt=baseline_aptt,
                    time_hours=times,
                    aptt_observations=aptts,
                    heparin_rate_u_per_hr=rates,
                    heparin_bolus_u=boluses,
                )
            )
        return cohorts

    # ------------------------------------------------------------------
    # Synthetic generator — clearly labelled, MIMIC-IV shape, NOT real data
    # ------------------------------------------------------------------

    @classmethod
    def synthetic_cohort(
        cls,
        n_patients: int = 20,
        seed: int = 42,
    ) -> list["MIMICHeparinCohort"]:
        """Generate a synthetic cohort that mimics MIMIC-IV shape.

        **This is synthetic data.** It is generated from population
        statistics broadly consistent with the MIMIC-IV heparin cohort
        (adult ICU, mean weight ~80 kg, renal function ~ N(1.0, 0.2),
        aPTT draws every 4-8 h for ~24 h under a weight-based Raschke
        nomogram) so that the calibration harness can be exercised
        end-to-end. Results from this generator must **never** be
        reported as MIMIC-IV calibration results.

        The synthetic trajectories are produced by forward-simulating
        HeparinPKPD with the v0.1 default parameters (vmax=400 scaled,
        km=0.4, aptt_alpha=2.5, aptt_c_ref=0.15) and adding Gaussian
        measurement noise with sigma = 5 s (clinically typical aPTT
        lab CV ~= 5-8%). That gives the optimizer a reproducible target
        whose ground-truth parameters are known, which is what the unit
        tests check.

        Parameters
        ----------
        n_patients : int
            Number of synthetic patients (>= 1). Default 20 — enough for
            the fit to converge on synthetic data while keeping test
            wall-time reasonable (< 5 s on CI).
        seed : int
            numpy RNG seed for reproducible synthetic cohorts.

        Returns
        -------
        list[MIMICHeparinCohort]
            Each patient carries `patient_id = f"synthetic_{i:04d}"`.
        """
        if n_patients <= 0:
            raise ValueError(
                f"synthetic_cohort: n_patients must be >= 1, got {n_patients!r}."
            )

        rng = np.random.default_rng(seed)
        cohorts: list[MIMICHeparinCohort] = []

        # Sampling grid — mimic real ICU cadence: q4-6h aPTT draws across
        # the first 24 hours post-heparin-start.
        obs_grid_h = np.array([0.0, 2.0, 6.0, 12.0, 18.0, 24.0], dtype=float)

        # Population priors (broadly MIMIC-IV adult-ICU shaped, not a
        # claim about any specific cohort).
        for i in range(n_patients):
            weight_kg = float(np.clip(rng.normal(80.0, 15.0), 45.0, 140.0))
            renal_function = float(np.clip(rng.normal(1.0, 0.2), 0.3, 1.4))
            baseline_aptt = float(np.clip(rng.normal(30.0, 3.0), 22.0, 42.0))

            # Weight-based Raschke: 80 U/kg bolus at t=0 + 18 U/kg/h infusion.
            bolus_t0 = 80.0 * weight_kg
            infusion_rate = 18.0 * weight_kg

            # Forward-simulate with default PKPD params, then sample at
            # the observation grid with added noise.
            model = HeparinPKPD(
                weight=weight_kg,
                renal_function=renal_function,
                baseline_aptt=baseline_aptt,
            )
            # Default (v0.1) parameters are the "ground truth" we're
            # pretending MIMIC-IV embodies for test purposes.
            model.reset()

            times: list[float] = []
            aptts: list[float] = []
            rates: list[float] = []
            boluses: list[float] = []

            # Generate per-step bolus history: bolus fires at t=0 only.
            dt = 0.5  # 30-min sim step — finer than 1h for smoother aPTT
            n_steps = int(round(obs_grid_h[-1] / dt))

            # Map observation grid to step indices for sampling.
            obs_step_idx = {int(round(t / dt)): t for t in obs_grid_h}

            # Hour 0 observation = baseline_aptt (pre-drug).
            current_t = 0.0
            times.append(current_t)
            aptts.append(float(baseline_aptt + rng.normal(0.0, 3.0)))
            rates.append(0.0)
            boluses.append(0.0)

            # Simulate the infusion+bolus period.
            for step_i in range(1, n_steps + 1):
                bolus_this_step = bolus_t0 if step_i == 1 else 0.0
                model.step(
                    infusion_rate_u_hr=infusion_rate,
                    bolus_u=bolus_this_step,
                    dt_hours=dt,
                )
                current_t = step_i * dt
                if step_i in obs_step_idx:
                    observed = model.get_aptt() + rng.normal(0.0, 5.0)
                    times.append(float(current_t))
                    aptts.append(float(max(observed, 20.0)))
                    rates.append(float(infusion_rate))
                    # Bolus attributed to the very first observation after t=0.
                    boluses.append(float(bolus_t0) if step_i == 1 else 0.0)

            cohorts.append(
                cls(
                    patient_id=f"synthetic_{i:04d}",
                    weight_kg=weight_kg,
                    renal_function=renal_function,
                    baseline_aptt=baseline_aptt,
                    time_hours=times,
                    aptt_observations=aptts,
                    heparin_rate_u_per_hr=rates,
                    heparin_bolus_u=boluses,
                )
            )

        return cohorts

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_csv_rows(self) -> list[dict[str, str]]:
        """Return this cohort as CSV-ready row dicts (string values).

        Used by tests for the round-trip property, and by callers who
        want to export a synthetic cohort for offline inspection.
        """
        rows: list[dict[str, str]] = []
        # Invert the CrCl transform applied in from_csv; emit the
        # nominal CrCl so a round-trip preserves renal function.
        crcl_export = self.renal_function * _DEFAULT_CRCL
        for t, a, r, b in zip(
            self.time_hours,
            self.aptt_observations,
            self.heparin_rate_u_per_hr,
            self.heparin_bolus_u,
            strict=True,
        ):
            rows.append(
                {
                    "subject_id": self.patient_id,
                    "time_hours": f"{t:.6f}",
                    "aptt": f"{a:.6f}",
                    "current_rate_u_per_hr": f"{r:.6f}",
                    "bolus_in_last_hour_u": f"{b:.6f}",
                    "weight_kg": f"{self.weight_kg:.6f}",
                    "crcl_recent": f"{crcl_export:.6f}",
                    "baseline_aptt": f"{self.baseline_aptt:.6f}",
                }
            )
        return rows


def write_cohort_csv(
    cohorts: list[MIMICHeparinCohort], path: str | Path
) -> Path:
    """Write a list of cohorts to a CSV file following `mimic_schema.md`.

    Used by `--synthetic` mode to materialize a test fixture on disk,
    and by tests for the round-trip property. Returns the written path.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(_CSV_COLUMNS))
        writer.writeheader()
        for cohort in cohorts:
            for row in cohort.to_csv_rows():
                writer.writerow(row)
    return dest


# ---------------------------------------------------------------------------
# Forward simulation helpers
# ---------------------------------------------------------------------------


def _simulate_patient(
    cohort: MIMICHeparinCohort,
    params: dict[str, float],
    model_class: type[HeparinPKPD] = HeparinPKPD,
) -> np.ndarray:
    """Forward-simulate one patient and return predicted aPTT trajectory.

    The simulation is driven at `_SIM_DT_HOURS` resolution; bolus and
    rate inputs from the cohort are attributed to the simulation step
    whose time bracket brackets the corresponding observation time.

    Parameters
    ----------
    cohort : MIMICHeparinCohort
        One patient's observations.
    params : dict
        Override values for tunable PKPD parameters. Expected keys
        are `vmax`, `km`, `aptt_alpha`, `aptt_c_ref`. Unspecified keys
        fall back to the model defaults.
    model_class : type
        HeparinPKPD-compatible class. Parameterized so the harness can
        be exercised against alternative model implementations if a
        collaborator wants to swap PK backends.

    Returns
    -------
    np.ndarray
        Predicted aPTT values (seconds) at each observation time,
        shape `(len(cohort.time_hours),)`.
    """
    model = model_class(
        weight=cohort.weight_kg,
        renal_function=cohort.renal_function,
        baseline_aptt=cohort.baseline_aptt,
    )
    # Apply tunable overrides.
    for key in ("vmax", "km", "aptt_alpha", "aptt_c_ref"):
        if key in params:
            setattr(model, key, float(params[key]))
    model.reset()

    times = np.asarray(cohort.time_hours, dtype=float)
    predictions = np.empty_like(times)

    # Observation at t=0 is the baseline draw — emit it directly.
    # (This matches how the synthetic generator labels its first row and
    # how MIMIC-IV typically reports the pre-drug aPTT.)
    predictions[0] = model.get_aptt()

    for i in range(1, len(times)):
        dt = float(times[i] - times[i - 1])
        # Attribute bolus/rate at observation i to the step leading up to
        # it. This is a conservative discretization: infusion is the
        # rate "in effect" during the interval; any bolus recorded in
        # the final hour (per the schema) fires at step start.
        rate = float(cohort.heparin_rate_u_per_hr[i])
        bolus = float(cohort.heparin_bolus_u[i])
        if dt <= 0:
            predictions[i] = model.get_aptt()
            continue
        model.step(infusion_rate_u_hr=rate, bolus_u=bolus, dt_hours=dt)
        predictions[i] = model.get_aptt()

    return predictions


def _total_sse(
    theta: np.ndarray,
    cohort: list[MIMICHeparinCohort],
    param_names: tuple[str, ...],
    model_class: type[HeparinPKPD],
) -> float:
    """SSE objective across the cohort for `scipy.optimize.minimize`."""
    # Physiologic bounds — outside these we return a steep penalty to
    # discourage Nelder-Mead from wandering.
    params = dict(zip(param_names, theta.tolist(), strict=True))
    bounds = {
        "vmax": (1.0, 5000.0),
        "km": (1e-3, 5.0),
        "aptt_alpha": (1e-3, 20.0),
        "aptt_c_ref": (1e-3, 2.0),
    }
    for name, value in params.items():
        if not np.isfinite(value):
            return 1e12
        lo, hi = bounds.get(name, (-np.inf, np.inf))
        if value <= lo or value >= hi:
            return 1e12

    total = 0.0
    for patient in cohort:
        obs = np.asarray(patient.aptt_observations, dtype=float)
        pred = _simulate_patient(patient, params, model_class=model_class)
        residuals = pred - obs
        total += float(np.sum(residuals**2))
    return total


# ---------------------------------------------------------------------------
# CalibrationResult
# ---------------------------------------------------------------------------


_DEFAULT_INITIAL_PARAMS: dict[str, float] = {
    "vmax": 400.0,
    "km": 0.4,
    "aptt_alpha": 2.5,
    "aptt_c_ref": 0.15,
}


@dataclass
class CalibrationResult:
    """Outcome of fitting PK/PD parameters against a MIMIC heparin cohort.

    Attributes:
        fitted_params: Dict of fitted parameter name -> value.
        initial_params: Dict of the starting guess.
        rmse_per_patient: List of dicts, one per patient: `patient_id`,
            `n_observations`, `rmse`, `mean_observed_aptt`,
            `mean_predicted_aptt`.
        overall_rmse: RMSE aggregated across every (patient, obs) pair.
        n_patients: Number of patients in the fit.
        n_observations: Total aPTT observations across the cohort.
        convergence_info: Dict echoing the optimizer's status ---
            `message`, `converged`, `n_iterations`, `loss`, `method`.
        data_source: "synthetic" or "mimic_iv"; stamped explicitly so
            that no downstream artifact can lose provenance.
        seed: Seed passed to the calibration (for audit reproducibility).
        param_names: Ordered tuple of parameters that were fit (same
            order as `initial_params` and `fitted_params` iterate).
    """

    fitted_params: dict[str, float]
    initial_params: dict[str, float]
    rmse_per_patient: list[dict[str, Any]]
    overall_rmse: float
    n_patients: int
    n_observations: int
    convergence_info: dict[str, Any]
    data_source: str = "unspecified"
    seed: int = 0
    param_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation suitable for `json.dumps`."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def markdown_report(self) -> str:
        """Render a publication-ready Markdown summary.

        The section header stamps the data source explicitly so the
        paper cannot accidentally print synthetic-cohort numbers under
        a heading that implies MIMIC-IV calibration.
        """
        lines: list[str] = []
        lines.append("## MIMIC-IV heparin PK/PD calibration")
        lines.append("")
        lines.append(f"- Data source: **{self.data_source}**")
        lines.append(f"- Patients: {self.n_patients}")
        lines.append(f"- Total aPTT observations: {self.n_observations}")
        lines.append(f"- Overall RMSE: **{self.overall_rmse:.3f} s**")
        conv = self.convergence_info
        status = "converged" if conv.get("converged") else "DID NOT CONVERGE"
        lines.append(
            f"- Optimizer: {conv.get('method', '?')} — **{status}** "
            f"after {conv.get('n_iterations', '?')} iterations "
            f"(final loss: {conv.get('loss', float('nan')):.4g})"
        )
        lines.append(f"- Seed: {self.seed}")
        lines.append("")

        # Explicit scaffold disclosure when running on synthetic data.
        if self.data_source == "synthetic":
            lines.append(
                "> **Scaffold notice.** This run used the "
                "`synthetic_cohort` generator, not real MIMIC-IV "
                "extracts. These numbers verify that the calibration "
                "pipeline is wired correctly end-to-end; they are not a "
                "validation claim. Real-data calibration is Phase 2 of "
                "the collaboration and requires PhysioNet credentialed "
                "access (see `mimic_schema.md`)."
            )
            lines.append("")

        lines.append("### Fitted parameters")
        lines.append("")
        lines.append("| Parameter | Initial | Fitted |")
        lines.append("|-----------|---------|--------|")
        for name in self.fitted_params:
            init_v = self.initial_params.get(name, float("nan"))
            fit_v = self.fitted_params[name]
            lines.append(f"| `{name}` | {init_v:.4g} | {fit_v:.4g} |")
        lines.append("")

        lines.append("### Per-patient RMSE")
        lines.append("")
        lines.append(
            "| Patient | N obs | RMSE (s) | Mean observed (s) | Mean predicted (s) |"
        )
        lines.append(
            "|---------|-------|----------|-------------------|--------------------|"
        )
        # Stable ordering for diffability.
        for row in self.rmse_per_patient:
            lines.append(
                f"| `{row['patient_id']}` | {row['n_observations']} | "
                f"{row['rmse']:.3f} | {row['mean_observed_aptt']:.2f} | "
                f"{row['mean_predicted_aptt']:.2f} |"
            )
        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Calibration entry point
# ---------------------------------------------------------------------------


def calibrate_heparin_pkpd(
    cohort: list[MIMICHeparinCohort],
    model_class: type[HeparinPKPD] = HeparinPKPD,
    initial_params: dict[str, float] | None = None,
    *,
    method: str = "Nelder-Mead",
    max_iter: int = 400,
    data_source: str = "unspecified",
    seed: int = 42,
) -> CalibrationResult:
    """Fit heparin PK/PD parameters against a MIMIC-shaped cohort.

    Minimizes total sum-of-squared-residuals across all patients between
    simulated and observed aPTT trajectories. The fit is performed over
    the four "tunable" parameters in `HeparinPKPD`: `vmax`, `km`,
    `aptt_alpha`, `aptt_c_ref`.

    Parameters
    ----------
    cohort : list[MIMICHeparinCohort]
        At least one patient. Each patient's parallel time series must
        validate per the dataclass's `__post_init__` checks.
    model_class : type
        HeparinPKPD-compatible class (default: the v0.1 HeparinPKPD).
    initial_params : dict or None
        Starting guess; defaults to the v0.1 HeparinPKPD defaults
        (`vmax=400.0`, `km=0.4`, `aptt_alpha=2.5`, `aptt_c_ref=0.15`).
    method : str
        scipy.optimize.minimize method — Nelder-Mead (default) is
        robust for small dimensional fits with non-smooth objectives
        arising from clipped physiologic bounds. `L-BFGS-B` is also
        accepted.
    max_iter : int
        Maximum iterations passed to the optimizer.
    data_source : str
        Free-form provenance string stamped onto the result and its
        Markdown report. Canonical values: `"synthetic"` (generator
        output) or `"mimic_iv"` (real MIMIC extract).
    seed : int
        Seed recorded onto the result. Deterministic ODE models don't
        consume it but the CLI logs it for reproducibility.

    Returns
    -------
    CalibrationResult
        Fully populated; `to_dict()` is JSON-safe,
        `markdown_report()` is report-safe.
    """
    if not cohort:
        raise ValueError(
            "calibrate_heparin_pkpd: cohort must contain at least one patient."
        )
    if method not in {"Nelder-Mead", "L-BFGS-B"}:
        raise ValueError(
            f"calibrate_heparin_pkpd: unsupported optimizer method "
            f"{method!r}. Use 'Nelder-Mead' or 'L-BFGS-B'."
        )

    initial = dict(_DEFAULT_INITIAL_PARAMS)
    if initial_params:
        # Only override known tunable keys; silently ignore extraneous
        # keys so callers can pass the full dataclass dict if they want.
        for key in initial:
            if key in initial_params:
                initial[key] = float(initial_params[key])

    param_names = tuple(initial.keys())
    x0 = np.array([initial[k] for k in param_names], dtype=float)

    # L-BFGS-B takes explicit bounds; Nelder-Mead does not — we rely on
    # the in-loss penalty to keep it physiologic.
    if method == "L-BFGS-B":
        bounds_list = [
            (1.0, 5000.0),    # vmax
            (1e-3, 5.0),      # km
            (1e-3, 20.0),     # aptt_alpha
            (1e-3, 2.0),      # aptt_c_ref
        ]
        result = minimize(
            _total_sse,
            x0,
            args=(cohort, param_names, model_class),
            method=method,
            bounds=bounds_list,
            options={"maxiter": max_iter, "ftol": 1e-8},
        )
    else:
        result = minimize(
            _total_sse,
            x0,
            args=(cohort, param_names, model_class),
            method=method,
            options={"maxiter": max_iter, "xatol": 1e-4, "fatol": 1e-6},
        )

    fitted = {name: float(result.x[i]) for i, name in enumerate(param_names)}

    # Build per-patient diagnostics using the fitted parameters.
    per_patient: list[dict[str, Any]] = []
    total_sq = 0.0
    total_obs = 0
    for patient in cohort:
        obs = np.asarray(patient.aptt_observations, dtype=float)
        pred = _simulate_patient(patient, fitted, model_class=model_class)
        resid = pred - obs
        rmse = float(np.sqrt(np.mean(resid**2))) if resid.size else 0.0
        per_patient.append(
            {
                "patient_id": patient.patient_id,
                "n_observations": int(obs.size),
                "rmse": rmse,
                "mean_observed_aptt": float(np.mean(obs)) if obs.size else 0.0,
                "mean_predicted_aptt": float(np.mean(pred)) if pred.size else 0.0,
            }
        )
        total_sq += float(np.sum(resid**2))
        total_obs += int(obs.size)

    overall_rmse = (
        float(np.sqrt(total_sq / max(total_obs, 1))) if total_obs else 0.0
    )

    convergence_info: dict[str, Any] = {
        "method": method,
        "converged": bool(result.success),
        "message": str(result.message),
        "n_iterations": int(getattr(result, "nit", 0)),
        "loss": float(result.fun),
    }

    return CalibrationResult(
        fitted_params=fitted,
        initial_params=initial,
        rmse_per_patient=per_patient,
        overall_rmse=overall_rmse,
        n_patients=len(cohort),
        n_observations=total_obs,
        convergence_info=convergence_info,
        data_source=data_source,
        seed=seed,
        param_names=list(param_names),
    )


# ---------------------------------------------------------------------------
# Small utility used by the CLI and by tests
# ---------------------------------------------------------------------------


def cohort_summary(cohort: list[MIMICHeparinCohort]) -> dict[str, Any]:
    """Return cohort-level summary statistics (N, mean weight, etc.).

    Kept separate from `CalibrationResult` so it can be logged before
    any fitting is done — useful for the CLI's "heard you; loaded N
    patients, weights look like X" progress line.
    """
    if not cohort:
        return {
            "n_patients": 0,
            "n_observations": 0,
            "mean_weight_kg": 0.0,
            "mean_baseline_aptt": 0.0,
        }
    weights = [c.weight_kg for c in cohort]
    baselines = [c.baseline_aptt for c in cohort]
    n_obs = sum(len(c.time_hours) for c in cohort)
    return {
        "n_patients": len(cohort),
        "n_observations": int(n_obs),
        "mean_weight_kg": float(np.mean(weights)),
        "mean_baseline_aptt": float(np.mean(baselines)),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "CalibrationResult",
    "MIMICHeparinCohort",
    "calibrate_heparin_pkpd",
    "cohort_summary",
    "write_cohort_csv",
]


# Round-trip smoke reader used by tests to avoid touching disk.
def _parse_csv_string(text: str) -> list[MIMICHeparinCohort]:
    """Parse CSV content from a string (testing helper, not public API)."""
    rows = list(csv.DictReader(io.StringIO(text)))
    return MIMICHeparinCohort._from_row_dicts(rows)
