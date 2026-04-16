# MIMIC-IV Heparin Cohort — Export Schema

This document specifies the cohort extraction needed to run
`hemosim.validation.mimic_calibration.calibrate_heparin_pkpd` against
real individual-patient trajectories from the MIMIC-IV database.

> **Access requirement.** MIMIC-IV is a credentialed-access resource
> distributed on PhysioNet. Use requires an active HIPAA + CITI-compliant
> Data Use Agreement (DUA) and credentialed PhysioNet access. The
> hemosim project does *not* redistribute MIMIC-IV data; this document
> describes the query a credentialed collaborator runs on their end,
> and the export format that the hemosim harness ingests.

Dataset citation:

> Johnson AEW, Bulgarelli L, Shen L, Gayles A, Shammout A, Horng S,
> Pollard TJ, Hao S, Moody B, Gow B, Lehman L-H, Celi LA, Mark RG.
> MIMIC-IV, a freely accessible electronic health record dataset.
> *Scientific Data* **10**, 1 (2023). doi:10.1038/s41597-022-01899-x

## 1. Cohort definition

Include an ICU stay when **all** of the following are true:

1. Adult (>= 18 years at ICU admission).
2. At least one continuous heparin infusion order active during the
   ICU stay (see §2 for item IDs).
3. At least three aPTT measurements (labevents `itemid = 220560` or
   equivalent; see §3) recorded while a heparin order is active or in
   the 6 h immediately following discontinuation.
4. Documented admission weight (`chartevents` items 226512, 224639, or
   patient `weight_admit` when available).
5. Recent creatinine clearance computable from `labevents` serum
   creatinine plus `patients.anchor_age`, `chartevents` admission
   weight, and sex via Cockcroft–Gault.

Exclude stays with:

- Low-molecular-weight heparin (enoxaparin, dalteparin) as the only
  anticoagulant — this cohort is **unfractionated heparin only**.
- Active argatroban, bivalirudin, or lepirudin orders during the
  window (direct thrombin inhibitors confound aPTT response).
- aPTT values recorded on ECMO while the ACT is the primary monitor.

## 2. Source tables and items

| Signal                      | Table                                   | Filter                                                                                               |
|-----------------------------|------------------------------------------|------------------------------------------------------------------------------------------------------|
| aPTT lab values             | `mimiciv_hosp.labevents`                | `itemid = 220560` (aPTT, seconds)                                                                    |
| Heparin continuous infusion | `mimiciv_icu.inputevents`               | `itemid IN (225152, 221319)` (Heparin Sodium — continuous) or `ordercategoryname ILIKE '%heparin%'`  |
| Heparin IV bolus            | `mimiciv_icu.inputevents`               | same `itemid` set, `ordercategoryname LIKE '%bolus%'` OR derived from `prescriptions`                |
| Heparin prescriptions       | `mimiciv_hosp.prescriptions`            | `drug ILIKE 'heparin%'` AND `drug NOT ILIKE '%lovenox%' AND drug NOT ILIKE '%enoxaparin%'`           |
| Creatinine                  | `mimiciv_hosp.labevents`                | `itemid = 50912` (serum creatinine, mg/dL)                                                           |
| Admission weight            | `mimiciv_icu.chartevents`               | `itemid IN (226512, 224639)` (admission weight, kg)                                                  |
| Demographics / age / sex    | `mimiciv_hosp.patients`                 | `subject_id`, `anchor_age`, `gender`                                                                 |

The `itemid` integers above are the canonical MIMIC-IV IDs at the time
of schema v2.2. **Collaborators must confirm IDs against `d_items` /
`d_labitems`** at the version they are querying; MIMIC-IV periodically
re-maps itemids across minor releases.

## 3. Time alignment

The calibration harness expects **one row per aPTT observation** with
the concurrent infusion state attached. Let `t_obs` be the charttime of
the aPTT draw. For each observation compute:

1. `current_rate_u_per_hr` — the **most recent active** heparin
   infusion rate at `t_obs`. Prefer `inputevents.rate` when
   `rateuom = 'units/hour'`; convert `'units/kg/hour'` by multiplying
   by the admission weight. If no infusion is active at `t_obs`, emit
   `0`.
2. `bolus_in_last_hour_u` — sum of heparin boluses given in
   `[t_obs - 60 min, t_obs]`. Combine `inputevents` bolus rows and
   `prescriptions` entries whose `starttime` falls in the window,
   expressed in **units** (not U/kg; pre-multiply by weight).
3. `aptt` — the aPTT value in seconds from the labevents row.
4. `weight_kg` — admission weight as above.
5. `crcl_recent` — Cockcroft–Gault creatinine clearance (mL/min)
   computed from the most recent serum creatinine within the preceding
   48 h. If unavailable, emit NaN (the harness then treats renal
   function as 1.0).

Within a stay, compute `time_hours = (t_obs - stay_start_time) / 1h`
using the first eligible aPTT as `stay_start_time = 0`. This gives a
monotonically increasing time vector per patient.

## 4. Export CSV format

The harness (`MIMICHeparinCohort.from_csv`) expects a **UTF-8 CSV with
a header row** and the following columns. Rows are sorted by
`subject_id` then `time_hours` ascending. Missing values are the empty
string (not `NaN`, not `NULL`).

```
subject_id,time_hours,aptt,current_rate_u_per_hr,bolus_in_last_hour_u,weight_kg,crcl_recent,baseline_aptt
```

| Column                   | Type    | Units                 | Notes                                                                |
|--------------------------|---------|-----------------------|----------------------------------------------------------------------|
| `subject_id`             | string  | —                     | MIMIC subject_id; pseudonymous per PhysioNet DUA.                    |
| `time_hours`             | float   | hours since cohort t0 | Monotone ascending within a subject.                                 |
| `aptt`                   | float   | seconds               | Measured aPTT.                                                       |
| `current_rate_u_per_hr`  | float   | U/hr                  | Active continuous infusion rate; 0 if paused.                        |
| `bolus_in_last_hour_u`   | float   | U                     | Sum of boluses in preceding 60 min; 0 if none.                       |
| `weight_kg`              | float   | kg                    | Admission weight; carried forward within a stay.                     |
| `crcl_recent`            | float   | mL/min                | Cockcroft–Gault; empty = unknown (harness treats as normal).         |
| `baseline_aptt`          | float   | seconds               | Pre-heparin aPTT if available; else 30.0 (physiologic default).      |

### 4.1 Example rows

```csv
subject_id,time_hours,aptt,current_rate_u_per_hr,bolus_in_last_hour_u,weight_kg,crcl_recent,baseline_aptt
10001,0.0,32.1,0,0,78.5,92.0,32.1
10001,2.5,58.4,1400,5000,78.5,92.0,32.1
10001,6.0,78.3,1400,0,78.5,92.0,32.1
10001,12.0,71.0,1300,0,78.5,92.0,32.1
10002,0.0,29.8,0,0,64.2,48.0,29.8
10002,4.0,62.1,1100,4500,64.2,48.0,29.8
```

## 5. Reference SQL sketch

The following SQL sketch returns the export shape directly from
MIMIC-IV Postgres (BigQuery equivalents omit `mimiciv_*.` prefixes). It
is a **sketch**, not a production query: production runs should stage
each CTE into a materialized table and add the exclusion filters from
§1.

```sql
WITH heparin_infusions AS (
    SELECT
        stay_id,
        subject_id,
        starttime,
        endtime,
        rate,
        rateuom,
        amount,
        amountuom
    FROM mimiciv_icu.inputevents
    WHERE itemid IN (225152, 221319)  -- heparin sodium continuous
),
heparin_boluses AS (
    SELECT
        stay_id,
        subject_id,
        starttime,
        amount AS bolus_u
    FROM mimiciv_icu.inputevents
    WHERE itemid IN (225152, 221319)
      AND ordercategoryname ILIKE '%bolus%'
),
aptt AS (
    SELECT
        le.subject_id,
        le.hadm_id,
        le.charttime,
        le.valuenum AS aptt
    FROM mimiciv_hosp.labevents le
    WHERE le.itemid = 220560
      AND le.valuenum IS NOT NULL
),
weights AS (
    SELECT
        stay_id,
        AVG(valuenum) AS weight_kg
    FROM mimiciv_icu.chartevents
    WHERE itemid IN (226512, 224639)
    GROUP BY stay_id
),
cohort AS (
    SELECT DISTINCT
        a.subject_id,
        i.stay_id
    FROM aptt a
    JOIN heparin_infusions i
      ON a.subject_id = i.subject_id
     AND a.charttime BETWEEN i.starttime AND (i.endtime + INTERVAL '6 hours')
)
SELECT
    a.subject_id,
    EXTRACT(EPOCH FROM (a.charttime - MIN(a.charttime)
        OVER (PARTITION BY a.subject_id))) / 3600.0 AS time_hours,
    a.aptt,
    COALESCE((
        SELECT i.rate
        FROM heparin_infusions i
        WHERE i.subject_id = a.subject_id
          AND a.charttime BETWEEN i.starttime AND i.endtime
        ORDER BY i.starttime DESC
        LIMIT 1
    ), 0) AS current_rate_u_per_hr,
    COALESCE((
        SELECT SUM(b.bolus_u)
        FROM heparin_boluses b
        WHERE b.subject_id = a.subject_id
          AND b.starttime BETWEEN (a.charttime - INTERVAL '1 hour') AND a.charttime
    ), 0) AS bolus_in_last_hour_u,
    w.weight_kg,
    NULL::float AS crcl_recent,   -- fill from separate Cockcroft-Gault CTE
    (SELECT MIN(aptt) FROM aptt a2 WHERE a2.subject_id = a.subject_id) AS baseline_aptt
FROM aptt a
JOIN cohort c  ON c.subject_id = a.subject_id
JOIN weights w ON w.stay_id = c.stay_id
ORDER BY a.subject_id, a.charttime;
```

## 6. Data-use and publication

Collaborators must cite:

- Johnson AEW *et al.* **MIMIC-IV**, a freely accessible electronic
  health record dataset. *Scientific Data* **10**, 1 (2023).
  doi:10.1038/s41597-022-01899-x
- PhysioNet data use citation per the MIMIC-IV DUA.
- The hemosim calibration harness (this repository) for the fitting
  pipeline.

Any derived fitted parameters published by the hemosim team will
state explicitly which data were used (real MIMIC-IV vs synthetic
cohort generated by `MIMICHeparinCohort.synthetic_cohort`). The two
provenances are **never** mixed in a single reported result.
