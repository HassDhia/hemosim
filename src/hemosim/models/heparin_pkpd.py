"""Heparin (unfractionated) PK/PD model.

Nonlinear (saturable) clearance model with aPTT response, based on
clinical pharmacokinetic data from Hirsh et al. (2001) and the Raschke
weight-based dosing protocol.

State variables:
    0: heparin_concentration  - Plasma heparin concentration (U/mL)
    1: at3_heparin_complex    - AT-III:heparin complex level (fraction, 0-1)
    2: aptt                   - Activated partial thromboplastin time (seconds)
    3: platelet_count         - Platelet count (x10^3/uL)

References:
    Hirsh J, Warkentin TE, Shaughnessy SG, et al. Heparin and low-molecular-
    weight heparin: mechanisms of action, pharmacokinetics, dosing, monitoring,
    efficacy, and safety. Chest. 2001;119(1 Suppl):64S-94S.

    Raschke RA, Reilly BM, Guidry JR, et al. The weight-based heparin dosing
    nomogram compared with a standard care nomogram. Ann Intern Med. 1993.
"""

from __future__ import annotations

import numpy as np


class HeparinPKPD:
    """Nonlinear PK model for unfractionated heparin with aPTT response.

    Implements saturable (Michaelis-Menten) clearance kinetics:
        CL_total = CL_linear + Vmax / (Km + C)

    aPTT response follows a nonlinear (log-linear) relationship with
    heparin concentration.
    """

    N_STATES = 4

    def __init__(
        self,
        weight: float = 80.0,
        renal_function: float = 1.0,
        baseline_aptt: float = 30.0,
    ) -> None:
        self.weight = weight
        self.renal_function = renal_function  # fraction, 1.0 = normal
        self.baseline_aptt = baseline_aptt

        # PK parameters - weight-adjusted
        self.vd = 0.07 * weight * 1000.0    # mL - volume of distribution (~70 mL/kg)
        self.cl_linear = 20.0 * weight / 80.0  # mL/h - linear clearance component
        self.vmax = 400.0 * weight / 80.0       # U/h  - maximal saturable clearance
        self.km = 0.4                            # U/mL - Michaelis constant

        # Renal function affects clearance (minor for UFH, ~10-20%)
        self.cl_renal_factor = 0.85 + 0.15 * renal_function

        # aPTT response parameters (log-linear model)
        # aPTT = baseline * (1 + alpha * ln(1 + C/C_ref))
        self.aptt_alpha = 2.5          # sensitivity coefficient
        self.aptt_c_ref = 0.15         # U/mL - reference concentration

        # Platelet dynamics (simplified - for HIT monitoring)
        self.baseline_platelets = 250.0    # x10^3/uL
        self.platelet_consumption_rate = 0.5  # x10^3/uL per hour at therapeutic levels
        self.platelet_production_rate = 2.0   # x10^3/uL per hour (bone marrow)

        # AT-III dynamics
        self.at3_binding_rate = 0.5    # 1/h - rate of complex formation
        self.at3_dissociation = 0.1    # 1/h - complex dissociation

        # Initialize state
        self.state = np.array([
            0.0,                     # heparin concentration (U/mL)
            0.0,                     # AT-III:heparin complex
            self.baseline_aptt,      # aPTT (seconds)
            self.baseline_platelets, # platelet count
        ])

    def step(
        self,
        infusion_rate_u_hr: float,
        bolus_u: float = 0.0,
        dt_hours: float = 1.0,
    ) -> np.ndarray:
        """Advance the heparin model by one time step.

        Args:
            infusion_rate_u_hr: Continuous IV infusion rate in U/hr.
            bolus_u: One-time IV bolus dose in Units (0 = no bolus).
            dt_hours: Time step in hours.

        Returns:
            Updated state vector.
        """
        n_substeps = max(int(dt_hours / 0.25), 1)
        h = dt_hours / n_substeps

        # Apply bolus at start of step (instantaneous IV)
        if bolus_u > 0:
            self.state[0] += bolus_u / self.vd  # U/mL concentration increase

        for _ in range(n_substeps):
            conc = max(self.state[0], 0.0)
            at3_complex = np.clip(self.state[1], 0.0, 1.0)

            # Nonlinear clearance: dC/dt = infusion/Vd - (CL_linear + Vmax/(Km+C)) * C / Vd
            total_cl = (
                self.cl_linear + self.vmax / (self.km + conc + 1e-10)
            ) * self.cl_renal_factor

            d_conc = infusion_rate_u_hr / self.vd - total_cl * conc / self.vd

            # AT-III:heparin complex dynamics
            d_at3 = (
                self.at3_binding_rate * conc * (1.0 - at3_complex)
                - self.at3_dissociation * at3_complex
            )

            # aPTT response - nonlinear relationship
            # Higher heparin concentration -> longer aPTT
            new_conc = max(conc + d_conc * h, 0.0)
            target_aptt = self.baseline_aptt * (
                1.0 + self.aptt_alpha * np.log1p(new_conc / self.aptt_c_ref)
            )
            target_aptt = np.clip(target_aptt, 20.0, 200.0)

            # aPTT approaches target with time constant (~1h lab measurement delay)
            aptt = self.state[2]
            d_aptt = (target_aptt - aptt) * 0.5  # 1/h rate toward target

            # Platelet dynamics - consumption at therapeutic levels
            platelet_count = self.state[3]
            consumption = self.platelet_consumption_rate * conc / (conc + 0.3)
            production = self.platelet_production_rate * (
                1.0 + 0.5 * (self.baseline_platelets - platelet_count)
                / self.baseline_platelets
            )
            d_platelets = production - consumption

            # Update state
            self.state[0] = max(conc + d_conc * h, 0.0)
            self.state[1] = np.clip(at3_complex + d_at3 * h, 0.0, 1.0)
            self.state[2] = max(aptt + d_aptt * h, 20.0)
            self.state[3] = max(platelet_count + d_platelets * h, 0.0)

        return self.state.copy()

    def get_aptt(self) -> float:
        """Return current aPTT in seconds."""
        return float(max(self.state[2], 20.0))

    def get_concentration(self) -> float:
        """Return current heparin plasma concentration in U/mL."""
        return float(max(self.state[0], 0.0))

    def get_platelet_count(self) -> float:
        """Return current platelet count in x10^3/uL."""
        return float(max(self.state[3], 0.0))

    def reset(self) -> np.ndarray:
        """Reset model to initial (drug-free) state."""
        self.state = np.array([
            0.0,
            0.0,
            self.baseline_aptt,
            self.baseline_platelets,
        ])
        return self.state.copy()
