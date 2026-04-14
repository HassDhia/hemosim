"""Direct Oral Anticoagulant (DOAC) population PK models.

Two-compartment oral PK models with first-order absorption for:
- Rivaroxaban (Xarelto) - based on Mueck et al. (2007)
- Dabigatran (Pradaxa) - based on Liesenfeld et al. (2006)
- Apixaban (Eliquis) - based on Frost et al. (2014)

State variables:
    0: drug_gut         - Drug amount in gut/absorption compartment (mg)
    1: drug_central     - Drug amount in central compartment (mg)
    2: drug_peripheral  - Drug amount in peripheral compartment (mg)

References:
    Mueck W, Becka M, Kubitza D, et al. Population model of the pharmacokinetics
    and pharmacodynamics of rivaroxaban. Int J Clin Pharmacol Ther. 2007.

    Liesenfeld KH, Lehr T, Dansirikul C, et al. Population pharmacokinetic
    analysis of the oral thrombin inhibitor dabigatran etexilate. J Clin
    Pharmacol. 2006.

    Frost C, Nepal S, Wang J, et al. Safety, pharmacokinetics and
    pharmacodynamics of multiple oral doses of apixaban. Br J Clin Pharmacol.
    2014.
"""

from __future__ import annotations

import numpy as np


# Drug-specific population PK parameters
DRUG_PARAMS = {
    "rivaroxaban": {
        "ka": 1.3,          # 1/h  - absorption rate constant
        "cl_base": 9.2,     # L/h  - total clearance
        "vc": 55.0,         # L    - central volume
        "vp": 25.0,         # L    - peripheral volume
        "q": 2.5,           # L/h  - intercompartmental clearance
        "f_oral": 0.80,     # -    - oral bioavailability
        "renal_fraction": 0.33,  # fraction of CL that is renal
        "antixa_slope": 7.0,     # ng/mL per anti-Xa IU/mL
        "antixa_baseline": 0.0,
        "standard_dose": 20.0,   # mg
        "low_dose": 15.0,        # mg
        "high_dose": 20.0,       # mg - same for rivaroxaban
        "crcl_threshold": 50.0,  # mL/min for dose reduction
    },
    "dabigatran": {
        "ka": 2.0,
        "cl_base": 5.4,
        "vc": 60.0,
        "vp": 30.0,
        "q": 3.0,
        "f_oral": 0.065,    # very low bioavailability (prodrug)
        "renal_fraction": 0.80,
        "antixa_slope": 0.0,     # dabigatran is thrombin inhibitor, not anti-Xa
        "antixa_baseline": 0.0,
        "standard_dose": 150.0,
        "low_dose": 110.0,
        "high_dose": 150.0,
        "crcl_threshold": 30.0,
    },
    "apixaban": {
        "ka": 1.5,
        "cl_base": 3.3,
        "vc": 21.0,
        "vp": 15.0,
        "q": 1.8,
        "f_oral": 0.50,
        "renal_fraction": 0.27,
        "antixa_slope": 3.5,
        "antixa_baseline": 0.0,
        "standard_dose": 5.0,
        "low_dose": 2.5,
        "high_dose": 5.0,
        "crcl_threshold": 25.0,
    },
}


class DOACPKPD:
    """Two-compartment oral PK model for direct oral anticoagulants.

    Supports rivaroxaban, dabigatran, and apixaban with renal function
    adjustment and anti-Xa activity calculation.
    """

    N_STATES = 3

    def __init__(
        self,
        drug: str = "rivaroxaban",
        crcl: float = 90.0,
        age: float = 65.0,
        weight: float = 75.0,
    ) -> None:
        if drug not in DRUG_PARAMS:
            raise ValueError(
                f"Unknown drug: {drug}. Supported: {list(DRUG_PARAMS.keys())}"
            )

        self.drug = drug
        self.crcl = crcl    # creatinine clearance in mL/min
        self.age = age
        self.weight = weight

        # Load drug-specific parameters
        self.params = {**DRUG_PARAMS[drug]}

        # Adjust clearance for renal function
        # CL_adjusted = CL_nonrenal + CL_renal * (CrCl / 90)
        renal_frac = self.params["renal_fraction"]
        nonrenal_cl = self.params["cl_base"] * (1 - renal_frac)
        renal_cl = self.params["cl_base"] * renal_frac * (crcl / 90.0)
        self.cl = nonrenal_cl + renal_cl

        # Age effect on clearance (minor: ~5% per decade)
        self.cl *= (65.0 / max(age, 20.0)) ** 0.15

        # Weight effect on volume
        self.vc = self.params["vc"] * (weight / 75.0) ** 0.5
        self.vp = self.params["vp"] * (weight / 75.0) ** 0.5

        # State: [gut, central, peripheral] in mg
        self.state = np.zeros(self.N_STATES)

    def step(self, dose_mg: float, dt_hours: float = 12.0) -> np.ndarray:
        """Advance the DOAC model by one time step with an oral dose.

        Args:
            dose_mg: Oral dose in mg (0 for no dose this period).
            dt_hours: Time step in hours (default 12h for BID dosing).

        Returns:
            Updated state vector.
        """
        # Add dose to gut compartment (adjusted by bioavailability)
        if dose_mg > 0:
            self.state[0] += dose_mg * self.params["f_oral"]

        # Euler integration with substeps
        n_substeps = max(int(dt_hours / 0.25), 1)
        h = dt_hours / n_substeps

        ka = self.params["ka"]
        q = self.params["q"]

        for _ in range(n_substeps):
            gut = max(self.state[0], 0.0)
            central = max(self.state[1], 0.0)
            periph = max(self.state[2], 0.0)

            conc_central = central / self.vc
            conc_periph = periph / self.vp

            # Absorption from gut
            d_gut = -ka * gut

            # Central compartment: absorption in, clearance out, distribution
            d_central = (
                ka * gut
                - self.cl * conc_central
                - q * (conc_central - conc_periph)
            )

            # Peripheral compartment: distribution
            d_periph = q * (conc_central - conc_periph)

            self.state[0] = max(gut + d_gut * h, 0.0)
            self.state[1] = max(central + d_central * h, 0.0)
            self.state[2] = max(periph + d_periph * h, 0.0)

        return self.state.copy()

    def get_concentration(self) -> float:
        """Return current plasma concentration in mg/L (= ug/mL)."""
        return float(max(self.state[1] / self.vc, 0.0))

    def get_antixa_activity(self) -> float:
        """Return estimated anti-Xa activity in IU/mL.

        For factor Xa inhibitors (rivaroxaban, apixaban), anti-Xa activity
        correlates linearly with plasma concentration. Dabigatran is a
        thrombin inhibitor and does not have meaningful anti-Xa activity.
        """
        conc_ng_ml = self.get_concentration() * 1000.0  # mg/L -> ng/mL (ug/L)
        slope = self.params["antixa_slope"]
        if slope <= 0:
            return 0.0  # dabigatran - no anti-Xa
        # Linear relationship: anti-Xa = conc / slope
        return float(conc_ng_ml / slope)

    def get_dose_for_renal(self) -> float:
        """Return recommended dose based on current renal function."""
        p = self.params
        if self.crcl < p["crcl_threshold"]:
            return p["low_dose"]
        return p["standard_dose"]

    def reset(self) -> np.ndarray:
        """Reset model to initial (drug-free) state."""
        self.state = np.zeros(self.N_STATES)
        return self.state.copy()
