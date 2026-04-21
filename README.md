# hemosim

**Reinforcement-learning Gymnasium environments for hemostasis and anticoagulation management**

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Tests](https://img.shields.io/badge/tests-384%20passing-brightgreen.svg)
[![PyPI version](https://img.shields.io/pypi/v/hemosim.svg)](https://pypi.org/project/hemosim/)

---

`hemosim` provides four Gymnasium-compatible environments for training and evaluating RL agents on bedside anticoagulation-dosing decisions. Every environment is backed by mechanistic pharmacokinetic/pharmacodynamic (PK/PD) models derived from the published clinical literature (IWPC, Hamberg, Raschke, Hirsh, Hockin–Mann, RE-LY/ROCKET-AF/ARISTOTLE). A POMDP reformulation exposes the delayed, noisy, action-ordered laboratory-measurement process that any bedside policy must reason through.

## Scope

**What this package is.** A reproducible simulation substrate: environments, PK/PD models, a POMDP-reformulated lab-ordering interface, clinical-outcome metrics (Rosendaal TTR, ISTH 2005 major bleeding, Warkentin 4T HIT, thromboembolic event aggregator), a rule-based CDS safety layer, a published-data calibration harness, a SPIRIT 2013–compliant silent-deployment protocol, and an extensible baselines suite including a faithful reimplementation of the Nemati 2016 DQN architecture.

**What this package is not.** It does not ship a trained bedside-ready RL policy. End-to-end policy training against individual-patient MIMIC-IV trajectories is a separate (Phase 2) contribution with its own calibration and evaluation protocol.

## Installation

```bash
pip install hemosim
```

## Environments

| Environment | Task | Observation | Action |
|---|---|---|---|
| `hemosim/WarfarinDosing-v0` | 90-day warfarin titration, INR 2–3 target | Box(8) | Box(1) |
| `hemosim/HeparinInfusion-v0` | 5-day UFH infusion, aPTT 60–100 s target | Box(6) | Box(2) |
| `hemosim/HeparinInfusion-POMDP-v0` | Same, with delayed/noisy action-ordered labs | Box(10) | Box(5) |
| `hemosim/DOACManagement-v0` | 365-day DOAC selection for atrial fibrillation | Box(8) | MultiDiscrete(3,3) |
| `hemosim/DICManagement-v0` | 7-day DIC component-therapy management | Box(8) | MultiDiscrete(4,4,3,3) |

## Quick Start — clinical baseline on warfarin

```python
import gymnasium as gym
import hemosim
from hemosim.agents.baselines import WarfarinClinicalBaseline

env = gym.make("hemosim/WarfarinDosing-v0")
obs, info = env.reset(seed=100000)
agent = WarfarinClinicalBaseline(seed=42)

for day in range(1, 91):
    action = agent.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if day % 14 == 0:
        print(f"day={day:3d}  INR={info['inr']:.2f}  dose={info['dose_mg']:.1f} mg  "
              f"therapeutic={info['therapeutic']}")
    if terminated or truncated:
        break
```

The fixed-dose INR-adjusted clinical baseline titrates to the 2.0–3.0 INR therapeutic range over the 90-day episode. For the POMDP environment, the same pattern applies but the agent must also emit a lab-order action on each step (see §7 of the paper for the `HeparinInfusionPOMDPEnv` observation/action schema).

## Reproducibility

Every number in the paper and in `results/EXPECTED_RESULTS.json` can be reproduced from a fresh clone with a single command:

```bash
./scripts/reproduce.sh
```

The harness creates a clean venv, installs the package, runs the test suite, re-runs the baseline evaluation on 100 held-out patient seeds, and diffs the output against `results/EXPECTED_RESULTS.json` with a 5% relative tolerance on `mean_reward`. Exit code 0 means PASS.

**Train/held-out seed discipline.** Training uses seeds in `range(0, 10000)`; evaluation uses seeds in `range(100000, 101000)`. The 90,000-seed buffer is a deliberate guard against boundary errors. See [`HELDOUT_SEEDS.md`](HELDOUT_SEEDS.md) and `src/hemosim/reproducibility.py` for the `assert_held_out` / `assert_train` guards.

## Documentation

- **Paper** (30 pages): [`paper/hemosim.pdf`](paper/hemosim.pdf) — environments, PK/PD models, POMDP, calibration residuals, CDS harness, clinical translation pathway.
- **Silent-deployment protocol** (SPIRIT 2013–compliant, 21 pages): [`paper/silent_deployment_protocol.pdf`](paper/silent_deployment_protocol.pdf) — IRB-ready for partner-site submission.
- **Protocol summary** (400-word executive): [`paper/protocol_summary.md`](paper/protocol_summary.md).
- **Seed-pool policy**: [`HELDOUT_SEEDS.md`](HELDOUT_SEEDS.md).
- **Changelog**: [`CHANGELOG.md`](CHANGELOG.md).

## Citation

Cite both the software and the methods paper:

```bibtex
@misc{dhia2026hemosim_paper,
  author       = {Dhia, Hass},
  title        = {{hemosim}: Reproducible Simulation Infrastructure for Reinforcement
                  Learning in Anticoagulation Management, with a POMDP Reformulation
                  and a Clinical Translation Pathway},
  year         = {2026},
  howpublished = {\url{https://github.com/HassDhia/hemosim/blob/main/paper/hemosim.pdf}},
  note         = {Preprint. Smart Technology Investments Research Institute.},
}

@software{dhia2026hemosim_software,
  author    = {Dhia, Hass},
  title     = {{hemosim}: Gymnasium Environments for Reinforcement Learning in
               Hemostasis and Anticoagulation Management},
  year      = {2026},
  version   = {0.2.2},
  publisher = {Smart Technology Investments Research Institute},
  url       = {https://github.com/HassDhia/hemosim},
}
```

## License

MIT. See [LICENSE](LICENSE). Copyright © Smart Technology Investments Research Institute.

## Contact

- Hass Dhia — Smart Technology Investments Research Institute
- Email: partners@smarttechinvest.com
- Web: [smarttechinvest.com/research](https://smarttechinvest.com/research)
