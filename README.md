# hemosim

**Reinforcement learning Gymnasium environments for hemostasis and anticoagulation management**

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Tests](https://img.shields.io/badge/tests-142%20passing-brightgreen.svg)
[![PyPI version](https://img.shields.io/pypi/v/hemosim.svg)](https://pypi.org/project/hemosim/)

---

hemosim provides four Gymnasium-compatible environments for training and evaluating reinforcement learning agents on anticoagulation therapy tasks. Each environment is backed by mechanistic pharmacokinetic/pharmacodynamic (PK/PD) models derived from published clinical literature, including warfarin dose titration with CYP2C9/VKORC1 pharmacogenomics, heparin infusion management, direct oral anticoagulant (DOAC) selection, and disseminated intravascular coagulation (DIC) management.

## Installation

```bash
pip install hemosim
```

## Environments

| Environment | Description | Observation | Action |
|---|---|---|---|
| `hemosim/WarfarinDosing-v0` | 90-day warfarin titration | Box(8) | Box(1) |
| `hemosim/HeparinInfusion-v0` | 5-day heparin infusion | Box(6) | Box(2) |
| `hemosim/DOACManagement-v0` | 365-day DOAC selection | Box(8) | MultiDiscrete(3,3) |
| `hemosim/DICManagement-v0` | 7-day DIC management | Box(8) | MultiDiscrete(4,4,3,3) |

## Quick Start

```python
import gymnasium as gym
import hemosim

env = gym.make("hemosim/WarfarinDosing-v0")
obs, info = env.reset(seed=42)

for _ in range(90):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

## Reproducibility

Every number in the paper and in `results/EXPECTED_RESULTS.json` can be
regenerated from a fresh clone with a single command:

```bash
./scripts/reproduce.sh
```

The harness creates a clean venv, installs the package, runs the test
suite, runs the ISC-12 self-test (if present), re-runs the honest baseline
evaluation on 100 episodes of the **held-out** patient seed pool, and
compares the output to `results/EXPECTED_RESULTS.json` with a relative
tolerance of 5% on `mean_reward`. End-to-end in well under 30 minutes on a
laptop.

Exit code 0 means PASS; non-zero means divergence (with a per-metric diff
table printed to stdout). No number in this repo is allowed to come from a
formula — everything must survive `./scripts/reproduce.sh`.

**Train/held-out seed discipline.** Training uses seeds in `range(0, 10000)`;
evaluation uses seeds in `range(100000, 101000)`. The two pools are disjoint
by 90,000 seeds so that training on one pool cannot accidentally reveal a
patient from the other. See `HELDOUT_SEEDS.md` for the full policy and
`src/hemosim/reproducibility.py` for the `assert_held_out` / `assert_train`
guards.

## Paper

The accompanying paper is available at:
- [PDF (GitHub)](https://github.com/HassDhia/hemosim/blob/main/paper/hemosim.pdf)

## Citation

If you use hemosim in your research, please cite:

```bibtex
@software{dhia2026hemosim,
  author = {Dhia, Hass},
  title = {hemosim: Gymnasium Environments for Reinforcement Learning in Hemostasis and Anticoagulation Management},
  year = {2026},
  publisher = {Smart Technology Investments Research Institute},
  url = {https://github.com/HassDhia/hemosim}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

Hass Dhia - Smart Technology Investments Research Institute
- Email: partners@smarttechinvest.com
- Web: [smarttechinvest.com/research](https://smarttechinvest.com/research)
