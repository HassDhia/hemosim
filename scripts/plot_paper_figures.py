"""Generate paper-quality figures for hemosim v2 (ISC-9 follow-up).

All three figures are produced from real data in ``results/`` — zero
synthetic content. Re-run anytime via:

    .venv/bin/python scripts/plot_paper_figures.py

Outputs:
- figures/fig1_baseline_comparison.png — honest clinical vs random
- figures/fig2_calibration_residuals.png — published-data fit residuals
- figures/fig3_pomdp_flow.png — POMDP observation-flow schematic

Publication-quality standards per AppliedResearch skill ReviewChecklist:
- 300 DPI
- serif font
- colorblind-safe palette
- captions with comparative/quantitative takeaways (in paper .tex)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Publication-quality style (AppliedResearch ReviewChecklist).
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Colorblind-safe palette (Wong 2011 / Nat Methods)
COLOR_CLINICAL = "#0072B2"   # blue
COLOR_RANDOM = "#E69F00"     # orange
COLOR_PPO = "#56B4E9"        # light blue (unused until ISC-8)
COLOR_EXPECTED = "#009E73"   # green
COLOR_RESIDUAL_POS = "#D55E00"  # vermillion
COLOR_RESIDUAL_NEG = "#CC79A7"  # reddish purple
COLOR_NEUTRAL = "#999999"


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure 1 — Baseline performance across four environments
# ---------------------------------------------------------------------------


def fig1_baseline_comparison() -> Path:
    """Clinical vs random mean reward across the four environments.

    Source: results/EXPECTED_RESULTS.json (held-out seed pool).
    """
    data = _load_json(RESULTS_DIR / "EXPECTED_RESULTS.json")
    envs = [
        ("hemosim/WarfarinDosing-v0", "Warfarin"),
        ("hemosim/HeparinInfusion-v0", "Heparin"),
        ("hemosim/DOACManagement-v0", "DOAC"),
        ("hemosim/DICManagement-v0", "DIC"),
    ]

    clinical_means, clinical_stds = [], []
    random_means, random_stds = [], []
    for env_id, _ in envs:
        env_data = data["environments"][env_id]
        clinical_means.append(env_data["clinical_baseline"]["mean_reward"])
        clinical_stds.append(env_data["clinical_baseline"]["std_reward"])
        random_means.append(env_data["random"]["mean_reward"])
        random_stds.append(env_data["random"]["std_reward"])

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(envs))
    width = 0.38

    ax.bar(
        x - width / 2, clinical_means, width, yerr=clinical_stds,
        label="Clinical guideline",
        color=COLOR_CLINICAL, capsize=3, edgecolor="black", linewidth=0.5,
    )
    ax.bar(
        x + width / 2, random_means, width, yerr=random_stds,
        label="Random policy",
        color=COLOR_RANDOM, capsize=3, edgecolor="black", linewidth=0.5,
    )

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in envs])
    ax.set_ylabel("Mean episode reward (100 eps, held-out seeds)")
    ax.set_title("Clinical guideline vs. random policy across hemosim v2 environments")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=10,
    )

    # Annotate DOAC gap (largest)
    doac_gap = clinical_means[2] - random_means[2]
    ax.annotate(
        f"Δ = {doac_gap:.1f}",
        xy=(2, clinical_means[2]),
        xytext=(2.2, clinical_means[2] + 8),
        fontsize=9, color=COLOR_CLINICAL,
        arrowprops=dict(arrowstyle="-", color=COLOR_CLINICAL, lw=0.5),
    )

    plt.tight_layout()
    out = FIG_DIR / "fig1_baseline_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2 — Calibration residuals
# ---------------------------------------------------------------------------


def fig2_calibration_residuals() -> Path:
    """Normalized residual (simulated − expected) / expected per benchmark.

    Source: results/published_calibration.json.
    """
    cal = _load_json(RESULTS_DIR / "published_calibration.json")

    # Collect labels, expected, observed, trial
    rows: list[tuple[str, float, float, str]] = []

    # Heparin residuals (in heparin_fit)
    for r in cal.get("heparin_fit", {}).get("residuals", []):
        label = {
            "raschke_aptt_6h":            "Raschke\naPTT 6h",
            "hirsh_therapeutic_conc_mid": "Hirsh\nconc",
            "wan_aptt_ttr_standard_of_care": "Wan\naPTT-TTR",
        }.get(r["key"], r["key"])
        rows.append((label, r["expected"], r["observed"], "Heparin"))

    # Warfarin residuals (in warfarin_fit)
    for r in cal.get("warfarin_fit", {}).get("residuals", []):
        label = {
            "iwpc_mean_maintenance_dose": "IWPC\nmaintdose",
            "iwpc_days_to_therapeutic":   "IWPC\nd-to-INR",
            "hamberg_ss_inr_wildtype":    "Hamberg\nSS INR",
        }.get(r["key"], r["key"])
        rows.append((label, r["expected"], r["observed"], "Warfarin"))

    # DOAC event rate validation
    doac_drugs = cal.get("doac_validation", {}).get("drugs", {})
    for drug, drug_data in doac_drugs.items():
        rows.append((
            f"{drug[:5]}\nstroke",
            drug_data["expected_stroke"],
            drug_data["stroke_rate_per_100py"],
            "DOAC",
        ))
        rows.append((
            f"{drug[:5]}\nbleed",
            drug_data["expected_bleed"],
            drug_data["bleed_rate_per_100py"],
            "DOAC",
        ))

    if not rows:
        raise RuntimeError("no calibration rows found — check JSON schema")

    labels = [r[0] for r in rows]
    expected = np.array([r[1] for r in rows])
    observed = np.array([r[2] for r in rows])
    trial = [r[3] for r in rows]

    # Normalized residual: (obs - exp) / max(|exp|, epsilon)
    residuals = (observed - expected) / np.maximum(np.abs(expected), 1e-9)

    trial_colors = {
        "Heparin": COLOR_CLINICAL,
        "Warfarin": COLOR_EXPECTED,
        "DOAC": COLOR_RANDOM,
    }
    bar_colors = [trial_colors[t] for t in trial]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(labels))
    bars = ax.bar(
        x, residuals, color=bar_colors,
        edgecolor="black", linewidth=0.5, alpha=0.9,
    )

    ax.axhline(0, color="black", linewidth=0.7)
    # Reference tolerance band: ±10% normalized residual
    ax.axhspan(-0.10, 0.10, color=COLOR_NEUTRAL, alpha=0.12,
               label="±10% tolerance")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Normalized residual  (obs − exp) / |exp|")
    ax.set_title("Published-data calibration residuals by benchmark")

    # Legend: one patch per domain
    patches = [
        mpatches.Patch(color=trial_colors["Heparin"], label="Heparin"),
        mpatches.Patch(color=trial_colors["Warfarin"], label="Warfarin"),
        mpatches.Patch(color=trial_colors["DOAC"], label="DOAC"),
        mpatches.Patch(color=COLOR_NEUTRAL, alpha=0.3, label="±10% tol."),
    ]
    ax.legend(
        handles=patches,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=10,
    )

    # Annotate the Wan aPTT-TTR outlier
    for i, (label, r) in enumerate(zip(labels, residuals)):
        if "aPTT-TTR" in label:
            ax.annotate(
                f"{r:+.2f}",
                xy=(i, r),
                xytext=(i + 0.3, r - 0.04),
                fontsize=8, color=COLOR_RESIDUAL_POS,
                arrowprops=dict(arrowstyle="-", color=COLOR_RESIDUAL_POS, lw=0.5),
            )

    plt.tight_layout()
    out = FIG_DIR / "fig2_calibration_residuals.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 3 — POMDP observation flow schematic
# ---------------------------------------------------------------------------


def fig3_pomdp_flow() -> Path:
    """POMDP order→delay→noise→return schematic for HeparinInfusion-POMDP-v0."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, color, ec="black", fontsize=9, weight="normal"):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.15",
            linewidth=1, edgecolor=ec, facecolor=color, alpha=0.92,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize, weight=weight,
        )

    def arrow(x1, y1, x2, y2, text="", color="black", textabove=True):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.3),
        )
        if text:
            tx, ty = (x1 + x2) / 2, (y1 + y2) / 2
            offset = 0.22 if textabove else -0.22
            ax.text(
                tx, ty + offset, text,
                ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="none", alpha=0.9),
            )

    # Agent (left)
    box(0.1, 2.2, 1.8, 1.4, "Policy\n(PPO, DQN,\nor clinical)",
        color="#D0E4F5", fontsize=10, weight="bold")

    # Environment (right)
    box(7.9, 2.2, 2.0, 1.4, "HeparinPKPD\n(true state,\nhidden)",
        color="#FDE2C4", fontsize=10, weight="bold")

    # Lab queue (middle top)
    box(4.2, 4.6, 2.8, 1.1, "LabOrderQueue\n(TAT + analytical noise)",
        color="#E8F3E8", fontsize=9, weight="bold")

    # Action channels (bottom middle)
    box(3.2, 0.4, 1.7, 0.9, "Dose action\n(rate, bolus)",
        color="#F5F5F5", fontsize=8)
    box(5.3, 0.4, 1.7, 0.9, "Lab-order action\n(aPTT, anti-Xa,\nplatelets)",
        color="#F5F5F5", fontsize=7.5)

    # Observation buffer (center)
    box(4.1, 2.5, 2.9, 1.0, "Observation:\nreturned lab history",
        color="#E4D4F4", fontsize=9, weight="bold")

    # Arrows
    # Agent → dose/lab-order actions
    arrow(1.0, 2.2, 4.0, 1.3, "", color="#555")
    arrow(1.0, 2.2, 6.0, 1.3, "", color="#555")
    # Dose action → env
    arrow(4.9, 0.9, 8.0, 2.3, "apply dose", color="#555", textabove=False)
    # Lab-order action → queue
    arrow(6.2, 1.3, 5.6, 4.5, "order lab", color="#555")
    # Env true state sampled → queue (ground truth at order time)
    arrow(8.0, 3.0, 7.0, 4.7, "GT at t_order", color="#009E73")
    # Queue → observation (delayed + noisy)
    arrow(5.6, 4.5, 5.5, 3.5, "return after\nTAT + CV noise", color="#D55E00")
    # Observation → agent
    arrow(4.1, 3.0, 1.9, 2.9, "obs history", color="#0072B2")

    ax.set_title(
        "hemosim/HeparinInfusion-POMDP-v0 — partial observability via lab ordering",
        fontsize=11, pad=8,
    )

    # Legend at bottom
    legend_y = -0.3
    ax.text(0.2, legend_y,
            "Lab specs (cited): aPTT 45 min / CV 8% • anti-Xa 180 min / CV 12% "
            "• platelets 45 min / CV 5%",
            fontsize=7.5, color="#444")

    out = FIG_DIR / "fig3_pomdp_flow.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    # fig3_pomdp_flow is now authored as inline TikZ in paper/hemosim.tex.
    # The PNG generator is retained for archival reproducibility but is no
    # longer invoked by default — regenerating it would produce dead cruft.
    outs = [
        fig1_baseline_comparison(),
        fig2_calibration_residuals(),
    ]
    print("Generated:")
    for p in outs:
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
