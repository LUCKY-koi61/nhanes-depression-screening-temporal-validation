"""Create the primary regularised-model case-capture figure for submission.

The development-selected regularised LightGBM is the primary maintenance
analysis. The source results are frozen in ``outputs/review/regularised_models``;
this script intentionally does not refit any model.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "review" / "regularised_models.csv"
OUTPUT = ROOT / "figures" / "Figure2_primary_capture_gain.png"

INK = "#1f2933"
MUTED = "#6b7280"
TEAL = "#2c9c95"


def main() -> None:
    frame = pd.read_csv(INPUT)
    frame = frame[frame["arm"] == "retrained"].copy()
    frame = frame[frame["cycle"].isin([2009, 2011, 2013, 2015, 2017])]
    labels = ["09–10", "11–12", "13–14", "15–16", "17–18"]
    cycles = [2009, 2011, 2013, 2015, 2017]

    figure, axes = plt.subplots(1, 2, figsize=(7.25, 3.55), sharey=True)
    for axis, endpoint, title, letter in (
        (axes[0], "dep10", "Primary endpoint · PHQ-9 ≥10", "A"),
        (axes[1], "dep5", "Milder endpoint · PHQ-9 ≥5", "B"),
    ):
        series = frame[frame["endpoint"] == endpoint].set_index("cycle").loc[cycles]
        point = 100 * series["gain"].to_numpy()
        low = 100 * series["gain_lo"].to_numpy()
        high = 100 * series["gain_hi"].to_numpy()
        axis.errorbar(
            range(len(cycles)), point, yerr=[point - low, high - point],
            fmt="s", color=TEAL, ms=4.2, lw=1.1, capsize=2.2,
        )
        axis.axhline(0, color=MUTED, lw=0.7, ls=":")
        axis.set_title(title, fontsize=8.5, color=INK, pad=7, fontweight="bold")
        axis.set_xticks(range(len(cycles)), labels, fontsize=8.8)
        axis.set_ylim(-6, 12)
        axis.grid(axis="y", color="#e5e7eb", lw=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(-0.16, 1.14, letter, transform=axis.transAxes, fontsize=8,
                  fontweight="bold", color=INK, va="bottom")
    axes[0].set_ylabel("Capture gain (percentage points)", fontsize=8.5)
    figure.text(0.08, 0.975,
                "Retraining effects for the development-selected regularised model",
                ha="left", va="top", fontsize=10, fontweight="bold", color=INK)
    figure.text(0.08, 0.915,
                "Expanding-window retrained minus fixed capture at 10% record-based review capacity",
                ha="left", va="top", fontsize=7.6, color=MUTED)
    figure.legend(
        [plt.Line2D([], [], color=TEAL, marker="s", ls="none", ms=4.2),
         plt.Line2D([], [], color=MUTED, ls=":", lw=0.9)],
        ["Retrained − fixed (95% CI)", "No difference"],
        loc="center", ncol=2, fontsize=7.2, frameon=False,
        bbox_to_anchor=(0.64, 0.865), handletextpad=0.5, columnspacing=1.8,
    )
    figure.subplots_adjust(left=0.10, right=0.985, top=0.70, bottom=0.18, wspace=0.20)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
