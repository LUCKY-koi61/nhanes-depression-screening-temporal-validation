"""Draw Supplementary Figure S2 from primary regularised-model sensitivities."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "review" / "regularised_capacity_jackknife.csv"
OUTPUT = ROOT / "figures" / "FigureS2_capacity_sensitivity.png"


def main() -> None:
    frame = pd.read_csv(INPUT)
    frame = frame[frame["cycle"] > 2009].copy()
    capacities = np.array([5, 10, 20, 30])
    colours = {"dep10": "#2c7fb8", "dep5": "#d95f0e"}
    labels = {"dep10": "PHQ-9 ≥10", "dep5": "PHQ-9 ≥5"}

    fig, ax = plt.subplots(figsize=(6.8, 3.55))
    for endpoint in ("dep10", "dep5"):
        block = frame[frame.endpoint == endpoint].copy()
        points, low, high = [], [], []
        for capacity in capacities / 100:
            values = 100 * block.loc[np.isclose(block.capacity, capacity), "point"]
            points.append(values.median())
            low.append(values.min())
            high.append(values.max())
        points = np.asarray(points)
        ax.plot(capacities, points, marker="o", ms=4.0, lw=1.2,
                color=colours[endpoint], label=labels[endpoint])
        ax.fill_between(capacities, low, high, color=colours[endpoint], alpha=0.14,
                        linewidth=0)

    ax.axhline(0, color="#6b7280", ls=":", lw=0.8)
    ax.axvline(10, color="#6b7280", ls="--", lw=0.7)
    ax.set_xlabel("Record-based review capacity (%)")
    ax.set_ylabel("Retrained − fixed capture (percentage points)")
    ax.set_xticks(capacities)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", lw=0.6)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Primary regularised-model capacity sensitivity", loc="left",
                 fontsize=10, fontweight="bold")
    ax.text(10.3, ax.get_ylim()[0] + 0.05 * np.diff(ax.get_ylim())[0],
            "primary capacity", fontsize=7.2, color="#4b5563")
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
