"""Shared loader for the frozen NHANES analytic panel.

The repository distributes the de-identified, analysis-ready CSV used for the
standard-cycle analyses. It avoids an optional parquet dependency and
normalises the historical weight-column name on input.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PANEL = ROOT / "data" / "frozen" / "panel_w5.csv"

FEATURES = ["RIDAGEYR", "RIAGENDR", "INDFMPIR", "educ", "BMXBMI", "smoke",
            "sleep", "PAQ605", "PAQ620", "PAQ635", "PAQ650", "PAQ665"]
WEIGHT = "WTMEC2YR"


def load_panel(prefer: str = "csv") -> pd.DataFrame:
    """Return the frozen analysis panel with a ``WTMEC2YR`` weight column."""

    del prefer  # Retained for backwards-compatible calls from the analysis scripts.
    if not CSV_PANEL.exists():
        raise RuntimeError(f"missing frozen analysis panel: {CSV_PANEL}")
    panel = pd.read_csv(CSV_PANEL)
    if "wt" in panel.columns and WEIGHT not in panel.columns:
        panel = panel.rename(columns={"wt": WEIGHT})
    return panel


def weighted_quantiles(values, weight, probabilities):
    """Weighted quantiles of a continuous variable (step interpolation)."""

    import numpy as np

    values = np.asarray(values, dtype=float)
    weight = np.asarray(weight, dtype=float)
    order = np.argsort(values)
    cumulative = np.cumsum(weight[order]) / weight.sum()
    return np.interp(probabilities, cumulative, values[order])


def kish_effective_size(weight) -> float:
    """Kish effective sample size for a weighted mean."""

    import numpy as np

    weight = np.asarray(weight, dtype=float)
    return float(weight.sum() ** 2 / np.square(weight).sum())
