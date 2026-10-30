"""Stratified delete-one-PSU jackknife, including strata with three PSUs."""

import numpy as np
from scipy.stats import t


def replicate_multipliers(frame):
    """Return JKn multipliers, variance factors and design degrees of freedom."""
    strata = sorted(frame.SDMVSTRA.unique())
    multipliers, factors = [], []
    df = 0
    for stratum in strata:
        psus = sorted(frame.loc[frame.SDMVSTRA == stratum, "SDMVPSU"].unique())
        n = len(psus)
        if n < 2:
            raise ValueError(f"Stratum {stratum} has fewer than two PSUs")
        df += n - 1
        for psu in psus:
            multiplier = np.ones(len(frame))
            multiplier[(frame.SDMVSTRA == stratum).to_numpy()] = n / (n - 1)
            multiplier[((frame.SDMVSTRA == stratum) & (frame.SDMVPSU == psu)).to_numpy()] = 0
            multipliers.append(multiplier)
            factors.append((n - 1) / n)
    return np.array(multipliers), np.array(factors), df


def interval(point, replicates, factors, design_df):
    """Stratified jackknife variance and a design-df t interval."""
    values = np.asarray(replicates, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite jackknife replicate")
    se = float(np.sqrt(np.sum(factors * (values - point) ** 2)))
    allowance = float(t.ppf(0.975, design_df) * se)
    return {
        "point": float(point), "lo": float(point - allowance),
        "hi": float(point + allowance), "se": se, "df": int(design_df),
        "replicates": len(values), "method": "stratified delete-one-PSU jackknife, design-df t interval",
    }


def allocation(y, p, w, share=0.10, population=False, counts=None):
    """Return capture, PPV and reviewed population share.

    The population rule fractionally allocates the boundary record so that
    represented population capacity is exactly `share`. Full-sample score
    order is retained when zero-weight records are removed in replicates.
    """
    y, p, w = (np.asarray(value) for value in (y, p, w))
    if counts is None:
        counts = np.ones(len(y))
    counts = np.asarray(counts)
    # Sort before deletion so tied retained records cannot swap in a replicate.
    order = np.argsort(-p)
    keep = (w[order] > 0) & (counts[order] > 0)
    order = order[keep]
    y, p, w, counts = y[order], p[order], w[order], counts[order]
    order = np.arange(len(y))
    selected = np.zeros(len(y))
    if population:
        before = np.cumsum(w[order]) - w[order]
        selected[order] = np.clip((share * w.sum() - before) / w[order], 0, 1)
    else:
        before = np.cumsum(counts[order]) - counts[order]
        target = max(1, int(share * counts.sum()))
        selected[order] = np.clip((target - before) / counts[order], 0, 1)
    cases = float(np.sum(w * y * selected))
    reviewed = float(np.sum(w * selected))
    return {
        "capture": cases / np.sum(w * y),
        "ppv": cases / reviewed,
        "reviewed_population": reviewed / w.sum(),
    }


def capture_contrast(frame, y, fixed, updated, w, share=0.10, population=False):
    point = (allocation(y, updated, w, share, population)["capture"]
             - allocation(y, fixed, w, share, population)["capture"])
    multipliers, factors, df = replicate_multipliers(frame)
    values = []
    for multiplier in multipliers:
        values.append(
            allocation(y, updated, w * multiplier, share, population, multiplier)["capture"]
            - allocation(y, fixed, w * multiplier, share, population, multiplier)["capture"]
        )
    return interval(point, values, factors, df)
