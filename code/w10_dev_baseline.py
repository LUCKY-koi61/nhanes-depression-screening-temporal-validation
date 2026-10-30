"""Development-cycle baseline for the fixed screening specification.

The main analysis compares each evaluation cycle's observed prevalence with
the fixed model's mean prediction in that cycle. That difference mixes two
components: a probability-scale deficit the fixed specification already has
when it is applied to data it has not seen, and whatever the later populations
add on top of it. Reporting only the evaluation-cycle gap therefore attributes
the whole difference to population change.

This script isolates the first component by cross-validating the fixed
specification inside the 2007-2008 development cycle itself. Folds are formed
on survey primary sampling units, so no cluster is split across a fold
boundary and the held-out predictions respect the survey's clustering.
Reported alongside the cross-validated values are the in-sample values, which
show how much of the apparent deficit is overfitting rather than drift.

Outputs ``outputs/w10/dev_baseline.csv`` and ``outputs/w10/dev_baseline.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "w10"
PANEL = ROOT / "data" / "frozen" / "panel_w5.csv"

FEATURES = ["RIDAGEYR", "RIAGENDR", "INDFMPIR", "educ", "BMXBMI", "smoke",
            "sleep", "PAQ605", "PAQ620", "PAQ635", "PAQ650", "PAQ665"]
LABELS = ["dep10", "dep5"]
DEV_CYCLE = 2007
FOLDS = (5, 10)
SEED = 20260929


def make_model() -> LGBMClassifier:
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                          min_child_samples=40, verbose=-1, random_state=0,
                          n_jobs=2)


def fit(frame: pd.DataFrame, label: str) -> LGBMClassifier:
    weight = frame["wt"].to_numpy(dtype=float)
    model = make_model()
    model.fit(frame[FEATURES], frame[label], sample_weight=weight / weight.mean())
    return model


def ece(y: np.ndarray, p: np.ndarray, w: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0] -= 1e-9
    b = np.digitize(p, edges) - 1
    out = 0.0
    for k in range(bins):
        mask = b == k
        if mask.sum():
            out += w[mask].sum() / w.sum() * abs(
                np.average(y[mask], weights=w[mask])
                - np.average(p[mask], weights=w[mask])
            )
    return float(out)


def capture(y: np.ndarray, p: np.ndarray, w: np.ndarray, share: float = 0.10) -> float:
    k = max(1, int(share * len(y)))
    order = np.argsort(-p)[:k]
    denominator = w[y == 1].sum()
    return float(w[order][y[order] == 1].sum() / denominator) if denominator else float("nan")


def psu_grouped_folds(frame: pd.DataFrame, k: int, rng: np.random.Generator) -> np.ndarray:
    """Assign whole primary sampling units to folds, balancing fold sizes."""

    clusters = (
        frame.groupby(["SDMVSTRA", "SDMVPSU"], sort=True).size()
        .reset_index(name="n").to_numpy()
    )
    order = rng.permutation(len(clusters))
    sizes = clusters[order, 2].astype(int)
    fold = np.zeros(len(clusters), dtype=int)
    load = np.zeros(k, dtype=int)
    for i, size in zip(order, sizes):
        target = int(np.argmin(load))
        fold[i] = target
        load[target] += size
    key = {(int(s), int(p)): int(f) for (s, p, _), f in zip(clusters, fold)}
    return np.array([key[(int(s), int(p))] for s, p in
                     zip(frame["SDMVSTRA"], frame["SDMVPSU"])])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL, low_memory=False)
    dev = panel[panel.cycle == DEV_CYCLE].reset_index(drop=True)
    rows: list[dict[str, object]] = []
    summary: dict[str, dict[str, object]] = {}

    for label in LABELS:
        w = dev["wt"].to_numpy(dtype=float)
        y = dev[label].to_numpy(dtype=float)
        observed = float(np.average(y, weights=w))

        in_sample = fit(dev, label).predict_proba(dev[FEATURES])[:, 1]
        rows.append({
            "endpoint": label, "scheme": "in_sample", "folds": 0, "n": len(dev),
            "observed": round(observed, 4),
            "mean_prediction": round(float(np.average(in_sample, weights=w)), 4),
            "gap_pp": round(100 * (observed - float(np.average(in_sample, weights=w))), 2),
            "ece": round(ece(y, in_sample, w), 4),
            "auc": round(float(roc_auc_score(y, in_sample, sample_weight=w)), 3),
            "capture": round(capture(y, in_sample, w), 3),
        })
        summary[label] = {"observed": observed}

        for k in FOLDS:
            rng = np.random.default_rng(SEED + k)
            fold = psu_grouped_folds(dev, k, rng)
            oof = np.zeros(len(dev))
            for f in range(k):
                train = dev[fold != f]
                test = dev[fold == f]
                oof[fold == f] = fit(train, label).predict_proba(test[FEATURES])[:, 1]
            mean_prediction = float(np.average(oof, weights=w))
            rows.append({
                "endpoint": label, "scheme": "cross_validated", "folds": k,
                "n": len(dev), "observed": round(observed, 4),
                "mean_prediction": round(mean_prediction, 4),
                "gap_pp": round(100 * (observed - mean_prediction), 2),
                "ece": round(ece(y, oof, w), 4),
                "auc": round(float(roc_auc_score(y, oof, sample_weight=w)), 3),
                "capture": round(capture(y, oof, w), 3),
            })
            if k == FOLDS[0]:
                summary[label].update({
                    "cv_mean_prediction": mean_prediction,
                    "cv_gap_pp": 100 * (observed - mean_prediction),
                    "cv_ece": ece(y, oof, w),
                    "cv_auc": float(roc_auc_score(y, oof, sample_weight=w)),
                    "cv_capture": capture(y, oof, w),
                })

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "dev_baseline.csv", index=False)

    # Increment in the evaluation-cycle gap that is not already present at
    # development, computed from the primary and secondary log tables.
    for key, name in (("dep10", "w2v2_dep10.csv"), ("dep5", "w2v2_dep5.csv")):
        later = pd.read_csv(ROOT / "outputs" / "w2" / name)
        baseline = summary[key]["cv_gap_pp"]
        summary[key]["evaluation_cycles"] = {
            str(int(row["cycle"])): {
                "observed": float(row["prev_w"]),
                "fixed_mean_prediction": float(row["fixed_mpred"]),
                "fixed_gap_pp": 100 * (float(row["prev_w"]) - float(row["fixed_mpred"])),
                "gap_minus_development_pp": 100 * (
                    float(row["prev_w"]) - float(row["fixed_mpred"])
                ) - baseline,
                "fixed_ece": float(row["fixed_ece"]),
            }
            for _, row in later.iterrows()
        }

    (OUT / "dev_baseline.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8"
    )
    print(table.to_string(index=False))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
