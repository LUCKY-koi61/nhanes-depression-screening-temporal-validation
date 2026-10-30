"""Final-review benchmark tie sensitivity and primary population capacity.

Historical results are never overwritten. All fits use the already selected
development-only parameters, and are held fixed for jackknife replication.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from panel_io import FEATURES, WEIGHT, load_panel
from review_reanalysis import fit
from survey_jackknife import allocation, capture_contrast, interval, replicate_multipliers

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/review"
RULES = ("seqn_ascending", "seqn_descending", "equal_fraction", "historical_numpy")


def ordered_scores(values, ids, descending=False):
    """Rank lower values first, resolving ties by a unique, outcome-free ID."""
    values, ids = np.asarray(values), np.asarray(ids)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Tie-breaking IDs must be unique")
    order = np.lexsort((-ids if descending else ids, values))
    scores = np.empty(len(order))
    scores[order] = -np.arange(len(order))
    return scores


def fractional_capture(y, scores, weights, counts, share=.1):
    """Equal selection probability within a boundary score tie."""
    y, scores, weights, counts = map(np.asarray, (y, scores, weights, counts))
    keep = (weights > 0) & (counts > 0)
    y, scores, weights, counts = (x[keep] for x in (y, scores, weights, counts))
    target = max(1, int(share * counts.sum()))
    order = np.argsort(-scores)
    boundary = scores[order[np.searchsorted(np.cumsum(counts[order]), target)]]
    higher, tied = scores > boundary, scores == boundary
    probability = (target - counts[higher].sum()) / counts[tied].sum()
    selected = higher.astype(float) + tied * probability
    return float(np.sum(weights * y * selected) / np.sum(weights * y))


def main():
    panel = load_panel()
    cycles = (2009, 2011, 2013, 2015, 2017)
    panel = panel[panel.cycle.isin((2007, *cycles))].copy()
    settings = json.loads((OUT / "development_selection.json").read_text())
    tie_rows, population_rows = [], []
    for endpoint in ("dep10", "dep5"):
        parameters = settings[endpoint]["selected"]
        fixed = fit(panel[panel.cycle == 2007], endpoint, parameters=parameters)
        for cycle in cycles:
            test = panel[panel.cycle == cycle].reset_index(drop=True)
            y, w = test[endpoint].to_numpy(), test[WEIGHT].to_numpy()
            retrained = fit(panel[panel.cycle < cycle], endpoint, parameters=parameters)
            predictions = {
                "fixed": fixed.predict_proba(test[FEATURES])[:, 1],
                "retrained": retrained.predict_proba(test[FEATURES])[:, 1],
            }
            multipliers, factors, df = replicate_multipliers(test)
            poverty = -test.INDFMPIR.to_numpy()
            poverty_scores = {
                "historical_numpy": poverty,
                "seqn_ascending": ordered_scores(test.INDFMPIR, test.SEQN),
                "seqn_descending": ordered_scores(test.INDFMPIR, test.SEQN, True),
            }
            for arm, scores in predictions.items():
                for rule in RULES:
                    if rule == "equal_fraction":
                        capture = fractional_capture(y, poverty, w, np.ones(len(y)))
                        point = allocation(y, scores, w)["capture"] - capture
                        replicates = [
                            allocation(y, scores, w * m, counts=m)["capture"]
                            - fractional_capture(y, poverty, w * m, m)
                            for m in multipliers
                        ]
                        contrast = interval(point, replicates, factors, df)
                    else:
                        comparator = poverty_scores[rule]
                        capture = allocation(y, comparator, w)["capture"]
                        contrast = capture_contrast(test, y, comparator, scores, w)
                    tie_rows.append({
                        "endpoint": endpoint, "cycle": cycle, "arm": arm, "rule": rule,
                        "model_capture": allocation(y, scores, w)["capture"],
                        "poverty_capture": capture, **contrast,
                    })
            contrast = capture_contrast(
                test, y, predictions["fixed"], predictions["retrained"], w, population=True)
            for arm, scores in predictions.items():
                metrics = allocation(y, scores, w, population=True)
                population_rows.append({
                    "endpoint": endpoint, "cycle": cycle, "arm": arm, **metrics,
                    "missed": 1 - metrics["capture"], "gain": contrast["point"],
                    "gain_lo": contrast["lo"], "gain_hi": contrast["hi"],
                    "df": contrast["df"],
                })
            print(f"completed final-review analyses: {endpoint} {cycle}", flush=True)
    ties = pd.DataFrame(tie_rows)
    ties.to_csv(OUT / "final_benchmark_ties.csv", index=False)
    ties[ties.rule == "seqn_ascending"].to_csv(OUT / "final_benchmark.csv", index=False)
    pd.DataFrame(population_rows).to_csv(OUT / "final_population_capacity.csv", index=False)
    (OUT / "final_submission_protocol.json").write_text(json.dumps({
        "stage": "post-review, not preregistered",
        "benchmark_rule": "income-to-poverty ascending; ties by SEQN ascending",
        "reason": "outcome-independent reproducible allocation, not significance selection",
        "sensitivity_rules": RULES,
        "model_record_rule": "historical fitted-score ranking retained",
        "population_capacity": "exact 10% weighted mass, fractional boundary record",
        "inference": "stratified delete-one-PSU jackknife, design-df t",
        "models": "fixed during replication; development-selected settings unchanged",
        "training_uncertainty_included": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
