"""Reanalysis after manuscript review; writes separate, traceable outputs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from panel_io import FEATURES, WEIGHT, load_panel
from survey_jackknife import allocation, capture_contrast
from w10_dev_baseline import psu_grouped_folds

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "review"
BASE = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=40)
CANDIDATES = [
    dict(n_estimators=100, learning_rate=0.03, num_leaves=7, min_child_samples=80, reg_lambda=10),
    dict(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=80, reg_lambda=10),
    dict(n_estimators=100, learning_rate=0.03, num_leaves=15, min_child_samples=80, reg_lambda=10),
    dict(n_estimators=200, learning_rate=0.03, num_leaves=15, min_child_samples=80, reg_lambda=10),
]


def fit(frame, label, features=FEATURES, parameters=None):
    w = frame[WEIGHT].to_numpy()
    model = LGBMClassifier(**(BASE if parameters is None else parameters),
                          verbosity=-1, random_state=0, n_jobs=2)
    return model.fit(frame[features], frame[label], sample_weight=w / w.mean())


def metrics(frame, label, predictions):
    y, w = frame[label].to_numpy(), frame[WEIGHT].to_numpy()
    ece = 0.0
    for rows in np.array_split(np.argsort(predictions), 10):
        ece += w[rows].sum() / w.sum() * abs(np.average(y[rows] - predictions[rows], weights=w[rows]))
    return dict(
        n=len(frame), observed=float(np.average(y, weights=w)),
        mean_prediction=float(np.average(predictions, weights=w)),
        auc=float(roc_auc_score(y, predictions, sample_weight=w)),
        brier=float(np.average((y - predictions) ** 2, weights=w)), ece=float(ece),
        **allocation(y, predictions, w),
    )


def select_on_development(frame, label, folds=3, seed=102):
    fold = psu_grouped_folds(frame, folds, np.random.default_rng(seed))
    losses = []
    for parameters in CANDIDATES:
        predictions = np.zeros(len(frame))
        for number in range(folds):
            model = fit(frame[fold != number], label, parameters=parameters)
            predictions[fold == number] = model.predict_proba(frame.loc[fold == number, FEATURES])[:, 1]
        losses.append(float(np.average((frame[label] - predictions) ** 2, weights=frame[WEIGHT])))
    return CANDIDATES[int(np.argmin(losses))], losses


def regularisation_analysis(panel):
    dev = panel[panel.cycle == 2007].reset_index(drop=True)
    records, selections = [], {}
    for label in ("dep10", "dep5"):
        fold = psu_grouped_folds(dev, 5, np.random.default_rng(20260934))
        predictions = np.zeros(len(dev))
        outer_selections = []
        for number in range(5):
            training = dev[fold != number].reset_index(drop=True)
            parameters, _ = select_on_development(training, label, seed=100 + number)
            outer_selections.append(parameters)
            model = fit(training, label, parameters=parameters)
            predictions[fold == number] = model.predict_proba(dev.loc[fold == number, FEATURES])[:, 1]
        records.append(dict(endpoint=label, cycle=2007, arm="nested_cv", **metrics(dev, label, predictions)))
        parameters, losses = select_on_development(dev, label, folds=5, seed=20260934)
        selections[label] = dict(selected=parameters, brier_by_candidate=losses,
                                 candidates=CANDIDATES, outer_selections=outer_selections)
        fixed = fit(dev, label, parameters=parameters)
        records.append(dict(endpoint=label, cycle=2007, arm="in_sample",
                            **metrics(dev, label, fixed.predict_proba(dev[FEATURES])[:, 1])))
        for cycle in (2009, 2011, 2013, 2015, 2017):
            test = panel[panel.cycle == cycle].reset_index(drop=True)
            pf = fixed.predict_proba(test[FEATURES])[:, 1]
            pr = fit(panel[panel.cycle < cycle], label, parameters=parameters).predict_proba(test[FEATURES])[:, 1]
            contrast = capture_contrast(test, test[label].to_numpy(), pf, pr, test[WEIGHT].to_numpy())
            for arm, p in (("fixed", pf), ("retrained", pr)):
                records.append(dict(endpoint=label, cycle=cycle, arm=arm,
                                    gain=contrast["point"], gain_lo=contrast["lo"], gain_hi=contrast["hi"],
                                    **metrics(test, label, p)))
        print("regularisation", label, selections[label]["selected"], flush=True)
    pd.DataFrame(records).to_csv(OUT / "regularised_models.csv", index=False)
    (OUT / "development_selection.json").write_text(json.dumps(selections, indent=2))


def standard_inference(panel):
    contrasts, capacities, clinical = [], [], []
    for label in ("dep10", "dep5"):
        fixed = fit(panel[panel.cycle == 2007], label)
        for cycle in (2009, 2011, 2013, 2015, 2017):
            test = panel[panel.cycle == cycle].reset_index(drop=True)
            y, w = test[label].to_numpy(), test[WEIGHT].to_numpy()
            pf = fixed.predict_proba(test[FEATURES])[:, 1]
            prior = panel[panel.cycle < cycle]
            pr = fit(prior, label).predict_proba(test[FEATURES])[:, 1]
            for capacity in (0.05, 0.10, 0.20, 0.30):
                contrast = capture_contrast(test, y, pf, pr, w, capacity)
                capacities.append(dict(endpoint=label, cycle=cycle, capacity=capacity,
                                       fixed_capture=allocation(y, pf, w, capacity)["capture"],
                                       retrained_capture=allocation(y, pr, w, capacity)["capture"], **contrast))
                if capacity == 0.10:
                    contrasts.append(dict(endpoint=label, cycle=cycle, arm="expanding", **contrast))
            if cycle > 2009:
                latest = fit(panel[panel.cycle == cycle - 2], label).predict_proba(test[FEATURES])[:, 1]
                contrasts.append(dict(endpoint=label, cycle=cycle, arm="single_prior",
                                      **capture_contrast(test, y, pf, latest, w)))
            population_contrast = capture_contrast(test, y, pf, pr, w, population=True)
            for rule in ("records", "population"):
                for arm, p in (("fixed", pf), ("retrained", pr)):
                    stats = allocation(y, p, w, population=rule == "population")
                    clinical.append(dict(endpoint=label, cycle=cycle, rule=rule, arm=arm,
                                         missed_case_share=1 - stats["capture"], **stats,
                                         population_gain=population_contrast["point"],
                                         population_gain_lo=population_contrast["lo"],
                                         population_gain_hi=population_contrast["hi"]))
            print("JKn", label, cycle, contrasts[-1], flush=True)
    pd.DataFrame(contrasts).to_csv(OUT / "capture_jackknife.csv", index=False)
    pd.DataFrame(capacities).to_csv(OUT / "capacity_jackknife.csv", index=False)
    pd.DataFrame(clinical).to_csv(OUT / "population_capacity.csv", index=False)


def reduced_inference():
    panel = pd.read_csv(ROOT / "data/frozen/panel_w7.csv").rename(columns={"wt": WEIGHT})
    features = FEATURES[:7]
    records = []
    for label in ("dep10", "dep5"):
        fixed = fit(panel[panel.cycle == 2007], label, features)
        for cycle in sorted(panel.cycle.unique()):
            if cycle <= 2007:
                continue
            test = panel[panel.cycle == cycle].reset_index(drop=True)
            pf = fixed.predict_proba(test[features])[:, 1]
            pr = fit(panel[panel.cycle < cycle], label, features).predict_proba(test[features])[:, 1]
            contrast = capture_contrast(test, test[label].to_numpy(), pf, pr, test[WEIGHT].to_numpy())
            records.append(dict(endpoint=label, cycle=str(test.cycle_label.iloc[0]), **contrast))
    pd.DataFrame(records).to_csv(OUT / "reduced_capture_jackknife.csv", index=False)


def main():
    OUT.mkdir(exist_ok=True)
    panel = load_panel()
    panel = panel[panel.cycle.isin([2007, 2009, 2011, 2013, 2015, 2017])].copy()
    standard_inference(panel)
    reduced_inference()
    regularisation_analysis(panel)
    (OUT / "analysis_protocol.json").write_text(json.dumps({
        "status": "post-review sensitivity analyses",
        "inference": "stratified delete-one-PSU jackknife, t interval with PSU-minus-strata df",
        "models_held_fixed": True,
        "population_capacity": "exact weighted decile with fractional boundary allocation",
        "regularisation_selection": "development-only PSU-grouped Brier selection; nested internal validation",
        "no_future_labels_for_selection": True,
    }, indent=2))


if __name__ == "__main__":
    main()
