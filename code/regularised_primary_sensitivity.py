"""Primary-model sensitivity analyses for the regularised LightGBM.

The manuscript treats the development-selected regularised model as its primary
maintenance analysis.  This script supplies the two directly relevant
sensitivities using that *same frozen specification*: capture contrasts at 5%,
10%, 20%, and 30% record capacity, and comparisons with a transparent
income-to-poverty-ratio ranking benchmark at the primary 10% capacity.

Models are fitted once for each full-sample arm and held fixed while the
stratified delete-one-PSU jackknife recomputes the capacity cut-off.  Thus the
intervals describe evaluation-sample uncertainty conditional on the fitted
models, consistently with the main analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from panel_io import FEATURES, WEIGHT, load_panel
from review_reanalysis import fit
from survey_jackknife import allocation, capture_contrast


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "review"
CYCLES = (2009, 2011, 2013, 2015, 2017)
CAPACITIES = (0.05, 0.10, 0.20, 0.30)


def selected_parameters() -> dict[str, dict]:
    source = OUT / "development_selection.json"
    return {endpoint: details["selected"]
            for endpoint, details in json.loads(source.read_text(encoding="utf-8")).items()}


def main() -> None:
    panel = load_panel()
    panel = panel[panel.cycle.isin((2007, *CYCLES))].copy()
    development = panel[panel.cycle == 2007].reset_index(drop=True)
    parameters = selected_parameters()
    capacity_rows: list[dict] = []
    benchmark_rows: list[dict] = []

    for endpoint in ("dep10", "dep5"):
        fixed_model = fit(development, endpoint, parameters=parameters[endpoint])
        for cycle in CYCLES:
            test = panel[panel.cycle == cycle].reset_index(drop=True)
            y = test[endpoint].to_numpy()
            weights = test[WEIGHT].to_numpy()
            fixed_scores = fixed_model.predict_proba(test[FEATURES])[:, 1]
            retrained_model = fit(panel[panel.cycle < cycle], endpoint,
                                  parameters=parameters[endpoint])
            retrained_scores = retrained_model.predict_proba(test[FEATURES])[:, 1]

            for capacity in CAPACITIES:
                contrast = capture_contrast(
                    test, y, fixed_scores, retrained_scores, weights, capacity)
                capacity_rows.append({
                    "endpoint": endpoint,
                    "cycle": cycle,
                    "capacity": capacity,
                    "fixed_capture": allocation(y, fixed_scores, weights, capacity)["capture"],
                    "retrained_capture": allocation(y, retrained_scores, weights, capacity)["capture"],
                    **contrast,
                })

            # Lower income-to-poverty ratio ranks first.  This is a transparent
            # one-variable prioritisation benchmark, not a clinical standard.
            poverty_scores = -test["INDFMPIR"].to_numpy()
            poverty_capture = allocation(y, poverty_scores, weights, 0.10)["capture"]
            for arm, scores in (("fixed", fixed_scores), ("retrained", retrained_scores)):
                contrast = capture_contrast(test, y, poverty_scores, scores, weights, 0.10)
                benchmark_rows.append({
                    "endpoint": endpoint,
                    "cycle": cycle,
                    "arm": arm,
                    "capacity": 0.10,
                    "model_capture": allocation(y, scores, weights, 0.10)["capture"],
                    "poverty_capture": poverty_capture,
                    "random_expected_capture": 0.10,
                    **contrast,
                })
            print(f"completed {endpoint} {cycle}", flush=True)

    pd.DataFrame(capacity_rows).to_csv(OUT / "regularised_capacity_jackknife.csv", index=False)
    pd.DataFrame(benchmark_rows).to_csv(OUT / "regularised_benchmark_jackknife.csv", index=False)
    (OUT / "regularised_primary_sensitivity_protocol.json").write_text(
        json.dumps({
            "model": "development-selected regularised LightGBM",
            "capacities": CAPACITIES,
            "benchmark": "income-to-poverty ratio ranked lowest first",
            "random_benchmark": "expected capture equals record capacity",
            "inference": "stratified delete-one-PSU jackknife; models held fixed",
        }, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
