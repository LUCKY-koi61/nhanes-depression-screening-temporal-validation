"""Publication-grade figures for the psych01 Journal of Affective Disorders submission.

Each display item is written in two formats:

* a vector PDF, which is the file to upload as the journal figure (Elsevier
  prefers vector art for line drawings and plots);
* a 600 dpi PNG, which is embedded in the manuscript DOCX and satisfies
  raster resolution requirements where a bitmap is needed.

Panels are drawn at the journal's final print width rather than scaled down
afterwards (double column 190 mm, single column 90 mm), with Arial at
6-8 pt so that the lettering stays at the intended size on the printed page.
All series are distinguished by marker and line style as well as colour, so
the figures remain readable in greyscale print.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mc  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "outputs" / "w2"
RESULTS = ROOT / "outputs" / "results"

MM = 1 / 25.4
# 148 mm is the text width that pandoc's reference DOCX allows, so a figure
# drawn at this width is embedded in the manuscript at 1:1 and the 7 pt
# lettering stays at 7 pt for reviewers. The supplied PDF is vector, so the
# journal can place it at its own column width without any loss.
# Journal-legal widths, chosen in inches as the reference figure script does:
# 7.4 in = 188 mm (full double column), 6.2 in = 157 mm (one-and-a-half column).
WIDE_IN, MEDIUM_IN = 7.4, 6.2
FIGURE_WIDTH_MM = 188

# House palette, taken from the figure implementation in project 18
# (`make_design_amplification_figures.py`): four muted hues, each tied to a
# role rather than to a colour name, so that one colour means one thing in
# every panel of the paper.
#   C_REFERENCE  steel blue -- the observed series and reference-model diagnostics
#   C_AUDIT      crimson    -- the fixed 2007-2008 model under audit (and its
#                              development-cycle cross-validation baseline)
#   C_REPAIR     ochre      -- recalibration, i.e. repair of the probability scale
#   C_REFRESH    teal       -- retraining, i.e. refresh of the ranking
C_REFERENCE, C_AUDIT, C_REPAIR, C_REFRESH = "#33658a", "#c1121f", "#e09f3e", "#2a9d8f"
# Short aliases kept so the drawing routines read naturally.
BLUE, ORANGE, AQUA = C_REFERENCE, C_AUDIT, C_REFRESH
INK, SEC, MUT, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURF = "#ffffff"

CYCLES = [2009, 2011, 2013, 2015, 2017]
# Readable labels for the covariate-stability panel; the analysis logs keep the
# NHANES variable names.
VARIABLE_LABELS = {
    "sleep": "Sleep duration",
    "educ": "Education",
    "INDFMPIR": "Income-to-poverty ratio",
    "RIDAGEYR": "Age",
    "BMXBMI": "Body mass index",
    "smoke": "Smoking",
    "RIAGENDR": "Sex",
    "PAQ605": "Vigorous work activity",
    "PAQ620": "Moderate work activity",
    "PAQ635": "Walking or cycling",
    "PAQ650": "Vigorous recreation",
    "PAQ665": "Moderate recreation",
}
ENDPOINT_LABELS = ["PHQ-9 \u2265 10", "PHQ-9 \u2265 5"]
# The post-pandemic cycle is plotted as an extra point after the standard
# series and marked with an open symbol; it uses the reduced predictor set.
CYCLE_TICKS = ["2009\u201310", "2011\u201312", "2013\u201314", "2015\u201316",
               "2017\u201318", "2021\u201323\u2021"]
EXTERNAL_X = 5
# Two-digit tick labels for the six-panel figure, where six full cycle labels do
# not fit side by side; the caption states what they mean.
SHORT_CYCLE_TICKS = ["09\u201310", "11\u201312", "13\u201314", "15\u201316",
                     "17\u201318", "21\u201323\u2021"]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            # Type and grid follow the figure implementation in project 18
            # (make_design_amplification_figures.py): 9 pt base text, 10 pt
            # titles, a 0.25-alpha grid on both axes. Arial replaces DejaVu Sans
            # because the target journals ask for Arial/Helvetica, and the
            # raster is written at 600 dpi rather than 450.
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "text.color": INK,
            "axes.edgecolor": BASE,
            "axes.linewidth": 0.6,
            "axes.labelcolor": INK,
            "xtick.color": SEC,
            "ytick.color": SEC,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": SURF,
            "axes.facecolor": SURF,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_data() -> dict[str, object]:
    corrected = pd.read_csv(ROOT / "outputs/review/capture_jackknife.csv")
    corrected_post = pd.read_csv(ROOT / "outputs/review/reduced_capture_jackknife.csv")
    corrected_capacity = pd.read_csv(ROOT / "outputs/review/capacity_jackknife.csv")

    def intervals(endpoint):
        block = corrected[(corrected.endpoint == endpoint) & (corrected.arm == "expanding")]
        return {str(int(row.cycle)): {"point": row.point, "lo": row.lo, "hi": row.hi}
                for _, row in block.iterrows()}

    return {
        "dev_baseline": json.loads(
            (ROOT / "outputs" / "w10" / "dev_baseline.json").read_text(encoding="utf-8")
        ),
        "d10": pd.read_csv(LOGS / "w2v2_dep10.csv"),
        "d5": pd.read_csv(LOGS / "w2v2_dep5.csv"),
        "r10": pd.read_csv(LOGS / "w4_recal_dep10.csv"),
        "r5": pd.read_csv(LOGS / "w4_recal_dep5.csv"),
        "psi": pd.read_csv(LOGS / "w3_psi.csv", index_col=0),
        "ci10": intervals("dep10"),
        "ci5": intervals("dep5"),
        "ltt": json.loads((LOGS / "w3_ltt.json").read_text(encoding="utf-8")),
        "logit": json.loads((LOGS / "w4_logit_arm.json").read_text(encoding="utf-8")),
        "flow": json.loads((RESULTS / "nhanes_sample_flow.json").read_text(encoding="utf-8")),
        "external": pd.read_csv(LOGS.parent / "w7" / "postpandemic_metrics.csv"),
        "external_boot": {
            row.endpoint: {"point": row.point, "lo": row.lo, "hi": row.hi}
            for _, row in corrected_post[corrected_post.cycle == "2021-2023"].iterrows()
        },
        "capacity": pd.read_csv(LOGS.parent / "w5" / "capacity_summary.csv"),
        "capacity_sensitivity": corrected_capacity.rename(columns={"point": "gain"}),
    }


def performance_frame(data: dict, endpoint: str) -> pd.DataFrame:
    """Standard series plus the later-release point on a shared x axis."""

    dep = data["d10"] if endpoint == "dep10" else data["d5"]
    recal = data["r10"] if endpoint == "dep10" else data["r5"]
    external = data["external"]
    row = external[external.cycle == "2021-2023"].set_index("endpoint").loc[endpoint]
    dev = data["dev_baseline"][endpoint]
    frame = pd.DataFrame(
        {
            "x": list(range(len(dep))) + [EXTERNAL_X],
            "prev_w": list(dep.prev_w) + [float(row["observed_w"])],
            "fixed_mpred": list(dep.fixed_mpred) + [float(row["fixed_pred"])],
            "fixed_auc": list(dep.fixed_auc) + [float(row["fixed_auc"])],
            "rolling_auc": list(dep.rolling_auc) + [float(row["retrained_auc"])],
            "fixed_ece": list(dep.fixed_ece) + [float(row["fixed_ece"])],
            "recal_ece": list(recal.recal_ece) + [float(row["recalibrated_ece"])],
            "rolling_ece": list(dep.rolling_ece) + [float(row["retrained_ece"])],
            "external": [False] * len(dep) + [True],
            # The fixed specification's out-of-sample baseline, estimated by
            # cross-validation inside the development cycle; drawn so that the
            # reader can see how much of the later gap was present at
            # development rather than added by population change.
            "dev_cv_prediction": [dev["cv_mean_prediction"]] * (len(dep) + 1),
        }
    )
    return frame


def capture_series(data: dict, endpoint: str) -> pd.DataFrame:
    """Capture contrast by cycle, with the later release appended."""

    intervals = data["ci10"] if endpoint == "dep10" else data["ci5"]
    rows = []
    for position, year in enumerate(CYCLES):
        item = intervals[str(year)]
        rows.append(
            {
                "x": position,
                "point": item["point"] * 100,
                "lo": item["lo"] * 100,
                "hi": item["hi"] * 100,
                "external": False,
            }
        )
    boot = data["external_boot"][endpoint]
    rows.append(
        {
            "x": EXTERNAL_X,
            "point": boot["point"] * 100,
            "lo": boot["lo"] * 100,
            "hi": boot["hi"] * 100,
            "external": True,
        }
    )
    return pd.DataFrame(rows)


def save(figure: plt.Figure, destination_stem: Path, tight: bool = True) -> None:
    """Write the vector PDF and the 600 dpi PNG for one display item.

    ``tight=False`` keeps the canvas at the declared print width; the
    sample-flow diagram is drawn to fill its canvas, and cropping it would
    shrink the figure and its lettering below the other display items.
    """

    box = "tight" if tight else None
    figure.savefig(destination_stem.with_suffix(".pdf"), bbox_inches=box,
                   facecolor=SURF)
    figure.savefig(destination_stem.with_suffix(".png"), dpi=600,
                   bbox_inches=box, facecolor=SURF)
    plt.close(figure)


def panel_title(axis: plt.Axes, text: str) -> None:
    axis.set_title(text, fontsize=9.0, color=INK, pad=4)


# --------------------------------------------------------------------------
# Figure 1: sample construction (flow diagram plus per-cycle counts)
# --------------------------------------------------------------------------


def draw_figure1(flow: dict) -> plt.Figure:
    # The diagram is saved on its full canvas (no tight crop), so the axes have
    # to fill that canvas; otherwise matplotlib's default margins shrink the
    # drawing and its lettering.
    figure, axis = plt.subplots(figsize=(WIDE_IN, 3.9))
    figure.subplots_adjust(left=0.02, right=0.985, top=0.93, bottom=0.035)
    axis.set_xlim(0, 14)
    axis.set_ylim(1.8, 10.0)
    axis.axis("off")
    figure.patch.set_facecolor(SURF)
    axis.set_facecolor(SURF)
    figure.text(
        0.025, 0.985,
        "One development cycle and five temporal evaluation cycles comprised the analytic sample",
        ha="left", va="top", fontsize=10.0, fontweight="bold", color=INK,
    )
    axis.text(0.3, 9.52, "Sample construction", fontsize=8.5,
              fontweight="bold", color=SEC, va="bottom")
    axis.text(7.25, 9.52, "Analytic sample by survey cycle", fontsize=8.5,
              fontweight="bold", color=SEC, va="bottom")
    axis.text(7.25, 9.23, "2007–08 development sample highlighted; later cycles used for temporal evaluation",
              fontsize=6.9, color=MUT, va="bottom")

    def box(x, y, w, h, lines, title=None, strong=False):
        face = "#eef2f7" if strong else "white"
        edge = BLUE if strong else BASE
        axis.add_patch(
            FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.05,rounding_size=0.08",
                           fc=face, ec=edge, lw=0.9 if strong else 0.6)
        )
        if title:
            axis.text(x + 0.22, y + h - 0.30, title, fontsize=8.5,
                      fontweight="bold", color=INK, va="center")
            axis.text(x + 0.22, y + (h - 0.60) / 2 - 0.02, lines, fontsize=8.0,
                      color=INK, va="center", linespacing=1.45)
        else:
            axis.text(x + 0.22, y + h / 2, lines, fontsize=8.0, color=INK,
                      va="center", linespacing=1.45)

    def arrow(x, y1, y2, note):
        axis.annotate("", xy=(x, y2), xytext=(x, y1),
                      arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8))
        lines = note.split("\n")
        axis.text(x + 0.22, (y1 + y2) / 2 + 0.14, lines[0], fontsize=9.0,
                  color=MUT, va="center")
        if len(lines) > 1:
            axis.text(x + 0.22, (y1 + y2) / 2 - 0.24, lines[1], fontsize=9.0,
                      color=MUT, va="center")

    box(0.3, 7.95, 5.4, 1.12,
        "NHANES 2007\u20132018, six cycles\n"
        "Adults 18+: 36,580",
        title="Source population", strong=True)
    arrow(2.4, 7.90, 6.84, "no complete PHQ-9: \u22124,969")
    box(0.3, 5.55, 5.4, 1.12,
        "31,611 respondents\n"
        "DPQ100 was not required",
        title="PHQ-9-complete respondents")
    arrow(2.6, 5.50, 4.44, "incomplete covariates: \u22124,830")
    box(0.3, 3.15, 5.4, 1.12,
        "26,781 adults analysed\n"
        "Development 4,652 \u00b7 Evaluation 22,129",
        title="Analytic sample", strong=True)

    rows = [
        ("Cycle", "Adults", "Complete PHQ-9", "Analytic"),
        ("2007\u201308", "6,228", "5,447", "4,652"),
        ("2009\u201310", "6,527", "5,576", "4,773"),
        ("2011\u201312", "5,864", "4,944", "4,221"),
        ("2013\u201314", "6,113", "5,393", "4,638"),
        ("2015\u201316", "5,992", "5,161", "4,342"),
        ("2017\u201318", "5,856", "5,090", "4,155"),
    ]
    tx, ty, rh = 7.25, 8.58, 0.62
    widths = [1.45, 1.30, 1.95, 1.45]
    total_width = sum(widths)
    development_y = ty - rh
    axis.add_patch(
        Rectangle((tx - 0.05, development_y - 0.33), total_width + 0.10, 0.61,
                  facecolor="#eef2f7", edgecolor="none", zorder=0)
    )
    for index, width in enumerate(widths):
        x = tx + sum(widths[:index])
        axis.text(x + width / 2, ty, rows[0][index], ha="center", fontsize=7.7,
                  fontweight="bold", color=INK)
    axis.plot([tx, tx + total_width], [ty + 0.24, ty + 0.24], color=INK, lw=0.9)
    axis.plot([tx, tx + total_width], [ty - 0.14, ty - 0.14], color=INK, lw=0.5)
    for index, row in enumerate(rows[1:]):
        y = ty - (index + 1) * rh
        for column, value in enumerate(row):
            x = tx + sum(widths[:column])
            axis.text(x + widths[column] / 2, y, value, ha="center",
                      fontsize=8.0, color=INK)
    axis.plot([tx, tx + total_width], [ty - 6 * rh - 0.26, ty - 6 * rh - 0.26],
              color=INK, lw=0.9)

    axis.text(0.4, 2.82,
              "PHQ-9 total: DPQ010\u2013DPQ090. DPQ100 was not required because of its "
              "skip pattern.",
              fontsize=7.0, color=SEC, va="top")
    return figure


# --------------------------------------------------------------------------
# Figure 2: performance overview (prediction gap, discrimination, calibration)
# --------------------------------------------------------------------------


def mark_external(axis) -> None:
    """Shade the later release instead of marking every panel separately.

    One pale band plus a single label in the top row carries the same
    information as six repeated annotations, and keeps the panels themselves
    free of explanatory text.
    """

    axis.axvspan(EXTERNAL_X - 0.5, EXTERNAL_X + 0.55, color="#f4f4f1", zorder=0)
    axis.axvline(EXTERNAL_X - 0.5, color=GRID, lw=0.8, ls="--", zorder=1)


def _prevalence_panel(axis, frame, label):
    series = frame[~frame.external]
    external = frame[frame.external]
    axis.plot(series.x, series.prev_w * 100, "o-", color=C_REFERENCE, lw=1.65, ms=4.1,
              label="Observed")
    axis.plot(series.x, series.fixed_mpred * 100, "o-", color=C_AUDIT, lw=1.65,
              ms=4.1, label="Fixed model (2007\u201308)")
    axis.plot(external.x, external.prev_w * 100, "o", color=C_REFERENCE, ms=5.0,
              mfc="white", mew=1.35)
    # The post-pandemic cycle is not on the same measurement footing as the
    # standard series, so it is drawn as a detached symbol rather than joined
    # to the last standard point.

    axis.plot(external.x, external.fixed_mpred * 100, "o", color=C_AUDIT, ms=5.0,
              mfc="white", mew=1.35)
    axis.plot(series.x, series.dev_cv_prediction * 100, ":", color=C_AUDIT, lw=1.4,
              label="Fixed model, development cross-validation")

    mark_external(axis)
    axis.set_xticks(range(len(CYCLE_TICKS)))
    axis.set_xticklabels(SHORT_CYCLE_TICKS, fontsize=8.3)
    axis.set_ylabel("Prevalence (%)", fontsize=9.5)


def _discrimination_panel(axis, frame, label):
    series = frame[~frame.external]
    external = frame[frame.external]
    axis.plot(series.x, series.fixed_auc, "o-", color=C_AUDIT, lw=1.65, ms=4.1,
              label="Fixed")
    axis.plot(series.x, series.rolling_auc, "s--", color=C_REFRESH, lw=1.65, ms=4.1,
              label="Retrained")
    axis.plot(external.x, external.fixed_auc, "o", color=C_AUDIT, ms=5.0, mfc="white",
              mew=1.35)
    axis.plot(external.x, external.rolling_auc, "s", color=C_REFRESH, ms=5.0,
              mfc="white",
              mew=1.35)
    mark_external(axis)
    axis.set_xticks(range(len(CYCLE_TICKS)))
    axis.set_xticklabels(SHORT_CYCLE_TICKS, fontsize=8.3)
    axis.set_ylim(0.55, 0.82)
    axis.set_ylabel("Weighted AUC", fontsize=9.5)


def _calibration_panel(axis, frame, label):
    series = frame[~frame.external]
    external = frame[frame.external]
    axis.plot(series.x, series.fixed_ece, "o-", color=C_AUDIT, lw=1.65, ms=4.1,
              label="Fixed")
    axis.plot(series.x, series.recal_ece, "^-", color=C_REPAIR, lw=1.65, ms=4.6,
              label="Recalibrated")
    axis.plot(series.x, series.rolling_ece, "s--", color=C_REFRESH, lw=1.65, ms=4.1,
              label="Retrained")
    axis.plot(external.x, external.fixed_ece, "o", color=C_AUDIT, ms=5.0, mfc="white",
              mew=1.35)
    axis.plot(external.x, external.recal_ece, "^", color=C_REPAIR, ms=5.4,
              mfc="white",
              mew=1.35)
    axis.plot(external.x, external.rolling_ece, "s", color=C_REFRESH, ms=5.0,
              mfc="white",
              mew=1.35)
    mark_external(axis)
    axis.set_xticks(range(len(CYCLE_TICKS)))
    axis.set_xticklabels(SHORT_CYCLE_TICKS, fontsize=8.3)
    axis.set_ylabel("Expected calibration error", fontsize=9.5)


def draw_figure2(data: dict) -> plt.Figure:
    """Small multiples with conclusion-led rows and row-specific shared keys."""

    figure, axes = plt.subplots(3, 2, figsize=(WIDE_IN, 7.8))
    for column, endpoint in enumerate(("dep10", "dep5")):
        frame = performance_frame(data, endpoint)
        _prevalence_panel(axes[0, column], frame, None)
        _discrimination_panel(axes[1, column], frame, None)
        _calibration_panel(axes[2, column], frame, None)
    figure.subplots_adjust(left=0.085, right=0.985, top=0.865, bottom=0.075,
                           hspace=0.98, wspace=0.95)
    figure.text(
        0.085, 0.985,
        "Diagnostic high-complexity model: temporal performance",
        ha="left", va="top", fontsize=11.5, fontweight="bold", color=INK,
    )

    positions = [[axis.get_position() for axis in row] for row in axes]
    for column, label in enumerate(
        ("Primary endpoint · PHQ-9 ≥10", "Milder endpoint · PHQ-9 ≥5")
    ):
        position = positions[0][column]
        figure.text((position.x0 + position.x1) / 2, 0.925, label,
                    ha="center", va="bottom", fontsize=9.7, fontweight="bold", color=SEC)
    row_claims = (
        "Underprediction was already evident at development",
        "Ranking changed non-monotonically",
        "Recalibration reduced error without changing rank",
    )
    for row, claim in enumerate(row_claims):
        position = positions[row][0]
        figure.text(position.x0, position.y1 + 0.017, claim,
                    fontsize=9.5, fontweight="bold", color=INK, ha="left", va="bottom")
    for row, letters in enumerate((("A", "D"), ("B", "E"), ("C", "F"))):
        for column, letter in enumerate(letters):
            axes[row, column].text(0.02, 0.96, letter,
                                   transform=axes[row, column].transAxes,
                                   fontsize=8.5, fontweight="bold", color=INK,
                                   ha="left", va="top")

    external_x = figure.transFigure.inverted().transform(
        axes[0, 1].transData.transform((EXTERNAL_X, 0))
    )[0]
    figure.text(external_x, positions[0][1].y1 + 0.018,
                "2021–2023‡ reduced-predictor stress test",
                ha="center", va="bottom", fontsize=7.6, color=MUT)

    prevalence_handles = [
        Line2D([], [], color=C_REFERENCE, marker="o", ls="-", ms=4.1, lw=1.65,
               label="Observed prevalence"),
        Line2D([], [], color=C_AUDIT, marker="o", ls="-", ms=4.1, lw=1.65,
               label="Fixed model"),
        Line2D([], [], color=C_AUDIT, ls=":", lw=1.4,
               label="Development cross-validation baseline"),
    ]
    discrimination_handles = [
        Line2D([], [], color=C_AUDIT, marker="o", ls="-", ms=4.1, lw=1.65,
               label="Fixed model"),
        Line2D([], [], color=C_REFRESH, marker="s", ls="--", ms=4.1, lw=1.65,
               label="Retrained model"),
    ]
    calibration_handles = [
        Line2D([], [], color=C_AUDIT, marker="o", ls="-", ms=4.1, lw=1.65,
               label="Fixed model"),
        Line2D([], [], color=C_REPAIR, marker="^", ls="-", ms=4.6, lw=1.65,
               label="Recalibrated"),
        Line2D([], [], color=C_REFRESH, marker="s", ls="--", ms=4.1, lw=1.65,
               label="Retrained model"),
    ]
    legend_ys = [
        (positions[0][0].y0 + positions[1][0].y1) / 2,
        (positions[1][0].y0 + positions[2][0].y1) / 2,
        0.030,
    ]
    for handles, legend_y in zip(
        (prevalence_handles, discrimination_handles, calibration_handles), legend_ys
    ):
        figure.legend(handles=handles, loc="center", ncol=len(handles), fontsize=8.2,
                      frameon=False, bbox_to_anchor=(0.53, legend_y),
                      bbox_transform=figure.transFigure, handletextpad=0.5,
                      columnspacing=2.0)
    return figure


# --------------------------------------------------------------------------
# Figure 3: case capture gain with design-based intervals
# --------------------------------------------------------------------------


def draw_figure3(data: dict) -> plt.Figure:
    """Case capture gain at the prespecified capacity.

    The capacity sensitivity lives in Supplementary Figure S2, so the main
    figure keeps one claim: whether retraining changes who is reached at a
    fixed review capacity.
    """

    figure = plt.figure(figsize=(WIDE_IN, 3.55))
    grid = figure.add_gridspec(1, 2)
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    for axis, endpoint, label in [
        (axes[0], "dep10", "Primary endpoint · PHQ-9 ≥10"),
        (axes[1], "dep5", "Milder endpoint · PHQ-9 ≥5"),
    ]:
        series = capture_series(data, endpoint)
        standard = series[~series.external]
        external = series[series.external]
        for block, open_marker in ((standard, False), (external, True)):
            points = block.point.to_numpy()
            lows = block.lo.to_numpy()
            highs = block.hi.to_numpy()
            axis.errorbar(
                block.x, points, yerr=[points - lows, highs - points],
                fmt="s", color=C_REFRESH, ms=4.0, lw=1.1, capsize=2,
                mfc="white" if open_marker else C_REFRESH,
                mew=1.1 if open_marker else 0.8,
            )
        axis.axhline(0, color=MUT, lw=0.6, ls=":")
        mark_external(axis)
        axis.set_title(label, fontsize=8.5, color=SEC, pad=7, fontweight="bold")
        axis.set_xticks(range(len(CYCLE_TICKS)))
        axis.set_xticklabels(SHORT_CYCLE_TICKS, fontsize=9.0)
        axis.set_ylabel("Capture gain (pp)")
        axis.set_ylim(-6, 12)
    figure.subplots_adjust(left=0.08, right=0.985, top=0.70, bottom=0.18, wspace=0.23)
    figure.text(0.08, 0.975,
                "Capture changes varied by endpoint and evaluation cycle",
                ha="left", va="top", fontsize=10.0, fontweight="bold", color=INK)
    figure.text(0.08, 0.915, "Difference from the fixed model at 10% review capacity",
                ha="left", va="top", fontsize=7.7, color=SEC)
    figure.legend(
        handles=[
            Line2D([], [], color=C_REFRESH, marker="s", ls="none", ms=4.0,
                   label="Retrained \u2212 fixed (95% CI)"),
            Line2D([], [], color=MUT, ls=":", lw=0.8, label="No difference"),
        ],
        loc="center", ncol=2, fontsize=7.3, frameon=False,
        bbox_to_anchor=(0.65, 0.865), bbox_transform=figure.transFigure,
        handletextpad=0.5, columnspacing=1.8,
    )
    for axis, letter in zip(axes, "AB"):
        axis.text(-0.16, 1.14, letter, transform=axis.transAxes, fontsize=8,
                  fontweight="bold", color=INK, va="bottom")
    axes[1].text(EXTERNAL_X, 10.55, "2021–2023‡\nstress test",
                 ha="center", va="top", fontsize=6.8, color=MUT)
    return figure


# --------------------------------------------------------------------------
# Figure 4: covariate stability and recalibration coefficients
# --------------------------------------------------------------------------


def draw_figure4(data: dict) -> plt.Figure:
    psi = data["psi"]
    # The reference column (2007) is a structural zero; the CSV stores column
    # labels as strings, so match on the string form.
    # 2007 is the reference (all zeros) and 2020 is the non-representative
    # transition release, which the paper does not use to interpret population
    # drift; only the five standard evaluation cycles are shown.
    matrix = psi.drop(columns=[c for c in psi.columns if str(c) in ("2007", "2020")])
    order = (
        matrix.apply(lambda row: max(float(v) for v in row), axis=1)
        .sort_values(ascending=False)
        .index
    )
    # The main figure shows the variables that carry the finding; the complete
    # matrix is Supplementary Figure S3.
    matrix = matrix.reindex(order).head(5)
    figure = plt.figure(figsize=(WIDE_IN, 5.7))
    grid = figure.add_gridspec(2, 2, height_ratios=[1.35, 1.0])
    heat = figure.add_subplot(grid[0, :])
    slope = figure.add_subplot(grid[1, 0])
    intercept = figure.add_subplot(grid[1, 1])
    figure.subplots_adjust(left=0.085, right=0.985, top=0.84, bottom=0.15,
                           hspace=0.56, wspace=0.34)
    figure.text(0.085, 0.975,
                "Predictor shift and diagnostic-model recalibration",
                ha="left", va="top", fontsize=10.0, fontweight="bold", color=INK)

    # Single-hue sequential ramp built from the house steel blue, so the
    # heatmap belongs to the same palette as the line panels.
    cmap = mc.LinearSegmentedColormap.from_list(
        "house_steel", ["#f2f5f8", "#b9cbdb", "#7fa0ba", "#547b9b", "#33658a"]
    )
    image = heat.imshow(matrix.values.astype(float), cmap=cmap, aspect="auto",
                        vmin=0, vmax=0.25)
    heat.set_xticks(range(len(matrix.columns)))
    heat.set_xticklabels([CYCLE_TICKS[list(CYCLES).index(int(c))] for c in matrix.columns],
                         fontsize=8.0)
    heat.set_yticks(range(len(matrix.index)))
    heat.set_yticklabels([VARIABLE_LABELS.get(str(name), str(name)) for name in matrix.index],
                         fontsize=8.0)
    heat.grid(False)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = float(matrix.values[row, column])
            if value >= 0.1 and not (row == 0 and column == 3):
                heat.text(column, row, f"{value:.2f}", ha="center", va="center",
                          fontsize=9.0, color="white" if value > 0.12 else INK,
                          fontweight="bold")
    colourbar = figure.colorbar(image, ax=heat, shrink=0.9, pad=0.015)
    colourbar.set_label("PSI vs 2007\u201308", fontsize=7.4)
    colourbar.outline.set_visible(False)
    colourbar.ax.tick_params(labelsize=6)
    heat.set_title("Covariate distribution shift (PSI vs 2007–08)", loc="left",
                   fontsize=8.5, color=SEC, fontweight="bold", pad=6)
    heat.annotate("Sleep duration, 2015–16:\nPSI 0.22", xy=(3, 0),
                  xytext=(0.20, 0.36), textcoords="data", fontsize=6.9, color=INK,
                  ha="left", va="center",
                  arrowprops=dict(arrowstyle="->", color=MUT, lw=0.65),
                  annotation_clip=False)
    heat.text(-0.10, 1.14, "A", transform=heat.transAxes, fontsize=8,
              fontweight="bold", color=INK, va="bottom")
    for axis, column, title, ylabel, ideal, letter in [
        (slope, "recal_a", "Slope", "Recalibration slope", 1, "B"),
        (intercept, "recal_b", "Intercept", "Recalibration intercept", 0, "C"),
    ]:
        axis.plot(data["r10"].cycle, data["r10"][column], "o-", color=C_REFERENCE,
                  lw=1.4, ms=3.6, label=ENDPOINT_LABELS[0])
        axis.plot(data["r5"].cycle, data["r5"][column], "s--", color=C_REFRESH,
                  lw=1.4, ms=3.6, label=ENDPOINT_LABELS[1])
        axis.axhline(ideal, color=MUT, lw=0.6, ls=":")
        axis.text(0.98, ideal, f"Ideal = {ideal}", transform=axis.get_yaxis_transform(),
                  ha="right", va="bottom", fontsize=6.6, color=MUT)
        axis.set_xticks(CYCLES)
        axis.set_xticklabels([CYCLE_TICKS[list(CYCLES).index(c)] for c in CYCLES],
                             fontsize=8.0)
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontsize=8.5, color=SEC, fontweight="bold", pad=6)
    for axis, letter in zip([slope, intercept], "BC"):
        axis.text(-0.20, 1.16, letter, transform=axis.transAxes, fontsize=8,
                  fontweight="bold", color=INK, va="bottom")
    lower_top = slope.get_position().y1
    figure.text(0.085, lower_top + 0.055,
                "High-complexity diagnostic-model recalibration coefficients",
                fontsize=8.5, fontweight="bold", color=INK, ha="left", va="bottom")
    figure.legend(
        handles=[
            Line2D([], [], color=C_REFERENCE, marker="o", ls="-", ms=3.6, lw=1.4,
                   label=ENDPOINT_LABELS[0]),
            Line2D([], [], color=C_REFRESH, marker="s", ls="--", ms=3.6, lw=1.4,
                   label=ENDPOINT_LABELS[1]),
        ],
        loc="center", ncol=2, fontsize=7.3, frameon=False,
        bbox_to_anchor=(0.55, 0.045), bbox_transform=figure.transFigure,
        handletextpad=0.5, columnspacing=1.8,
    )
    return figure


# --------------------------------------------------------------------------
# Supplementary Figure S1: one-sided risk control and model class
# --------------------------------------------------------------------------


def draw_figureS1(data: dict) -> plt.Figure:
    """Supplementary diagnostic: risk control and model-class robustness.

    Panel A shows the exempted share on its own 0-100% scale because a ~95%
    line would otherwise compress the two case-finding series, which live in
    the 5-25% band and are shown together in panel B.
    """

    rule = pd.DataFrame(data["ltt"]["0.1"])
    logit = data["logit"]
    # The two multi-series panels use compact, below-axis legends.  Keeping
    # legends out of the plotting area matters here: at final manuscript size
    # an in-panel legend either masks the 2011--2015 values or dominates the
    # data.  A slightly taller canvas preserves readable 7-pt lettering.
    figure, axes = plt.subplots(1, 3, figsize=(WIDE_IN, 3.35))

    # --- A: the share the rule exempts ------------------------------------
    axes[0].plot(rule.cycle, rule.approve_pct, "o-", color=C_REFERENCE, lw=1.4,
                 ms=3.6, label="Exempted from review")
    axes[0].axhline(10, color=MUT, lw=0.7, ls="-.")
    axes[0].set_ylim(0, 104)
    axes[0].set_xlim(2008.6, 2017.9)
    axes[0].set_xticks(CYCLES)
    axes[0].set_ylabel("Percent of population")
    panel_title(axes[0], "Exempted share (\u03c4 = 10% rule)")
    # Panel A has only one series, so a direct label is clearer than a legend.
    axes[0].text(2009.2, 54, "Exempted from review", fontsize=7.2,
                 color=C_REFERENCE, ha="left", va="center")
    axes[0].text(2009.2, 11.4, "cap 10%", fontsize=7.0, color=MUT, ha="left",
                 va="bottom")
    axes[0].grid(axis="x", visible=False)

    # --- B: the two case-finding series, on their own scale ---------------
    axes[1].plot(rule.cycle, rule.obs_dep_approved * 100, "s--", color=C_AUDIT,
                 lw=1.4, ms=3.6, label="Exempted-group rate")
    axes[1].plot(rule.cycle, rule.positives_in_review, "^:", color=C_REFRESH,
                 lw=1.4, ms=4.0, label="Cases retained for review")
    axes[1].axhline(10, color=MUT, lw=0.7, ls="-.")
    axes[1].set_ylim(0, 32)
    axes[1].set_xlim(2008.6, 2017.9)
    axes[1].set_xticks(CYCLES)
    axes[1].set_ylabel("Percent")
    panel_title(axes[1], "Case finding under the same rule")
    axes[1].legend(fontsize=6.8, loc="upper center", ncol=1,
                   bbox_to_anchor=(0.5, -0.25), borderaxespad=0.1,
                   handletextpad=0.45, labelspacing=0.25)
    axes[1].grid(axis="x", visible=False)

    # --- C: model-class robustness --------------------------------------
    for arm, style, colour, label in [
        ("logit_dep10", "o-", C_REFERENCE, "Logistic / \u2265 10"),
        ("logit_dep5", "s--", C_REFERENCE, "Logistic / \u2265 5"),
    ]:
        series = pd.DataFrame(logit[arm])
        axes[2].plot(series.cycle, series.logit_fixed_auc, style, color=colour,
                     lw=1.4, ms=3.6, label=label)
    axes[2].plot(data["d10"].cycle, data["d10"].fixed_auc, "^:", color=C_AUDIT,
                 lw=1.4, ms=4.0, alpha=0.85, label="LightGBM / \u2265 10")
    axes[2].plot(data["d5"].cycle, data["d5"].fixed_auc, "v:", color=C_AUDIT,
                 lw=1.4, ms=4.0, alpha=0.85, label="LightGBM / \u2265 5")
    axes[2].set_xticks(CYCLES)
    axes[2].set_ylim(0.55, 0.80)
    axes[2].set_ylabel("Fixed-model AUC")
    panel_title(axes[2], "AUC by model class")
    axes[2].legend(fontsize=6.7, loc="upper center", ncol=2,
                   bbox_to_anchor=(0.5, -0.25), borderaxespad=0.1,
                   handletextpad=0.4, columnspacing=0.75, labelspacing=0.25)
    axes[2].grid(axis="x", visible=False)

    for axis, letter in zip(axes, "ABC"):
        axis.text(-0.30, 1.18, letter, transform=axis.transAxes, fontsize=8,
                  fontweight="bold", color=INK, va="bottom")
    figure.subplots_adjust(left=0.075, right=0.99, top=0.84, bottom=0.33,
                            wspace=0.46)
    return figure


def draw_figureS2(data: dict) -> plt.Figure:
    """Capacity sensitivity: median with the range across standard cycles.

    The range is drawn as a light band with the median as a line, so that the
    panel does not read as a set of confidence intervals.
    """

    per_cycle = data["capacity_sensitivity"]
    per_cycle = per_cycle[per_cycle.cycle != "2017-2020"]
    figure, axis = plt.subplots(figsize=(MEDIUM_IN, 2.6))
    for endpoint, colour, marker, label in [
        ("dep10", C_REFRESH, "o-", ENDPOINT_LABELS[0]),
        ("dep5", C_REFRESH, "s--", ENDPOINT_LABELS[1]),
    ]:
        block = per_cycle[per_cycle.endpoint == endpoint]
        rows = []
        for capacity_value, group in block.groupby("capacity"):
            gains = 100 * group.gain.to_numpy()
            rows.append((100 * float(capacity_value), float(np.median(gains)),
                         float(gains.min()), float(gains.max())))
        rows.sort()
        x = np.array([row[0] for row in rows])
        median = np.array([row[1] for row in rows])
        low = np.array([row[2] for row in rows])
        high = np.array([row[3] for row in rows])
        axis.fill_between(x, low, high, color=colour, alpha=0.10, lw=0,
                          label=None)
        axis.plot(x, median, marker, color=colour, lw=1.6, ms=4.0, label=label)
    axis.axhline(0, color=MUT, lw=0.6, ls=":")
    axis.set_xticks([5, 10, 20, 30])
    axis.set_xlabel("Review capacity (% of sample reviewed)")
    axis.set_ylabel("Retraining gain (pp)")
    panel_title(axis, "Capacity sensitivity")
    axis.legend(fontsize=8.0, loc="upper left")
    figure.tight_layout()
    return figure


def draw_figureS3(data: dict) -> plt.Figure:
    """The complete covariate-stability matrix (all variables)."""

    psi = data["psi"]
    matrix = psi.drop(columns=[c for c in psi.columns if str(c) in ("2007", "2020")])
    order = (matrix.apply(lambda row: max(float(v) for v in row), axis=1)
             .sort_values(ascending=False).index)
    matrix = matrix.reindex(order)
    figure, axis = plt.subplots(figsize=(MEDIUM_IN, 3.2))
    cmap = mc.LinearSegmentedColormap.from_list(
        "house_steel", ["#f2f5f8", "#b9cbdb", "#7fa0ba", "#547b9b", "#33658a"])
    image = axis.imshow(matrix.values.astype(float), cmap=cmap, aspect="auto",
                        vmin=0, vmax=0.25)
    axis.set_xticks(range(len(matrix.columns)))
    axis.set_xticklabels([CYCLE_TICKS[list(CYCLES).index(int(c))]
                          for c in matrix.columns], fontsize=8.0)
    axis.set_yticks(range(len(matrix.index)))
    axis.set_yticklabels([VARIABLE_LABELS.get(str(name), str(name))
                          for name in matrix.index], fontsize=8.0)
    axis.grid(False)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = float(matrix.values[row, column])
            if value >= 0.05:
                axis.text(column, row, f"{value:.2f}", ha="center", va="center",
                          fontsize=9.0, color="white" if value > 0.12 else INK,
                          fontweight="bold")
    colourbar = figure.colorbar(image, ax=axis, shrink=0.9, pad=0.015)
    colourbar.set_label("PSI vs 2007\u201308", fontsize=8.0)
    colourbar.outline.set_visible(False)
    colourbar.ax.tick_params(labelsize=6)
    panel_title(axis, "All covariates, five standard evaluation cycles")
    figure.tight_layout()
    return figure


DISPLAY_ITEMS = {
    "Figure1_sample_flow": lambda data: draw_figure1(data["flow"]),
    "Figure3_diagnostic_performance": draw_figure2,
    "Figure4_attribution_and_recalibration": draw_figure4,
    "FigureS1_risk_control_and_model_class": draw_figureS1,
    "FigureS3_covariate_stability_full": draw_figureS3,
}


def build_all(destination: Path) -> list[Path]:
    """Write every display item as a vector PDF and a 600 dpi PNG."""

    destination.mkdir(parents=True, exist_ok=True)
    apply_style()
    data = load_data()
    written: list[Path] = []
    full_canvas = {"Figure1_sample_flow"}
    for name, draw in DISPLAY_ITEMS.items():
        save(draw(data), destination / name, tight=name not in full_canvas)
        written.extend([destination / f"{name}.pdf", destination / f"{name}.png"])
    return written


if __name__ == "__main__":
    for path in build_all(ROOT / "figures"):
        print("wrote", path.relative_to(ROOT))
