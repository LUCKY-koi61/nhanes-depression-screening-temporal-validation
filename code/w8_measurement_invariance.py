"""Formal multi-group measurement invariance of the PHQ-9 across NHANES cycles.

The manuscript's assessment-validity argument rests on three claims, one of
which is that the PHQ-9 itself kept measuring the same construct across cycles.
The item-level descriptive analyses in ``w6_measurement_stability.py`` support
that claim but do not test it in the framework in which such claims are
normally evaluated. This script fits the ordinal (graded-response) multi-group
measurement model directly, by full-information maximum likelihood, and
compares the nested invariance restrictions:

    configural  slopes and thresholds free in every cycle; latent mean 0 and
                latent SD 1 in every cycle (item parameters on each cycle's own
                standardised metric)
    metric      slopes equal across cycles; thresholds free; latent SD free in
                all cycles but the reference (latent mean fixed at 0)
    scalar      slopes and thresholds equal across cycles; latent mean and SD
                free in all cycles but the reference
    partial     scalar with the thresholds of selected items released, the
                remaining items acting as anchors

Models are estimated with the EM algorithm (Bock-Aitkin style: fixed
Gauss-Hermite quadrature over the latent trait, item-wise Newton/Quasi-Newton
M-steps, latent-mean/SD block updated by direct optimisation). Because the
models are nested, they are compared by likelihood-ratio tests and by AIC/BIC,
and because a sample of 27,000 makes even trivial departures significant, the
practical size of any non-invariance is reported alongside the tests: the
expected item-score difference at matched latent trait (points on the 0-3 item
scale) and the model-implied shift in PHQ-9 threshold prevalence.

The estimator is validated against simulated data with known parameters in
``--selftest`` before its output is used (``outputs/w8/simulation_validation.json``).

Outputs are written to ``outputs/w8/``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from scipy.stats import chi2, norm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "w8"
PANEL = ROOT / "data" / "frozen" / "panel_w7.csv"

ITEMS = [f"DPQ{i:03d}" for i in range(10, 91, 10)]
N_CAT = 4
STANDARD_CYCLES = ["2007-2008", "2009-2010", "2011-2012",
                   "2013-2014", "2015-2016", "2017-2018"]
POST_CYCLE = ["2021-2023"]
N_NODES = 21


# --------------------------------------------------------------------------
# quadrature and item response functions
# --------------------------------------------------------------------------

def gh_nodes(n_nodes: int = N_NODES) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights for integrating over a standard normal density."""
    x, w = np.polynomial.hermite.hermgauss(n_nodes)
    return x * math.sqrt(2.0), w / math.sqrt(math.pi)


QUAD_Z, QUAD_W = gh_nodes()
N_QUAD = QUAD_Z.shape[0]


def cat_probs(a: float, b: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Category probabilities of a graded response item at latent values eta.

    ``b`` holds three ordered thresholds (b1 < b2 < b3); the result has shape
    (len(eta), 4) with columns P(X = 0) ... P(X = 3).
    """
    cumulative = expit(a * (eta[:, None] - b[None, :]))
    probs = np.empty((eta.shape[0], N_CAT))
    probs[:, 0] = 1.0 - cumulative[:, 0]
    probs[:, 1] = cumulative[:, 0] - cumulative[:, 1]
    probs[:, 2] = cumulative[:, 1] - cumulative[:, 2]
    probs[:, 3] = cumulative[:, 2]
    probs = np.clip(probs, 1e-12, None)
    return probs / probs.sum(axis=1, keepdims=True)


def ordered_thresholds(params: np.ndarray) -> np.ndarray:
    """Map three unconstrained values to ordered thresholds."""
    b1 = params[0]
    b2 = b1 + math.exp(params[1])
    b3 = b2 + math.exp(params[2])
    return np.array([b1, b2, b3])


def threshold_params(b: np.ndarray) -> np.ndarray:
    """Map ordered thresholds to the (b1, log gap, log gap) parameterisation."""
    b = np.asarray(b, dtype=float)
    gap1 = max(b[1] - b[0], 1e-3)
    gap2 = max(b[2] - b[1], 1e-3)
    return np.array([b[0], math.log(gap1), math.log(gap2)])


def initial_thresholds(values: np.ndarray, slope: float = 1.6) -> np.ndarray:
    """Rough thresholds from the marginal category proportions."""
    cumulative = [(values >= k).mean() for k in (1, 2, 3)]
    cumulative = [min(max(p, 1e-3), 1 - 1e-3) for p in cumulative]
    b = -norm.ppf(cumulative) / slope
    return np.array([b[0], b[1], b[2]])


# --------------------------------------------------------------------------
# data container
# --------------------------------------------------------------------------

class Group:
    """One survey cycle: item responses plus (optionally) case weights."""

    def __init__(self, label: str, frame: pd.DataFrame, weighted: bool):
        self.label = label
        self.x = frame[ITEMS].to_numpy(dtype=np.int64)
        self.n = self.x.shape[0]
        if weighted:
            w = frame["wt"].to_numpy(dtype=float)
            self.w = w / w.mean()
        else:
            self.w = np.ones(self.n)


def load_groups(cycles: list[str], weighted: bool = False) -> list[Group]:
    panel = pd.read_csv(PANEL)
    groups = []
    for label in cycles:
        block = panel[panel.cycle_label == label].dropna(subset=ITEMS)
        groups.append(Group(label, block, weighted))
    return groups


# --------------------------------------------------------------------------
# model container
# --------------------------------------------------------------------------

class Model:
    """Multi-group graded-response model under an invariance restriction.

    ``kind`` is one of ``configural``, ``metric``, ``scalar`` or ``partial``.
    In ``partial`` the thresholds of ``free_items`` are released while all
    other items act as anchors. Slopes are equal across groups in every model
    except ``configural``.
    """

    def __init__(self, groups: list[Group], kind: str, free_items: tuple[str, ...] = ()):
        self.groups = groups
        self.kind = kind
        self.free_items = tuple(free_items)
        self.labels = [g.label for g in groups]
        self.n_groups = len(groups)
        self.slope_shared = kind != "configural"
        self.threshold_shared = kind in ("scalar", "partial")
        self.free_sd = kind in ("metric", "scalar", "partial")
        self.free_mean = kind in ("scalar", "partial")

        self.slopes = np.full((self.n_groups, len(ITEMS)), 1.6)
        self.thresholds = np.zeros((self.n_groups, len(ITEMS), 3))
        self.mean = np.zeros(self.n_groups)
        self.sd = np.ones(self.n_groups)
        for gi, group in enumerate(groups):
            for ii, item in enumerate(ITEMS):
                self.thresholds[gi, ii] = initial_thresholds(group.x[:, ii])

    # -- parameter bookkeeping ------------------------------------------------

    def n_parameters(self) -> int:
        if self.kind == "configural":
            n = self.n_groups * len(ITEMS) * 4
        elif self.kind == "metric":
            n = len(ITEMS) + self.n_groups * len(ITEMS) * 3
        else:
            n_free = len(self.free_items) if self.kind == "partial" else 0
            n = len(ITEMS) + (len(ITEMS) - n_free) * 3 + n_free * 3 * self.n_groups
        n += (self.n_groups - 1) * (int(self.free_sd) + int(self.free_mean))
        return int(n)

    def item_is_shared(self, item: str) -> bool:
        if self.kind == "configural":
            return False
        if self.kind == "metric":
            return True  # slope only; thresholds still group-specific
        return item not in self.free_items

    def thresholds_of(self, gi: int, ii: int) -> np.ndarray:
        item = ITEMS[ii]
        if self.threshold_shared and self.item_is_shared(item):
            return self.thresholds[0, ii]
        return self.thresholds[gi, ii]

    def slope_of(self, gi: int, ii: int) -> float:
        if self.slope_shared:
            return float(self.slopes[0, ii])
        return float(self.slopes[gi, ii])

    def eta(self, gi: int) -> np.ndarray:
        return self.mean[gi] + self.sd[gi] * QUAD_Z

    def log_density(self) -> float:
        """Marginal log-likelihood under the current parameters."""
        total = 0.0
        for gi, group in enumerate(self.groups):
            total += np.dot(self._group_loglik(gi)[0], group.w)
        return float(total)

    def _group_loglik(self, gi: int) -> tuple[np.ndarray, np.ndarray]:
        eta = self.eta(gi)
        x = self.groups[gi].x
        log_p = np.zeros((self.groups[gi].n, N_QUAD))
        for ii in range(len(ITEMS)):
            probs = cat_probs(self.slope_of(gi, ii), self.thresholds_of(gi, ii), eta)
            log_p += np.log(probs.T[x[:, ii]])
        log_w = np.log(QUAD_W)[None, :]
        log_lik = logsumexp(log_p + log_w, axis=1)
        return log_lik, log_p

    # -- estimation -----------------------------------------------------------

    def fit(self, max_iter: int = 300, tol: float = 1e-6, verbose: bool = False) -> dict:
        history = []
        previous = -np.inf
        decreases = 0
        start = time.time()
        for iteration in range(max_iter):
            loglik, stats = self._estep()
            history.append(loglik)
            # Relative stopping rule: an absolute tolerance of 1e-6 is not
            # reachable in reasonable time at this sample size, and a change
            # smaller than 1e-6 of the log-likelihood is far below any
            # inferential consequence.
            if iteration and abs(loglik - previous) < tol * max(1.0, abs(loglik)):
                break
            if loglik < previous - 1e-6:
                # block coordinate ascent should be monotone; a tiny numerical
                # dip is tolerated and recorded rather than fatal
                decreases += 1
                if decreases > 3:
                    raise RuntimeError("EM likelihood decreased repeatedly")
            previous = loglik
            self._mstep(stats)
            if verbose and iteration % 10 == 0:
                print(f"    iter {iteration:4d} loglik {loglik:.4f}", flush=True)
        return {
            "loglik": float(loglik),
            "iterations": iteration + 1,
            "likelihood_decreases": decreases,
            "seconds": round(time.time() - start, 1),
            "history_tail": [round(v, 6) for v in history[-3:]],
        }

    def _estep(self) -> tuple[float, list[dict]]:
        total = 0.0
        stats = []
        for gi, group in enumerate(self.groups):
            log_lik, log_p = self._group_loglik(gi)
            total += float(np.dot(log_lik, group.w))
            posterior = np.exp(log_p + np.log(QUAD_W)[None, :] - log_lik[:, None])
            posterior *= group.w[:, None] / group.w.mean()
            counts = []
            for ii in range(len(ITEMS)):
                onehot = np.zeros((group.n, N_CAT))
                onehot[np.arange(group.n), group.x[:, ii]] = 1.0
                counts.append(onehot.T @ posterior)
            stats.append({"counts": counts, "group": group})
        return total, stats

    def _mstep(self, stats: list[dict]) -> None:
        for ii, item in enumerate(ITEMS):
            shared = self.item_is_shared(item)
            if self.kind == "configural":
                for gi in range(self.n_groups):
                    self._fit_item_group(ii, gi, stats[gi]["counts"][ii])
            elif self.kind == "metric":
                self._fit_item_shared_slope(ii, stats)
            elif shared:
                self._fit_item_shared(ii, stats)
            else:
                self._fit_item_partial(ii, stats)
        if self.free_sd:
            for gi in range(1, self.n_groups):
                self._fit_group_scale(gi)
        if self.free_mean:
            for gi in range(1, self.n_groups):
                self._fit_group_mean(gi)

    # M-step helpers ----------------------------------------------------------

    @staticmethod
    def _q(counts: np.ndarray, probs: np.ndarray) -> float:
        return float(np.sum(counts * np.log(probs).T))

    def _fit_item_group(self, ii: int, gi: int, counts: np.ndarray) -> None:
        eta = self.eta(gi)

        def objective(p):
            a = math.exp(p[3])
            probs = cat_probs(a, ordered_thresholds(p[0:3]), eta)
            return -self._q(counts, probs)

        x0 = np.concatenate([threshold_params(self.thresholds[gi, ii]),
                             [math.log(self.slopes[gi, ii])]])
        result = minimize(objective, x0, method="L-BFGS-B",
                          bounds=[(-8, 8), (-8, 8), (-8, 8), (math.log(0.3), math.log(8))])
        self.thresholds[gi, ii] = ordered_thresholds(result.x[0:3])
        self.slopes[gi, ii] = math.exp(result.x[3])

    def _fit_item_shared_slope(self, ii: int, stats: list[dict]) -> None:
        """Metric model: one slope, group-specific thresholds."""

        def objective(p):
            a = math.exp(p[0])
            total = 0.0
            offset = 1
            for gi, st in enumerate(stats):
                probs = cat_probs(a, ordered_thresholds(p[offset:offset + 3]), self.eta(gi))
                total += self._q(st["counts"][ii], probs)
                offset += 3
            return -total

        x0 = [math.log(self.slopes[0, ii])]
        for gi in range(self.n_groups):
            x0.extend(threshold_params(self.thresholds[gi, ii]))
        bounds = [(math.log(0.3), math.log(8))] + [(-8, 8), (-8, 8), (-8, 8)] * self.n_groups
        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
        self.slopes[0, ii] = math.exp(result.x[0])
        for gi in range(self.n_groups):
            self.thresholds[gi, ii] = ordered_thresholds(result.x[1 + 3 * gi:4 + 3 * gi])

    def _fit_item_shared(self, ii: int, stats: list[dict]) -> None:
        """Scalar/partial model, anchor item: one slope and one threshold set."""

        def objective(p):
            a = math.exp(p[3])
            b = ordered_thresholds(p[0:3])
            total = 0.0
            for gi, st in enumerate(stats):
                probs = cat_probs(a, b, self.eta(gi))
                total += self._q(st["counts"][ii], probs)
            return -total

        x0 = np.concatenate([threshold_params(self.thresholds[0, ii]),
                             [math.log(self.slopes[0, ii])]])
        result = minimize(objective, x0, method="L-BFGS-B",
                          bounds=[(-8, 8), (-8, 8), (-8, 8), (math.log(0.3), math.log(8))])
        self.thresholds[0, ii] = ordered_thresholds(result.x[0:3])
        self.slopes[0, ii] = math.exp(result.x[3])

    def _fit_item_partial(self, ii: int, stats: list[dict]) -> None:
        """Partial model, released item: shared slope, group-specific thresholds."""
        self._fit_item_shared_slope(ii, stats)

    def _fit_group_mean(self, gi: int) -> None:
        group = self.groups[gi]
        x = group.x

        def objective(p):
            model_mean = float(p[0])
            eta = model_mean + self.sd[gi] * QUAD_Z
            log_p = np.zeros((group.n, N_QUAD))
            for ii in range(len(ITEMS)):
                probs = cat_probs(self.slope_of(gi, ii),
                                  np.asarray(self.thresholds_of(gi, ii), dtype=float), eta)
                log_p += np.log(probs.T[x[:, ii]])
            weighted = logsumexp(log_p + np.log(QUAD_W)[None, :], axis=1)
            return -float((weighted * group.w).sum())

        result = minimize(objective, [self.mean[gi]], method="L-BFGS-B",
                          bounds=[(-4, 4)])
        self.mean[gi] = float(result.x[0])

    def _fit_group_scale(self, gi: int) -> None:
        group = self.groups[gi]
        x = group.x

        def objective(p):
            sd = math.exp(p[0])
            eta = self.mean[gi] + sd * QUAD_Z
            log_p = np.zeros((group.n, N_QUAD))
            for ii in range(len(ITEMS)):
                probs = cat_probs(self.slope_of(gi, ii), self.thresholds_of(gi, ii), eta)
                log_p += np.log(probs.T[x[:, ii]])
            weighted = logsumexp(log_p + np.log(QUAD_W)[None, :], axis=1)
            return -float((weighted * group.w).sum())

        result = minimize(objective, [math.log(self.sd[gi])], method="L-BFGS-B",
                          bounds=[(math.log(0.3), math.log(3.0))])
        self.sd[gi] = math.exp(float(result.x[0]))


# --------------------------------------------------------------------------
# derived quantities
# --------------------------------------------------------------------------

def expected_item_score(model: Model, gi: int, ii: int, eta: np.ndarray) -> np.ndarray:
    b = np.asarray(model.thresholds_of(gi, ii), dtype=float)
    probs = cat_probs(model.slope_of(gi, ii), b, eta)
    return probs @ np.arange(N_CAT)


def dif_effect_sizes(model: Model, reference: int = 0) -> pd.DataFrame:
    """Expected item-score difference vs the reference cycle at matched trait."""
    eta = model.mean[reference] + model.sd[reference] * QUAD_Z
    weights = QUAD_W
    reference_of = {}
    rows = []
    for ii, item in enumerate(ITEMS):
        reference_of[item] = expected_item_score(model, reference, ii, eta)
    for gi, group in enumerate(model.groups):
        if gi == reference:
            continue
        for ii, item in enumerate(ITEMS):
            focal = expected_item_score(model, gi, ii, eta)
            delta = float(np.sum(weights * (focal - reference_of[item])) / weights.sum())
            if abs(delta) < 1e-9:
                continue
            rows.append({
                "cycle": group.label,
                "item": item,
                "expected_score_diff": round(delta, 4),
                "absolute_diff": round(abs(delta), 4),
            })
    return pd.DataFrame(rows)


def implied_prevalence(model: Model, gi: int, reference: int = 0) -> dict:
    """Model-implied PHQ-9 threshold prevalence under own vs reference items."""
    eta = model.eta(gi)
    weights = QUAD_W
    own = np.zeros((N_QUAD, 1))
    own[:, 0] = 1.0
    ref = np.zeros((N_QUAD, 1))
    ref[:, 0] = 1.0
    for ii in range(len(ITEMS)):
        probs_own = cat_probs(model.slope_of(gi, ii), model.thresholds_of(gi, ii), eta)
        probs_ref = cat_probs(model.slope_of(reference, ii),
                              model.thresholds_of(reference, ii), eta)
        own = np.array([np.convolve(own[row], probs_own[row]) for row in range(N_QUAD)])
        ref = np.array([np.convolve(ref[row], probs_ref[row]) for row in range(N_QUAD)])
    own_dist = (own * weights[:, None]).sum(axis=0)
    ref_dist = (ref * weights[:, None]).sum(axis=0)
    total = np.arange(own_dist.shape[0])
    return {
        "cycle": model.groups[gi].label,
        "implied_mean_own_items": round(float((own_dist * total).sum()), 3),
        "implied_mean_reference_items": round(float((ref_dist * total).sum()), 3),
        "implied_prevalence_10_own_items": round(float(own_dist[total >= 10].sum()), 4),
        "implied_prevalence_10_reference_items": round(float(ref_dist[total >= 10].sum()), 4),
        "implied_prevalence_5_own_items": round(float(own_dist[total >= 5].sum()), 4),
        "implied_prevalence_5_reference_items": round(float(ref_dist[total >= 5].sum()), 4),
    }


# --------------------------------------------------------------------------
# model comparison
# --------------------------------------------------------------------------

def fit_sequence(groups: list[Group], max_iter: int, warm: Model | None = None,
                 verbose: bool = True) -> dict[str, Model]:
    models: dict[str, Model] = {}
    order = ["configural", "metric", "scalar"]
    for kind in order:
        start = time.time()
        model = Model(groups, kind)
        if warm is not None and kind == "scalar":
            model.thresholds[0] = warm.thresholds[0].copy()
            model.slopes[0] = warm.slopes[0].copy()
            model.mean = warm.mean.copy()
            model.sd = warm.sd.copy()
        info = model.fit(max_iter=max_iter, verbose=verbose)
        model.info = info
        if verbose:
            print(f"  {kind:11s} loglik {info['loglik']:.3f} "
                  f"par {model.n_parameters():3d} ({info['seconds']}s, "
                  f"{info['iterations']} iters)", flush=True)
        models[kind] = model
        warm = model
    return models


def compare(models: dict[str, Model], reference: str = "configural") -> pd.DataFrame:
    rows = []
    n_obs = sum(g.n for g in next(iter(models.values())).groups)
    base = models[reference]
    base_df = base.n_parameters()
    for kind, model in models.items():
        loglik = model.info["loglik"]
        df = model.n_parameters()
        lr = 2 * (base.info["loglik"] - model.info["loglik"])
        delta_df = base_df - df
        p = float(chi2.sf(lr, delta_df)) if delta_df > 0 else np.nan
        rows.append({
            "model": kind,
            "free_items": ",".join(model.free_items),
            "n_parameters": df,
            "loglik": round(loglik, 3),
            "AIC": round(-2 * loglik + 2 * df, 2),
            "BIC": round(-2 * loglik + df * math.log(n_obs), 2),
            "aBIC": round(-2 * loglik + df * math.log((n_obs + 2) / 24), 2),
            "df_vs_configural": delta_df if delta_df > 0 else 0,
            "chisq_vs_configural": round(lr, 2) if delta_df > 0 else 0.0,
            "p_vs_configural": p,
        })
    return pd.DataFrame(rows)


def partial_sequence(groups: list[Group], scalar: Model, max_iter: int,
                     max_free: int = 4, verbose: bool = True) -> tuple[list[dict], list[Model]]:
    """Release the items with the largest single-item DIF, one at a time."""
    baseline = scalar.info["loglik"]
    base_df = scalar.n_parameters()
    single = []
    for item in ITEMS:
        model = Model(groups, "partial", free_items=(item,))
        model.thresholds[0] = scalar.thresholds[0].copy()
        model.slopes[0] = scalar.slopes[0].copy()
        model.mean = scalar.mean.copy()
        model.sd = scalar.sd.copy()
        info = model.fit(max_iter=max_iter, tol=1e-7)
        model.info = info
        lr = 2 * (info["loglik"] - baseline)
        df = model.n_parameters() - base_df
        single.append({
            "item": item,
            "loglik": round(info["loglik"], 3),
            "chisq": round(lr, 2),
            "df": df,
            "p": float(chi2.sf(lr, df)),
            "seconds": info["seconds"],
        })
        if verbose:
            print(f"    single-item release {item}: "
                  f"chisq {lr:8.1f} df {df:3d}", flush=True)
    single.sort(key=lambda row: -row["chisq"])

    sequence = [{
        "n_free": 0,
        "free_items": "",
        "loglik": round(baseline, 3),
        "chisq_vs_scalar": 0.0,
        "AIC": round(-2 * baseline + 2 * base_df, 2),
        "BIC": round(-2 * baseline + base_df * math.log(sum(g.n for g in groups)), 2),
    }]
    models = [scalar]
    free: list[str] = []
    for item in [row["item"] for row in single][:max_free]:
        free.append(item)
        model = Model(groups, "partial", free_items=tuple(sorted(free)))
        model.thresholds[0] = scalar.thresholds[0].copy()
        model.slopes[0] = scalar.slopes[0].copy()
        model.mean = scalar.mean.copy()
        model.sd = scalar.sd.copy()
        info = model.fit(max_iter=max_iter, tol=1e-7)
        model.info = info
        df = model.n_parameters()
        n_obs = sum(g.n for g in groups)
        sequence.append({
            "n_free": len(free),
            "free_items": ",".join(sorted(free)),
            "loglik": round(info["loglik"], 3),
            "chisq_vs_scalar": round(2 * (info["loglik"] - baseline), 2),
            "AIC": round(-2 * info["loglik"] + 2 * df, 2),
            "BIC": round(-2 * info["loglik"] + df * math.log(n_obs), 2),
        })
        models.append(model)
        if verbose:
            print(f"    cumulative release {sorted(free)}: "
                  f"BIC {sequence[-1]['BIC']:.1f}", flush=True)
    return single, sequence


# --------------------------------------------------------------------------
# simulation self-test
# --------------------------------------------------------------------------

def simulate(rng, groups_spec, n_per_group=1200, n_items=6):
    """Simulate responses from a graded response model with known parameters."""
    slopes = rng.uniform(1.0, 2.2, size=n_items)
    thresholds = np.sort(rng.uniform(-1.4, 1.4, size=(n_items, 3)), axis=1)
    data = {}
    items = [f"I{i}" for i in range(n_items)]
    for label, (mean, sd, dif_item, dif_shift) in groups_spec.items():
        z = rng.normal(mean, sd, size=n_per_group)
        x = np.zeros((n_per_group, n_items), dtype=int)
        for ii in range(n_items):
            b = thresholds[ii].copy()
            if dif_item is not None and ii == dif_item:
                b = b + dif_shift
            probs = cat_probs(slopes[ii], b, z)
            x[:, ii] = [rng.choice(N_CAT, p=probs[row]) for row in range(n_per_group)]
        frame = pd.DataFrame(x, columns=items)
        frame["wt"] = 1.0
        data[label] = frame
    return data, slopes, thresholds, items


def run_selftest() -> dict:
    rng = np.random.default_rng(20260929)

    class Fake(Group):
        def __init__(self, label, frame, items):
            self.label = label
            self.x = frame[items].to_numpy(dtype=np.int64)
            self.n = self.x.shape[0]
            self.w = np.ones(self.n)

    global ITEMS
    keep = ITEMS
    results = {}
    scenarios = {
        "A_scalar_truth": (
            {"g1": (0.0, 1.0, None, 0.0), "g2": (0.35, 1.0, None, 0.0),
             "g3": (-0.25, 1.0, None, 0.0)}, "scalar"),
        "B_scalar_unequal_variance_truth": (
            {"g1": (0.0, 1.0, None, 0.0), "g2": (0.0, 1.35, None, 0.0),
             "g3": (0.0, 0.80, None, 0.0)}, "scalar"),
        "C_dif_truth": (
            {"g1": (0.0, 1.0, None, 0.0), "g2": (0.0, 1.0, 2, 0.9),
             "g3": (0.0, 1.0, 2, 0.9)}, "partial"),
    }
    for name, (spec, expected) in scenarios.items():
        data, slopes, thresholds, items = simulate(rng, spec)
        ITEMS = items
        try:
            groups = [Fake(label, data[label], items) for label in data]
            models = fit_sequence(groups, max_iter=800, verbose=False)
            table = compare(models)
            single, sequence = partial_sequence(groups, models["scalar"],
                                                max_iter=400, max_free=2, verbose=False)
            ll = {kind: models[kind].info["loglik"] for kind in models}
            par = {kind: models[kind].n_parameters() for kind in models}
            chisq_metric = 2 * (ll["configural"] - ll["metric"])
            df_metric = par["configural"] - par["metric"]
            chisq_scalar = 2 * (ll["metric"] - ll["scalar"])
            df_scalar = par["metric"] - par["scalar"]
            best_partial = min(sequence, key=lambda row: row["BIC"])
            flagged = [row["item"] for row in single][:2]
            recovered_sd = [round(float(v), 3) for v in models["metric"].sd]
            checks = {
                "expect_scalar_rejected": bool(chi2.sf(chisq_scalar, df_scalar) < 0.01),
                "expect_metric_rejected": bool(chi2.sf(chisq_metric, df_metric) < 0.01),
                "BIC_order": {
                    "configural": round(table.iloc[0]["BIC"], 1),
                    "metric": round(float(table[table.model == "metric"].iloc[0]["BIC"]), 1),
                    "scalar": round(float(table[table.model == "scalar"].iloc[0]["BIC"]), 1),
                },
                "BIC_best": table.loc[table.BIC.idxmin(), "model"],
                "best_partial_model": best_partial["free_items"] or "none",
                "top_single_item_releases": flagged,
                "metric_sd_recovery": recovered_sd,
                "scalar_latent_mean_recovery": [round(float(v), 3)
                                                for v in models["scalar"].mean],
                "max_abs_slope_error_configural": round(float(np.nanmax(np.abs(
                    models["configural"].slopes - slopes))), 3),
                "max_abs_threshold_error_configural": round(float(np.nanmax(np.abs(
                    models["configural"].thresholds - thresholds))), 3),
            }
            if expected == "scalar":
                checks["verdict"] = "pass" if (
                    not checks["expect_scalar_rejected"]
                    and not checks["expect_metric_rejected"]) else "FAIL"
            elif expected == "metric":
                checks["verdict"] = "pass" if (
                    checks["expect_scalar_rejected"]
                    and not checks["expect_metric_rejected"]) else "FAIL"
            else:
                checks["verdict"] = "pass" if (
                    checks["expect_scalar_rejected"]
                    and best_partial["free_items"] == "I2") else "FAIL"
            checks["expected"] = expected
            results[name] = checks
        finally:
            ITEMS = keep
    results["all_passed"] = all(row.get("verdict") == "pass" for row in results.values())
    return results


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def analyse(cycles: list[str], tag: str, max_iter: int, weighted: bool = False,
            verbose: bool = True) -> dict:
    groups = load_groups(cycles, weighted=weighted)
    if verbose:
        print(f"[{tag}] groups: "
              + ", ".join(f"{g.label} n={g.n}" for g in groups), flush=True)
    models = fit_sequence(groups, max_iter=max_iter, verbose=verbose)
    table = compare(models)
    aux_iter = max_iter
    single, sequence = partial_sequence(groups, models["scalar"], max_iter=aux_iter,
                                        verbose=verbose)
    best_index = int(np.argmin([row["BIC"] for row in sequence]))
    if best_index == 0:
        final = models["scalar"]
        final_name = "scalar"
    else:
        final = Model(groups, "partial",
                      free_items=tuple(sequence[best_index]["free_items"].split(",")))
        final.thresholds[0] = models["scalar"].thresholds[0].copy()
        final.slopes[0] = models["scalar"].slopes[0].copy()
        final.mean = models["scalar"].mean.copy()
        final.sd = models["scalar"].sd.copy()
        final.info = final.fit(max_iter=aux_iter, tol=1e-6)
        final_name = sequence[best_index]["free_items"]

    # Practical size of the departures. The scalar model is preferred by BIC, so
    # its item parameters are identical across cycles and carry no information
    # about the size of the departures that the likelihood-ratio tests detect.
    # Two specifications do carry it: the metric model (every item's thresholds
    # free, which is the least restrictive model in which a common metric is
    # still defined) and a partial model that releases the single item with the
    # largest departure, with the other eight items as anchors.
    top_item = single[0]["item"]
    partial_one = Model(groups, "partial", free_items=(top_item,))
    partial_one.thresholds[0] = models["scalar"].thresholds[0].copy()
    partial_one.slopes[0] = models["scalar"].slopes[0].copy()
    partial_one.mean = models["scalar"].mean.copy()
    partial_one.sd = models["scalar"].sd.copy()
    partial_one.info = partial_one.fit(max_iter=aux_iter, tol=1e-6)

    dif = dif_effect_sizes(models["metric"])
    dif_partial = dif_effect_sizes(partial_one)
    scale = pd.DataFrame([implied_prevalence(models["metric"], gi)
                          for gi in range(len(groups))])
    scale_partial = pd.DataFrame([implied_prevalence(partial_one, gi)
                                  for gi in range(len(groups))])

    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / f"invariance_models_{tag}.csv", index=False)
    pd.DataFrame(single).to_csv(OUT / f"single_item_release_{tag}.csv", index=False)
    pd.DataFrame(sequence).to_csv(OUT / f"partial_sequence_{tag}.csv", index=False)
    dif.to_csv(OUT / f"dif_effect_sizes_{tag}.csv", index=False)
    scale.to_csv(OUT / f"scale_impact_{tag}.csv", index=False)
    dif_partial.to_csv(OUT / f"dif_effect_sizes_partial_{tag}.csv", index=False)
    scale_partial.to_csv(OUT / f"scale_impact_partial_{tag}.csv", index=False)
    params = {
        "cycles": [g.label for g in groups],
        "model": final_name,
        "slopes": [round(float(v), 4) for v in final.slopes[0]],
        "thresholds_reference": [[round(float(v), 4) for v in row]
                                 for row in final.thresholds[0]],
        "latent_means": [round(float(v), 4) for v in final.mean],
        "latent_sds": [round(float(v), 4) for v in final.sd],
        "metric_slopes": [round(float(v), 4) for v in models["metric"].slopes[0]],
        "metric_thresholds": [[[round(float(v), 4) for v in row]
                               for row in block] for block in models["metric"].thresholds],
        "metric_latent_sds": [round(float(v), 4) for v in models["metric"].sd],
        "top_dif_item": top_item,
        "max_abs_dif_points_metric": round(float(dif.absolute_diff.max()), 4)
        if len(dif) else 0.0,
        "max_abs_dif_points_partial": round(float(dif_partial.absolute_diff.max()), 4)
        if len(dif_partial) else 0.0,
        "max_implied_prevalence_shift_metric": round(float(
            (scale.implied_prevalence_10_own_items
             - scale.implied_prevalence_10_reference_items).abs().max()), 5),
        "max_implied_prevalence_shift_partial": round(float(
            (scale_partial.implied_prevalence_10_own_items
             - scale_partial.implied_prevalence_10_reference_items).abs().max()), 5),
        "loglik": round(float(final.info["loglik"]), 3),
        "n_parameters": final.n_parameters(),
    }
    (OUT / f"final_params_{tag}.json").write_text(json.dumps(params, indent=1), encoding="utf-8")
    return {"table": table, "single": single, "sequence": sequence, "dif": dif,
            "dif_partial": dif_partial, "scale": scale, "scale_partial": scale_partial,
            "final_name": final_name, "top_item": top_item, "models": models,
            "params": params}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--max-iter", type=int, default=1200)
    parser.add_argument("--skip-post", action="store_true")
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    globals()["OUT"] = args.out
    OUT.mkdir(parents=True, exist_ok=True)
    if args.selftest:
        print("running simulation self-test", flush=True)
        result = run_selftest()
        (OUT / "simulation_validation.json").write_text(
            json.dumps(result, indent=1), encoding="utf-8")
        print(json.dumps(result, indent=1))
        return

    summary = {}
    tag_suffix = "_weighted" if args.weighted else ""

    def summarise(result: dict) -> dict:
        table = result["table"]
        metric_row = table[table.model == "metric"].iloc[0]
        scalar_row = table[table.model == "scalar"].iloc[0]
        return {
            "n": sum(g.n for g in result["models"]["scalar"].groups),
            "n_by_cycle": {g.label: g.n for g in result["models"]["scalar"].groups},
            "convergence": {kind: model.info for kind, model in result["models"].items()},
            "loglik": {row["model"]: row["loglik"] for _, row in table.iterrows()},
            "n_parameters": {row["model"]: int(row["n_parameters"])
                             for _, row in table.iterrows()},
            "BIC": {row["model"]: row["BIC"] for _, row in table.iterrows()},
            "AIC": {row["model"]: row["AIC"] for _, row in table.iterrows()},
            "metric_vs_configural": {
                "chisq": float(metric_row["chisq_vs_configural"]),
                "df": int(metric_row["df_vs_configural"]),
                "p": float(metric_row["p_vs_configural"]),
            },
            "scalar_vs_metric": {
                "chisq": round(2 * (metric_row["loglik"] - scalar_row["loglik"]), 2),
                "df": int(metric_row["n_parameters"] - scalar_row["n_parameters"]),
                "p": float(chi2.sf(2 * (metric_row["loglik"] - scalar_row["loglik"]),
                                   metric_row["n_parameters"] - scalar_row["n_parameters"])),
            },
            "BIC_preferred_model": table.loc[table.BIC.idxmin(), "model"],
            "top_dif_item": result["top_item"],
            "single_item_chisq": {row["item"]: row["chisq"] for row in result["single"]},
            "max_abs_dif_points_metric": result["params"]["max_abs_dif_points_metric"],
            "max_abs_dif_points_partial": result["params"]["max_abs_dif_points_partial"],
            "max_implied_prevalence_shift_metric": result["params"]["max_implied_prevalence_shift_metric"],
            "max_implied_prevalence_shift_partial": result["params"]["max_implied_prevalence_shift_partial"],
            "latent_means_scalar": [round(float(v), 4) for v in result["models"]["scalar"].mean],
            "latent_sds_metric": [round(float(v), 4) for v in result["models"]["metric"].sd],
            "latent_sds_scalar": [round(float(v), 4) for v in result["models"]["scalar"].sd],
        }

    primary = analyse(STANDARD_CYCLES, "standard6" + tag_suffix, args.max_iter,
                      weighted=args.weighted)
    summary["standard6" + tag_suffix] = summarise(primary)
    if not args.skip_post:
        post = analyse(STANDARD_CYCLES + POST_CYCLE, "with2021" + tag_suffix,
                       args.max_iter, weighted=args.weighted)
        summary["with2021" + tag_suffix] = summarise(post)
    (OUT / "invariance_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
