"""Canonical smoothed-subgrid / N_eff PSS and manuscript SC-CV.

No boundary extrapolation, pseudocount, density normalization, or truth tuning.
Coordinate ties follow the manuscript's null-event zero convention.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
VERSION = "smoothed_subgrid_neff_v2"
REASONS = ["covered", "outside_global_range", "empty_cell", "singleton_cell",
           "outside_local_range", "zero_spacing", "degenerate_training_sample"]


@lru_cache(None)
def library():
    lib = ctypes.CDLL(str(HERE / ("libpss_v2.dylib" if sys.platform == "darwin" else "libpss_v2.so")))
    double = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    integer = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    lib.pss_v2_evaluate.argtypes = [double, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   double, ctypes.c_int, double, integer, integer, double]
    lib.pss_v2_evaluate.restype = ctypes.c_int
    return lib


def matrix(x):
    x = np.ascontiguousarray(x, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError("A nonempty finite n-by-d matrix is required")
    return x


def evaluate(train, query, ell):
    train, query = matrix(train), matrix(query)
    n, d = train.shape
    if query.shape[1] != d or ell < 1 or int(ell) != ell or ell**d >= 2**63:
        raise ValueError("Invalid dimensions or partition count")
    logf = np.empty(len(query)); sizes = np.empty(len(query), dtype=np.int32)
    reasons = np.empty(len(query), dtype=np.int32); summary = np.empty(5)
    status = library().pss_v2_evaluate(train, n, d, int(ell), query, len(query),
                                      logf, sizes, reasons, summary)
    if status:
        raise RuntimeError(f"PSS v2 core status {status}")
    return dict(log_density=logf, cell_size=sizes, reason=reasons,
                occupied_cells=int(summary[0]), min_cell_size=int(summary[1]),
                singleton_cells=int(summary[2]), integrated_mass=summary[3],
                degenerate=bool(summary[4]))


def estimate(x, ell):
    x = matrix(x)
    values = evaluate(x, x, ell)
    valid = np.isfinite(values["log_density"])
    return dict(estimate=-float(values["log_density"][valid].mean()) if valid.any() else 0.0,
                coverage=float(valid.mean()), n_valid=int(valid.sum()),
                skipped_point_fraction=float(1-valid.mean()),
                mean_cell_size=len(x)/values["occupied_cells"] if values["occupied_cells"] else 0.,
                **{k: values[k] for k in ["occupied_cells", "min_cell_size", "singleton_cells",
                                          "integrated_mass", "degenerate"]})


def select_from_table(table, tau=.99):
    if not 0 <= tau <= 1:
        raise ValueError("tau must be in [0, 1]")
    finite = [r for r in table if np.isfinite(r["cv_score"])]
    feasible = [r for r in finite if r["stable_validation_coverage"] >= tau]
    if feasible:
        best = min(feasible, key=lambda r: (r["cv_score"], r["ell"]))
        mode = "stable_coverage_constrained"
    elif finite:
        best = min(finite, key=lambda r: (-r["stable_validation_coverage"], r["cv_score"], r["ell"]))
        mode = "fallback_max_stable_coverage"
    else:
        best = min(table, key=lambda r: r["ell"])
        mode = "failed_no_covered_validation_points"
    return dict(ell_star=int(best["ell"]), selection_rule=mode,
                fallback_mode=not bool(feasible), feasible_candidates=len(feasible), **
                {k: best[k] for k in ["cv_score", "cv_coverage", "stable_validation_coverage"]})


def select_sc_cv(x, candidates, n_folds=3, tau=.99, n_min=10, seed=42, fold_id=None):
    x = matrix(x); n = len(x)
    if int(n_min) != n_min or n_min < 2 or int(n_folds) != n_folds or not 2 <= n_folds <= n:
        raise ValueError("n_min >= 2 and 2 <= n_folds <= n must be integers")
    candidates = list(candidates)
    if not candidates or any(int(v) != v or v < 1 for v in candidates):
        raise ValueError("Nonempty positive integer candidate grid required")
    candidates = sorted(set(map(int, candidates)))
    if fold_id is None:
        fold_id = np.random.default_rng(seed).permutation(np.arange(n) % n_folds)
    fold_id = np.asarray(fold_id)
    if fold_id.shape != (n,) or set(fold_id.tolist()) != set(range(n_folds)):
        raise ValueError("fold_id must contain every label 0,...,n_folds-1")
    start = time.perf_counter(); table = []
    folds = [(x[fold_id != k], x[fold_id == k]) for k in range(n_folds)]
    for ell in candidates:
        log_sum = 0.; covered = stable = 0; reason_counts = np.zeros(len(REASONS), int)
        fold_coverage = []
        for train, validation in folds:
            result = evaluate(train, validation, ell)
            logf = result["log_density"]; ok = np.isfinite(logf)
            covered += int(ok.sum()); log_sum += logf[ok].sum()
            good = ok & (result["cell_size"] >= n_min)
            stable += int(good.sum()); fold_coverage.append(float(good.mean()))
            reason_counts += np.bincount(result["reason"], minlength=len(REASONS))
        table.append(dict(ell=ell, cv_score=-log_sum/covered if covered else float("inf"),
                          cv_coverage=covered/n, stable_validation_coverage=stable/n,
                          min_fold_stable_coverage=min(fold_coverage),
                          feasible=bool(covered and stable/n >= tau),
                          **{f"fraction_{name}": count/n for name, count in zip(REASONS, reason_counts)}))
    return dict(**select_from_table(table, tau), cv_table=table, tau=tau, n_min=n_min,
                n_folds=n_folds, selection_seconds=time.perf_counter()-start,
                definition=VERSION)


def mixed_mi(x, y, ell):
    """Entropy-difference diagnostic; not clipped to the binary MI bounds."""
    x = matrix(x); y = np.asarray(y)
    if y.shape != (len(x),):
        raise ValueError("One discrete label per observation required")
    labels, counts = np.unique(y, return_counts=True)
    if len(labels) < 2 or min(counts) < 2:
        raise ValueError("At least two classes with at least two samples each required")
    joint = estimate(x, ell)
    conditional = [estimate(x[y == label], ell) for label in labels]
    probabilities = counts/len(x)
    entropy = -float(np.dot(probabilities, np.log(probabilities)))
    value = joint["estimate"]-sum(p*c["estimate"] for p,c in zip(probabilities,conditional))
    return dict(mi=value, label_entropy=entropy, outside_mi_bounds=not 0 <= value <= entropy,
                coverage_all=joint["coverage"],
                min_conditional_coverage=min(c["coverage"] for c in conditional),
                degenerate=joint["degenerate"] or any(c["degenerate"] for c in conditional))
