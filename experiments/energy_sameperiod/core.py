"""Same-period Energy evaluation; all selection inputs are training-only."""
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma, gammaln
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "PSS"))
from pss_v2 import estimate, evaluate
from experiments.energy_v2.core import ross_mi

TAUS = [.90, .95, .99]
MINIMUMS = [5, 10]
NOISES = [1e-5, 1e-4]
KS = [1, 3, 5, 10, 15, 20]
ELLS = [1, 2, 3, 4, 5]
COUNTS = [5, 10, 20]
MAIN = ["PSS SC-CV", "Joint Ross MI", "Univariate Ross MI"]
ABLATIONS = ["PSS default SC-CV", "PSS component guard", "PSS ell=1",
             "PSS ell=2", "PSS global SC-CV", "KL difference k=1"]


def configs(method):
    if method == "PSS SC-CV":
        return [dict(method=method, noise=noise, tau=tau, n_min=minimum)
                for noise in NOISES for tau in TAUS for minimum in MINIMUMS]
    return [dict(method=method, noise=noise, k=k) for noise in NOISES for k in KS]


def split_indices(indices, seed, fraction=.7):
    # No target labels or full-data median are used to form random holdouts.
    shuffled = np.random.default_rng(seed).permutation(np.asarray(indices))
    n_train = int(np.floor(len(shuffled) * fraction))
    return np.sort(shuffled[:n_train]), np.sort(shuffled[n_train:])


def labels(train_target, other_target):
    threshold = float(np.median(train_target))
    return ((np.asarray(train_target) > threshold).astype(int),
            (np.asarray(other_target) > threshold).astype(int), threshold)


def selection_data(train_x, noise, seed):
    x = np.asarray(train_x, dtype=float)
    span = np.ptp(x, axis=0)
    active = np.flatnonzero(span > 0)
    if len(active) < 20 or not np.isfinite(x).all():
        raise ValueError("Require at least 20 finite, nonconstant predictors")
    z = (x - x.min(0)) / np.where(span > 0, span, 1.)
    z += np.random.default_rng(seed).normal(0., noise, size=z.shape)
    if np.any(np.diff(np.sort(z[:, active], axis=0), axis=0) == 0):
        raise ValueError("Unresolved coordinate ties")
    return z, active


def choose(table, config, guard=False):
    prefix = "minimum_stable" if guard else "pooled_stable"
    key = f"{prefix}_{config['n_min']}"
    finite = [r for r in table if np.isfinite(r["cv_score"]) and np.isfinite(r["score"])]
    feasible = [r for r in finite if r[key] >= config["tau"]]
    if not finite:
        raise ValueError("No usable SC-CV candidates")
    if feasible:
        best = min(feasible, key=lambda r: (r["cv_score"], r["ell"]))
    else:
        best = min(finite, key=lambda r: (-r[key], r["cv_score"], r["ell"]))
    return dict(**best, stable_coverage=best[key], fallback=not bool(feasible),
                feasible_candidates=len(feasible), guard=guard)


def kl_entropy(x, k):
    x = np.asarray(x)
    n, d = x.shape
    radius = cKDTree(x).query(x, k=k+1, workers=1)[0][:, -1]
    if np.any(radius <= 0) or k >= n:
        raise ValueError("Invalid KL radii")
    return float(digamma(n) - digamma(k) + d/2*np.log(np.pi) - gammaln(1+d/2)
                 + d*np.log(radius).mean())


class Selector:
    def __init__(self, x, y, active, seed):
        self.x = x
        self.y = y
        self.active = list(map(int, active))
        self.pss_cache = {}
        self.ross_cache = {}
        self.fixed_cache = {}
        self.kl_cache = {}
        self.fold_id = np.empty(len(y), dtype=int)
        rng = np.random.default_rng(seed + 91)
        for label in np.unique(y):
            rows = np.flatnonzero(y == label)
            self.fold_id[rows] = rng.permutation(np.arange(len(rows)) % 3)

    def fixed(self, features, ell):
        key = (tuple(sorted(features)), ell)
        if key in self.fixed_cache:
            return self.fixed_cache[key]
        z = self.x[:, key[0]]
        full = [estimate(z, ell)] + [estimate(z[self.y == label], ell) for label in [0, 1]]
        if min(r["n_valid"] for r in full) == 0:
            raise ValueError("Entropy component has no valid points")
        score = full[0]["estimate"] - sum(np.mean(self.y == label)*full[label+1]["estimate"]
                                          for label in [0, 1])
        value = dict(ell=ell, score=score, training_coverage=full[0]["coverage"],
                     conditional_training_coverage=min(r["coverage"] for r in full[1:]),
                     integrated_mass=full[0]["integrated_mass"])
        self.fixed_cache[key] = value
        return value

    def table(self, features):
        key = tuple(sorted(features))
        if key in self.pss_cache:
            return self.pss_cache[key]
        x = self.x[:, key]
        components = [np.ones(len(x), bool), self.y == 0, self.y == 1]
        table = []
        for ell in ELLS:
            stats = []
            for mask in components:
                z = x[mask]
                folds = self.fold_id[mask]
                covered = 0
                logsum = 0.
                stable = {minimum: 0 for minimum in MINIMUMS}
                for fold in range(3):
                    result = evaluate(z[folds != fold], z[folds == fold], ell)
                    good = np.isfinite(result["log_density"])
                    covered += int(good.sum())
                    logsum += float(result["log_density"][good].sum())
                    for minimum in MINIMUMS:
                        stable[minimum] += int((good & (result["cell_size"] >= minimum)).sum())
                stats.append(dict(nll=-logsum/covered if covered else float("inf"),
                                  coverage=covered/len(z),
                                  **{f"stable_{m}": count/len(z) for m, count in stable.items()}))
            value = dict(**self.fixed(key, ell), cv_score=stats[0]["nll"],
                         pooled_cv_coverage=stats[0]["coverage"],
                         conditional_cv_coverage=min(r["coverage"] for r in stats[1:]))
            for minimum in MINIMUMS:
                value[f"pooled_stable_{minimum}"] = stats[0][f"stable_{minimum}"]
                value[f"minimum_stable_{minimum}"] = min(r[f"stable_{minimum}"] for r in stats)
            table.append(value)
        self.pss_cache[key] = table
        return table

    def ross(self, features, k):
        key = (tuple(sorted(features)), k)
        if key not in self.ross_cache:
            self.ross_cache[key] = ross_mi(self.x[:, key[0]], self.y, k)
        return self.ross_cache[key]

    def kl(self, features):
        key = tuple(sorted(features))
        if key not in self.kl_cache:
            z = self.x[:, key]
            self.kl_cache[key] = kl_entropy(z, 1) - sum(
                np.mean(self.y == label)*kl_entropy(z[self.y == label], 1) for label in [0, 1])
        return self.kl_cache[key]

    def path(self, config, max_features=20):
        start = time.perf_counter()
        method = config["method"]
        selected = []
        history = []
        global_choice = choose(self.table(self.active), config) if method == "PSS global SC-CV" else None
        if method in ["PSS ell=1", "Univariate Ross MI"]:
            values = [(self.fixed([j], 1)["score"] if method == "PSS ell=1"
                       else self.ross([j], config["k"]), j) for j in self.active]
            total = 0.
            for score, j in sorted(values, key=lambda row: (-row[0], row[1]))[:max_features]:
                selected.append(j)
                total += score
                history.append(dict(step=len(selected), feature=j, features=selected.copy(),
                                    score=total if method == "PSS ell=1" else score,
                                    **({"ell": 1} if method == "PSS ell=1" else {})))
        else:
            for step in range(1, max_features+1):
                options = []
                for j in self.active:
                    if j in selected:
                        continue
                    features = selected + [j]
                    if method in ["PSS SC-CV", "PSS default SC-CV", "PSS component guard"]:
                        result = choose(self.table(features), config, method == "PSS component guard")
                    elif method in ["PSS ell=2", "PSS global SC-CV"]:
                        result = self.fixed(features, 2 if global_choice is None else global_choice["ell"])
                    elif method == "KL difference k=1":
                        result = dict(score=self.kl(features))
                    else:
                        result = dict(score=self.ross(features, config["k"]))
                    options.append((result["score"], j, result))
                _, j, result = min(options, key=lambda row: (-row[0], row[1]))
                selected.append(j)
                history.append(dict(**result, step=step, feature=j, features=selected.copy()))
        # Every PSS path gets coverage diagnostics, including fixed-ell ablations.
        # This post-selection work is NOT included in the selector runtime.
        seconds = time.perf_counter()-start
        if method.startswith("PSS"):
            for row in history:
                diag = next(r for r in self.table(row["features"]) if r["ell"] == row["ell"])
                row.update(diag)
        return dict(history=history, selection_seconds=seconds, n_selection=len(self.x),
                    global_choice=global_choice)


def classify(train_x, train_y, test_x, test_y, features):
    features = sorted(features)
    a = train_x[:, features]
    b = test_x[:, features]
    # e1071-style sample-SD scaling, C=1, gamma=1/number_of_features.
    center = a.mean(0)
    scale = np.where(a.std(0, ddof=1) > 0, a.std(0, ddof=1), 1.)
    model = SVC(C=1., gamma=1/len(features), class_weight=None, cache_size=256)
    start = time.perf_counter()
    model.fit((a-center)/scale, train_y)
    fit_seconds = time.perf_counter()-start
    start = time.perf_counter()
    prediction = model.predict((b-center)/scale)
    decision = model.decision_function((b-center)/scale)
    return dict(accuracy=float(accuracy_score(test_y, prediction)),
                balanced_accuracy=float(balanced_accuracy_score(test_y, prediction)),
                auc=float(roc_auc_score(test_y, decision)), fit_seconds=fit_seconds,
                prediction_seconds=time.perf_counter()-start,
                prediction=prediction, decision=decision)
