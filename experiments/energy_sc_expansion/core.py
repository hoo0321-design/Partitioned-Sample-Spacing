"""Coverage/fold sensitivity without changing the canonical PSS density."""
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.energy_sameperiod import core as original

TAUS = [.80, .85, .90, .95, .99]
FOLDS = [3, 5, 10]
KS = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
ELLS = [1, 2, 3, 4, 5]
NOISE = 1e-5
N_MIN = 5
COUNTS = [5, 10, 20]
PRIMARY = ["PSS expanded SC-CV", "KL tuned", "Ross tuned", "Univariate tuned"]


def configurations(method):
    if method == PRIMARY[0]:
        return [dict(estimator="pss", folds=k, tau=tau, n_min=N_MIN)
                for k in FOLDS for tau in TAUS]
    estimator = {"KL tuned": "kl", "Ross tuned": "ross", "Univariate tuned": "univariate"}[method]
    return [dict(estimator=estimator, k=k) for k in KS]


class Selector:
    def __init__(self, x, y, active, seed):
        self.x, self.y = np.asarray(x), np.asarray(y)
        self.active = list(map(int, active))
        self.base = original.Selector(self.x, self.y, self.active, seed)
        self.tables, self.kl_cache, self.fold_ids = {}, {}, {}
        for k in FOLDS:
            ids = np.empty(len(y), dtype=int)
            rng = np.random.default_rng(seed+91)
            for label in np.unique(y):
                rows = np.flatnonzero(y == label)
                if len(rows) < k:
                    raise ValueError("Each label must occur in every validation fold")
                ids[rows] = rng.permutation(np.arange(len(rows)) % k)
            self.fold_ids[k] = ids

    def cv(self, x, fold_id, folds, ell):
        covered, stable, logsum = 0, 0, 0.
        for fold in range(folds):
            values = original.evaluate(x[fold_id != fold], x[fold_id == fold], ell)
            valid = np.isfinite(values["log_density"])
            covered += int(valid.sum())
            stable += int((valid & (values["cell_size"] >= N_MIN)).sum())
            logsum += float(values["log_density"][valid].sum())
        return dict(nll=-logsum/covered if covered else float("inf"),
                    coverage=covered/len(x), stable=stable/len(x))

    def table(self, features, folds):
        key = (tuple(sorted(features)), folds)
        if key not in self.tables:
            x = self.x[:, key[0]]
            rows = []
            for ell in ELLS:
                stats = self.cv(x, self.fold_ids[folds], folds, ell)
                rows.append(dict(**self.base.fixed(key[0], ell), cv_score=stats["nll"],
                                 pooled_cv_coverage=stats["coverage"], pooled_stable_5=stats["stable"]))
            self.tables[key] = rows
        return self.tables[key]

    def kl(self, features, k):
        key = (tuple(sorted(features)), k)
        if key not in self.kl_cache:
            x = self.x[:, key[0]]
            value = original.kl_entropy(x, k) - sum(
                np.mean(self.y == label)*original.kl_entropy(x[self.y == label], k)
                for label in [0, 1])
            self.kl_cache[key] = value
        return self.kl_cache[key]

    def path(self, config, max_features=20):
        start = time.perf_counter()
        estimator = config["estimator"]
        selected, history = [], []
        for step in range(1, max_features+1):
            options = []
            for j in self.active:
                if j in selected:
                    continue
                features = selected+[j]
                if estimator == "pss":
                    result = original.choose(self.table(features, config["folds"]), config)
                elif estimator == "fixed":
                    result = self.base.fixed(features, config["ell"])
                elif estimator == "kl":
                    result = dict(score=self.kl(features, config["k"]))
                elif estimator in ["ross", "univariate"]:
                    result = dict(score=self.base.ross(features if estimator == "ross" else [j], config["k"]))
                else:
                    raise ValueError(estimator)
                options.append((result["score"], j, result))
            _, j, result = min(options, key=lambda row: (-row[0], row[1]))
            selected.append(j)
            history.append(dict(**result, step=step, feature=j, features=selected.copy()))
        return dict(history=history, selection_seconds=time.perf_counter()-start,
                    n_selection=len(self.x))

    def diagnostics(self, path, config):
        if config["estimator"] not in ["pss", "fixed"]:
            return
        folds = config.get("folds", 3)
        for row in path["history"]:
            diag = next(r for r in self.table(row["features"], folds) if r["ell"] == row["ell"])
            row.update(diag)
            x = self.x[:, row["features"]]
            conditional = [self.cv(x[self.y == label], self.fold_ids[folds][self.y == label], folds, row["ell"])
                           for label in [0, 1]]
            row["conditional_cv_coverage"] = min(r["coverage"] for r in conditional)
            row["minimum_stable_5"] = min(row["pooled_stable_5"], *(r["stable"] for r in conditional))
            row["diagnostic_folds"] = folds
