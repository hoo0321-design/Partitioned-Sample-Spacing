"""Class-aware coverage guard, without changing the PSS density or MI score."""
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.energy_sc_expansion.core import (Selector as PooledSelector, ELLS,
    FOLDS, TAUS, N_MIN, NOISE, COUNTS, configurations, original)

POOLED = "PSS pooled SC-CV"
GUARDED = "PSS class-aware SC-CV"
MATCHED = "PSS class-aware (pooled settings)"


class Selector(PooledSelector):
    def __init__(self, x, y, active, seed):
        super().__init__(x, y, active, seed)
        self.component_cache = {}

    def table(self, features, folds):
        key = (tuple(sorted(features)), folds)
        if key not in self.tables:
            rows = []
            for ell in ELLS:
                stats = self.cv(self.x[:, key[0]], self.fold_ids[folds], folds, ell)
                try:
                    entropy = self.base.fixed(key[0], ell)
                except ValueError as error:
                    if str(error) != "Entropy component has no valid points":
                        raise
                    entropy = dict(ell=ell, score=float("nan"))
                rows.append(dict(**entropy, cv_score=stats["nll"],
                                 pooled_cv_coverage=stats["coverage"], pooled_stable_5=stats["stable"]))
            self.tables[key] = rows
        return self.tables[key]

    def component_row(self, features, folds, row):
        key = (tuple(sorted(features)), folds, row["ell"])
        if key not in self.component_cache:
            x = self.x[:, key[0]]
            conditional = [self.cv(x[self.y == label], self.fold_ids[folds][self.y == label],
                                   folds, row["ell"]) for label in [0, 1]]
            self.component_cache[key] = dict(conditional_cv_coverage=min(c["coverage"] for c in conditional),
                conditional_stable_5=min(c["stable"] for c in conditional),
                minimum_stable_5=min(row["pooled_stable_5"], *(c["stable"] for c in conditional)),
                class0_stable=conditional[0]["stable"], class1_stable=conditional[1]["stable"])
        return dict(**row, **self.component_cache[key])

    def select_guard(self, features, config):
        rows = self.table(features, config["folds"])
        finite = [r for r in rows if np.isfinite(r["cv_score"]) and np.isfinite(r["score"])]
        # A class-aware feasible candidate must first pass the pooled constraint.
        potential = [self.component_row(features, config["folds"], r)
                     for r in finite if r["pooled_stable_5"] >= config["tau"]]
        if any(r["minimum_stable_5"] >= config["tau"] for r in potential):
            evaluated = potential
        else:
            evaluated = [self.component_row(features, config["folds"], r) for r in finite]
        selected = original.choose(evaluated, config, guard=True)
        return dict(**selected, invalid_candidates=len(rows)-len(finite))

    def path_guard(self, config, maximum=20):
        start = time.perf_counter()
        selected, history = [], []
        for step in range(1, maximum+1):
            candidates = []
            for feature in self.active:
                if feature in selected:
                    continue
                row = self.select_guard(selected+[feature], config)
                candidates.append((row["score"], feature, row))
            _, feature, row = min(candidates, key=lambda r: (-r[0], r[1]))
            selected.append(feature)
            history.append(dict(**row, step=step, feature=feature, features=selected.copy()))
        return dict(history=history, selection_seconds=time.perf_counter()-start, n_selection=len(self.x))

    def diagnose(self, path, config):
        for row in path["history"]:
            candidate = next(r for r in self.table(row["features"], config["folds"]) if r["ell"] == row["ell"])
            row.update(self.component_row(row["features"], config["folds"], candidate))
            p = np.bincount(self.y, minlength=2)/len(self.y)
            bound = -float(np.dot(p, np.log(p)))
            row.update(label_entropy=bound, outside_mi_bounds=not 0 <= row["score"] <= bound)
