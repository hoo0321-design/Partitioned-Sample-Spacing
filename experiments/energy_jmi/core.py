"""Pairwise JMI with the unchanged class-aware PSS and KL MI estimators.

The first variable maximizes univariate MI. Each later candidate maximizes
the mean of I((candidate, selected_variable); Y) over selected variables.
The full singleton/pair score table is retained to audit every greedy choice.
"""
from itertools import combinations
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.energy_component_guard.core import (
    Selector as GuardSelector, COUNTS, NOISE, N_MIN, FOLDS, TAUS, ELLS,
    original,
)
from experiments.energy_sc_expansion.core import KS

PSS_JMI = "PSS-JMI"
KL_JMI = "KL-JMI"
BASE_PSS = "PSS class-aware SC-CV"
BASE_KL = "KL tuned"
PRIMARY = [BASE_PSS, PSS_JMI, BASE_KL, KL_JMI]


def configurations(method):
    if method == PSS_JMI:
        return [dict(estimator="pss_jmi", folds=folds, tau=tau, n_min=N_MIN)
                for folds in FOLDS for tau in TAUS]
    if method == KL_JMI:
        return [dict(estimator="kl_jmi", k=k) for k in KS]
    raise ValueError(f"No new configuration grid for {method!r}")


class Selector(GuardSelector):
    def __init__(self, x, y, active, seed):
        super().__init__(x, y, active, seed)
        self.jmi_cache = {}

    def jmi_component(self, features, config):
        """Return a canonical singleton/pair estimate, preserving diagnostics."""
        features = tuple(sorted(map(int, features)))
        if len(features) not in (1, 2) or len(set(features)) != len(features):
            raise ValueError("JMI components must contain one or two distinct features")
        estimator = config["estimator"]
        if estimator == "pss_jmi":
            setting = (estimator, config["folds"], config["tau"], config["n_min"])
        elif estimator == "kl_jmi":
            setting = (estimator, config["k"])
        else:
            raise ValueError(f"Unsupported JMI estimator: {estimator!r}")
        key = (features, setting)
        if key not in self.jmi_cache:
            if estimator == "pss_jmi":
                row = self.select_guard(features, config)
            else:
                row = dict(score=self.kl(features, config["k"]))
            if not np.isfinite(row["score"]):
                raise ValueError(f"Nonfinite JMI component for {features}")
            self.jmi_cache[key] = dict(row)
        # A caller may annotate a returned row without altering cached estimates.
        return dict(features=list(features), **self.jmi_cache[key])

    def path_jmi(self, config, max_features=20):
        start = time.perf_counter()
        if not 1 <= max_features <= len(self.active) or int(max_features) != max_features:
            raise ValueError("max_features must be between one and the active feature count")
        active = sorted(self.active)
        if len(set(active)) != len(active):
            raise ValueError("Active feature indices must be distinct")
        table = dict(
            singletons=[self.jmi_component([j], config) for j in active],
            pairs=[self.jmi_component(pair, config) for pair in combinations(active, 2)],
        )
        lookup = {tuple(row["features"]): row
                  for rows in table.values() for row in rows}
        selected, history = [], []
        for step in range(1, int(max_features) + 1):
            candidates = []
            for feature in active:
                if feature in selected:
                    continue
                keys = ([tuple(sorted((feature, old))) for old in selected]
                        if selected else [(feature,)])
                components = [dict(lookup[key], features=list(key)) for key in keys]
                score = float(np.mean([row["score"] for row in components]))
                candidates.append((score, feature, components))
            score, feature, components = min(candidates, key=lambda row: (-row[0], row[1]))
            selected.append(feature)
            history.append(dict(step=step, feature=feature, features=selected.copy(),
                                score=score, components=components))
        return dict(history=history, score_table=table, n_selection=len(self.x),
                    selection_seconds=time.perf_counter()-start)
