"""Full-joint greedy PSS selection with a single theory-rule coefficient.

The prescribed ell uses the full selection-training sample size and the
candidate subset dimension. That same ell is used for the pooled and both
conditional entropy components. There is no CV coverage guard or fallback.
"""
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.energy_sameperiod import core as original

COEFFICIENTS = [1, 1.5, 2, 2.5, 3, 4]
COUNTS = [5, 10, 20]
NOISE = 1e-5
TUNED = "PSS theory C tuned"
FIXED = "PSS theory C=1"
BASE_PSS = "PSS class-aware SC-CV"
BASE_KL = "KL tuned"


def theory_ell(n, d, coefficient):
    """Round C*(n/(d**6*log(n)**2))**(1/(d+8)) to nearest, half upward."""
    if not math.isfinite(n) or int(n) != n or n <= 1:
        raise ValueError("n must be an integer larger than one")
    if not math.isfinite(d) or int(d) != d or d < 1:
        raise ValueError("d must be a positive integer")
    if not math.isfinite(coefficient) or coefficient <= 0:
        raise ValueError("The coefficient must be positive and finite")
    rate = (n / (d**6 * math.log(n)**2)) ** (1 / (d+8))
    return max(1, math.floor(coefficient * rate + .5))


class Selector(original.Selector):
    def __init__(self, x, y, active, seed):
        super().__init__(np.asarray(x), np.asarray(y), active, seed)
        self.invalid_cache = {}

    def coefficient_candidate(self, selected, feature, ell):
        """Evaluate the unchanged full-joint MI and retain explicit failures."""
        features = tuple(sorted([*selected, int(feature)]))
        key = (features, ell)
        if key in self.invalid_cache:
            result = self.invalid_cache[key]
        else:
            try:
                result = dict(self.fixed(features, ell), valid=True)
            except ValueError as error:
                if str(error) != "Entropy component has no valid points":
                    raise
                result = dict(ell=ell, score=None, valid=False, error=str(error),
                              training_coverage=None, conditional_training_coverage=None,
                              integrated_mass=None)
            if result["valid"] and not np.isfinite(result["score"]):
                result.update(score=None, valid=False, error="Nonfinite mutual-information score")
            if not result["valid"]:
                self.invalid_cache[key] = result
        return dict(result, feature=int(feature), features=list(features))

    def path_coefficient(self, coefficient, max_features=20):
        start = time.perf_counter()
        if (int(max_features) != max_features
                or not 1 <= max_features <= len(self.active)):
            raise ValueError("max_features must be between one and the active feature count")
        active = sorted(self.active)
        if len(set(active)) != len(active):
            raise ValueError("Active feature indices must be distinct")
        # Validate the coefficient before starting any estimator work.
        theory_ell(len(self.x), 1, coefficient)
        selected, history = [], []
        failure = None
        for step in range(1, int(max_features)+1):
            ell = theory_ell(len(self.x), step, coefficient)
            candidates = [self.coefficient_candidate(selected, feature, ell)
                          for feature in active if feature not in selected]
            usable = [row for row in candidates if row["valid"]]
            if not usable:
                failure = dict(step=step, ell=ell, features=selected.copy(),
                               candidates=candidates, error="No valid candidates at required ell")
                break
            chosen = min(usable, key=lambda row: (-row["score"], row["feature"]))
            selected.append(chosen["feature"])
            history.append(dict(chosen, step=step, features=selected.copy(),
                                candidates=candidates))
        return dict(coefficient=coefficient, status="failed" if failure else "complete",
                    failure=failure, history=history, n_selection=len(self.x),
                    selection_seconds=time.perf_counter()-start)
