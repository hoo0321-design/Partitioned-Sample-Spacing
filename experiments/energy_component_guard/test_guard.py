import unittest
from unittest.mock import patch

import numpy as np

from core import Selector, PooledSelector, original, FOLDS
from PSS.pss_v2 import select_sc_cv


class GuardTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(932)
        self.x = rng.normal(size=(800, 4))
        self.y = (self.x[:, 0] + .7*self.x[:, 1] > 0).astype(int)
        self.selector = Selector(self.x, self.y, range(4), 51)

    def test_pooled_tables_and_paths_unchanged(self):
        old = PooledSelector(self.x, self.y, range(4), 51)
        for folds in FOLDS:
            self.assertEqual(self.selector.table([0, 1], folds), old.table([0, 1], folds))
        config = dict(estimator="pss", folds=3, tau=.9, n_min=5)
        a, b = self.selector.path(config, 4), old.path(config, 4)
        self.assertEqual(a["history"], b["history"])

    def test_each_conditional_cv_matches_canonical(self):
        for folds in FOLDS:
            ids = self.selector.fold_ids[folds]
            candidates = self.selector.table([0, 1], folds)
            for label in [0, 1]:
                mask = self.y == label
                reference = select_sc_cv(self.x[mask, :2], [1, 2, 3, 4, 5], n_folds=folds,
                                         n_min=5, fold_id=ids[mask])
                for a, b in zip(candidates, reference["cv_table"]):
                    value = self.selector.component_row([0, 1], folds, a)
                    self.assertEqual(value[f"class{label}_stable"], b["stable_validation_coverage"])

    def test_lazy_gate_equals_full_evaluation_including_fallback(self):
        for features in [[0], [0, 1], [0, 1, 2, 3]]:
            for folds in FOLDS:
                full = [self.selector.component_row(features, folds, r) for r in self.selector.table(features, folds)]
                for tau in [.8, .85, .9, .95, .99, 1.]:
                    config = dict(folds=folds, tau=tau, n_min=5)
                    expected = original.choose(full, config, guard=True)
                    actual = self.selector.select_guard(features, config)
                    for field in expected:
                        self.assertEqual(actual[field], expected[field], (features, folds, tau, field))

    def test_guard_path_satisfies_all_components_when_feasible(self):
        config = dict(folds=3, tau=.95, n_min=5)
        path = self.selector.path_guard(config, 4)
        previous = []
        for row in path["history"]:
            self.assertEqual(row["features"][:-1], previous)
            previous = row["features"]
            if not row["fallback"]:
                for field in ["pooled_stable_5", "class0_stable", "class1_stable"]:
                    self.assertGreaterEqual(row[field], .95)

    def test_fixed_one_score_unchanged(self):
        old = PooledSelector(self.x, self.y, range(4), 51)
        self.assertEqual(self.selector.base.fixed([0, 1, 2], 1), old.base.fixed([0, 1, 2], 1))

    def test_pooled_feasible_can_fail_conditional_guard(self):
        rows = [dict(ell=1, score=.2, cv_score=1., pooled_stable_5=.99, minimum_stable_5=.96),
                dict(ell=2, score=.3, cv_score=0., pooled_stable_5=.99, minimum_stable_5=.50)]
        config = dict(folds=3, tau=.95, n_min=5)
        with patch.object(self.selector, "table", return_value=rows), \
             patch.object(self.selector, "component_row", side_effect=lambda f, k, r: r):
            self.assertEqual(original.choose(rows, config)["ell"], 2)
            self.assertEqual(self.selector.select_guard([0], config)["ell"], 1)

    def test_zero_effective_component_is_inadmissible(self):
        previous = self.selector.base.fixed
        def fixed(features, ell):
            if ell == 2:
                raise ValueError("Entropy component has no valid points")
            return previous(features, ell)
        with patch.object(self.selector.base, "fixed", side_effect=fixed):
            result = self.selector.select_guard([0, 1], dict(folds=3, tau=.8, n_min=5))
            self.assertEqual(result["invalid_candidates"], 1)
            self.assertNotEqual(result["ell"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
