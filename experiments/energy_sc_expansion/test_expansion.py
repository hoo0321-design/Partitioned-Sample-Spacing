import unittest

import numpy as np

from core import FOLDS, KS, Selector, configurations, original
from pss_v2 import select_sc_cv


class ExpansionTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(441).normal(size=(500, 3))
        self.y = (self.x[:, 0]+self.x[:, 1] > 0).astype(int)
        self.new = Selector(self.x, self.y, range(3), 92)

    def test_old_threefold_path_unchanged(self):
        old = original.Selector(self.x, self.y, range(3), 92)
        for tau in [.80, .95, .99]:
            old_path = old.path(dict(method="PSS SC-CV", tau=tau, n_min=5), 3)
            config = dict(estimator="pss", tau=tau, n_min=5, folds=3)
            new_path = self.new.path(config, 3)
            self.new.diagnostics(new_path, config)
            for a, b in zip(old_path["history"], new_path["history"]):
                self.assertEqual(a["features"], b["features"])
                for field in ["ell", "score", "cv_score", "pooled_stable_5", "minimum_stable_5",
                              "pooled_cv_coverage", "conditional_cv_coverage", "training_coverage",
                              "conditional_training_coverage", "fallback"]:
                    self.assertAlmostEqual(a[field], b[field], places=10, msg=field)

    def test_all_fold_tables_match_canonical(self):
        for folds in FOLDS:
            ids = self.new.fold_ids[folds]
            for label in [0, 1]:
                counts = np.bincount(ids[self.y == label], minlength=folds)
                self.assertLessEqual(counts.max()-counts.min(), 1)
            reference = select_sc_cv(self.x[:, :2], [1, 2, 3, 4, 5], tau=.85, n_min=5,
                                     n_folds=folds, fold_id=ids)
            table = self.new.table([0, 1], folds)
            for a, b in zip(table, reference["cv_table"]):
                self.assertAlmostEqual(a["cv_score"], b["cv_score"], places=10)
                self.assertAlmostEqual(a["pooled_stable_5"], b["stable_validation_coverage"], places=12)

    def test_kl_k1_path_unchanged(self):
        old = original.Selector(self.x, self.y, range(3), 92)
        a = old.path(dict(method="KL difference k=1"), 3)
        b = self.new.path(dict(estimator="kl", k=1), 3)
        for left, right in zip(a["history"], b["history"]):
            self.assertEqual(left["features"], right["features"])
            self.assertAlmostEqual(left["score"], right["score"], places=10)
        for k in KS:
            expected = original.kl_entropy(self.x, k)-sum(
                np.mean(self.y == label)*original.kl_entropy(self.x[self.y == label], k)
                for label in [0, 1])
            self.assertAlmostEqual(self.new.kl([0, 1, 2], k), expected, places=12)

    def test_fixed_and_univariate_paths(self):
        a = self.new.path(dict(estimator="fixed", ell=1), 3)
        b = self.new.base.path(dict(method="PSS ell=1"), 3)
        for left, right in zip(a["history"], b["history"]):
            self.assertEqual(left["features"], right["features"])
            self.assertAlmostEqual(left["score"], right["score"], places=10)
        a = self.new.path(dict(estimator="univariate", k=5), 3)
        b = self.new.base.path(dict(method="Univariate Ross MI", k=5), 3)
        self.assertEqual(a["history"], b["history"])

    def test_candidate_budgets_and_no_forced_ell(self):
        pss = configurations("PSS expanded SC-CV")
        self.assertEqual(len(pss), 15)
        self.assertEqual(sum(c["folds"] in [3, 5] for c in pss), 10)
        for method in ["KL tuned", "Ross tuned", "Univariate tuned"]:
            self.assertEqual(len(configurations(method)), 10)
        self.assertEqual(self.new.table([0], 3)[0]["ell"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
