"""Independent selection oracles and canonical estimator checks for JMI."""
import unittest
from unittest.mock import patch

import numpy as np

from experiments.energy_jmi.core import (
    Selector, GuardSelector, configurations, PSS_JMI, KL_JMI, FOLDS, TAUS, KS,
)
from PSS.pss_v2 import mixed_mi, select_sc_cv


class JMITests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(8713)
        self.x = rng.normal(size=(400, 4))
        self.y = (self.x[:, 0] + .6*self.x[:, 1] > 0).astype(int)
        self.selector = Selector(self.x, self.y, [3, 2, 1, 0], 92)
        self.kl_config = dict(estimator="kl_jmi", k=3)
        self.pss_config = dict(estimator="pss_jmi", folds=3, tau=.90, n_min=5)
        # Deliberate singleton and pair ties, followed by a different best
        # joint contribution; scores also exceed log(2) to catch clipping.
        self.scores = {(0,): .40, (1,): .55, (2,): .55, (3,): .10,
                       (0, 1): .70, (0, 2): .60, (0, 3): .95,
                       (1, 2): .80, (1, 3): .80, (2, 3): .99}

    def fake_kl(self, features, k):
        return self.scores[tuple(features)]

    def test_nontrivial_greedy_oracle_and_low_index_ties(self):
        with patch.object(self.selector, "kl", side_effect=self.fake_kl):
            path = self.selector.path_jmi(self.kl_config, 4)
        self.assertEqual([row["feature"] for row in path["history"]], [1, 2, 3, 0])
        np.testing.assert_allclose([row["score"] for row in path["history"]],
                                   [.55, .80, .895, .75], rtol=0, atol=1e-14)
        self.assertEqual(path["history"][2]["components"],
                         [dict(features=[1, 3], score=.80), dict(features=[2, 3], score=.99)])
        self.assertEqual(path["n_selection"], 400)
        self.assertEqual(len(path["score_table"]["singletons"]), 4)
        self.assertEqual(len(path["score_table"]["pairs"]), 6)

    def test_symmetry_active_permutation_cache_and_incremental_prefix(self):
        with patch.object(self.selector, "kl", side_effect=self.fake_kl) as kl:
            a = self.selector.jmi_component([3, 1], self.kl_config)
            b = self.selector.jmi_component([1, 3], self.kl_config)
            self.assertEqual(a, b)
            self.assertEqual(kl.call_count, 1)
            small = self.selector.path_jmi(self.kl_config, 2)
            full = self.selector.path_jmi(self.kl_config, 4)
            self.selector.active.reverse()
            permuted = self.selector.path_jmi(self.kl_config, 4)
            self.assertEqual(kl.call_count, 10)
        self.assertEqual(small["history"], full["history"][:2])
        self.assertEqual(permuted["history"], full["history"])
        self.assertEqual(permuted["score_table"], full["score_table"])
        a["features"][0] = 999
        a["score"] = 999
        self.assertEqual(self.selector.jmi_component([1, 3], self.kl_config), b)

    def test_negative_component_scores_are_retained(self):
        with patch.object(self.selector, "kl", return_value=-.25):
            path = self.selector.path_jmi(self.kl_config, 4)
        self.assertEqual([row["feature"] for row in path["history"]], [0, 1, 2, 3])
        self.assertTrue(all(row["score"] == -.25 for row in path["history"]))

    def test_two_dimensional_pss_score_and_all_guard_diagnostics(self):
        canonical_guard = GuardSelector(self.x, self.y, range(4), 92)
        expected = canonical_guard.select_guard([0, 1], self.pss_config)
        actual = self.selector.jmi_component([1, 0], self.pss_config)
        self.assertEqual(actual, dict(features=[0, 1], **expected))
        canonical_mi = mixed_mi(self.x[:, :2], self.y, actual["ell"])
        self.assertAlmostEqual(actual["score"], canonical_mi["mi"], places=13)
        for label in (None, 0, 1):
            mask = np.ones(len(self.y), bool) if label is None else self.y == label
            reference = select_sc_cv(
                self.x[mask, :2], [actual["ell"]], n_folds=3, n_min=5,
                fold_id=self.selector.fold_ids[3][mask],
            )
            coverage_key = "pooled_stable_5" if label is None else f"class{label}_stable"
            self.assertEqual(actual[coverage_key], reference["stable_validation_coverage"])
            if label is None:
                self.assertEqual(actual["cv_score"], reference["cv_score"])

    def test_pss_estimation_tables_reused_across_coverage_thresholds(self):
        # tau=1 forces a fallback and evaluates every finite ell component.
        self.selector.jmi_component([0, 1], dict(self.pss_config, tau=1.))
        with patch.object(self.selector, "cv", side_effect=AssertionError("Repeated CV")), \
             patch.object(self.selector.base, "fixed", side_effect=AssertionError("Repeated MI")):
            other = self.selector.jmi_component([1, 0], dict(self.pss_config, tau=.80))
        self.assertEqual(other["features"], [0, 1])

    def test_first_two_steps_match_original_full_joint_selectors(self):
        guard = self.selector.path_guard(self.pss_config, 2)
        pss_jmi = self.selector.path_jmi(self.pss_config, 2)
        kl = self.selector.path(dict(estimator="kl", k=3), 2)
        kl_jmi = self.selector.path_jmi(self.kl_config, 2)
        for baseline, jmi in [(guard, pss_jmi), (kl, kl_jmi)]:
            for a, b in zip(baseline["history"], jmi["history"]):
                self.assertEqual(a["features"], b["features"])
                self.assertEqual(a["score"], b["score"])

    def test_ell_one_jmi_degenerates_to_univariate_ranking(self):
        univariate = {j: self.selector.base.fixed([j], 1)["score"] for j in range(4)}
        for a in range(4):
            for b in range(a+1, 4):
                pair = self.selector.base.fixed([a, b], 1)["score"]
                self.assertAlmostEqual(pair, univariate[a]+univariate[b], places=13)
        # This is an explicit test-only ell=1 intervention; production has no
        # fixed-ell branch. If pair estimates collapse, JMI cannot fix redundancy.
        with patch.object(self.selector, "select_guard",
                          side_effect=lambda features, config: self.selector.base.fixed(features, 1)):
            path = self.selector.path_jmi(self.pss_config, 4)
        expected = sorted(range(4), key=lambda j: (-univariate[j], j))
        self.assertEqual([row["feature"] for row in path["history"]], expected)
        for row in path["history"][1:]:
            previous = row["features"][:-1]
            score = univariate[row["feature"]]+np.mean([univariate[j] for j in previous])
            self.assertAlmostEqual(row["score"], score, places=13)

    def test_original_parameter_grids(self):
        pss, kl = configurations(PSS_JMI), configurations(KL_JMI)
        self.assertEqual(len(pss), 15)
        self.assertEqual({(c["folds"], c["tau"]) for c in pss},
                         {(folds, tau) for folds in FOLDS for tau in TAUS})
        self.assertTrue(all(c["n_min"] == 5 and c["estimator"] == "pss_jmi" for c in pss))
        self.assertEqual([c["k"] for c in kl], KS)
        self.assertTrue(all(c["estimator"] == "kl_jmi" for c in kl))


if __name__ == "__main__":
    unittest.main(verbosity=2)
