"""Selection oracles, prescribed ell checks, and canonical PSS equivalence."""
import math
import unittest
from unittest.mock import patch

import numpy as np

from experiments.energy_theory_c.core import Selector, theory_ell, COEFFICIENTS, original
from PSS.pss_v2 import mixed_mi


class TheoryCoefficientTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(8933)
        self.x = rng.normal(size=(400, 4))
        self.y = (self.x[:, 0] + .7*self.x[:, 1] > 0).astype(int)
        self.selector = Selector(self.x, self.y, [3, 2, 1, 0], 46)

    @staticmethod
    def row(score, ell):
        return dict(score=score, ell=ell, training_coverage=1.,
                    conditional_training_coverage=1., integrated_mass=.99)

    def test_rule_grid_coefficient_outside_root_and_refit_sample_size(self):
        # Independently computed at 50-digit Decimal precision. In particular,
        # the final entry changes on refit even though C remains fixed at four.
        dimensions = [1, 2, 3, 5, 10, 20]
        expected_inner = [[2, 1, 1, 1, 1, 1], [3, 2, 1, 1, 1, 1],
                          [3, 2, 2, 1, 1, 1], [4, 3, 2, 2, 2, 2],
                          [5, 3, 3, 2, 2, 2], [7, 4, 3, 3, 2, 2]]
        expected_outer = [row.copy() for row in expected_inner]
        expected_outer[-1][-1] = 3
        for n, expected in [(9669, expected_inner), (13814, expected_outer)]:
            actual = [[theory_ell(n, d, c) for d in dimensions] for c in COEFFICIENTS]
            self.assertEqual(actual, expected)

    def test_nearest_rounding_half_up_and_lower_bound(self):
        rate = (9669/(math.log(9669)**2))**(1/9)
        self.assertEqual(theory_ell(9669, 1, 2.5/rate), 3)
        self.assertEqual(theory_ell(9669, 1, (2.5-1e-10)/rate), 2)
        self.assertEqual(theory_ell(9669, 1, (2.5+1e-10)/rate), 3)
        self.assertEqual(theory_ell(9669, 20, 1e-8), 1)

    def test_greedy_joint_oracle_low_index_ties_and_no_clipping(self):
        scores = {(0,): .4, (1,): .55, (2,): .55, (3,): .1,
                  (0, 1): .7, (1, 2): .8, (1, 3): .8,
                  (0, 1, 2): -.3, (1, 2, 3): -.1, (0, 1, 2, 3): 1.7}
        with patch.object(self.selector, "fixed",
                          side_effect=lambda f, ell: self.row(scores[tuple(f)], ell)):
            path = self.selector.path_coefficient(4, 4)
        self.assertEqual(path["status"], "complete")
        self.assertIsNone(path["failure"])
        self.assertEqual([r["feature"] for r in path["history"]], [1, 2, 3, 0])
        self.assertEqual([r["score"] for r in path["history"]], [.55, .8, -.1, 1.7])
        self.assertEqual([len(r["candidates"]) for r in path["history"]], [4, 3, 2, 1])
        for row in path["history"]:
            self.assertEqual(row["ell"], theory_ell(len(self.x), row["step"], 4))
            self.assertTrue(all(c["ell"] == row["ell"] for c in row["candidates"]))

    def test_c1_first_two_steps_match_direct_canonical_full_joint_scores(self):
        rng = np.random.default_rng(394)
        x = rng.normal(size=(9669, 4))
        y = (x[:, 0]+.7*x[:, 1] > 0).astype(int)
        selector = Selector(x, y, [3, 2, 1, 0], 46)
        canonical = original.Selector(x, y, range(4), 46)
        path = selector.path_coefficient(1, 2)
        selected = []
        for step, row in enumerate(path["history"], 1):
            ell = theory_ell(len(x), step, 1)
            estimates = [(canonical.fixed(selected+[j], ell), j)
                         for j in range(4) if j not in selected]
            expected, feature = min(estimates, key=lambda item: (-item[0]["score"], item[1]))
            selected.append(feature)
            self.assertEqual(row["features"], selected)
            for field, value in expected.items():
                self.assertEqual(row[field], value)
            self.assertAlmostEqual(row["score"], mixed_mi(x[:, selected], y, ell)["mi"], places=13)
        self.assertEqual([r["ell"] for r in path["history"]], [2, 1])

    def test_prefix_permutation_and_shared_cache_across_coefficients(self):
        full = self.selector.path_coefficient(1, 4)
        keys = set(self.selector.fixed_cache)
        with patch.object(original, "estimate", side_effect=AssertionError("Recomputed PSS")):
            short = self.selector.path_coefficient(1, 2)
            self.selector.active.reverse()
            repeated = self.selector.path_coefficient(1, 4)
            # At this sample size these coefficients round to the same ell=1
            # at all four dimensions, so full estimator work must be reused.
            same_ells = self.selector.path_coefficient(.9, 4)
        self.assertEqual(keys, set(self.selector.fixed_cache))
        self.assertEqual(short["history"], full["history"][:2])
        self.assertEqual(repeated["history"], full["history"])
        self.assertEqual(same_ells["history"], full["history"])

    def test_invalid_candidate_retained_but_other_candidate_selected(self):
        def fixed(features, ell):
            if features == (0,):
                raise ValueError("Entropy component has no valid points")
            return self.row(-float(features[0]), ell)
        with patch.object(self.selector, "fixed", side_effect=fixed):
            path = self.selector.path_coefficient(3, 1)
        self.assertEqual(path["status"], "complete")
        self.assertEqual(path["history"][0]["feature"], 1)
        rejected = path["history"][0]["candidates"][0]
        self.assertFalse(rejected["valid"])
        self.assertIsNone(rejected["score"])
        self.assertEqual(rejected["features"], [0])
        self.assertEqual(rejected["error"], "Entropy component has no valid points")

    def test_all_invalid_failure_retains_partial_path_without_changing_ell(self):
        def fixed(features, ell):
            if len(features) > 1:
                raise ValueError("Entropy component has no valid points")
            return self.row(-float(features[0]), ell)
        with patch.object(self.selector, "fixed", side_effect=fixed) as calls:
            path = self.selector.path_coefficient(4, 4)
            self.assertEqual(calls.call_count, 7)
            cached_invalid = self.selector.coefficient_candidate([0], 1, theory_ell(400, 2, 4))
            self.assertEqual(calls.call_count, 7)
        self.assertFalse(cached_invalid["valid"])
        self.assertEqual(path["status"], "failed")
        self.assertEqual(len(path["history"]), 1)
        self.assertEqual(path["failure"]["features"], [0])
        self.assertEqual(path["failure"]["step"], 2)
        self.assertEqual(path["failure"]["ell"], theory_ell(400, 2, 4))
        self.assertEqual(len(path["failure"]["candidates"]), 3)
        self.assertTrue(all(not r["valid"] for r in path["failure"]["candidates"]))

    def test_nonfinite_scores_inadmissible_and_unrelated_errors_not_hidden(self):
        with patch.object(self.selector, "fixed", return_value=self.row(float("nan"), 1)):
            path = self.selector.path_coefficient(1, 2)
        self.assertEqual(path["status"], "failed")
        self.assertEqual(path["history"], [])
        self.assertTrue(all(r["score"] is None for r in path["failure"]["candidates"]))
        self.selector.invalid_cache.clear()
        with patch.object(self.selector, "fixed", side_effect=ValueError("Unexpected error")):
            with self.assertRaisesRegex(ValueError, "Unexpected error"):
                self.selector.path_coefficient(1, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
