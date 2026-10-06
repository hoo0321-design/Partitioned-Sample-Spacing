import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
from sklearn.feature_selection import mutual_info_classif

from core import (Selector, choose, classify, configs, estimate, kl_entropy, labels,
                  ross_mi, selection_data, split_indices)
from pss_v2 import select_sc_cv
from run import SEEDS, splits


class Tests(unittest.TestCase):
    def test_nested_splits(self):
        for seed in SEEDS:
            tr, te, it, iv = splits(seed)
            self.assertEqual(len(tr), 13814)
            self.assertEqual(len(te), 5921)
            self.assertEqual(len(np.intersect1d(tr, te)), 0)
            self.assertEqual(len(np.intersect1d(it, iv)), 0)
            np.testing.assert_array_equal(np.union1d(it, iv), tr)
            np.testing.assert_array_equal(np.union1d(tr, te), np.arange(19735))

    def test_median_training_only(self):
        y, q, threshold = labels([1, 2, 3, 4, 5], [-100, 3, 100])
        self.assertEqual(threshold, 3)
        np.testing.assert_array_equal(y, [0, 0, 0, 1, 1])
        np.testing.assert_array_equal(q, [0, 0, 1])

    def test_deterministic_smoothing_and_all_rows(self):
        x = np.random.default_rng(18).integers(0, 10, (100, 25)).astype(float)
        z, active = selection_data(x, 1e-5, 19)
        q, _ = selection_data(x*2+3, 1e-5, 19)
        np.testing.assert_allclose(z, q)
        self.assertEqual(len(z), len(x))
        self.assertEqual(len(active), 25)
        self.assertTrue(np.all(np.diff(np.sort(z, axis=0), axis=0) > 0))

    def test_equal_search_budgets(self):
        for method in ["PSS SC-CV", "Joint Ross MI", "Univariate Ross MI"]:
            self.assertEqual(len(configs(method)), 12)

    def test_pooled_versus_guard(self):
        table = [dict(ell=1, score=.1, cv_score=3., pooled_stable_10=.995, minimum_stable_10=.99),
                 dict(ell=2, score=.2, cv_score=1., pooled_stable_10=.96, minimum_stable_10=.80)]
        config = dict(tau=.95, n_min=10)
        self.assertEqual(choose(table, config)["ell"], 2)
        self.assertEqual(choose(table, config, True)["ell"], 1)
        result = choose(table, dict(tau=1., n_min=10))
        self.assertTrue(result["fallback"])
        self.assertEqual(result["ell"], 1)

    def test_canonical_sc_agreement(self):
        rng = np.random.default_rng(26)
        x = rng.normal(size=(600, 2))
        y = np.arange(len(x)) % 2
        selector = Selector(x, y, [0, 1], 27)
        actual = selector.table([0, 1])
        for minimum in [5, 10]:
            reference = select_sc_cv(x, [1, 2, 3, 4, 5], tau=.95, n_min=minimum, fold_id=selector.fold_id)
            for a, b in zip(actual, reference["cv_table"]):
                self.assertAlmostEqual(a["cv_score"], b["cv_score"], places=10)
                self.assertAlmostEqual(a[f"pooled_stable_{minimum}"], b["stable_validation_coverage"], places=12)
            self.assertEqual(choose(actual, dict(tau=.95, n_min=minimum))["ell"], reference["ell_star"])

    def test_pss_affine_and_additivity(self):
        rng = np.random.default_rng(31)
        x = rng.normal(size=(800, 3))
        y = np.arange(800) % 2
        a = Selector(x, y, range(3), 5)
        b = Selector(x*np.array([2, 3, 4])+5, y, range(3), 5)
        for ell in [1, 2, 3]:
            self.assertAlmostEqual(a.fixed([0, 1, 2], ell)["score"], b.fixed([0, 1, 2], ell)["score"], places=9)
        self.assertAlmostEqual(a.fixed([0, 1, 2], 1)["score"], sum(a.fixed([j], 1)["score"] for j in range(3)), places=9)

    def test_ross_matches_sklearn_scalar(self):
        rng = np.random.default_rng(68)
        y = np.arange(1000) % 2
        x = (rng.normal(size=1000)+y*2)[:, None]
        for k in [1, 3, 10]:
            reference = mutual_info_classif(x, y, discrete_features=False, n_neighbors=k, random_state=1)[0]
            self.assertAlmostEqual(ross_mi(x, y, k), reference, places=10)

    def test_kl_matches_r_fnn(self):
        rng = np.random.default_rng(79)
        x = rng.normal(size=(180, 3))
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/"x.csv"
            np.savetxt(p, x, delimiter=",")
            result = subprocess.run(["/usr/local/bin/Rscript", "-e",
                f'x<-as.matrix(read.csv("{p}",header=FALSE));cat(sprintf("%.14f",FNN::entropy(x,k=5)))'],
                check=True, text=True, capture_output=True)
            values = list(map(float, result.stdout.split()))
            for k in [1, 3, 5]:
                self.assertAlmostEqual(kl_entropy(x, k), values[k-1], places=10)

    def test_svm_matches_r_defaults(self):
        rng = np.random.default_rng(82)
        x = rng.normal(size=(300, 4))
        y = (x[:, 0]+x[:, 1]**2 > .5).astype(int)
        ours = classify(x[:200], y[:200], x[200:], y[200:], list(range(4)))
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/"data.csv"
            np.savetxt(p, np.column_stack([x, y]), delimiter=",")
            script = f'z<-read.csv("{p}",header=FALSE);m<-e1071::svm(x=z[1:200,1:4],y=as.factor(z[1:200,5]),kernel="radial");cat(as.character(predict(m,z[201:300,1:4])))'
            result = subprocess.run(["/usr/local/bin/Rscript", "-e", script], check=True, text=True, capture_output=True)
            np.testing.assert_array_equal(ours["prediction"], list(map(int, result.stdout.split())))


if __name__ == "__main__":
    unittest.main(verbosity=2)
