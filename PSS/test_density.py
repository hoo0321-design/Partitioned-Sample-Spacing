"""Independent literal subgrid reference, without calling the C++ core."""
import unittest
import numpy as np
from pss_v2 import estimate, evaluate


def reference(train, query, ell):
    n, d = train.shape
    lo, hi = train.min(0), train.max(0)
    keys = np.clip(np.ceil(ell*(train-lo)/(hi-lo)).astype(int)-1, 0, ell-1)
    logs = []
    for point in query:
        if np.any(point < lo) or np.any(point > hi):
            logs.append(-np.inf)
            continue
        key = np.clip(np.ceil(ell*(point-lo)/(hi-lo)).astype(int)-1, 0, ell-1)
        cell = train[np.all(keys == key, axis=1)]
        size = len(cell)
        if size < 2:
            logs.append(-np.inf)
            continue
        m = int(np.sqrt(size)+.5)
        value = np.log(size/n)
        for j in range(d):
            t = sorted(cell[:, j])
            T = lambda r: t[max(1, min(size, r))-1]
            if point[j] < t[0] or point[j] > t[-1]:
                value = -np.inf
                break
            xi = [t[0]] + [sum(T(u) for u in range(r-m, r+m))/(2*m)
                           for r in range(1, size+1)] + [t[-1]]
            a = max(0, np.searchsorted(xi, point[j], side="left")-1)
            gap = T(a+m)-T(a-m)
            if gap <= 0:
                value = -np.inf
                break
            value += np.log(2*m/(size*gap))
        logs.append(value)
    return np.asarray(logs)


class DensityTests(unittest.TestCase):
    def test_literal_density_and_neff(self):
        rng = np.random.default_rng(912)
        for n in [3, 9, 45, 200]:
            for d in [1, 2, 5]:
                x = rng.random((n, d))
                query = np.vstack([rng.random((30, d)), x])
                for ell in [1, 2, 4]:
                    expected = reference(x, query, ell)
                    np.testing.assert_allclose(evaluate(x, query, ell)["log_density"], expected, atol=1e-10)
                    good = np.isfinite(expected[-n:])
                    entropy = estimate(x, ell)
                    self.assertEqual(entropy["n_valid"], good.sum())
                    expected_h = -expected[-n:][good].mean() if good.any() else 0.
                    self.assertAlmostEqual(entropy["estimate"], expected_h, places=10)

    def test_no_extrapolation(self):
        x = np.array([0., .1, .3, .7, .8, 1.])[:, None]
        q = np.array([-.01, 1.01, .4, .6, .1])[:, None]
        np.testing.assert_array_equal(evaluate(x, q, 2)["reason"], [1, 1, 4, 4, 0])

    def test_exact_ties_are_not_silently_jittered(self):
        result = estimate(np.array([[0., 0.], [0., 1.], [1., 2.]]), 1)
        self.assertTrue(result["degenerate"])
        self.assertEqual(result["n_valid"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
