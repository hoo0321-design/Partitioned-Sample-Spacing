# ==============================================================================
#  REVISED: k-NN SIMULATION FOR MVN (VARYING CORRELATION)
#  - Fixes Dimension (d=6) and Sample Size (N=10000).
#  - Varies Correlation (rho) -> [0.0, 0.3, 0.6, 0.9].
#  - Finds best k, Measures Time, Saves to "knn_mvn_results_rho.csv".
# ==============================================================================

# --- compatibility shims & imports -------------------------------------------
import sys, os, warnings, time

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    import numpy as np

    if not hasattr(np, "bool"): np.bool = bool

import builtins, itertools

if not hasattr(itertools, "izip"): itertools.izip = zip
if not hasattr(builtins, "xrange"): builtins.xrange = range
if not hasattr(builtins, "unicode"): builtins.unicode = str
if not hasattr(builtins, "long"): builtins.long = int

os.makedirs("util/util/figs", exist_ok=True)
os.makedirs("util/util/temp_data", exist_ok=True)

from scipy import stats
import matplotlib
# matplotlib.use("Agg") # PyCharm에서 그래프 창을 띄우려면 주석 처리
import matplotlib.pyplot as plt
import pandas as pd

# 사용자 라이브러리 (환경에 맞게 유지)
from ent_est.entropy import kl, ksg, tkl, tksg


# --- Simulator Class (Local Implementation) -----------------------------------
def equicorr(d, rho):
    """Generates an equi-correlated covariance matrix."""
    R = np.full((d, d), rho, dtype=float)
    np.fill_diagonal(R, 1.0)
    return R


class GaussianSimulator:
    def __init__(self, dim_x, rho=0.0):
        self.d = int(dim_x)
        self.rho = float(rho)
        self.mean = np.zeros(self.d)
        self.cov = equicorr(self.d, self.rho)

        # Check PD
        w, _ = np.linalg.eigh(self.cov)
        if np.any(w <= 0): raise ValueError("Covariance matrix is not positive definite")

    def true_entropy(self):
        """Analytical Entropy for Multivariate Normal"""
        sign, logdet = np.linalg.slogdet(self.cov)
        return 0.5 * np.log((2 * np.pi * np.e) ** self.d * np.exp(logdet))

    def sim(self, n):
        return np.random.multivariate_normal(self.mean, self.cov, size=int(n))


# --- Experiment Settings ----------------------------------------------------
# FIXED Parameters
fixed_d = 5
fixed_n = 20000

# VARYING Parameters (Target)
rhos = [0,0.5,0.8]

n_trials = 10
n_trials_for_timing = 10
k_grid = [1,5,10,15]

# --- Storage Arrays ---
shape_tuple = (len(rhos),)
mse1_best, mse2_best, mse3_best, mse4_best = (np.empty(shape_tuple) for _ in range(4))
bestk1, bestk2, bestk3, bestk4 = (np.empty(shape_tuple, dtype=int) for _ in range(4))
time1_best, time2_best, time3_best, time4_best = (np.empty(shape_tuple) for _ in range(4))


def _argmin(m, order):
    best, bk = float('inf'), -1
    for k in order:
        v = m[k]
        if v < best or (np.isclose(v, best) and k < bk):
            best, bk = v, k
    return bk, best


# --- Run Simulation ---------------------------------------------------------
if __name__ == '__main__':
    print(f"=== Starting MVN Robustness Experiment (d={fixed_d}, N={fixed_n}) ===")

    for ri, rho in enumerate(rhos):
        # Initialize simulator with CURRENT rho
        sim = GaussianSimulator(fixed_d, rho=rho)
        H = sim.true_entropy()

        print(f"\n[Iter {ri + 1}/{len(rhos)}] Processing rho={rho}... True H={H:.4f}")

        # --- Part 1: Find best k by minimizing MSE ---
        cal1, cal2, cal3, cal4 = ({k: np.empty(n_trials) for k in k_grid} for _ in range(4))

        for t in range(n_trials):
            X = sim.sim(fixed_n)

            # For UM estimators (Uniformizing Mapping)
            # 1. Transform to Uniform via marginal CDF (Standard Normal)
            Z = stats.norm.cdf(X)
            # 2. Correction term: - Sum(log(marginal_pdf))
            # Since margins are N(0,1), we compute logpdf sum
            logpdf_sum = np.sum(stats.norm.logpdf(X), axis=1)
            corr_term = -np.mean(logpdf_sum)

            for k in k_grid:
                cal1[k][t] = kl(X, k=k)
                cal2[k][t] = ksg(X, k=k)
                # UM estimator formula: H(U) + Sum(H(marginals))
                # tkl(Z) estimates H(Copula), corr_term estimates Sum(H(marginals))
                cal3[k][t] = tkl(Z, k=k) + corr_term
                cal4[k][t] = tksg(Z, k=k) + corr_term

        mse1, mse2, mse3, mse4 = ({k: np.mean((v - H) ** 2) for k, v in cal.items()} for cal in
                                  [cal1, cal2, cal3, cal4])

        k1, m1 = _argmin(mse1, k_grid)
        k2, m2 = _argmin(mse2, k_grid)
        k3, m3 = _argmin(mse3, k_grid)
        k4, m4 = _argmin(mse4, k_grid)

        bestk1[ri] = k1;
        mse1_best[ri] = m1
        bestk2[ri] = k2;
        mse2_best[ri] = m2
        bestk3[ri] = k3;
        mse3_best[ri] = m3
        bestk4[ri] = k4;
        mse4_best[ri] = m4

        print(f"  -> Best k: KL:{k1} KSG:{k2} UM-tKL:{k3} UM-tKSG:{k4}")

        # --- Part 2: Measure average time at the optimal k ---
        print(f"  -> Timing {n_trials_for_timing} reps...")
        times1, times2, times3, times4 = ([] for _ in range(4))

        for _ in range(n_trials_for_timing):
            # KL
            X = sim.sim(fixed_n)
            t0 = time.perf_counter()
            _ = kl(X, k=k1)
            times1.append(time.perf_counter() - t0)

            # KSG
            X = sim.sim(fixed_n)
            t0 = time.perf_counter()
            _ = ksg(X, k=k2)
            times2.append(time.perf_counter() - t0)

            # UM-tKL
            X = sim.sim(fixed_n)
            Z = stats.norm.cdf(X)
            t0 = time.perf_counter()
            _ = tkl(Z, k=k3)
            times3.append(time.perf_counter() - t0)

            # UM-tKSG
            X = sim.sim(fixed_n)
            Z = stats.norm.cdf(X)
            t0 = time.perf_counter()
            _ = tksg(Z, k=k4)
            times4.append(time.perf_counter() - t0)

        time1_best[ri] = np.mean(times1)
        time2_best[ri] = np.mean(times2)
        time3_best[ri] = np.mean(times3)
        time4_best[ri] = np.mean(times4)

    # === Build and Save DataFrame ===
    results_list = []

    for ri, rho in enumerate(rhos):
        # KL
        results_list.append({
            "Distribution": "Normal", "Dimensions": fixed_d, "N_Samples": fixed_n, "Correlation": rho,
            "Method": "KL", "Optimal_Param": bestk1[ri], "Eval_Time_s": time1_best[ri],
            "RMSE": np.sqrt(mse1_best[ri])
        })
        # KSG
        results_list.append({
            "Distribution": "Normal", "Dimensions": fixed_d, "N_Samples": fixed_n, "Correlation": rho,
            "Method": "KSG", "Optimal_Param": bestk2[ri], "Eval_Time_s": time2_best[ri],
            "RMSE": np.sqrt(mse2_best[ri])
        })
        # UM-tKL
        results_list.append({
            "Distribution": "Normal", "Dimensions": fixed_d, "N_Samples": fixed_n, "Correlation": rho,
            "Method": "UM-tKL", "Optimal_Param": bestk3[ri], "Eval_Time_s": time3_best[ri],
            "RMSE": np.sqrt(mse3_best[ri])
        })
        # UM-tKSG
        results_list.append({
            "Distribution": "Normal", "Dimensions": fixed_d, "N_Samples": fixed_n, "Correlation": rho,
            "Method": "UM-tKSG", "Optimal_Param": bestk4[ri], "Eval_Time_s": time4_best[ri],
            "RMSE": np.sqrt(mse4_best[ri])
        })

    df_long = pd.DataFrame(results_list)

    print(f"\n=== MVN Results (Varying Correlation) ===")
    print(df_long.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # Save to distinct filename
    csv_path = os.path.join("util", "temp_data", "knn_mvn_results_rho.csv")
    df_long.to_csv(csv_path, index=False)
    print(f"\nSaved correlation analysis CSV -> {csv_path}")

    print("\nDone.")