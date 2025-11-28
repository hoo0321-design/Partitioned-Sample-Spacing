# ==============================================================================
#  REVISED: k-NN SIMULATION FOR GAMMA-COPULA (VARYING CORRELATION)
#  - Fixes Dimension (d=6) and Sample Size (N=10000).
#  - Varies Correlation (rho) -> [0.0, 0.3, 0.6, 0.9].
#  - Finds best k, Measures Time, Saves to "knn_gamma_results_rho.csv".
# ==============================================================================

# --- compatibility shims & imports -------------------------------------------
import sys, os, warnings, time

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    import numpy as np

    if not hasattr(np, "bool"): np.bool = bool
import pickle as _pickle;

sys.modules['cPickle'] = _pickle
import builtins, itertools

if not hasattr(itertools, "izip"): itertools.izip = zip
if not hasattr(builtins, "xrange"): builtins.xrange = range
if not hasattr(builtins, "unicode"): builtins.unicode = str
if not hasattr(builtins, "long"): builtins.long = int

os.makedirs("util/figs", exist_ok=True);
os.makedirs("util/temp_data", exist_ok=True)

from scipy import stats, special
import matplotlib;

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from ent_est.entropy import kl, ksg, tkl, tksg


# --- Simulator Class ----------------------------------------------------------
def equicorr(d, rho):
    R = np.full((d, d), rho, dtype=float);
    np.fill_diagonal(R, 1.0);
    return R


class GammaGaussCopula:
    def __init__(self, dim_x, rho=0.0, shape=4.0, scale=6.0):
        self.d = int(dim_x);
        self.rho = float(rho)
        self.shape = np.full(self.d, shape) if np.isscalar(shape) else np.asarray(shape, float)
        self.scale = np.full(self.d, scale) if np.isscalar(scale) else np.asarray(scale, float)
        self.R = equicorr(self.d, self.rho)
        w, _ = np.linalg.eigh(self.R);
        if np.any(w <= 0): raise ValueError("R not PD")
        s, logdet = np.linalg.slogdet(self.R);
        if s <= 0: raise ValueError("det(R)<=0")
        self.det_term = 0.5 * logdet

    @staticmethod
    def _hgamma(k, theta):
        return k + np.log(theta) + special.gammaln(k) + (1.0 - k) * special.digamma(k)

    def true_entropy(self):
        return float(np.sum(self._hgamma(self.shape, self.scale)) + self.det_term)

    def sim(self, n):
        Z = np.random.multivariate_normal(np.zeros(self.d), self.R, size=int(n))
        U = stats.norm.cdf(Z)
        X = np.empty_like(U)
        for j in range(self.d):
            X[:, j] = stats.gamma.ppf(U[:, j], a=self.shape[j], scale=self.scale[j])
        return X


# --- experiment settings ----------------------------------------------------
# FIXED Parameters
fixed_d = 7
fixed_n = 50000
shape_param, scale_param = 0.4, 0.3

# VARYING Parameters (Target of this experiment)
rhos = [0.0,0.1, 0.3,0.8]

n_trials = 10  # Slightly reduced for speed, increase if needed
n_trials_for_timing = 10
k_grid = [1,3,7,12]

# --- storage arrays ---
# Shape is simply (len(rhos),) since we only iterate over rho
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


# --- run simulation ---------------------------------------------------------
print(f"=== Starting Robustness Experiment: Varying Correlation (d={fixed_d}, N={fixed_n}) ===")

for ri, rho in enumerate(rhos):
    # Initialize simulator with CURRENT rho
    sim = GammaGaussCopula(fixed_d, rho=rho, shape=shape_param, scale=scale_param)
    H = sim.true_entropy()

    print(f"\n[Iter {ri + 1}/{len(rhos)}] Processing rho={rho}... True H={H:.4f}")

    # --- Part 1: Find best k by minimizing MSE ---
    cal1, cal2, cal3, cal4 = ({k: np.empty(n_trials) for k in k_grid} for _ in range(4))

    for t in range(n_trials):
        X = sim.sim(fixed_n);
        # Pre-compute U and logpdf for UM estimators
        U = stats.gamma.cdf(X, a=shape_param, scale=scale_param)
        logpdf = np.sum(stats.gamma.logpdf(X, a=shape_param, scale=scale_param), axis=1)

        for k in k_grid:
            cal1[k][t] = kl(X, k=k)
            cal2[k][t] = ksg(X, k=k)
            cal3[k][t] = tkl(U, k=k) - np.mean(logpdf)
            cal4[k][t] = tksg(U, k=k) - np.mean(logpdf)

    mse1, mse2, mse3, mse4 = ({k: np.mean((v - H) ** 2) for k, v in cal.items()} for cal in
                              [cal1, cal2, cal3, cal4])

    k1, m1 = _argmin(mse1, k_grid);
    k2, m2 = _argmin(mse2, k_grid);
    k3, m3 = _argmin(mse3, k_grid);
    k4, m4 = _argmin(mse4, k_grid)

    bestk1[ri] = k1;
    mse1_best[ri] = m1;
    bestk2[ri] = k2;
    mse2_best[ri] = m2
    bestk3[ri] = k3;
    mse3_best[ri] = m3;
    bestk4[ri] = k4;
    mse4_best[ri] = m4

    print(f"  -> Best k found: KL:{k1} KSG:{k2} UM-tKL:{k3} UM-tKSG:{k4}")

    # --- Part 2: Measure average time at the optimal k ---
    print(f"  -> Timing {n_trials_for_timing} reps...")
    times1, times2, times3, times4 = ([] for _ in range(4))

    for _ in range(n_trials_for_timing):
        X = sim.sim(fixed_n);
        t0 = time.perf_counter();
        _ = kl(X, k=k1);
        t1 = time.perf_counter();
        times1.append(t1 - t0)

        X = sim.sim(fixed_n);
        t0 = time.perf_counter();
        _ = ksg(X, k=k2);
        t1 = time.perf_counter();
        times2.append(t1 - t0)

        X = sim.sim(fixed_n);
        U = stats.gamma.cdf(X, a=shape_param, scale=scale_param);
        t0 = time.perf_counter();
        _ = tkl(U, k=k3);
        t1 = time.perf_counter();
        times3.append(t1 - t0)

        X = sim.sim(fixed_n);
        U = stats.gamma.cdf(X, a=shape_param, scale=scale_param);
        t0 = time.perf_counter();
        _ = tksg(U, k=k4);
        t1 = time.perf_counter();
        times4.append(t1 - t0)

    time1_best[ri] = np.mean(times1);
    time2_best[ri] = np.mean(times2)
    time3_best[ri] = np.mean(times3);
    time4_best[ri] = np.mean(times4)

# === REVISED SECTION: Build and save a long-format DataFrame ===
results_list = []

for ri, rho in enumerate(rhos):
    # KL
    results_list.append({
        "Distribution": "Gamma", "Dimensions": fixed_d, "N_Samples": fixed_n, "Correlation": rho,
        "Method": "KL", "Optimal_Param": bestk1[ri], "Eval_Time_s": time1_best[ri],
        "RMSE": np.sqrt(mse1_best[ri])
    })
    # KSG
    results_list.append({
        "Distribution": "Gamma", "Dimensions": fixed_d, "N_Samples": fixed_n, "Correlation": rho,
        "Method": "KSG", "Optimal_Param": bestk2[ri], "Eval_Time_s": time2_best[ri],
        "RMSE": np.sqrt(mse2_best[ri])
    })
    # UM-tKL
    results_list.append({
        "Distribution": "Gamma", "Dimensions": fixed_d, "N_Samples": fixed_n, "Correlation": rho,
        "Method": "UM-tKL", "Optimal_Param": bestk3[ri], "Eval_Time_s": time3_best[ri],
        "RMSE": np.sqrt(mse3_best[ri])
    })
    # UM-tKSG
    results_list.append({
        "Distribution": "Gamma", "Dimensions": fixed_d, "N_Samples": fixed_n, "Correlation": rho,
        "Method": "UM-tKSG", "Optimal_Param": bestk4[ri], "Eval_Time_s": time4_best[ri],
        "RMSE": np.sqrt(mse4_best[ri])
    })

df_long = pd.DataFrame(results_list)

print(f"\n=== Gamma Copula Results (Varying Correlation) ===")
print(df_long.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

# Save to a distinctive filename
csv_path = os.path.join("util", "temp_data", "knn_gamma_results_rho.csv")
df_long.to_csv(csv_path, index=False)
print(f"\nSaved correlation analysis CSV -> {csv_path}")

print("\nDone.")