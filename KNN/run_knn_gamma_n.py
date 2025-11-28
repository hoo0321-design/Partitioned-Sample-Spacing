# ==============================================================================
#  REVISED: k-NN SIMULATION FOR GAMMA-COPULA DATA (VARYING SAMPLE SIZES & TIMING)
#  - Finds the best k for each estimator by minimizing MSE.
#  - Measures the average evaluation time at that best k.
#  - Saves results to "knn_gamma_results_n.csv" in a long format.
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
# ... [GammaGaussCopula class definition remains unchanged] ...
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
ds = [5]
all_n_samples = [1000, 2000, 4000, 8000,16000,32000]
n_trials = 10
n_trials_for_timing = 10
k_grid = [1, 2, 4, 6, 8, 10, 13, 16]
shape_param, scale_param, fixed_rho = 0.4, 0.3, 0

# --- storage arrays ---
shape_tuple = (len(all_n_samples), len(ds))
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
for di, d in enumerate(ds):
    sim = GammaGaussCopula(d, rho=fixed_rho, shape=shape_param, scale=scale_param)
    H = sim.true_entropy()
    for ni, N in enumerate(all_n_samples):
        # --- Part 1: Find best k by minimizing MSE ---
        cal1, cal2, cal3, cal4 = ({k: np.empty(n_trials) for k in k_grid} for _ in range(4))
        for t in range(n_trials):
            X = sim.sim(N);
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

        bestk1[ni, di] = k1;
        mse1_best[ni, di] = m1;
        bestk2[ni, di] = k2;
        mse2_best[ni, di] = m2
        bestk3[ni, di] = k3;
        mse3_best[ni, di] = m3;
        bestk4[ni, di] = k4;
        mse4_best[ni, di] = m4
        print(f"[Gamma d={d} N={N}] MSE portion complete. Best k -> KL:{k1} KSG:{k2} UM-tKL:{k3} UM-tKSG:{k4}")

        # --- Part 2: Measure average time at the optimal k ---
        print(f"  -> Now timing {n_trials_for_timing} reps at optimal k...")
        times1, times2, times3, times4 = ([] for _ in range(4))

        for _ in range(n_trials_for_timing):
            X = sim.sim(N);
            t0 = time.perf_counter();
            _ = kl(X, k=k1);
            t1 = time.perf_counter();
            times1.append(t1 - t0)
            X = sim.sim(N);
            t0 = time.perf_counter();
            _ = ksg(X, k=k2);
            t1 = time.perf_counter();
            times2.append(t1 - t0)
            X = sim.sim(N);
            U = stats.gamma.cdf(X, a=shape_param, scale=scale_param);
            t0 = time.perf_counter();
            _ = tkl(U, k=k3);
            t1 = time.perf_counter();
            times3.append(t1 - t0)
            X = sim.sim(N);
            U = stats.gamma.cdf(X, a=shape_param, scale=scale_param);
            t0 = time.perf_counter();
            _ = tksg(U, k=k4);
            t1 = time.perf_counter();
            times4.append(t1 - t0)

        time1_best[ni, di] = np.mean(times1);
        time2_best[ni, di] = np.mean(times2)
        time3_best[ni, di] = np.mean(times3);
        time4_best[ni, di] = np.mean(times4)

# === REVISED SECTION: Build and save a long-format DataFrame ===
results_list = []
d_val = ds[0]

for ni, n_samples in enumerate(all_n_samples):
    results_list.append({
        "Distribution": "Gamma", "Dimensions": d_val, "N_Samples": n_samples, "Correlation": fixed_rho,
        "Method": "KL", "Optimal_Param": bestk1[ni, 0], "Eval_Time_s": time1_best[ni, 0],
        "RMSE": np.sqrt(mse1_best[ni, 0])
    })
    results_list.append({
        "Distribution": "Gamma", "Dimensions": d_val, "N_Samples": n_samples, "Correlation": fixed_rho,
        "Method": "KSG", "Optimal_Param": bestk2[ni, 0], "Eval_Time_s": time2_best[ni, 0],
        "RMSE": np.sqrt(mse2_best[ni, 0])
    })
    results_list.append({
        "Distribution": "Gamma", "Dimensions": d_val, "N_Samples": n_samples, "Correlation": fixed_rho,
        "Method": "UM-tKL", "Optimal_Param": bestk3[ni, 0], "Eval_Time_s": time3_best[ni, 0],
        "RMSE": np.sqrt(mse3_best[ni, 0])
    })
    results_list.append({
        "Distribution": "Gamma", "Dimensions": d_val, "N_Samples": n_samples, "Correlation": fixed_rho,
        "Method": "UM-tKSG", "Optimal_Param": bestk4[ni, 0], "Eval_Time_s": time4_best[ni, 0],
        "RMSE": np.sqrt(mse4_best[ni, 0])
    })

df_long = pd.DataFrame(results_list)

print(f"\n=== Gamma Copula Results in Long Format (d={d_val}, best-k) ===")
print(df_long.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

csv_path = os.path.join("util", "temp_data", "knn_gamma_results_n.csv")
df_long.to_csv(csv_path, index=False)
print(f"\nSaved compatible CSV -> {csv_path}")

# --- plotting and payload (unchanged) ---
# ...

print("\nDone.")