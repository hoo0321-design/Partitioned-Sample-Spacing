# ==============================================================================
#  REVISED: k-NN SIMULATION FOR GAMMA-COPULA DATA (VARYING DIMENSIONS & TIMING)
#  - Finds the best k for each estimator by minimizing MSE.
#  - Measures the average evaluation time at that best k.
#  - Saves results to "knn_gamma_results_d.csv" in a long format.
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
    # ... [GammaGaussCopula class definition remains unchanged] ...
    def __init__(self, dim_x, rho=0.0, shape=4.0, scale=6.0):
        self.d = int(dim_x);
        self.rho = float(rho)
        self.shape = np.full(self.d, shape, dtype=float)
        self.scale = np.full(self.d, scale, dtype=float)
        self.R = equicorr(self.d, self.rho)
        w, _ = np.linalg.eigh(self.R)
        if np.any(w <= 0): raise ValueError("Correlation matrix not positive definite.")
        self.chol = np.linalg.cholesky(self.R)
        sign, logdet = np.linalg.slogdet(self.R)
        if sign <= 0: raise ValueError("Non-positive determinant for R.")
        self.det_term = 0.5 * logdet

    @staticmethod
    def _gamma_entropy_marginal(k, theta):
        return k + np.log(theta) + special.gammaln(k) + (1.0 - k) * special.digamma(k)

    def mcmc_entropy(self):
        H_marg = np.sum(self._gamma_entropy_marginal(self.shape, self.scale))
        return float(H_marg + self.det_term)

    def sim(self, n_samples):
        n = int(n_samples)
        Z = np.random.randn(n, self.d) @ self.chol.T
        U = stats.norm.cdf(Z)
        X = np.empty_like(U)
        for j in range(self.d):
            X[:, j] = stats.gamma.ppf(U[:, j], a=self.shape[j], scale=self.scale[j])
        return X


# --- experiment settings -----------------------------------------------------
ds = [2, 5, 10, 15, 20]
n_trials = 10
n_trials_for_timing = 10
all_n_samples = [30000]
gamma_shape = 0.4
gamma_scale = 0.3
fixed_rho = 0.0
k_grid = [1, 2, 10, 15]

# --- Storage arrays ---
shape = (len(all_n_samples), len(ds))
mse1_best, mse2_best, mse3_best, mse4_best = (np.empty(shape) for _ in range(4))
bestk1, bestk2, bestk3, bestk4 = (np.empty(shape, dtype=int) for _ in range(4))
time1_best, time2_best, time3_best, time4_best = (np.empty(shape) for _ in range(4))


def _argmin_with_tiebreak(mse_byk, k_order):
    best_mse, best_k = float('inf'), -1
    for k in k_order:
        m = mse_byk[k]
        if (m < best_mse) or (np.isclose(m, best_mse) and k < best_k):
            best_mse, best_k = m, k
    return best_k, best_mse


# --- run simulation ----------------------------------------------------------
for di, d in enumerate(ds):
    sim_mdl = GammaGaussCopula(dim_x=d, rho=fixed_rho, shape=gamma_shape, scale=gamma_scale)

    for ni, n_samples in enumerate(all_n_samples):
        # --- Part 1: Find best k by minimizing MSE ---
        cal1_k, cal2_k, cal3_k, cal4_k = ({k: np.empty(n_trials) for k in k_grid} for _ in range(4))

        for i in range(n_trials):
            X = sim_mdl.sim(n_samples)
            U = np.empty_like(X)
            logpdf = np.zeros(n_samples, dtype=float)
            for j in range(d):
                U[:, j] = stats.gamma.cdf(X[:, j], a=gamma_shape, scale=gamma_scale)
                logpdf += stats.gamma.logpdf(X[:, j], a=gamma_shape, scale=gamma_scale)
            for k_val in k_grid:
                cal1_k[k_val][i] = kl(X, k=k_val)
                cal2_k[k_val][i] = ksg(X, k=k_val)
                cal3_k[k_val][i] = tkl(U, k=k_val) - np.mean(logpdf)
                cal4_k[k_val][i] = tksg(U, k=k_val) - np.mean(logpdf)

        true_val = sim_mdl.mcmc_entropy()
        mse1_byk, mse2_byk, mse3_byk, mse4_byk = ({k: np.mean((v - true_val) ** 2) for k, v in cal.items()} for cal in
                                                  [cal1_k, cal2_k, cal3_k, cal4_k])

        k1, m1 = _argmin_with_tiebreak(mse1_byk, k_grid);
        bestk1[ni, di] = k1;
        mse1_best[ni, di] = m1
        k2, m2 = _argmin_with_tiebreak(mse2_byk, k_grid);
        bestk2[ni, di] = k2;
        mse2_best[ni, di] = m2
        k3, m3 = _argmin_with_tiebreak(mse3_byk, k_grid);
        bestk3[ni, di] = k3;
        mse3_best[ni, di] = m3
        k4, m4 = _argmin_with_tiebreak(mse4_byk, k_grid);
        bestk4[ni, di] = k4;
        mse4_best[ni, di] = m4

        print(f"[Gamma d={d}, N={n_samples}] MSE portion complete. Best k -> "
              f"KL:{k1}  KSG:{k2}  UM-tKL:{k3}  UM-tKSG:{k4}")

        # --- Part 2: Measure average time at the optimal k ---
        print(f"  -> Now timing {n_trials_for_timing} reps at optimal k...")
        times1, times2, times3, times4 = ([] for _ in range(4))

        for _ in range(n_trials_for_timing):
            # KL Timing
            X = sim_mdl.sim(n_samples);
            t0 = time.perf_counter();
            _ = kl(X, k=k1);
            t1 = time.perf_counter();
            times1.append(t1 - t0)
            # KSG Timing
            X = sim_mdl.sim(n_samples);
            t0 = time.perf_counter();
            _ = ksg(X, k=k2);
            t1 = time.perf_counter();
            times2.append(t1 - t0)
            # UM-tKL Timing
            X = sim_mdl.sim(n_samples);
            U = stats.gamma.cdf(X, a=gamma_shape, scale=gamma_scale)
            t0 = time.perf_counter();
            _ = tkl(U, k=k3);
            t1 = time.perf_counter();
            times3.append(t1 - t0)
            # UM-tKSG Timing
            X = sim_mdl.sim(n_samples);
            U = stats.gamma.cdf(X, a=gamma_shape, scale=gamma_scale)
            t0 = time.perf_counter();
            _ = tksg(U, k=k4);
            t1 = time.perf_counter();
            times4.append(t1 - t0)

        time1_best[ni, di] = np.mean(times1)
        time2_best[ni, di] = np.mean(times2)
        time3_best[ni, di] = np.mean(times3)
        time4_best[ni, di] = np.mean(times4)

# --- plotting (unchanged) ---
# ... (plotting code remains the same) ...

# === REVISED SECTION: Build and save the final long-format DataFrame ===
results_list = []
n_s = all_n_samples[0]

for di, d in enumerate(ds):
    results_list.append({
        "Distribution": "Gamma", "Dimensions": d, "N_Samples": n_s, "Correlation": fixed_rho,
        "Method": "KL", "Optimal_Param": bestk1[0, di], "Eval_Time_s": time1_best[0, di],
        "RMSE": np.sqrt(mse1_best[0, di])
    })
    results_list.append({
        "Distribution": "Gamma", "Dimensions": d, "N_Samples": n_s, "Correlation": fixed_rho,
        "Method": "KSG", "Optimal_Param": bestk2[0, di], "Eval_Time_s": time2_best[0, di],
        "RMSE": np.sqrt(mse2_best[0, di])
    })
    results_list.append({
        "Distribution": "Gamma", "Dimensions": d, "N_Samples": n_s, "Correlation": fixed_rho,
        "Method": "UM-tKL", "Optimal_Param": bestk3[0, di], "Eval_Time_s": time3_best[0, di],
        "RMSE": np.sqrt(mse3_best[0, di])
    })
    results_list.append({
        "Distribution": "Gamma", "Dimensions": d, "N_Samples": n_s, "Correlation": fixed_rho,
        "Method": "UM-tKSG", "Optimal_Param": bestk4[0, di], "Eval_Time_s": time4_best[0, di],
        "RMSE": np.sqrt(mse4_best[0, di])
    })

df_long = pd.DataFrame(results_list)

print(f"\n=== Gamma Copula Results in Long Format (n={n_s}, best-k) ===")
print(df_long.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

csv_path = os.path.join("util", "temp_data", "knn_gamma_results_d.csv")
df_long.to_csv(csv_path, index=False)
print(f"\nSaved compatible CSV -> {csv_path}")

# --- (Optional) Payload Save ---
# ... (payload saving code remains the same) ...

print("\nDone.")