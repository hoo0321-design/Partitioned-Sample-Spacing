# ==============================================================================
#  REVISED: k-NN SIMULATION FOR MVN WITH VARYING SAMPLE SIZES & TIMING
#  - Finds the best k for each estimator by minimizing MSE.
#  - Measures the average evaluation time at that best k.
#  - Saves results to "knn_mvn_results_n.csv" in a long format.
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

from scipy import stats
import matplotlib;

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ent_est.entropy import kl, ksg, tkl, tksg
from simulators.complex import mvn

# --- experiment grids ---------------------------------------------------------
ds = [10]
all_n_samples = [1000, 2000, 4000, 6000]
n_trials = 10
n_trials_for_timing = 10
fixed_rho = 0.0
k_grid = [1, 2, 3, 7, 10, 12, 15, 18, 24, 26]

# --- storage arrays ---
shape_tuple = (len(all_n_samples), len(ds))
mse1_best, mse2_best, mse3_best, mse4_best = (np.empty(shape_tuple) for _ in range(4))
bestk1, bestk2, bestk3, bestk4 = (np.empty(shape_tuple, dtype=int) for _ in range(4))
time1_best, time2_best, time3_best, time4_best = (np.empty(shape_tuple) for _ in range(4))


def _argmin_with_tiebreak(mse_byk, k_order):
    best_mse, best_k = float('inf'), -1
    for k in k_order:
        m = mse_byk[k]
        if (m < best_mse) or (np.isclose(m, best_mse) and k < best_k):
            best_mse, best_k = m, k
    return best_k, best_mse


# --- run simulation -----------------------------------------------------------
for di, d in enumerate(ds):
    sim_mdl = mvn(rho=fixed_rho, dim_x=d)
    true_val = sim_mdl.mcmc_entropy()

    for ni, n_samples in enumerate(all_n_samples):
        # --- Part 1: Find best k by minimizing MSE ---
        cal1_k, cal2_k, cal3_k, cal4_k = ({k: np.empty(n_trials) for k in k_grid} for _ in range(4))

        for i in range(n_trials):
            X = sim_mdl.sim(n_samples=n_samples)
            Z = stats.norm.cdf(X)
            corr = -np.mean(np.log(np.prod(stats.norm.pdf(X), axis=1)))
            for k_val in k_grid:
                cal1_k[k_val][i] = kl(X, k=k_val)
                cal2_k[k_val][i] = ksg(X, k=k_val)
                cal3_k[k_val][i] = tkl(Z, k=k_val) + corr
                cal4_k[k_val][i] = tksg(Z, k=k_val) + corr

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

        print(f"[d={d}, N={n_samples}] MSE portion complete. Best k -> KL:{k1}  KSG:{k2}  UM-tKL:{k3}  UM-tKSG:{k4}")

        # --- Part 2: Measure average time at the optimal k ---
        print(f"  -> Now timing {n_trials_for_timing} reps at optimal k...")
        times1, times2, times3, times4 = ([] for _ in range(4))

        for _ in range(n_trials_for_timing):
            X = sim_mdl.sim(n_samples);
            t0 = time.perf_counter();
            _ = kl(X, k=k1);
            t1 = time.perf_counter();
            times1.append(t1 - t0)
            X = sim_mdl.sim(n_samples);
            t0 = time.perf_counter();
            _ = ksg(X, k=k2);
            t1 = time.perf_counter();
            times2.append(t1 - t0)
            X = sim_mdl.sim(n_samples);
            Z = stats.norm.cdf(X);
            t0 = time.perf_counter();
            _ = tkl(Z, k=k3);
            t1 = time.perf_counter();
            times3.append(t1 - t0)
            X = sim_mdl.sim(n_samples);
            Z = stats.norm.cdf(X);
            t0 = time.perf_counter();
            _ = tksg(Z, k=k4);
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
        "Distribution": "Normal", "Dimensions": d_val, "N_Samples": n_samples, "Correlation": fixed_rho,
        "Method": "KL", "Optimal_Param": bestk1[ni, 0], "Eval_Time_s": time1_best[ni, 0],
        "RMSE": np.sqrt(mse1_best[ni, 0])
    })
    results_list.append({
        "Distribution": "Normal", "Dimensions": d_val, "N_Samples": n_samples, "Correlation": fixed_rho,
        "Method": "KSG", "Optimal_Param": bestk2[ni, 0], "Eval_Time_s": time2_best[ni, 0],
        "RMSE": np.sqrt(mse2_best[ni, 0])
    })
    results_list.append({
        "Distribution": "Normal", "Dimensions": d_val, "N_Samples": n_samples, "Correlation": fixed_rho,
        "Method": "UM-tKL", "Optimal_Param": bestk3[ni, 0], "Eval_Time_s": time3_best[ni, 0],
        "RMSE": np.sqrt(mse3_best[ni, 0])
    })
    results_list.append({
        "Distribution": "Normal", "Dimensions": d_val, "N_Samples": n_samples, "Correlation": fixed_rho,
        "Method": "UM-tKSG", "Optimal_Param": bestk4[ni, 0], "Eval_Time_s": time4_best[ni, 0],
        "RMSE": np.sqrt(mse4_best[ni, 0])
    })

df_long = pd.DataFrame(results_list)

print(f"\n=== Results in Long Format (d={d_val}, best-k) ===")
print(df_long.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

csv_path = os.path.join("util", "temp_data", "knn_mvn_results_n.csv")
df_long.to_csv(csv_path, index=False)
print(f"\nSaved compatible CSV -> {csv_path}")

# --- Plotting and Payload Save (Unchanged) ---
# ...

print("\nDone.")