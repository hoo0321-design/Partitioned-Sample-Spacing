# ==============================================================================
#  REVISED: k-NN SIMULATION FOR MVN WITH VARYING DIMENSIONS & TIMING
#  - Finds the best k for each estimator by minimizing MSE.
#  - Measures the average evaluation time at that best k.
#  - Saves results to "knn_mvn_results_d.csv" in a long format.
# ==============================================================================

# --- compatibility shims -----------------------------------------------------
import sys, os, time
import numpy as np

if not hasattr(np, "bool"):
    np.bool = bool
import pickle as _pickle

sys.modules['cPickle'] = _pickle
import builtins, itertools

if not hasattr(itertools, "izip"):
    itertools.izip = zip
if not hasattr(builtins, "xrange"):
    builtins.xrange = range
if not hasattr(builtins, "unicode"):
    builtins.unicode = str
if not hasattr(builtins, "long"):
    builtins.long = int

print("PYTHON EXEC:", sys.executable)
print("NumPy ver  :", np.__version__)

# --- output dirs --------------------------------------------------------------
os.makedirs("util/figs", exist_ok=True)
os.makedirs("util/temp_data", exist_ok=True)

# --- core imports -------------------------------------------------------------
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ent_est.entropy import kl, ksg, tkl, tksg
from simulators.complex import mvn

# --- experiment settings ------------------------------------------------------
ds = [2, 5, 10, 20, 30, 40]
n_trials = 10
n_trials_for_timing = 10  # Use fewer trials for timing to speed things up
all_n_samples = [3000]
fixed_rho = 0.0
k_grid = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# --- Storage arrays for results ---
# MSE and Best K
mse1_best, mse2_best, mse3_best, mse4_best = (np.empty((len(all_n_samples), len(ds))) for _ in range(4))
bestk1, bestk2, bestk3, bestk4 = (np.empty((len(all_n_samples), len(ds)), dtype=int) for _ in range(4))
# Evaluation Times
time1_best, time2_best, time3_best, time4_best = (np.empty((len(all_n_samples), len(ds))) for _ in range(4))

# --- run main simulation ------------------------------------------------------
for di, d in enumerate(ds):
    sim_mdl = mvn(rho=fixed_rho, dim_x=d)

    for ni, n_samples in enumerate(all_n_samples):
        # --- Part 1: Find best k by minimizing MSE ---
        cal1_k, cal2_k, cal3_k, cal4_k = ({k: np.empty(n_trials) for k in k_grid} for _ in range(4))

        for i in range(n_trials):
            xs = sim_mdl.sim(n_samples=n_samples)
            zs = stats.norm.cdf(xs)
            corr = -np.mean(np.log(np.prod(stats.norm.pdf(xs), axis=1)))
            for k_val in k_grid:
                cal1_k[k_val][i] = kl(xs, k=k_val)
                cal2_k[k_val][i] = ksg(xs, k=k_val)
                cal3_k[k_val][i] = tkl(zs, k=k_val) + corr
                cal4_k[k_val][i] = tksg(zs, k=k_val) + corr

        true_val = sim_mdl.mcmc_entropy()
        mse1_byk, mse2_byk, mse3_byk, mse4_byk = (
            {k: np.mean((v - true_val) ** 2) for k, v in cal.items()}
            for cal in [cal1_k, cal2_k, cal3_k, cal4_k]
        )

        bestk1[ni, di] = min(mse1_byk, key=mse1_byk.get)
        bestk2[ni, di] = min(mse2_byk, key=mse2_byk.get)
        bestk3[ni, di] = min(mse3_byk, key=mse3_byk.get)
        bestk4[ni, di] = min(mse4_byk, key=mse4_byk.get)

        mse1_best[ni, di] = mse1_byk[bestk1[ni, di]]
        mse2_best[ni, di] = mse2_byk[bestk2[ni, di]]
        mse3_best[ni, di] = mse3_byk[bestk3[ni, di]]
        mse4_best[ni, di] = mse4_byk[bestk4[ni, di]]

        print(f"[d={d}, N={n_samples}] MSE portion complete. Best k -> "
              f"KL:{bestk1[ni, di]}  KSG:{bestk2[ni, di]}  "
              f"UM-tKL:{bestk3[ni, di]}  UM-tKSG:{bestk4[ni, di]}")

        # --- Part 2: Measure average time at the optimal k ---
        print(f"  -> Now timing {n_trials_for_timing} reps at optimal k...")
        times1, times2, times3, times4 = ([] for _ in range(4))

        for _ in range(n_trials_for_timing):
            # KL Timing
            X = sim_mdl.sim(n_samples=n_samples)
            t0 = time.perf_counter();
            _ = kl(X, k=bestk1[ni, di]);
            t1 = time.perf_counter()
            times1.append(t1 - t0)

            # KSG Timing
            X = sim_mdl.sim(n_samples=n_samples)
            t0 = time.perf_counter();
            _ = ksg(X, k=bestk2[ni, di]);
            t1 = time.perf_counter()
            times2.append(t1 - t0)

            # UM-tKL Timing
            X = sim_mdl.sim(n_samples=n_samples)
            Z = stats.norm.cdf(X)
            t0 = time.perf_counter();
            _ = tkl(Z, k=bestk3[ni, di]);
            t1 = time.perf_counter()
            times3.append(t1 - t0)

            # UM-tKSG Timing
            X = sim_mdl.sim(n_samples=n_samples)
            Z = stats.norm.cdf(X)
            t0 = time.perf_counter();
            _ = tksg(Z, k=bestk4[ni, di]);
            t1 = time.perf_counter()
            times4.append(t1 - t0)

        time1_best[ni, di] = np.mean(times1)
        time2_best[ni, di] = np.mean(times2)
        time3_best[ni, di] = np.mean(times3)
        time4_best[ni, di] = np.mean(times4)

# --- Plotting (Unchanged) ---
fig, ax = plt.subplots(1, 1, figsize=(6, 4.2), dpi=150)
ax.plot(ds, np.sqrt(mse3_best[0]), marker='o', linestyle=':', label='UM-tKL (best-k)', mfc='none')
ax.plot(ds, np.sqrt(mse4_best[0]), marker='o', linestyle='-', label='UM-tKSG (best-k)', mfc='none')
ax.plot(ds, np.sqrt(mse1_best[0]), marker='x', linestyle=':', label='KL (best-k)')
ax.plot(ds, np.sqrt(mse2_best[0]), marker='x', linestyle='-', label='KSG (best-k)')
ax.set_xlabel('dimension');
ax.set_ylabel('RMSE');
ax.legend();
fig.tight_layout()
plt.savefig('util/figs/RMSE_vs_d_mvn_bestk.png')

# --- Build and save the final long-format DataFrame ---
results_list = []
n_s = all_n_samples[0]

for di, d in enumerate(ds):
    results_list.append({
        "Distribution": "Normal", "Dimensions": d, "N_Samples": n_s, "Correlation": fixed_rho,
        "Method": "KL", "Optimal_Param": bestk1[0, di], "Eval_Time_s": time1_best[0, di],
        "RMSE": np.sqrt(mse1_best[0, di])
    })
    results_list.append({
        "Distribution": "Normal", "Dimensions": d, "N_Samples": n_s, "Correlation": fixed_rho,
        "Method": "KSG", "Optimal_Param": bestk2[0, di], "Eval_Time_s": time2_best[0, di],
        "RMSE": np.sqrt(mse2_best[0, di])
    })
    results_list.append({
        "Distribution": "Normal", "Dimensions": d, "N_Samples": n_s, "Correlation": fixed_rho,
        "Method": "UM-tKL", "Optimal_Param": bestk3[0, di], "Eval_Time_s": time3_best[0, di],
        "RMSE": np.sqrt(mse3_best[0, di])
    })
    results_list.append({
        "Distribution": "Normal", "Dimensions": d, "N_Samples": n_s, "Correlation": fixed_rho,
        "Method": "UM-tKSG", "Optimal_Param": bestk4[0, di], "Eval_Time_s": time4_best[0, di],
        "RMSE": np.sqrt(mse4_best[0, di])
    })

df_long = pd.DataFrame(results_list)

print(f"\n=== Results in Long Format (n={n_s}, best-k) ===")
print(df_long.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

csv_path = os.path.join("util", "temp_data", "knn_mvn_results_d.csv")
df_long.to_csv(csv_path, index=False)
print(f"\nSaved compatible CSV -> {csv_path}")

print("\nDone.")