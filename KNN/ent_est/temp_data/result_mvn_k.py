# inspect_results_bestk.py
import os
import pickle
import numpy as np
import pandas as pd

# --- locate file ---
ROOT = "/Users/hojeongwoo/PyCharmMiscProject/NFEE/NFEE-main"
PKL  = os.path.join(ROOT, "temp_data", "RMSE_vs_d_mvn_bestk.pkl")

with open(PKL, "rb") as f:
    payload = pickle.load(f)

ds         = payload["ds"]
all_n      = payload["all_n_samples"]
mse1_best  = payload["mse1_best"]
mse2_best  = payload["mse2_best"]
mse3_best  = payload["mse3_best"]
mse4_best  = payload["mse4_best"]
bestk1     = payload["bestk1"]
bestk2     = payload["bestk2"]
bestk3     = payload["bestk3"]
bestk4     = payload["bestk4"]

# 여기서는 all_n_samples = [5000] 이므로 index 0만 사용
n = all_n[0]

rmse1 = np.sqrt(mse1_best[0])
rmse2 = np.sqrt(mse2_best[0])
rmse3 = np.sqrt(mse3_best[0])
rmse4 = np.sqrt(mse4_best[0])

df = pd.DataFrame({
    "dimension": ds,
    "RMSE_KL": rmse1,
    "best_k_KL": bestk1[0],
    "RMSE_KSG": rmse2,
    "best_k_KSG": bestk2[0],
    "RMSE_UM_tKL": rmse3,
    "best_k_UM_tKL": bestk3[0],
    "RMSE_UM_tKSG": rmse4,
    "best_k_UM_tKSG": bestk4[0],
})

# pretty print
print(f"\n=== RMSE vs dimension (n={n}, best-k per method) ===")
print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

# save CSV
out_csv = os.path.join(ROOT, "temp_data", "RMSE_vs_d_mvn_bestk_summary.csv")
df.to_csv(out_csv, index=False)
print(f"\nSaved CSV -> {out_csv}")
