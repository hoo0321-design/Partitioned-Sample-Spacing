# inspect_results_gamma.py
import os
import pickle
import numpy as np
import pandas as pd

# --- locate file (adjust ROOT if needed) ---
ROOT = "/Users/hojeongwoo/PyCharmMiscProject/NFEE/NFEE-main"
PKL  = os.path.join(ROOT, "temp_data", "RMSE_vs_d_mgamma.pkl")

with open(PKL, "rb") as f:
    ds, mse1, mse2, mse3, mse4 = pickle.load(f)

# mse* are shape (len(all_n_samples), len(ds))
# your experiment used all_n_samples=[5000], so we index [0]
rmse1 = np.sqrt(mse1[0])
rmse2 = np.sqrt(mse2[0])
rmse3 = np.sqrt(mse3[0])
rmse4 = np.sqrt(mse4[0])

df = pd.DataFrame({
    "dimension": ds,
    "RMSE_KL":      rmse1,
    "RMSE_KSG":     rmse2,
    "RMSE_UM_tKL":  rmse3,
    "RMSE_UM_tKSG": rmse4,
})

# pretty print
print("\n=== RMSE vs dimension (Gamma, n=5000) ===")
print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

# save CSV next to the pickle
out_csv = os.path.join(ROOT, "temp_data", "RMSE_vs_d_mgamma_summary.csv")
df.to_csv(out_csv, index=False)
print(f"\nSaved CSV -> {out_csv}")
