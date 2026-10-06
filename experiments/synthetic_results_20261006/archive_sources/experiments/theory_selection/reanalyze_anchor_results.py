"""Apply fixed theory-inspired ell rules to saved original-estimator outcomes."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["Experiment", "Distribution", "Dimensions", "N_Samples", "Correlation"]


def bound_choice(n, d, delta):
    lam = 2*d*math.log(2*n+1)+math.log(96/delta)
    continuous = (8**4*n/(d**4*lam**2))**(1/(d+8))
    candidates = {max(1, math.floor(continuous)), max(1, math.ceil(continuous))}
    return min(candidates, key=lambda ell: (ell**(-2)+math.sqrt(lam)*(ell**d/n)**.25, ell))


def markdown(frame):
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"]*len(frame.columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    source, out = Path(args.source_dir).resolve(), Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    settings = pd.read_csv(source/"settings.csv")
    raw = pd.read_csv(source/"r_pss_cadee_estimates.csv")
    raw = raw[raw.Method == "PSS"].copy()
    raw["ell"] = raw.Optimal_Param.astype(int)
    if raw.duplicated(KEYS+["ell", "Replicate"]).any():
        raise ValueError("Duplicate saved estimates")
    if not np.isfinite(raw[["Estimate", "True_Entropy"]]).all().all():
        raise ValueError("Nonfinite saved estimates")
    counts = raw.groupby(KEYS+["ell"]).size()
    if not (counts == 30).all():
        raise ValueError("Expected 30 saved replicates for each candidate")
    raw["error"] = raw.Estimate-raw.True_Entropy
    computed = raw.groupby(KEYS+["ell"]).agg(
        RMSE=("error", lambda x: np.sqrt(np.mean(x*x))),
        Bias=("error", "mean"), Reps=("error", "size")).reset_index()
    per = pd.read_csv(source/"pss_per_ell.csv")
    check = computed.merge(per, on=KEYS+["ell"], validate="one_to_one", suffixes=("_raw", "_saved"))
    if len(check) != len(per) or not np.allclose(check.RMSE_raw, check.RMSE_saved, atol=1e-10, rtol=1e-10):
        raise ValueError("Saved per-ell RMSE does not match replicate estimates")
    old = pd.read_csv(source/"r_pss_cadee_summary.csv")
    old = old[old.Method == "PSS"]
    selections = []
    for key, group in per.groupby(KEYS, sort=True):
        experiment, distribution, d, n, rho = key
        stable = group[(group.Coverage_Fraction >= .95) &
                       (group.Skipped_Point_Fraction <= .05) &
                       (group.Mean_Occupied_Cell_Size >= 2)]
        if stable.empty:
            raise ValueError("No stable oracle candidate; do not silently change the original protocol")
        oracle = stable.sort_values(["RMSE", "ell"]).iloc[0]
        stored = old
        for col, value in zip(KEYS, key):
            stored = stored[stored[col] == value]
        if len(stored) != 1 or int(stored.iloc[0].Optimal_Param) != int(oracle.ell):
            raise ValueError("Stored oracle not reproduced")
        choices = {
            "Dimension-aware rate C=1": max(1, math.floor((n/(d**6*math.log(n)**2))**(1/(d+8))+.5)),
            "Fixed-d rate C=1": max(1, math.floor((n/math.log(n)**2)**(1/(d+8))+.5)),
            "Bound delta=n^-4": bound_choice(n, d, n**(-4.0)),
            "Bound delta=0.05": bound_choice(n, d, .05),
            "Saved coverage-filtered oracle": int(oracle.ell),
        }
        for rule, ell in choices.items():
            candidate = group[group.ell == ell]
            if len(candidate) != 1:
                raise ValueError(f"Required ell={ell} not stored for {key}; new estimation is required")
            row = candidate.iloc[0]
            selections.append(dict(zip(KEYS, key), Rule=rule, ell=ell, RMSE=float(row.RMSE),
                                   Bias=float(row.Bias), Coverage=float(row.Coverage_Fraction),
                                   Oracle_ell=int(oracle.ell), Oracle_RMSE=float(oracle.RMSE),
                                   RMSE_ratio=float(row.RMSE/oracle.RMSE),
                                   Oracle_at_grid_upper=bool(oracle.ell == group.ell.max()), Reps=30))
    selected = pd.DataFrame(selections)
    selected.to_csv(out/"rule_comparison.csv", index=False)
    rho = selected[(selected.Experiment == "rho scaling") & (selected.Rule == "Bound delta=0.05")]
    rho.to_csv(out/"rho_comparison.csv", index=False)
    oracle = selected[selected.Rule == "Saved coverage-filtered oracle"]
    oracle.to_csv(out/"original_oracle_all_settings.csv", index=False)
    board = selected.groupby("Rule").agg(
        Settings=("RMSE", "size"), Pooled_RMSE=("RMSE", lambda x: np.sqrt(np.mean(x*x))),
        Median_RMSE_ratio=("RMSE_ratio", "median"), Worst_RMSE_ratio=("RMSE_ratio", "max")).reset_index()
    board.to_csv(out/"pooled_summary.csv", index=False)
    config = pd.read_csv(source/"data_generation_config.csv").iloc[0].to_dict()
    audit = dict(source=str(source), settings=settings.setting_id.nunique(),
                 datasets=len(settings), pss_records=len(raw), per_ell_groups=len(per), reps=30,
                 all_required_candidates_available=True, replicate_rmse_matches_saved=True,
                 original_oracle_reproduced=True,
                 source_sha256={name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                                for name in ["settings.csv", "data_generation_config.csv", "r_pss_cadee_estimates.csv", "pss_per_ell.csv", "r_pss_cadee_summary.csv"]},
                 distribution_parameters=config)
    (out/"audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    report = f"""# Reanalysis of the original five-distribution experiment

No simulations or estimator fits were rerun. This applies fixed, uncalibrated
theory-inspired ell schedules to the saved original R-estimator outcomes.
All 975 candidate groups have 30 finite estimates, and their recalculated RMSEs
agree with pss_per_ell.csv. The original coverage-filtered oracle is reproduced
for all 55 settings. All candidates needed by the four rules were already saved.

## Original design

- Normal, Gamma(shape=.4, scale=.3), Beta(a=.5, b=2), Lognormal(meanlog=0, sdlog=1), Laplace(scale=1/sqrt(2)).
- Gaussian copula: rho is its latent Gaussian equicorrelation parameter, not necessarily the transformed margins' Pearson correlation.
- N scaling: n=1000,3000,10000,30000; d=5, rho=0.
- d scaling: d=2,5,10,20; n=20000, rho=0.
- rho scaling: rho=0,.5,.8; n=20000, d=5.
- 55 settings, 30 repetitions each, 1650 datasets.

## Reused results

The original oracle minimizes RMSE over candidates whose mean coverage is at
least .95, skipped fraction at most .05, and mean occupied-cell size at least 2.
It is a posthoc, coverage-filtered grid oracle, not a globally optimal or
independently validated selector. The original saved candidate grid is 1:20
at d<=5, 1:10 at d=10, and 1:5 at d=20. Lognormal at rho=.8 selects the grid
upper limit ell=20, so the unconstrained optimum is unknown.

At n=20000,d=5, both bound rules and the literal dimension-aware asymptotic
formula with coefficient one select ell=1, irrespective of family or rho.
The fixed-d rate without the d^6 factor selects ell=2.

{markdown(rho[['Distribution', 'Correlation', 'ell', 'RMSE', 'Oracle_ell', 'Oracle_RMSE', 'Oracle_at_grid_upper']])}

## Aggregate across the saved 55 settings

Pooled RMSE is the square root of the equally weighted setting MSEs, not mean
RMSE. It can be dominated by highly correlated settings and is not minimax risk.

{markdown(board)}

## Scope

These are results for the original R rank-spacing statistic, with division by n,
not the smoothed-subgrid/N_eff statistic in pss_convergence_proof-3.pdf. The
theory-inspired schedules can be empirically checked on this old statistic,
but this is not an exact theorem-estimator validation. The original datasets
are saved, so the exact PDF estimator can later be evaluated on identical data.
No original files were changed and no coefficients were fitted in this reanalysis.

The five distribution designs also do not directly satisfy this proof's
bounded-support, positive lower-bound and bounded-density assumptions. Treat
the comparison as an empirical extrapolation, not a contradiction of the theorem.
The ell schedules depend only on n,d and hence do not adapt to rho or marginal
shape. A large finite-sample gap does not by itself refute an asymptotic rate.

Source results: {source}
"""
    (out/"REPORT.md").write_text(report)
    print("REANALYSIS_COMPLETE", out)
    print(rho[["Distribution", "Correlation", "ell", "RMSE", "Oracle_ell", "Oracle_RMSE"]].to_string(index=False))
    print(board.to_string(index=False))


if __name__ == "__main__":
    main()
