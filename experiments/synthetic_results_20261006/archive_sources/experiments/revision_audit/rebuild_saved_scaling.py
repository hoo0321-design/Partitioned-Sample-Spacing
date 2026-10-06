"""Rebuild the four-row historical figure from audited replicate-level data.

This does not change the estimator or pretend the old experiment used SC-CV.
The manuscript itself is not edited. Output is explicitly labeled historical.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"experiments"/"anchor_grid"))
from plot_anchor_grid_results import COLORS, DISTRIBUTIONS, MARKERS, METHODS, configure_matplotlib

KEYS = ["Experiment", "Distribution", "Dimensions", "N_Samples", "Correlation", "Method"]


def audited_points(source):
    summary = pd.read_csv(source/"combined_summary.csv")
    raw = pd.concat([pd.read_csv(source/"r_pss_cadee_estimates.csv"),
                     pd.read_csv(source/"knn_um_estimates.csv")], ignore_index=True)
    if summary.duplicated(KEYS).any() or len(summary) != 330:
        raise ValueError("Expected exactly 55 settings x 6 methods")
    rows = []
    selections = []
    groups = raw.groupby(KEYS)
    for r in summary.itertuples(index=False):
        key = tuple(getattr(r, name) for name in KEYS)
        g = groups.get_group(key)
        if pd.notna(r.Optimal_Param):
            g = g[g.Optimal_Param == r.Optimal_Param]
        if len(g) != 30 or g.Replicate.nunique() != 30:
            raise ValueError(f"Missing or repeated replicates: {key}")
        if not np.isfinite(g[["Estimate", "True_Entropy", "Eval_Time_s", "Train_Time_s"]]).all().all():
            raise ValueError(f"Invalid replicate: {key}")
        errors = g.Estimate.to_numpy()-g.True_Entropy.to_numpy()
        rmse = np.sqrt(np.mean(errors**2))
        se = np.std(errors**2, ddof=1)/np.sqrt(len(errors))/(2*rmse)
        if not np.isclose(rmse, r.RMSE, atol=1e-9, rtol=1e-9):
            raise ValueError(f"Saved RMSE mismatch: {key}")
        if not np.isclose(se, r.RMSE_SE, atol=1e-9, rtol=1e-9):
            raise ValueError(f"Saved RMSE SE mismatch: {key}")
        total = g.Eval_Time_s.to_numpy()+g.Train_Time_s.to_numpy()
        rows.append(dict(zip(KEYS, key), Optimal_Param=r.Optimal_Param, N_Reps=30,
                         RMSE=rmse, RMSE_SE=se, Bias=np.mean(errors),
                         Mean_Time_s=np.mean(total), Time_SE_s=np.std(total, ddof=1)/np.sqrt(30),
                         Mean_Train_Time_s=g.Train_Time_s.mean(), Mean_Eval_Time_s=g.Eval_Time_s.mean()))
        selections.append(g)
    return pd.DataFrame(rows), pd.concat(selections, ignore_index=True)


def draw(points, out):
    configure_matplotlib()
    specs = [
        ("N scaling", "N_Samples", "n", "RMSE", "RMSE_SE", "(a) Sample size: d=5, rho=0"),
        ("N scaling", "N_Samples", "n", "Mean_Time_s", "Time_SE_s", "(b) Runtime: selected-parameter evaluation + NF training"),
        ("d scaling", "Dimensions", "d", "RMSE", "RMSE_SE", "(c) Dimension: n=20,000, rho=0"),
        ("rho scaling", "Correlation", "rho", "RMSE", "RMSE_SE", "(d) Dependence: n=20,000, d=5"),
    ]
    fig, axes = plt.subplots(4, 5, figsize=(15.5, 12.4))
    fig.subplots_adjust(left=.055, right=.99, bottom=.095, top=.89, hspace=.64, wspace=.28)
    handles = []
    for row, (experiment, x_col, x_label, metric, err, title) in enumerate(specs):
        subset = points[points.Experiment == experiment]
        for col, family in enumerate(DISTRIBUTIONS):
            ax = axes[row, col]
            part = subset[subset.Distribution == family]
            for method in METHODS:
                g = part[part.Method == method].sort_values(x_col)
                line = ax.errorbar(g[x_col], g[metric], yerr=g[err], label=method,
                                  color=COLORS[method], marker=MARKERS[method], markersize=3.5,
                                  linewidth=1.8 if method == "PSS" else 1.15, capsize=2,
                                  linestyle="--" if method == "UM-tKL" else "-",
                                  markerfacecolor="white" if method == "UM-tKL" else COLORS[method],
                                  zorder=5 if method == "UM-tKL" else 3)
                if row == 0 and col == 0:
                    handles.append(line)
            ax.set_yscale("log")
            if x_col != "Correlation":
                ax.set_xscale("log", base=10 if x_col == "N_Samples" else 2)
                ax.xaxis.set_major_formatter(plt.ScalarFormatter())
            ax.set_xticks(sorted(part[x_col].unique()))
            ax.set_title(family, pad=5)
            ax.set_xlabel(x_label)
            if col == 0:
                ax.set_ylabel("Runtime (s; log scale)" if row == 1 else "RMSE (nats; log scale)")
            ax.grid(True, which="both", alpha=.2)
        y = axes[row, 0].get_position().y1+.035
        fig.text(.055, y, title, fontsize=11, weight="bold")
    fig.suptitle("Historical oracle benchmark: corrected row ordering", x=.055, ha="left", y=.99, fontsize=15, weight="bold")
    fig.legend(handles, METHODS, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(.52, .965))
    fig.text(.055, .022,
             "30 repetitions; error bars are +/- 1 Monte Carlo SE, not 95% CI. PSS uses rank spacing / n; this is not the new PDF estimator or SC-CV.\n"
             "Runtime excludes hyperparameter search. UM variants share an NF fit; each curve includes that fit plus its own evaluation time.",
             fontsize=9)
    for suffix in ["png", "pdf"]:
        fig.savefig(out/f"fig2_historical_corrected.{suffix}", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    points, selected = audited_points(args.source)
    points.to_csv(args.out/"historical_scaling_audited.csv", index=False)
    selected.to_csv(args.out/"historical_selected_replicates.csv", index=False)
    protocol = dict(
        source=str(args.source.resolve()), settings=55, methods=6, summaries=len(points),
        selected_replicates=len(selected), repetitions=30,
        rows=["N scaling RMSE", "N scaling runtime", "d scaling RMSE", "rho scaling RMSE"],
        source_parameters=pd.read_csv(args.source/"data_generation_config.csv").iloc[0].to_dict(),
        uncertainty="plus/minus one delta-method Monte Carlo standard error for RMSE; one SE for mean runtime; conditional on selected oracle parameter",
        scope="historical rank/n PSS, NOT PDF/N_eff, NOT SC-CV; not a replacement for new-definition experiments",
        tuning="PSS coverage-filtered empirical RMSE oracle; KL/KSG/UM k empirical RMSE oracle; CADEE has no candidate parameter in saved experiment",
        runtime="selected-candidate raw evaluation time + training time; hyperparameter search excluded; not interchangeable with end-to-end SC-CV cost",
        sha256={f: hashlib.sha256((args.source/f).read_bytes()).hexdigest() for f in
                ["combined_summary.csv", "r_pss_cadee_estimates.csv", "knn_um_estimates.csv", "data_generation_config.csv"]},
    )
    (args.out/"historical_scaling_provenance.json").write_text(json.dumps(protocol, indent=2)+"\n")
    draw(points, args.out)
    print(json.dumps({k:protocol[k] for k in ["settings", "summaries", "selected_replicates", "uncertainty", "scope"]}, indent=2))
    print("FIGURE_REBUILT", args.out.resolve())


if __name__ == "__main__":
    main()
