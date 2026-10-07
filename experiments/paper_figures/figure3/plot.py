"""Reproduce Figure 3 from the bundled historical five-family results.

Verify frozen input hashes and recompute all summaries before plotting. No
simulation, estimator fitting, oracle reselection, or timing is performed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import runpy
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PACKAGE = REPO / "experiments/synthetic_results_20261006"
SOURCE = PACKAGE / "results/five_family_legacy_20261006"
STEM = "five_family_benchmark"
FAMILIES = ["Normal", "Gamma", "Beta", "Lognormal", "Laplace"]
METHODS = ["PSS", "CADEE", "KL", "KSG", "UM-tKL", "UM-tKSG"]
COLORS = dict(zip(METHODS, ["#C23B22", "#6B7280", "#2563EB", "#059669", "#7C3AED", "#D97706"]))
MARKERS = dict(zip(METHODS, ["o", "s", "^", "D", "v", "P"]))
SPECS = [
    ("N scaling", "N_Samples", "Sample size $n$", "RMSE", "RMSE_SE",
     "(a) Sample size: $d=5$, $\\rho=0$"),
    ("N scaling", "N_Samples", "Sample size $n$", "Mean_Time_s", "Time_SE_s",
     "(b) Runtime: selected-parameter evaluation plus flow training"),
    ("d scaling", "Dimensions", "Dimension $d$", "RMSE", "RMSE_SE",
     "(c) Dimension: $n=20,000$, $\\rho=0$"),
    ("rho scaling", "Correlation", "Copula correlation $\\rho$", "RMSE", "RMSE_SE",
     "(d) Dependence: $n=20,000$, $d=5$"),
]


def verify():
    # Load definitions without invoking main() or writing a bytecode cache.
    verifier = SimpleNamespace(**runpy.run_path(str(PACKAGE / "verify.py")))
    manifest = verifier.read_json(PACKAGE / "manifest.json")
    records = {record["path"]: record for record in manifest["files"]}
    hashes = {}
    for name in ["summary.csv", "selected_replicates.csv", "audit.json"]:
        path = SOURCE / name
        relative = path.relative_to(REPO).as_posix()
        expected = records[relative]
        actual = verifier.digest(path)
        verifier.require(path.stat().st_size == expected["bytes"] and actual == expected["sha256"],
                         f"Frozen Figure 3 input changed: {relative}")
        hashes[relative] = actual
    counts = verifier.verify_legacy()
    rows = verifier.read_csv(SOURCE / "summary.csv")
    keys = ["Experiment", "Distribution", "Dimensions", "N_Samples", "Correlation", "Method"]
    verifier.require(len({tuple(row[k] for k in keys) for row in rows}) == 330,
                     "Duplicate Figure 3 summary")
    verifier.require({row["Distribution"] for row in rows} == set(FAMILIES), "Unexpected families")
    verifier.require({row["Method"] for row in rows} == set(METHODS), "Unexpected methods")
    verifier.require(all(int(row["N_Reps"]) == 30 for row in rows), "Unexpected repetition count")
    for row in rows:
        for value, error in [("RMSE", "RMSE_SE"), ("Mean_Time_s", "Time_SE_s")]:
            y, se = float(row[value]), float(row[error])
            verifier.require(math.isfinite(y) and math.isfinite(se) and se >= 0 and y - se > 0,
                             "Nonpositive or invalid interval on logarithmic axis")
    audit = verifier.read_json(SOURCE / "audit.json")
    verifier.require(audit["status"] == "PASS" and "historical rank spacing / n" in audit["estimator"],
                     "Unexpected estimator provenance")
    report = dict(status="PASS", **counts, methods=METHODS, families=FAMILIES,
                  input_sha256=hashes, estimator="historical rank-spacing / n; not canonical v2",
                  summary_checks="RMSE, delta-method RMSE SE, bias, mean runtime and runtime SE recomputed from saved replicates",
                  uncertainty="plus/minus one Monte Carlo SE, conditional on saved oracle choice",
                  timing="selected-parameter evaluation plus flow training; parameter search excluded",
                  new_simulations=0, new_estimator_fits=0, retuned_parameters=0,
                  verifier_sha256=verifier.digest(PACKAGE / "verify.py"),
                  plotting_source_sha256=verifier.digest(Path(__file__)))
    return rows, report


def draw(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
                         "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.dpi": 200})
    fig, axes = plt.subplots(4, 5, figsize=(15.5, 12.6))
    fig.subplots_adjust(left=.06, right=.99, top=.887, bottom=.115, hspace=.72, wspace=.31)
    handles = []
    for row_index, (experiment, xcol, xlabel, ycol, errcol, title) in enumerate(SPECS):
        for col, family in enumerate(FAMILIES):
            ax = axes[row_index, col]
            part = [r for r in rows if r["Experiment"] == experiment and r["Distribution"] == family]
            for method in METHODS:
                values = sorted((r for r in part if r["Method"] == method), key=lambda r: float(r[xcol]))
                line = ax.errorbar([float(r[xcol]) for r in values], [float(r[ycol]) for r in values],
                                  yerr=[float(r[errcol]) for r in values], label=method,
                                  color=COLORS[method], marker=MARKERS[method], markersize=3.4,
                                  linewidth=1.9 if method == "PSS" else 1.2, capsize=2, elinewidth=.65,
                                  linestyle="--" if method == "UM-tKL" else "-",
                                  markerfacecolor="white" if method == "UM-tKL" else COLORS[method])
                if row_index == 0 and col == 0:
                    handles.append(line)
            ax.set_yscale("log")
            if xcol != "Correlation":
                ax.set_xscale("log", base=10 if xcol == "N_Samples" else 2)
                ax.xaxis.set_major_formatter(plt.ScalarFormatter())
            ax.set_xticks(sorted({float(r[xcol]) for r in part}))
            ax.set_title(family, pad=5)
            ax.set_xlabel(xlabel)
            if col == 0:
                ax.set_ylabel("Time (seconds)" if ycol == "Mean_Time_s" else "RMSE (nats)")
            ax.grid(True, which="major", alpha=.22)
            ax.set_axisbelow(True)
        fig.text(.06, axes[row_index, 0].get_position().y1 + .031, title, fontsize=11, weight="bold")
    fig.suptitle("Five-family synthetic benchmark", x=.06, ha="left", y=.988, fontsize=16, weight="bold")
    fig.legend(handles, METHODS, ncol=6, frameon=False, loc="upper center", bbox_to_anchor=(.53, .967), fontsize=10)
    fig.text(.06, .069, "Saved original results; 30 repetitions per condition. Error bars: +/- 1 Monte Carlo SE, conditional on the selected oracle parameter.", fontsize=9)
    fig.text(.06, .044, "PSS uses the historical rank-spacing / n estimator. PSS and kNN-based methods use same-repetition RMSE oracle selection.", fontsize=9)
    fig.text(.06, .019, "Runtime excludes parameter search; flow methods include training plus their own evaluation. rho is latent Gaussian-copula correlation.", fontsize=9)
    for suffix in ["pdf", "png"]:
        fig.savefig(output / f"{STEM}.{suffix}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Verify saved inputs and summaries without plotting or writing files.")
    parser.add_argument("--output-dir", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    rows, report = verify()
    if not args.verify_only:
        output = args.output_dir.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        draw(rows, output)
        report.update(output_directory=str(output), panels=20, plotted_points=450)
        (output / "figure_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
