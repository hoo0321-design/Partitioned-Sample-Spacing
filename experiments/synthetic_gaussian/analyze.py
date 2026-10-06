"""Gaussian partition sensitivity from existing canonical PSS estimates only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path("/Users/hojeongwoo/Documents/Codex/2026-05-03/d-dimensional-partitioned-sample-spacing-pss/Partitioned-Sample-Spacing")
PAIRS = [(20000, d) for d in [2, 5, 10, 20]] + [(n, 5) for n in [1000, 3000, 10000, 30000]]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--output", type=Path, default=ROOT/"results/synthetic_gaussian_20261006")
    args = p.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    archive = args.source/"results/sc_cv_v2_20261005"
    protocol = json.loads((archive/"protocol.json").read_text())
    assert protocol["definition"] == "smoothed_subgrid_neff_v2"
    for path in ["PSS/pss_v2.py", "PSS/pss_v2.cpp"]:
        assert sha(ROOT/path) == protocol["code_sha256"][path], path
    raw = pd.read_csv(archive/"candidates.csv", low_memory=False)
    raw = raw[(raw.family == "Normal") & (raw.rho == .5)].copy()
    raw = raw[[tuple(x) in PAIRS for x in raw[["n", "d"]].to_numpy()]]
    assert set(map(tuple, raw[["n", "d"]].drop_duplicates().to_numpy())) == set(PAIRS)
    assert not raw.duplicated(["n", "d", "ell", "replicate"]).any()
    assert raw.groupby(["n", "d", "ell"]).size().eq(30).all()
    assert np.isfinite(raw[["estimate", "truth", "coverage"]]).all().all()
    truth = .5*(raw.d*np.log(2*np.pi*np.e)+(raw.d-1)*np.log(.5)+np.log(1+(raw.d-1)*.5))
    np.testing.assert_allclose(raw.truth, truth, rtol=0, atol=1e-12)
    raw["error"] = raw.estimate-raw.truth
    rng = np.random.default_rng(2026100602)
    rows = []
    for (n, d, ell), group in raw.groupby(["n", "d", "ell"], sort=True):
        error = group.sort_values("replicate").error.to_numpy()
        boot = error[rng.integers(len(error), size=(2000, len(error)))]
        lower, upper = np.quantile(np.sqrt(np.mean(boot**2, axis=1)), [.025, .975])
        rows.append(dict(n=int(n), d=int(d), ell=int(ell), reps=len(error),
                         rmse=float(np.sqrt(np.mean(error**2))), rmse_low=lower, rmse_high=upper,
                         bias=float(error.mean()), coverage_mean=float(group.coverage.mean()),
                         coverage_min=float(group.coverage.min()), truth=float(group.truth.iloc[0])))
    per = pd.DataFrame(rows)
    previous_path = args.source/"results/dependent_five_pdf_20261003/per_ell.csv"
    previous = pd.read_csv(previous_path)
    previous = previous[(previous.family == "Normal") & (previous.rho == .5) &
                        (previous.definition == "PDF subgrid / N_eff")]
    merged = per.merge(previous[["n", "d", "ell", "rmse"]], on=["n", "d", "ell"], suffixes=("", "_previous"), validate="one_to_one")
    assert len(merged) == len(per) == 260
    np.testing.assert_allclose(merged.rmse, merged.rmse_previous, rtol=0, atol=1e-10)
    comparison = []
    for n, d in PAIRS:
        part = per[(per.n == n) & (per.d == d)].sort_values("ell")
        assert set(part.ell) == set(protocol["candidate_grids"][str(d)])
        best = part.sort_values(["rmse", "ell"]).iloc[0]
        lam = 2*d*np.log(2*n+1)+np.log(96/.05)
        bound_ell = min(part.ell, key=lambda ell: (ell**-2+np.sqrt(lam)*(float(ell)**d/n)**.25, ell))
        bound = part[part.ell == bound_ell].iloc[0]
        rate_continuous = (n/(d**6*np.log(n)**2))**(1/(d+8))
        rate_ell = max(1, int(np.floor(rate_continuous+.5)))
        rate = part[part.ell == rate_ell].iloc[0]
        comparison.append(dict(n=n, d=d, empirical_ell=int(best.ell), empirical_rmse=float(best.rmse),
                               empirical_coverage_mean=float(best.coverage_mean),
                               empirical_coverage_min=float(best.coverage_min),
                               bound_ell=int(bound_ell), bound_rmse=float(bound.rmse),
                               bound_to_best_rmse=float(bound.rmse/best.rmse),
                               rate_C1_ell=rate_ell, rate_C1_rmse=float(rate.rmse),
                               rate_C1_continuous=float(rate_continuous),
                               bound_matches=int(best.ell)==bound_ell,
                               rate_matches=int(best.ell)==rate_ell))
    comparison = pd.DataFrame(comparison)
    per.to_csv(out/"per_ell_summary.csv", index=False)
    comparison.to_csv(out/"optimal_ell_comparison.csv", index=False)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.titlesize":11,
                         "axes.spines.top":False, "axes.spines.right":False,
                         "legend.fontsize":9, "pdf.fonttype":42, "savefig.dpi":190})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.8))
    colors = ["#0072B2", "#D55E00", "#009E73", "#8A66AD"]
    panels = [[(20000, d) for d in [2, 5, 10, 20]], [(n, 5) for n in [1000, 3000, 10000, 30000]]]
    for i, panel in enumerate(panels):
        ax = axes[i]
        for (n, d), color in zip(panel, colors):
            part = per[(per.n == n) & (per.d == d)].sort_values("ell")
            selected = comparison[(comparison.n == n) & (comparison.d == d)].iloc[0]
            label = f"$d={d}$" if i == 0 else f"$n={n:,}$"
            x, y = part.ell.to_numpy(), part.rmse.to_numpy()
            ax.plot(x, y, color=color, lw=1.7, label=label)
            ax.fill_between(x, part.rmse_low.to_numpy(), part.rmse_high.to_numpy(), color=color, alpha=.12, linewidth=0)
            ax.scatter([selected.empirical_ell], [selected.empirical_rmse], color=color,
                       marker="*", s=125, edgecolors="black", linewidths=.45, zorder=6)
            ax.scatter([selected.bound_ell], [selected.bound_rmse], facecolors="none", edgecolors=color,
                       marker="s", s=100, linewidths=1.5, zorder=5)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlim(.84, 43)
        ax.xaxis.set_major_locator(FixedLocator([1, 2, 3, 4, 6, 8, 12, 20, 40]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: str(int(value))))
        ax.grid(True, which="major", color="#E0E3E7", lw=.65); ax.set_axisbelow(True)
        ax.set_xlabel("Partitions per coordinate $\\ell$")
        ax.set_ylabel("Entropy RMSE (nats)")
        ax.set_title("(a) Fixed $n=20,000$; varying dimension" if i == 0 else "(b) Fixed $d=5$; varying sample size")
        ax.legend(frameon=False, loc="upper left", ncol=2)
    fig.suptitle("Gaussian data: empirical versus bound-based partition choice", fontsize=13, y=.985)
    fig.text(.5, .922, "$X\\sim N_d(0,\\Sigma)$, $\\Sigma_{jj}=1$, $\\Sigma_{ij}=0.5$ ($i\\ne j$); 30 repetitions per condition",
             ha="center", fontsize=10)
    markers = [Line2D([], [], marker="*", color="black", linestyle="none", markersize=10, label="Empirical RMSE minimum"),
               Line2D([], [], marker="s", color="black", markerfacecolor="none", linestyle="none", markersize=8, label="Bound minimum ($\\delta=0.05$)")]
    fig.legend(handles=markers, loc="upper center", bbox_to_anchor=(.5, .888), ncol=2, frameon=False)
    fig.text(.065, .133, "Shading: pointwise 95% bootstrap intervals. All saved levels and repetitions are retained; no coverage filter.", fontsize=8.5)
    fig.text(.065, .091, "The rounded dimension-aware rate with C=1 selects ell=1 in all eight conditions. Stars are same-data grid minima.", fontsize=8.5)
    fig.text(.065, .049, "Gaussian densities are outside the bounded-support theorem assumptions; minimizing its bound need not minimize RMSE.", fontsize=8.5)
    fig.subplots_adjust(left=.075, right=.98, top=.765, bottom=.25, wspace=.25)
    pdf_dir = ROOT/"output/pdf/synthetic_gaussian_20261006"; pdf_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_dir/"gaussian_ell_rmse.pdf")
    fig.savefig(out/"gaussian_ell_rmse.png")
    plt.close(fig)
    notes = ["Gaussian partition-level comparison", "",
             "Source: existing canonical PSS v2 estimates. No simulations, tuning, or baseline reruns.",
             "Normal equicorrelation rho=.5; n fixed at 20000 or d fixed at 5; 30 independent repetitions per condition.",
             "Truth: 0.5*[d*log(2*pi*e)+(d-1)*log(1-rho)+log(1+(d-1)*rho)].",
             "Empirical optimum: minimum aggregate RMSE over EVERY saved ell, without any coverage constraint.",
             "Bound optimum: integer minimizer of ell^-2+sqrt(Lambda)*(ell^d/n)^.25,",
             "Lambda=2*d*log(2*n+1)+log(96/.05). Ties prefer smaller ell.",
             "Dimension-aware rate: max(1,floor((n/(d^6*log(n)^2))^(1/(d+8))+.5)); C=1 fixed.",
             "No unknown asymptotic coefficient is estimated or asserted to equal one.",
             "Both theoretical choices are explicit formula implementations, not proven finite-sample RMSE minimizers.",
             "Gaussian densities do not meet the theorem's compact-support/positive-lower-bound assumptions.",
             "High-dimensional ell=1 agreement does not imply small estimation error.",
             "The empirical minimum is selected and reported on the same 30 repetitions; it is an optimistic reference.",
             "Intervals are pointwise bootstrap intervals at each fixed ell, not confidence sets for the minimizer.",
             "Canonical estimates average covered points. Coverage diagnostics at empirical minima are saved in the table.", "",
             comparison.to_string(index=False), "",
             "Suggested caption", "RMSE of the canonical PSS entropy estimator as a function of partition level for equicorrelated",
             "Gaussian data (rho=0.5). Left: n=20000 with varying dimension; right: d=5 with varying sample size.",
             "Stars mark unrestricted empirical RMSE minima over the saved grid; open squares mark integer minimizers",
             "of the displayed theoretical bound with delta=0.05. Shading gives pointwise 95% bootstrap intervals over",
             "30 repetitions. The C=1 dimension-aware rate selects ell=1 throughout. No coverage filter is applied.",
             "This comparison is outside the theorem's distributional assumptions and does not test an identity",
             "between upper-bound minimization and finite-sample RMSE minimization."]
    (out/"notes.txt").write_text("\n".join(notes)+"\n")
    manifest = dict(status="PASS", source_files={str(path):sha(path) for path in [archive/"candidates.csv", archive/"protocol.json", previous_path]},
                    canonical_code_hashes={p:sha(ROOT/p) for p in ["PSS/pss_v2.py", "PSS/pss_v2.cpp"]},
                    analysis_sha256=sha(__file__), settings=8, datasets=240, candidate_estimates=len(raw),
                    groups_verified=len(merged), max_previous_rmse_difference=float(np.abs(merged.rmse-merged.rmse_previous).max()),
                    bound_matches=int(comparison.bound_matches.sum()), rate_matches=int(comparison.rate_matches.sum()),
                    bootstrap_replicates=2000, bootstrap_seed=2026100602, new_simulations=False,
                    empirical_selection_filter="none", normal_rho=.5)
    (out/"audit.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(comparison[["n", "d", "empirical_ell", "empirical_rmse", "bound_ell", "rate_C1_ell"]].to_string(index=False))
    print("Verified", len(raw), "saved values; matching minima:", manifest["bound_matches"], "of 8")


if __name__ == "__main__":
    main()
