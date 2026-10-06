"""Reuse archived bounded-ridge simulations for two compact paper figures.

No new samples, coefficient fitting, or changes to the estimator. Canonical
numerical spot checks are performed separately by audit_reuse.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path("/Users/hojeongwoo/Documents/Codex/2026-05-03/d-dimensional-partitioned-sample-spacing-pss/Partitioned-Sample-Spacing")
FAMILIES = {"ridge_medium": 1, "ridge_frequency2": 2}
RUNS = {"theory_selection_20261003": [1000, 3000, 10000, 30000, 100000],
        "theory_selection_large_n_20261003": [300000, 1000000]}
METHODS = {"ell=1": ("Fixed $\\ell=1$", "#777777", "s", "-"),
           "bound": ("Bound rule ($\\delta=0.05$)", "#0072B2", "o", "-"),
           "oracle": ("Oracle reference", "#B44C48", "^", "--")}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(group, rng, bootstrap=False):
    error = group.error.to_numpy()
    result = dict(reps=len(error), rmse=float(np.sqrt(np.mean(error**2))),
                  bias=float(error.mean()), estimate_sd=float(error.std(ddof=1)),
                  mae=float(np.mean(np.abs(error))),
                  coverage_mean=float(group.coverage.mean()),
                  coverage_min=float(group.coverage.min()),
                  mean_cell_size=float(group.mean_cell_size.mean()),
                  integrated_mass_mean=float(group.integrated_mass.mean()))
    if bootstrap:
        draws = error[rng.integers(len(error), size=(2000, len(error)))]
        for name, values in [("rmse", np.sqrt(np.mean(draws**2, axis=1))),
                             ("bias", draws.mean(axis=1))]:
            low, high = np.quantile(values, [.025, .975])
            result.update({name+"_low": float(low), name+"_high": float(high)})
    return result


def collect(source):
    groups, manifests = {}, {}
    raw_frames = []
    for run, sample_sizes in RUNS.items():
        run_path = source / "results" / run
        design = json.loads((run_path / "design.json").read_text())
        manifests[str(run_path / "design.json")] = sha(run_path / "design.json")
        for family in FAMILIES:
            for d in [2, 5]:
                for n in sample_sizes:
                    cfg = next(c for c in design["settings"] if (c["family"], c["n"], c["d"]) == (family, n, d))
                    path = run_path / "evaluation" / f"{family}_n{n}_d{d}.csv"
                    data = pd.read_csv(path)
                    if data.duplicated(["rep", "ell"]).any():
                        raise ValueError(f"Duplicate observations: {path}")
                    if set(data.ell) != set(cfg["ells"]):
                        raise ValueError(f"Candidate grid mismatch: {path}")
                    if not (data.groupby("ell").size() == design["eval_reps"]).all():
                        raise ValueError(f"Missing repetitions: {path}")
                    for name, value in [("family", family), ("n", n), ("d", d)]:
                        if not (data[name] == value).all():
                            raise ValueError(f"Setting mismatch: {path}")
                    truth = -(1-np.sqrt(1-.7**2)+np.log((1+np.sqrt(1-.7**2))/2))
                    np.testing.assert_allclose(data.true_entropy, truth, atol=1e-14, rtol=0)
                    np.testing.assert_allclose(data.error, data.estimate-truth, atol=1e-13, rtol=0)
                    if not np.isfinite(data[["estimate", "error", "coverage"]]).all().all():
                        raise ValueError(f"Nonfinite observations: {path}")
                    for ell, part in data.groupby("ell"):
                        groups[(family, d, n, int(ell))] = part.sort_values("rep")
                    manifests[str(path)] = sha(path)
                    raw_frames.append(data)
    return groups, manifests, pd.concat(raw_frames, ignore_index=True)


def summaries(groups):
    rng = np.random.default_rng(2026100601)
    rows = []
    for (family, d, n, ell), group in sorted(groups.items()):
        rows.append(dict(family=family, k=FAMILIES[family], d=d, n=n, ell=ell,
                         **stats(group, rng, bootstrap=n in [10000, 100000])))
    per_ell = pd.DataFrame(rows)
    selected = []
    for (family, d, n), part in per_ell.groupby(["family", "d", "n"], sort=True):
        lam = 2*d*np.log(2*n+1)+np.log(96/.05)
        candidates = sorted(part.ell.astype(int))
        bound = min(candidates, key=lambda ell: (ell**-2+np.sqrt(lam)*(ell**d/n)**.25, ell))
        eligible = part[part.coverage_min == 1]
        if eligible.empty:
            raise ValueError("No fully covered oracle candidate")
        oracle = int(eligible.sort_values(["rmse", "ell"]).iloc[0].ell)
        for method, ell in [("ell=1", 1), ("bound", bound), ("oracle", oracle)]:
            selected.append(dict(family=family, k=FAMILIES[family], d=d, n=n,
                                 method=method, ell=ell,
                                 **stats(groups[(family, d, n, ell)], rng, bootstrap=True)))
    return per_ell, pd.DataFrame(selected)


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 10, "axes.titlesize": 11,
                         "legend.fontsize": 9, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": .7,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "savefig.dpi": 190})


def finish_axes(ax):
    ax.grid(True, which="major", color="#E0E3E7", lw=.65)
    ax.set_axisbelow(True)


def scaling_figure(selected, d):
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 4.6), sharey=True)
    for index, (family, k) in enumerate(FAMILIES.items()):
        ax = axes[index]
        for method, (label, color, marker, linestyle) in METHODS.items():
            data = selected[(selected.family == family) & (selected.d == d) & (selected.method == method)].sort_values("n")
            x, y = data.n.to_numpy(), data.rmse.to_numpy()
            ax.plot(x, y, label=label, color=color, marker=marker, linestyle=linestyle,
                    markersize=4, lw=1.5)
            if method != "oracle":
                low, high = data.rmse_low.to_numpy(), data.rmse_high.to_numpy()
                ax.fill_between(x, low, high, color=color, alpha=.14, linewidth=0)
            if method == "bound":
                levels = ", ".join(map(str, data.ell.to_list()))
                ax.text(.02, .04, "Bound-rule $\\ell$: " + levels, transform=ax.transAxes,
                        fontsize=8, color=color,
                        bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none", "pad": 2})
        ax.set(xscale="log", yscale="log", xlabel="Sample size $n$", title=f"({chr(97+index)}) Frequency $k={k}$")
        ax.set_ylim(2e-4, .3)
        finish_axes(ax)
    axes[0].set_ylabel("Entropy RMSE (nats)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(.5, .88))
    fig.suptitle(f"Bounded dependent density: error versus sample size ($d={d}$)", y=.985, fontsize=12)
    fig.text(.5, .925, "$f(x)=1+0.7\\cos(2\\pi k(x_1-x_2))$ on $[0,1]^d$; true entropy = -0.131623 nats",
             ha="center", fontsize=9)
    fig.text(.065, .08, "$n\\leq10^5$: 100 repeats; $n=3\\times10^5,10^6$: 30 repeats. Shading: pointwise 95% bootstrap intervals.", fontsize=8)
    fig.text(.065, .035, "Oracle: same-repeat minimum RMSE among fully covered candidates; optimistic reference, without a band.", fontsize=8)
    fig.subplots_adjust(left=.085, right=.975, top=.74, bottom=.23, wspace=.16)
    return fig


def sensitivity_figure(per_ell, d):
    fig, axes = plt.subplots(2, 2, figsize=(8.3, 6.8), sharex="col", sharey="row")
    colors = {10000: "#0072B2", 100000: "#D97724"}
    for column, (family, k) in enumerate(FAMILIES.items()):
        for n, color in colors.items():
            data = per_ell[(per_ell.family == family) & (per_ell.d == d) & (per_ell.n == n)].sort_values("ell")
            for row, metric in [(0, "rmse"), (1, "bias")]:
                ax = axes[row, column]
                x, y = data.ell.to_numpy(), data[metric].to_numpy()
                ax.plot(x, y, color=color, marker="o", markersize=3, lw=1.5,
                        label="$n=10^4$" if n == 10000 else "$n=10^5$")
                sparse = data.coverage_min.to_numpy() < 1
                ax.scatter(x[sparse], y[sparse], facecolors="white", edgecolors=color,
                           s=13, linewidths=.8, zorder=4)
                ax.fill_between(x, data[metric+"_low"].to_numpy(), data[metric+"_high"].to_numpy(),
                                color=color, alpha=.15, linewidth=0)
        for row in range(2):
            ax = axes[row, column]
            ax.set_xscale("log", base=2)
            ticks = [1, 2, 4, 8, 16, 32, 40] if d == 2 else [1, 2, 3, 4, 5, 6, 7]
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: str(int(value))))
            ax.set_xlim(.92, 43 if d == 2 else 7.5)
            finish_axes(ax)
        axes[0, column].set(title=f"({chr(97+column)}) Frequency $k={k}$", yscale="log")
        axes[1, column].set(yscale="symlog", xlabel="Partitions per coordinate $\\ell$")
        axes[1, column].set_yscale("symlog", linthresh=.02, linscale=.5)
        axes[1, column].axhline(0, color="#555555", linestyle=":", lw=.9)
    axes[0, 0].set_ylabel("Entropy RMSE (nats)")
    axes[1, 0].set_ylabel("Signed bias (nats; symlog)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(.5, .925))
    fig.suptitle(f"Partition sensitivity under bounded dependence ($d={d}$)", y=.985, fontsize=12)
    minimum = per_ell[(per_ell.d == d) & per_ell.n.isin([10000, 100000])].coverage_min.min()
    fig.text(.065, .113, "All saved partition levels; 100 repeats. Shading: pointwise 95% bootstrap intervals.", fontsize=8)
    fig.text(.065, .073, f"Hollow markers: at least one repeat has coverage < 1; minimum training coverage = {minimum:.1%}.", fontsize=8)
    fig.text(.065, .033, "Bias = mean estimate - truth; symlog is linear near zero. The estimator averages covered points.", fontsize=8)
    fig.subplots_adjust(left=.1, right=.975, top=.835, bottom=.21, hspace=.16, wspace=.16)
    return fig


def write_notes(out, selected, per_ell, source):
    lines = ["Bounded dependent-density experiment: compact reanalysis", "",
             "Design", "f(x)=1+0.7*cos(2*pi*k*(x1-x2)), x in [0,1]^d, k=1,2; d=2 main, d=5 appendix.",
             "Bounds on the support: 0.3 <= f <= 1.7. Euclidean Lipschitz constant <= 2*pi*k*0.7*sqrt(2).",
             "All coordinate marginals are uniform; true entropy is -0.131623132177013 nats.",
             "For d=5, only the first two coordinates are dependent; three independent uniform coordinates are added.",
             "Lipschitz regularity is on the support, not across the zero-extension boundary.", "",
             "Rules", "The fixed rule uses ell=1. The bound rule minimizes ell^-2+sqrt(Lambda)*(ell^d/n)^(1/4)",
             "over each saved candidate grid, Lambda=2*d*log(2*n+1)+log(96/0.05). Ties prefer smaller ell.",
             "Delta=.05 is a bound parameter, not a certified finite-sample 95% guarantee.",
             "Oracle minimizes aggregate same-repeat RMSE over candidates with coverage=1 in EVERY repeat.",
             "The oracle is optimistic and does not represent a deployable selector. It is shown without confidence bands.",
             "No coefficients or candidates were retuned. This is a posthoc presentation subset of an existing study.", "",
             "Interpretation", "At ell=1 the population product approximation is uniform, with entropy zero.",
             "Its limiting approximation bias is therefore +0.131623132177013 nats.",
             "The finite-sample errors are nonmonotone. Increasing sample size does not change integer ell at every n.",
             "At a fixed n, a small RMSE may coincide with a crossing of signed bias through zero.",
             "The results illustrate partition sensitivity; they do not establish an exact asymptotic exponent",
             "or finite-sample optimality of the theoretical rule. Unknown admissibility constants are not certified.", "",
             "Uncertainty", "Pointwise 95% percentile bootstrap intervals use 2000 resamples of independent dataset repetitions.",
             "Primary settings have 100 repeats; the large-n extension has 30. The rule is frozen during resampling.",
             "No observations are discarded based on their estimation error. No normalized-density variant is used.", "",
             "Canonical verification", "See audit.json: source provenance and deterministic numerical spot checks.",
             "These are not a canonical rerun of every archived repetition; the figures summarize the archived sub-grid/N_eff estimates.", "",
             "Selected results (RMSE in nats)"]
    for family in FAMILIES:
        for d in [2, 5]:
            for n in [100000, 1000000]:
                part = selected[(selected.family == family) & (selected.d == d) & (selected.n == n)]
                values = "; ".join(f"{r.method}: ell={r.ell}, RMSE={r.rmse:.6f}" for r in part.itertuples())
                lines.append(f"k={FAMILIES[family]}, d={d}, n={n}: {values}")
    sensitivity = per_ell[per_ell.n.isin([10000, 100000])]
    lines += ["", f"Minimum per-repeat coverage in selected curves: {selected.coverage_min.min():.6f}",
              f"Minimum per-repeat coverage over sensitivity grids: {sensitivity.coverage_min.min():.6f}", "",
              "Suggested caption: sample-size figure", "Entropy RMSE for two bounded cosine-ridge densities sharing the same uniform marginals",
              "and true entropy. Fixed ell=1, a deterministic bound-based partition rule, and a same-repeat",
              "full-coverage oracle reference are compared. Shading denotes pointwise 95% bootstrap intervals",
              "for the two fixed rules. Results use 100 independent repetitions through n=100000 and 30 at",
              "n=300000 and 1000000. Page 1 is d=2; page 2 adds three independent coordinates (d=5).", "",
              "Suggested caption: sensitivity figure", "Entropy RMSE and signed bias versus partition level, at n=10000 and 100000, for k=1 and 2.",
              "Every archived candidate level is shown. Shading denotes pointwise 95% bootstrap intervals over",
              "100 repetitions. Hollow markers indicate coverage below one in at least one repetition; the canonical",
              "estimate averages the finite-log-density observations. Sparse candidates are not theorem-admissibility certificates.",
              "Signed bias uses a symmetric logarithmic axis, linear within +/-0.02 nats.",
              "An error minimum need not imply small individual bias components. Page 1 is d=2 and page 2 is d=5.",
              "", "Source archive", str(source)]
    (out / "notes.txt").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=ROOT / "results/synthetic_bounded_20261006")
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    audit_path = out / "audit.json"
    audit = json.loads(audit_path.read_text())
    if audit["status"] != "PASS" or audit["source_root"] != str(args.source.resolve()):
        raise ValueError("A passing audit for the same archive is required")
    for path, expected in audit["current_implementation_hashes"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("Canonical implementation changed since audit")
    plots = out / "plots"; plots.mkdir(exist_ok=True)
    pdf_dir = ROOT / "output/pdf/synthetic_bounded_20261006"; pdf_dir.mkdir(parents=True, exist_ok=True)
    groups, manifests, raw = collect(args.source)
    per_ell, selected = summaries(groups)
    per_ell.to_csv(out / "per_ell_summary.csv", index=False)
    selected.to_csv(out / "selected_summary.csv", index=False)
    style()
    for stem, fn, data in [("bounded_n_scaling", scaling_figure, selected),
                           ("bounded_ell_sensitivity", sensitivity_figure, per_ell)]:
        with PdfPages(pdf_dir / (stem+".pdf")) as pdf:
            pdf.infodict().update(Title=stem.replace("_", " "), Author="PSS experiment")
            for d in [2, 5]:
                fig = fn(data, d)
                pdf.savefig(fig)
                fig.savefig(plots / f"{stem}_d{d}.png")
                plt.close(fig)
    write_notes(out, selected, per_ell, args.source)
    manifest = dict(source_files=manifests, analysis_sha256=sha(__file__),
                    settings=len(raw.groupby(["family", "n", "d"])),
                    datasets=len(raw.groupby(["family", "n", "d", "rep"])),
                    candidate_estimates=len(raw), source="archived subgrid/N_eff estimates",
                    canonical_audit="audit.json; deterministic numerical spot checks only",
                    canonical_audit_sha256=sha(audit_path),
                    bootstrap_replicates=2000, bootstrap_seed=2026100601,
                    delta=.05, oracle_filter="minimum coverage across repetitions equals 1",
                    new_simulations=False, normalized_variant=False)
    (out / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "source_files"}, indent=2))


if __name__ == "__main__":
    main()
