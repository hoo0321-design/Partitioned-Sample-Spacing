"""Report every prespecified correlation in the exploratory Gaussian sweep."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"results/synthetic_gaussian_rho_20261006"
PAIRS = [(20000, d) for d in [2, 5, 10, 20]] + [(n, 5) for n in [1000, 3000, 10000, 30000]]
RHOS = [0., .1, .2, .3, .4, .5]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load():
    protocol = json.loads((OUT/"protocol.json").read_text())
    assert protocol["lower_rho_grid"] == RHOS[:-1] and protocol["baseline_rho"] == .5
    for path in ["PSS/pss_v2.py", "PSS/pss_v2.cpp", "experiments/synthetic_gaussian_rho/run.py"]:
        assert sha(ROOT/path) == protocol["code_sha256"][path], "Source changed after protocol freeze"
    checks = json.loads((OUT/"checkpoints/baseline_spotchecks.json").read_text())
    assert len(checks) == 8 and all(c["passed"] for c in checks)
    frames, sources = [], {}
    for path in sorted((OUT/"checkpoints").glob("*.csv")):
        frame = pd.read_csv(path)
        if {"n", "d", "rho", "ell", "estimate", "truth", "coverage", "replicate"}.issubset(frame.columns):
            frames.append(frame)
            sources[str(path)] = sha(path)
    raw = pd.concat(frames, ignore_index=True)
    assert not raw.duplicated(["rho", "n", "d", "replicate", "ell"]).any()
    expected = {(rho, n, d) for rho in RHOS for n, d in PAIRS}
    assert set(map(tuple, raw[["rho", "n", "d"]].drop_duplicates().to_numpy())) == expected
    assert raw.groupby(["rho", "n", "d", "ell"]).size().eq(30).all()
    assert raw.status.eq("ok").all(), "Retained failures require explicit analysis before plotting"
    assert np.isfinite(raw[["estimate", "truth", "coverage"]]).all().all()
    np.testing.assert_allclose(raw.coverage, raw.n_valid/raw.n, rtol=0, atol=1e-14)
    truth = .5*(raw.d*np.log(2*np.pi*np.e)+(raw.d-1)*np.log1p(-raw.rho)+np.log1p((raw.d-1)*raw.rho))
    np.testing.assert_allclose(raw.truth, truth, rtol=0, atol=1e-12)
    raw["error"] = raw.estimate-raw.truth
    for (_, _, d), part in raw.groupby(["rho", "n", "d"]):
        assert set(part.ell) == set(range(1, {2:40, 5:40, 10:12, 20:8}[d]+1))
    assert len(raw) == 46800
    return raw, sources


def summarize(raw):
    rows = []
    rng = np.random.default_rng(2026100603)
    for (rho, n, d, ell), group in raw.groupby(["rho", "n", "d", "ell"], sort=True):
        errors = group.sort_values("replicate").error.to_numpy()
        resampled = errors[rng.integers(len(errors), size=(2000, len(errors)))]
        low, high = np.quantile(np.sqrt(np.mean(resampled**2, axis=1)), [.025, .975])
        rows.append(dict(rho=rho, n=int(n), d=int(d), ell=int(ell), reps=len(errors),
                         rmse=float(np.sqrt(np.mean(errors**2))), rmse_low=low, rmse_high=high,
                         bias=float(errors.mean()), coverage_mean=float(group.coverage.mean()),
                         coverage_min=float(group.coverage.min()),
                         zero_valid_repeats=int((group.coverage == 0).sum())))
    per = pd.DataFrame(rows)
    choices = []
    for rho in RHOS:
        for column, (n, d) in enumerate(PAIRS):
            group = per[(per.rho == rho) & (per.n == n) & (per.d == d)].sort_values("ell")
            best = group.sort_values(["rmse", "ell"]).iloc[0]
            lam = 2*d*np.log(2*n+1)+np.log(96/.05)
            bound_ell = min(group.ell, key=lambda ell:(ell**-2+np.sqrt(lam)*(float(ell)**d/n)**.25, ell))
            rate_ell = max(1, int(np.floor((n/(d**6*np.log(n)**2))**(1/(d+8))+.5)))
            bound = group[group.ell == bound_ell].iloc[0]
            rate = group[group.ell == rate_ell].iloc[0]
            choices.append(dict(rho=rho, n=n, d=d, column=column, empirical_ell=int(best.ell),
                                empirical_rmse=float(best.rmse),
                                empirical_coverage_mean=float(best.coverage_mean),
                                empirical_coverage_min=float(best.coverage_min),
                                bound_ell=int(bound_ell), bound_rmse=float(bound.rmse),
                                bound_rmse_ratio=float(bound.rmse/best.rmse),
                                bound_match=int(best.ell)==bound_ell,
                                rate_C1_ell=rate_ell, rate_C1_rmse=float(rate.rmse),
                                rate_C1_rmse_ratio=float(rate.rmse/best.rmse),
                                rate_C1_match=int(best.ell)==rate_ell,
                                population_ell1_bias=float(-.5*((d-1)*np.log1p(-rho)+np.log1p((d-1)*rho)))))
    comparison = pd.DataFrame(choices)
    summary = comparison.groupby("rho", sort=True).agg(
        settings=("n", "size"), bound_matches=("bound_match", "sum"), rate_C1_matches=("rate_C1_match", "sum"),
        bound_rmse_ratio_median=("bound_rmse_ratio", "median"), bound_rmse_ratio_max=("bound_rmse_ratio", "max"),
        rate_C1_rmse_ratio_median=("rate_C1_rmse_ratio", "median"), rate_C1_rmse_ratio_max=("rate_C1_rmse_ratio", "max")).reset_index()
    return per, comparison, summary


def summary_figure(comparison, summary):
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.1), gridspec_kw={"height_ratios":[1.65, 1]})
    values = comparison.pivot(index="rho", columns="column", values="empirical_ell").loc[RHOS].to_numpy()
    matches = comparison.pivot(index="rho", columns="column", values="bound_match").loc[RHOS].to_numpy()
    ax = axes[0]
    ax.imshow(matches.astype(int), cmap=ListedColormap(["#F6DED9", "#D5EBDD"]), aspect="auto", vmin=0, vmax=1)
    for row in range(len(RHOS)):
        for column in range(len(PAIRS)):
            ax.text(column, row, str(int(values[row, column])), ha="center", va="center", fontsize=12)
    labels = [f"n=20k\nd={d}" for d in [2,5,10,20]] + [f"n={n//1000}k\nd=5" for n in [1000,3000,10000,30000]]
    ax.set_xticks(range(8), labels)
    ax.set_yticks(range(6), [f"{rho:.1f}"+ (" (control)" if rho == 0 else "") for rho in RHOS])
    ax.set_ylabel("Correlation $\\rho$")
    ax.set_title("Empirical RMSE-minimizing $\\ell$: green matches the fixed bound rule", pad=12)
    ax.set_xticks(np.arange(-.5,8), minor=True); ax.set_yticks(np.arange(-.5,6), minor=True)
    ax.grid(which="minor", color="white", lw=2); ax.tick_params(which="minor", bottom=False, left=False)
    ax.axvline(3.5, color="white", lw=4)
    ax.text(.5, -.3, "Bound-rule levels by column: 2, 1, 1, 1 | 1, 1, 1, 1.  Rate C=1: 1 in every column.",
            ha="center", transform=ax.transAxes, fontsize=9)
    ax = axes[1]
    x = np.arange(len(RHOS)); width=.33
    ax.bar(x-width/2, summary.bound_matches, width=width, color="#0072B2", label="Bound minimum ($\\delta=0.05$)")
    ax.bar(x+width/2, summary.rate_C1_matches, width=width, color="#9B8EBB", label="Dimension-aware rate ($C=1$)")
    for positions, counts in [(x-width/2, summary.bound_matches),(x+width/2, summary.rate_C1_matches)]:
        for pos, count in zip(positions, counts): ax.text(pos, count+.1, str(int(count)), ha="center", fontsize=9)
    ax.set(xticks=x, xticklabels=[f"{rho:.1f}" for rho in RHOS], xlabel="Correlation $\\rho$", ylabel="Matches / 8 settings", ylim=(0,11))
    ax.set_yticks(range(0,9,2)); ax.legend(frameon=False, ncol=2, loc="upper center", fontsize=9)
    ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    fig.suptitle("Exploratory Gaussian correlation sweep: all tested correlations", fontsize=13, y=.985)
    fig.text(.08, .067, "30 repetitions per condition; all saved ell candidates, no coverage filter. rho=0 is an independence control.", fontsize=8.5)
    fig.text(.08, .027, "Theoretical rules are unchanged. Greater agreement under weaker dependence is not a finite-sample optimality proof.", fontsize=8.5)
    fig.subplots_adjust(left=.13, right=.97, top=.89, bottom=.17, hspace=.6)
    return fig


def curves_figure(per, comparison):
    fig, axes = plt.subplots(2, 4, figsize=(12, 7.5))
    colors = ["#555555", "#0072B2", "#009E73", "#D69B00", "#D55E00", "#9568AC"]
    for column, (n,d) in enumerate(PAIRS):
        ax = axes.flat[column]
        for rho, color in zip(RHOS, colors):
            part = per[(per.rho == rho) & (per.n == n) & (per.d == d)].sort_values("ell")
            best = comparison[(comparison.rho == rho) & (comparison.n == n) & (comparison.d == d)].iloc[0]
            ax.plot(part.ell, part.rmse, color=color, lw=1.35, label=f"$\\rho={rho:.1f}$")
            ax.scatter([best.empirical_ell], [best.empirical_rmse], marker="*", s=44, color=color, zorder=4)
        bound_ell = int(comparison[(comparison.n == n) & (comparison.d == d)].bound_ell.iloc[0])
        ax.axvline(bound_ell, color="#333333", linestyle=":", lw=1.2)
        ax.set(xscale="log", yscale="log", title=f"$n={n:,},\\ d={d}$", xlabel="$\\ell$")
        cap={2:40,5:40,10:12,20:8}[d]
        ticks=[v for v in [1,2,4,8,20,40] if v<=cap]
        ax.set_xlim(.85,cap*1.08)
        ax.xaxis.set_major_locator(FixedLocator(ticks));ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:str(int(v))))
        ax.grid(alpha=.2); ax.set_axisbelow(True)
        if column%4==0: ax.set_ylabel("RMSE (nats)")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=6,loc="upper center",bbox_to_anchor=(.5,.945),frameon=False)
    fig.suptitle("Gaussian ell-RMSE curves across the full correlation sweep",fontsize=13,y=.985)
    fig.text(.065,.075,"Stars: empirical grid minima. Dotted vertical line: fixed bound minimum. Curves include all candidates and repetitions.",fontsize=8.5)
    fig.text(.065,.035,"Fine partitions may skip observations; the canonical estimator averages covered points. Gaussian data are outside theorem assumptions.",fontsize=8.5)
    fig.subplots_adjust(left=.07,right=.985,top=.85,bottom=.18,wspace=.3,hspace=.4)
    return fig


def main():
    raw,sources=load(); per,comparison,summary=summarize(raw)
    # Reproduce every archived rho=.5 result from the previous turn.
    previous=pd.read_csv(ROOT/"results/synthetic_gaussian_20261006/optimal_ell_comparison.csv")
    joined=comparison[comparison.rho==.5].merge(previous,on=["n","d"],suffixes=("","_previous"),validate="one_to_one")
    np.testing.assert_array_equal(joined.empirical_ell,joined.empirical_ell_previous)
    np.testing.assert_allclose(joined.empirical_rmse,joined.empirical_rmse_previous,rtol=0,atol=1e-12)
    per.to_csv(OUT/"per_ell_summary.csv",index=False)
    comparison.to_csv(OUT/"optimal_ell_comparison.csv",index=False)
    summary.to_csv(OUT/"agreement_summary.csv",index=False)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.titlesize":10,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"savefig.dpi":175})
    pdfdir=ROOT/"output/pdf/synthetic_gaussian_rho_20261006";pdfdir.mkdir(parents=True,exist_ok=True)
    with PdfPages(pdfdir/"gaussian_rho_comparison.pdf") as pdf:
        for stem,fig in [("agreement",summary_figure(comparison,summary)),("ell_rmse",curves_figure(per,comparison))]:
            pdf.savefig(fig);fig.savefig(OUT/(stem+".png"));plt.close(fig)
    notes=["Exploratory Gaussian lower-correlation sweep", "",
           "The full rho grid 0,.1,.2,.3,.4 was fixed before computing new outcomes; the existing .5 baseline is retained.",
           "The sweep was motivated by observed disagreement at rho=.5. It is exploratory, not independent confirmation.",
           "All checked correlations and n/d settings are reported. rho=0 is an independence control, not a dependent example.",
           "No theoretical coefficient, delta, or candidate grid is changed. Bound delta=.05; dimension-aware rate C=1.",
           "Empirical optimum means minimum aggregate RMSE over the saved grid, no coverage filter, same 30 repetitions.",
           "Exact match counts are descriptive over these eight settings, not estimates of population match probability.",
           "The RMSE ratio is RMSE at the fixed theoretical ell divided by the empirical grid minimum (optimistic denominator).",
           "No inferential claims are made from this postselection ratio. Pointwise bootstrap intervals are saved for fixed ell.",
           "At ell=1, the population Gaussian product-approximation bias is",
           "-.5*[(d-1)*log(1-rho)+log(1+(d-1)*rho)], which vanishes as rho tends to zero.",
           "For d=5 this bias is 0,.042485,.152394,.319121,.543896,.836988 nats for rho 0,.1,.2,.3,.4,.5.",
           "Thus more ell=1 matches under weak correlation do not establish theoretical finite-sample optimality.",
           "Gaussian densities are outside the stated compact-support/positive-lower-bound assumptions.","",
           summary.to_string(index=False),"","See protocol.json for generation, frozen sources, and seed reuse; no unfavorable draws were removed."]
    (OUT/"notes.txt").write_text("\n".join(notes)+"\n")
    audit=dict(status="PASS",settings=48,datasets=int(raw.groupby(["rho","n","d","replicate"]).ngroups),candidate_estimates=len(raw),
               protocol_sha256=sha(OUT/"protocol.json"),analysis_sha256=sha(__file__),source_files=sources,
               all_rhos_reported=RHOS,baseline_reproduced=True,bootstrap_seed=2026100603,bootstrap_replicates=2000,
               failures=int((raw.status != "ok").sum()),zero_valid_rows=int((raw.n_valid == 0).sum()),
               source_hashes_match=True,baseline_numerical_checks=8,
               selection="unrestricted minimum of same-repetition aggregate RMSE",interpretation="exploratory full grid; no correlation silently selected")
    (OUT/"analysis_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(summary.to_string(index=False))


if __name__=="__main__":main()
