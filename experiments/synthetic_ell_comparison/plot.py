"""Reproduce the four-panel comparison from frozen summary tables.

No simulation, tuning, or uncertainty estimation is performed here. The tables
retain the original numerical precision and pointwise bootstrap intervals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parent
COLORS = ("#0072B2", "#D55E00", "#009E73")
STEM = "bounded_gaussian_ell_rmse"


def load_study(name: str, pairs: list[tuple[int, int]], grid_max: int, reps: int):
    """Read and cross-check the frozen tables without changing their values."""
    summary = pd.read_csv(PACKAGE / "tables" / f"{name}_summary.csv")
    selection = pd.read_csv(PACKAGE / "tables" / f"{name}_selection.csv")
    if name == "gaussian":
        selection = selection.rename(columns={
            "bound_ell": "theory_ell", "bound_rmse": "theory_rmse", "bound_match": "match",
        })
    assert set(map(tuple, summary[["n", "d"]].drop_duplicates().to_numpy())) == set(pairs)
    assert not summary.duplicated(["n", "d", "ell"]).any()
    assert not selection.duplicated(["n", "d"]).any()
    assert summary.reps.eq(reps).all()
    assert np.isfinite(summary[["rmse", "rmse_low", "rmse_high"]]).all().all()
    assert summary.rmse_low.gt(0).all()
    assert (summary.rmse_low <= summary.rmse).all()
    assert (summary.rmse <= summary.rmse_high).all()
    for n, d in pairs:
        group = summary[(summary.n == n) & (summary.d == d)].sort_values("ell")
        chosen = selection[(selection.n == n) & (selection.d == d)].iloc[0]
        assert group.ell.tolist() == list(range(1, grid_max + 1))
        empirical = group.sort_values(["rmse", "ell"]).iloc[0]
        ell = group.ell.to_numpy(dtype=float)
        criterion = ell**-2 + np.sqrt(2*d*np.log(2*n+1) + np.log(96/.05)) * (ell**d/n)**.25
        theory = group.iloc[int(np.argmin(criterion))]
        assert int(chosen.empirical_ell) == int(empirical.ell)
        assert int(chosen.theory_ell) == int(theory.ell)
        np.testing.assert_allclose(chosen.empirical_rmse, empirical.rmse, rtol=0, atol=1e-15)
        np.testing.assert_allclose(chosen.theory_rmse, theory.rmse, rtol=0, atol=1e-15)
    return summary, selection


def draw_panel(ax, summary, selection, pairs, varying, title, grid_max):
    for (n, d), color in zip(pairs, COLORS):
        group = summary[(summary.n == n) & (summary.d == d)].sort_values("ell")
        chosen = selection[(selection.n == n) & (selection.d == d)].iloc[0]
        exponent = int(np.floor(np.log10(n)))
        coefficient = n // 10**exponent
        sample_label = rf"$n=10^{{{exponent}}}$" if coefficient == 1 else rf"$n={coefficient}\times10^{{{exponent}}}$"
        label = rf"$d={d}$" if varying == "d" else sample_label
        x = group.ell.to_numpy()
        y = group.rmse.to_numpy()
        ax.plot(x, y, color=color, lw=1.05, label=label, zorder=3)
        # The original coverage diagnostic remains visible in all panels.
        # Empty circles denote any replicate with less than full coverage.
        complete = group.coverage_min.to_numpy() >= 1.0
        for mask, fill in ((complete, color), (~complete, "white")):
            ax.scatter(x[mask], y[mask], s=7, marker="o", facecolors=fill,
                       edgecolors=color, linewidths=.6, zorder=4)
        ax.fill_between(x, group.rmse_low.to_numpy(), group.rmse_high.to_numpy(),
                        color=color, alpha=.14, lw=0, zorder=2)
        ax.scatter([chosen.theory_ell], [chosen.theory_rmse], facecolors="none",
                   edgecolors=color, marker="s", s=45, linewidths=1, zorder=5)
        ax.scatter([chosen.empirical_ell], [chosen.empirical_rmse], color=color,
                   marker="*", s=60, edgecolors="black", linewidths=.45, zorder=6)
    ax.set(yscale="log", xticks=range(1, grid_max+1))
    ax.set_title(title, pad=6)
    ax.grid(alpha=.22, linewidth=.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", fontsize=7.5, handlelength=1.15,
              handletextpad=.35, labelspacing=.18, borderaxespad=.25)
    ax.tick_params(axis="both", which="major", pad=2, length=3)
    ax.margins(x=.065, y=.20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PACKAGE / "figures")
    args = parser.parse_args()
    provenance = json.loads((PACKAGE / "provenance.json").read_text())
    for source in provenance["inputs"]:
        path = PACKAGE / source["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError(f"Frozen input changed: {source['file']}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bounded_pairs = [(100000, 2), (100000, 3), (100000, 4), (1000, 2), (10000, 2)]
    gaussian_pairs = [(100000, 2), (100000, 3), (100000, 4), (30000, 3), (300000, 3)]
    bounded = load_study("bounded", bounded_pairs, 7, 100)
    gaussian = load_study("gaussian", gaussian_pairs, 8, 30)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 8, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "savefig.dpi": 300,
    })
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.8))
    panels = [
        (axes[0], bounded, [(100000, d) for d in [2, 3, 4]], "d",
         r"(a) Fixed $n=10^5$", 7),
        (axes[1], bounded, [(n, 2) for n in [1000, 10000, 100000]], "n",
         r"(b) Fixed $d=2$", 7),
        (axes[2], gaussian, [(100000, d) for d in [2, 3, 4]], "d",
         r"(c) Fixed $n=10^5$", 8),
        (axes[3], gaussian, [(n, 3) for n in [30000, 100000, 300000]], "n",
         r"(d) Fixed $d=3$", 8),
    ]
    for ax, (summary, selection), pairs, varying, title, grid_max in panels:
        draw_panel(ax, summary, selection, pairs, varying, title, grid_max)
    # Reserve vertical space for the longest legend without hiding the curves.
    axes[3].set_ylim(top=3.0)
    handles = [
        Line2D([], [], color="black", marker="*", linestyle="none", markersize=8,
               label=r"RMSE-minimizing $\ell$"),
        Line2D([], [], color="black", marker="s", markerfacecolor="none",
               linestyle="none", markersize=6,
               label=r"Theory-selected $\ell$ ($\delta=0.05$)"),
    ]
    fig.legend(handles=handles, ncol=2, frameon=False, loc="upper center",
               bbox_to_anchor=(.5, 1.0), fontsize=8.5, columnspacing=2)
    fig.subplots_adjust(left=.075, right=.99, top=.70, bottom=.20, wspace=.40)
    for start, title in [(0, "Bounded dependent density"), (2, r"Gaussian ($\rho=0.2$)")]:
        center = (axes[start].get_position().x0 + axes[start+1].get_position().x1)/2
        fig.text(center, .835, title, ha="center", fontsize=9, weight="bold")
    fig.text(.012, .45, "Entropy RMSE (nats)", va="center", rotation="vertical", fontsize=8.5)
    fig.text(.53, .035, r"Partitions per coordinate $\ell$", ha="center", fontsize=8.5)
    pdf = args.output_dir / f"{STEM}.pdf"
    png = args.output_dir / f"{STEM}.png"
    fig.savefig(pdf, metadata={"Title": "Partition selection for bounded and Gaussian densities",
                              "Creator": "Matplotlib; frozen summary tables"})
    fig.savefig(png)
    plt.close(fig)
    audit = {
        "status": "PASS", "new_simulations": 0, "recomputed_intervals": False,
        "bounded_unique_settings": 5, "bounded_matches": int(bounded[1]["match"].sum()),
        "gaussian_unique_settings": 5, "gaussian_matches": int(gaussian[1]["match"].sum()),
        "bounded_summary_rows": len(bounded[0]), "gaussian_summary_rows": len(gaussian[0]),
        "layout": "one row, four panels",
        "page_size_inches": [7.2, 2.8], "minimum_fontsize_points": 7.5,
        "source_sha256": {str(p.relative_to(PACKAGE)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((PACKAGE / "tables").glob("*.csv"))},
    }
    (args.output_dir / "figure_checks.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"pdf": str(pdf), "png": str(png), "validation": audit}, indent=2))


if __name__ == "__main__":
    main()
