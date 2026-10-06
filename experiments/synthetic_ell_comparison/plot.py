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
        label = rf"$d={d}$" if varying == "d" else rf"$n={n:,}$"
        x = group.ell.to_numpy()
        y = group.rmse.to_numpy()
        ax.plot(x, y, color=color, lw=1.3, label=label, zorder=3)
        # The original coverage diagnostic remains visible on both rows.
        # Empty circles denote any replicate with less than full coverage.
        complete = group.coverage_min.to_numpy() >= 1.0
        for mask, fill in ((complete, color), (~complete, "white")):
            ax.scatter(x[mask], y[mask], s=10, marker="o", facecolors=fill,
                       edgecolors=color, linewidths=.7, zorder=4)
        ax.fill_between(x, group.rmse_low.to_numpy(), group.rmse_high.to_numpy(),
                        color=color, alpha=.14, lw=0, zorder=2)
        ax.scatter([chosen.theory_ell], [chosen.theory_rmse], facecolors="none",
                   edgecolors=color, marker="s", s=72, linewidths=1.2, zorder=5)
        ax.scatter([chosen.empirical_ell], [chosen.empirical_rmse], color=color,
                   marker="*", s=90, edgecolors="black", linewidths=.55, zorder=6)
    ax.set(yscale="log", xticks=range(1, grid_max+1),
           xlabel=r"Partitions per coordinate $\ell$", ylabel="Entropy RMSE (nats)")
    ax.set_title(title, pad=7)
    ax.grid(alpha=.22, linewidth=.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5, handlelength=1.6,
              handletextpad=.5, labelspacing=.3, borderaxespad=.45)
    ax.margins(x=.045, y=.13)


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
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "savefig.dpi": 300,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.3))
    panels = [
        (axes[0, 0], bounded, [(100000, d) for d in [2, 3, 4]], "d",
         r"(a) Fixed $n=100,000$; varying dimension", 7),
        (axes[0, 1], bounded, [(n, 2) for n in [1000, 10000, 100000]], "n",
         r"(b) Fixed $d=2$; varying sample size", 7),
        (axes[1, 0], gaussian, [(100000, d) for d in [2, 3, 4]], "d",
         r"(c) Fixed $n=100,000$; varying dimension", 8),
        (axes[1, 1], gaussian, [(n, 3) for n in [30000, 100000, 300000]], "n",
         r"(d) Fixed $d=3$; varying sample size", 8),
    ]
    for ax, (summary, selection), pairs, varying, title, grid_max in panels:
        draw_panel(ax, summary, selection, pairs, varying, title, grid_max)
    handles = [
        Line2D([], [], color="black", marker="*", linestyle="none", markersize=9,
               label=r"RMSE-minimizing $\ell$"),
        Line2D([], [], color="black", marker="s", markerfacecolor="none",
               linestyle="none", markersize=7,
               label=r"Theory-selected $\ell$ ($\delta=0.05$)"),
    ]
    fig.legend(handles=handles, ncol=2, frameon=False, loc="upper center",
               bbox_to_anchor=(.5, .998), fontsize=9, columnspacing=2)
    fig.text(.535, .92, "Bounded dependent density", ha="center", fontsize=10, weight="bold")
    fig.text(.535, .449, r"Gaussian ($\rho=0.2$)", ha="center", fontsize=10, weight="bold")
    fig.subplots_adjust(left=.087, right=.985, top=.86, bottom=.095,
                        wspace=.32, hspace=.59)
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
        "page_size_inches": [7.2, 6.3], "minimum_fontsize_points": 8.5,
        "source_sha256": {str(p.relative_to(PACKAGE)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((PACKAGE / "tables").glob("*.csv"))},
    }
    (args.output_dir / "figure_checks.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"pdf": str(pdf), "png": str(png), "validation": audit}, indent=2))


if __name__ == "__main__":
    main()
