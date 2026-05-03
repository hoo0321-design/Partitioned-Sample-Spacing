"""Create paper-style scaling plots for anchor-grid experiments."""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ["PSS", "CADEE", "KL", "KSG", "UM-tKL", "UM-tKSG"]
DISTRIBUTIONS = ["Normal", "Gamma", "Beta", "Lognormal", "Laplace"]
COLORS = {
    "PSS": "#C23B22",
    "CADEE": "#6B7280",
    "KL": "#2563EB",
    "KSG": "#059669",
    "UM-tKL": "#7C3AED",
    "UM-tKSG": "#D97706",
}
MARKERS = {
    "PSS": "o",
    "CADEE": "s",
    "KL": "^",
    "KSG": "D",
    "UM-tKL": "v",
    "UM-tKSG": "P",
}


def configure_matplotlib():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def combine_if_needed(base_dir: str):
    combined_path = os.path.join(base_dir, "combined_summary.csv")
    if os.path.exists(combined_path):
        return combined_path

    r_path = os.path.join(base_dir, "r_pss_cadee_summary.csv")
    py_path = os.path.join(base_dir, "knn_um_summary.csv")
    if not os.path.exists(py_path):
        py_path = os.path.join(base_dir, "knn_nf_summary.csv")

    if not os.path.exists(r_path) or not os.path.exists(py_path):
        raise FileNotFoundError(
            "Expected r_pss_cadee_summary.csv and knn_um_summary.csv "
            f"under {base_dir}"
        )

    r = pd.read_csv(r_path)
    py = pd.read_csv(py_path)
    combined = pd.concat([r, py], ignore_index=True, sort=False)
    combined.to_csv(combined_path, index=False)
    return combined_path


def load_summary(base_dir: str):
    combined_path = combine_if_needed(base_dir)
    df = pd.read_csv(combined_path)
    df = df[df["Method"].isin(METHODS)].copy()
    df["Distribution"] = pd.Categorical(df["Distribution"], DISTRIBUTIONS, ordered=True)
    df["Method"] = pd.Categorical(df["Method"], METHODS, ordered=True)
    df["RMSE_plot"] = df["RMSE"].clip(lower=1.0e-4)
    df["RMSE_SE"] = df["RMSE_SE"].fillna(0.0).clip(lower=0.0)
    return df.sort_values(["Distribution", "Experiment", "Method"])


def savefig(fig, out_dir: str, stem: str):
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, f"{stem}.png")
    pdf = os.path.join(out_dir, f"{stem}.pdf")
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_scaling(df, experiment: str, x_col: str, x_label: str, out_dir: str, stem: str):
    part_all = df[df["Experiment"] == experiment].copy()
    fig, axes = plt.subplots(1, len(DISTRIBUTIONS), figsize=(16, 3.7), sharey=False)

    for ax, distribution in zip(axes, DISTRIBUTIONS):
        part_dist = part_all[part_all["Distribution"] == distribution]
        ax.set_title(distribution)
        if part_dist.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        for method in METHODS:
            part = part_dist[part_dist["Method"] == method].sort_values(x_col)
            if part.empty:
                continue
            ax.errorbar(
                part[x_col],
                part["RMSE_plot"],
                yerr=part["RMSE_SE"],
                label=method,
                color=COLORS[method],
                marker=MARKERS[method],
                markersize=4.2,
                linewidth=2.3 if method == "PSS" else 1.4,
                alpha=1.0 if method == "PSS" else 0.82,
                capsize=2,
                elinewidth=0.65,
            )

        ax.set_yscale("log")
        ax.set_xlabel(x_label)
        if x_col in ["N_Samples", "Dimensions"]:
            ax.set_xscale("log", base=10 if x_col == "N_Samples" else 2)
            ax.set_xticks(sorted(part_dist[x_col].unique()))
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        else:
            ax.set_xticks(sorted(part_dist[x_col].unique()))

        ymin = max(0.004, part_dist["RMSE_plot"].min() / 1.7)
        ymax = part_dist["RMSE_plot"].max() * 1.9
        ax.set_ylim(ymin, ymax)
        ax.grid(True, which="both", alpha=0.25)

    axes[0].set_ylabel("RMSE (log scale)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(experiment, y=1.18, fontsize=13, fontweight="bold")
    return savefig(fig, out_dir, stem)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    configure_matplotlib()
    out_dir = args.out_dir or os.path.join(args.base_dir, "plots")
    df = load_summary(args.base_dir)

    outputs = []
    outputs.extend(plot_scaling(df, "N scaling", "N_Samples", "n", out_dir, "fig_n_scaling"))
    outputs.extend(plot_scaling(df, "d scaling", "Dimensions", "d", out_dir, "fig_d_scaling"))
    outputs.extend(plot_scaling(df, "rho scaling", "Correlation", "rho", out_dir, "fig_rho_scaling"))

    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
