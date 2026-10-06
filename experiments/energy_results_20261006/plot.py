"""Plot compact Energy comparisons from frozen exported summary tables.

This checks export integrity and reads saved means/SDs. It does not repeat the
original training, feature selection, prediction checks, or scientific audits.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
COUNTS = (5, 10, 20)
PANELS = (
    ("energy_component_guard_20261005", "(a) Coverage constraint", (
        ("PSS pooled SC-CV", "PSS pooled", "#777777", "s"),
        ("PSS class-aware SC-CV", "PSS class-aware", "#0072B2", "o"),
        ("KL tuned", "KL tuned", "#D55E00", "^"),
    )),
    ("energy_jmi_20261006", "(b) Pairwise JMI", (
        ("PSS class-aware SC-CV", "PSS class-aware", "#0072B2", "o"),
        ("PSS-JMI", "PSS-JMI", "#009E73", "D"),
        ("KL tuned", "KL tuned", "#D55E00", "^"),
        ("KL-JMI", "KL-JMI", "#CC79A7", "v"),
    )),
    ("energy_theory_c_20261006", "(c) Theory coefficient", (
        ("PSS class-aware SC-CV", "PSS class-aware", "#0072B2", "o"),
        ("PSS theory C tuned", "PSS theory, tuned C", "#009E73", "D"),
        ("PSS theory C=1", "PSS theory, C=1", "#777777", "s"),
        ("KL tuned", "KL tuned", "#D55E00", "^"),
    )),
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_export():
    manifest = json.loads((HERE / "manifest.json").read_text())
    counts = {}
    for group, root in (("source_files", REPO), ("saved_artifacts", HERE)):
        counts[group] = len(manifest[group])
        for relative, expected in manifest[group].items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError(f"Manifest path escapes its root: {relative}")
            actual = sha256(path)
            if actual != expected:
                raise ValueError(f"Export SHA-256 mismatch: {relative}")
    return counts


def load_summary(run):
    path = HERE / "results" / run / "summary.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = {}
    for row in rows:
        count = int(row["count"])
        if count not in COUNTS:
            continue
        mean, sd = float(row["accuracy"]), float(row["sd"])
        if not (math.isfinite(mean) and 0 <= mean <= 1 and math.isfinite(sd) and sd >= 0):
            raise ValueError(f"Invalid accuracy summary in {path}: {row}")
        if "repeats" in row and int(row["repeats"]) != 3:
            raise ValueError(f"Expected three splits in {path}")
        key = (row["method"], count)
        if key in selected:
            raise ValueError(f"Duplicate summary row in {path}: {key}")
        selected[key] = (mean, sd)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "figures",
                        help="Output directory for the new PDF, PNG and verification record.")
    parser.add_argument("--verify-only", action="store_true",
                        help="Verify copied source/artifact hashes without importing plotting packages.")
    args = parser.parse_args()
    checked = verify_export()
    if args.verify_only:
        print(json.dumps({"export_hashes_verified": checked}, indent=2))
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.7), sharey=True)
    row_count = 0
    inputs = {}
    for ax, (run, title, methods) in zip(axes, PANELS):
        summary = load_summary(run)
        inputs[run] = sha256(HERE / "results" / run / "summary.csv")
        for name, label, color, marker in methods:
            values = [summary[(name, count)] for count in COUNTS]
            row_count += len(values)
            ax.errorbar(COUNTS, [100 * x[0] for x in values],
                        yerr=[100 * x[1] for x in values], label=label,
                        color=color, marker=marker, markersize=5,
                        linewidth=1.4, capsize=3, elinewidth=0.8)
        ax.set_title(title, loc="left", fontsize=11, pad=11)
        ax.set_xticks(COUNTS)
        ax.set_xlabel("Selected features")
        ax.set_xlim(3.5, 21.5)
        ax.set_ylim(69, 86)
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower right", fontsize=8.3, frameon=False)
    axes[0].set_ylabel("Test accuracy (%)")
    fig.suptitle("Energy: exploratory comparisons on the same three splits", fontsize=13, y=0.98)
    fig.text(0.5, 0.018,
             "Seeds 42–44; means ± sample SD across overlapping random holdouts. SD is not a confidence interval.",
             ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94), w_pad=1.7)
    fig.savefig(output / "energy_accuracy_comparison.pdf")
    fig.savefig(output / "energy_accuracy_comparison.png", dpi=180)
    plt.close(fig)
    record = {
        "passed": True,
        "scope": "Export/source SHA-256 verification and plotting saved summary means/SDs only; no model reruns or independent replay of historical audits.",
        "export_hashes_verified": checked,
        "summary_rows_plotted": row_count,
        "summary_sha256": inputs,
        "seeds": [42, 43, 44],
        "features": list(COUNTS),
        "uncertainty": "Saved sample SD over overlapping splits; not confidence intervals.",
    }
    (output / "plot_checks.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
