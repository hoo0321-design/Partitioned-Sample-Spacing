"""Verify saved Energy predictions and reproduce the four-method Figure 5."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
STEM = "energy_four_methods"
SEEDS = [42, 43, 44]
METHODS = [
    ("PSS class-aware SC-CV", "PSS (tuned SC-CV)", "#0072B2", "-", "o"),
    ("KL tuned", r"KL-kNN (tuned $k$)", "#D55E00", "-", "^"),
    ("PSS theory C=1", r"PSS (theory-guided $\ell$, $C=1$)", "#0072B2", "--", "o"),
    ("KL k=1", r"KL-kNN ($k=1$)", "#D55E00", "--", "^"),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(np.allclose(actual, expected, atol=1e-14, rtol=0), label)


def load_verified(data_dir=HERE / "data"):
    provenance = json.loads((data_dir / "provenance.json").read_text())
    method_names = [method[0] for method in METHODS]
    counts = list(range(1, 21))
    require(provenance["schema_version"] == 1, "Unsupported provenance schema")
    require(provenance["methods"] == method_names and provenance["seeds"] == SEEDS
            and provenance["counts"] == counts, "Unexpected experiment coordinates")
    expected_files = {"predictions.npz", "metrics.csv", "summary.csv"}
    require(set(provenance["artifact_sha256"]) == expected_files, "Unexpected data files")
    for name, expected in provenance["artifact_sha256"].items():
        require(sha(data_dir / name) == expected, f"SHA-256 mismatch: {name}")
    with np.load(data_dir / "predictions.npz", allow_pickle=False) as archive:
        require(set(archive.files) == {"predictions", "outer_train", "outer_test", "targets",
                                      "methods", "seeds", "counts"}, "Unexpected NPZ arrays")
        predictions = archive["predictions"]
        train, test, target = archive["outer_train"], archive["outer_test"], archive["targets"]
        require(archive["methods"].tolist() == method_names, "NPZ method order mismatch")
        require(archive["seeds"].tolist() == SEEDS, "NPZ seed order mismatch")
        require(archive["counts"].tolist() == counts, "NPZ feature-count order mismatch")
    require(predictions.shape == (4, 3, 20, 5921), "Prediction array shape mismatch")
    require(np.isin(predictions, [0, 1]).all(), "Predictions must be binary")
    require(train.shape == (3, 13814) and test.shape == (3, 5921), "Split shape mismatch")
    require(np.issubdtype(train.dtype, np.integer) and np.issubdtype(test.dtype, np.integer),
            "Split indices must be integers")
    require(target.shape == (19735,) and np.isfinite(target).all(), "Invalid Appliances targets")
    labels, thresholds = [], []
    for i, seed in enumerate(SEEDS):
        require(np.array_equal(np.sort(np.concatenate([train[i], test[i]])), np.arange(19735)),
                f"Seed {seed}: train/test must partition every row exactly once")
        threshold = float(np.median(target[train[i]]))
        thresholds.append(threshold)
        labels.append(target[test[i]] > threshold)
    # The complete repository also contains the original attributed Energy CSV.
    # A standalone copy of this directory uses the hash-checked target vector.
    dataset = HERE.parent.parent.parent / "data/energydata_complete.csv"
    dataset_checked = dataset.is_file()
    if dataset_checked:
        require(sha(dataset) == provenance["source_file_sha256"]["data/energydata_complete.csv"],
                "Full Energy CSV hash mismatch")
        require(np.array_equal(pd.read_csv(dataset).Appliances.to_numpy(), target),
                "Exported targets differ from full Energy CSV")
    measured = (predictions == np.asarray(labels)[None, :, None, :]).mean(axis=-1)
    metrics = pd.read_csv(data_dir / "metrics.csv")
    summary = pd.read_csv(data_dir / "summary.csv")
    require(len(metrics) == 240 and not metrics.duplicated(["method", "seed", "count"]).any(),
            "Expected 240 unique per-split metrics")
    require(len(summary) == 80 and not summary.duplicated(["method", "count"]).any(),
            "Expected 80 unique summary points")
    expected_metric_keys = {(method, seed, count) for method in method_names for seed in SEEDS for count in counts}
    require(set(metrics[["method", "seed", "count"]].itertuples(index=False, name=None))
            == expected_metric_keys, "Missing or unexpected metric coordinates")
    expected_summary_keys = {(method, count) for method in method_names for count in counts}
    require(set(summary[["method", "count"]].itertuples(index=False, name=None))
            == expected_summary_keys, "Missing or unexpected summary coordinates")
    stats = []
    for i, method in enumerate(method_names):
        for j, seed in enumerate(SEEDS):
            rows = metrics[(metrics.method == method) & (metrics.seed == seed)].sort_values("count")
            close(measured[i, j], rows.accuracy.to_numpy(),
                  f"Saved accuracy differs from predictions: {method}, seed {seed}")
        means, sds = measured[i].mean(axis=0), measured[i].std(axis=0, ddof=1)
        reference = summary[summary.method == method].sort_values("count")
        require(reference.repeats.eq(3).all(), "Every summary must contain three splits")
        close(means, reference.accuracy.to_numpy(), f"Saved mean differs: {method}")
        close(sds, reference.sd.to_numpy(), f"Saved sample SD differs: {method}")
        stats.extend(dict(method=method, count=count, mean=means[count-1], sd=sds[count-1],
                          repeats=3, accuracy_percent=100*means[count-1], sd_percent=100*sds[count-1])
                     for count in counts)
    checks = dict(status="PASS", methods=method_names, seeds=SEEDS, feature_counts=counts,
                  prediction_metrics_rechecked=240, summarized_points=80, sd_ddof=1,
                  training_rows=13814, test_rows=5921, training_medians=thresholds,
                  full_energy_csv_checked=dataset_checked, new_model_fits=0,
                  scope="Recompute labels, accuracy and mean/sample SD from saved predictions; no retraining or replay of feature selection.",
                  artifact_sha256=provenance["artifact_sha256"],
                  provenance_sha256=sha(data_dir / "provenance.json"))
    return pd.DataFrame(stats), checks


def draw(stats, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#404650", "axes.linewidth": .8,
        "xtick.color": "#30343B", "ytick.color": "#30343B",
        "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 300,
    })
    fig, ax = plt.subplots(figsize=(8.4, 5.7))
    fig.subplots_adjust(left=.105, right=.975, bottom=.18, top=.865)
    for method, label, color, linestyle, marker in METHODS:
        group = stats[stats.method == method].sort_values("count")
        x = group["count"].to_numpy()
        y, sd = group.accuracy_percent.to_numpy(), group.sd_percent.to_numpy()
        ax.fill_between(x, y-sd, y+sd, color=color, alpha=.085, linewidth=0, zorder=1)
        ax.plot(x, y, label=label, color=color, ls=linestyle, lw=1.9,
                marker=marker, markevery=[0, 4, 9, 14, 19], markersize=4.6,
                markerfacecolor=color if linestyle == "-" else "white",
                markeredgewidth=1.0, zorder=3)
    ax.set(xlim=(.7, 20.3), ylim=(54, 86.2), xticks=[1, 5, 10, 15, 20],
           yticks=np.arange(55, 86, 5), xlabel="Number of selected features",
           ylabel="Test accuracy (%)")
    ax.set_xticks(range(1, 21), minor=True)
    ax.tick_params(which="major", length=4, width=.7, pad=5)
    ax.tick_params(which="minor", length=2, width=.5)
    ax.grid(axis="y", alpha=.22, linewidth=.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", bbox_to_anchor=(.99, .035), frameon=False,
              fontsize=10, handlelength=3.1, labelspacing=.85, borderaxespad=.3)
    fig.text(.105, .942, "Appliances Energy", fontsize=15, weight="semibold", ha="left")
    fig.text(.105, .903, "Feature selection with a shared RBF SVM classifier", fontsize=10.5,
             color="#545B66", ha="left")
    fig.text(.105, .075, "Lines: mean test accuracy. Shading: ±1 sample SD across 3 random splits.",
             fontsize=9, color="#545B66", ha="left")
    fig.text(.105, .043, "Exploratory results on previously inspected splits; bands are not confidence intervals.",
             fontsize=8.6, color="#545B66", ha="left")
    for suffix in ["png", "pdf", "svg"]:
        path = destination / f"{STEM}.{suffix}"
        fig.savefig(path, facecolor="white",
                    metadata={"Title": "Energy feature selection: four-method comparison"})
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Verify without generating files")
    parser.add_argument("--output-dir", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    stats, checks = load_verified()
    if not args.verify_only:
        destination = args.output_dir.resolve()
        destination.mkdir(parents=True, exist_ok=True)
        stats.to_csv(destination / f"{STEM}_data.csv", index=False)
        draw(stats, destination)
        checks["plotting_source_sha256"] = sha(__file__)
        names = [f"{STEM}.{suffix}" for suffix in ["png", "pdf", "svg"]]
        names.append(f"{STEM}_data.csv")
        checks["output_sha256"] = {name: sha(destination / name) for name in names}
        (destination / "figure_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
