"""Paired audit and plots for the Energy class-aware coverage experiment."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from core import ROOT, POOLED, GUARDED, MATCHED, COUNTS, FOLDS, TAUS, NOISE, original
from run import (REFERENCE, SEEDS, CONTROLS, configurations, key, load_data,
                 reference_record, save, sha, slug)
from PSS.pss_v2 import mixed_mi, select_sc_cv

COLORS = {POOLED: "#2155cd", GUARDED: "#d2641c", MATCHED: "#9863a4", "KL tuned": "#c52f32",
          "PSS ell=1": "#6e747c", "Univariate tuned": "#258366"}
LABELS = {POOLED: "PSS: pooled coverage", GUARDED: "PSS: class-aware coverage",
          MATCHED: "Class-aware: pooled settings", "KL tuned": "KL: tuned k",
          "PSS ell=1": r"PSS: fixed $\ell=1$", "Univariate tuned": "Univariate ranking"}


def prediction_metrics(path, labels):
    with np.load(path) as p:
        if "y" in p:
            np.testing.assert_array_equal(p["y"], labels)
        return dict(accuracy=accuracy_score(labels, p["prediction"]),
                    balanced_accuracy=balanced_accuracy_score(labels, p["prediction"]),
                    auc=roc_auc_score(labels, p["decision"]))


def check_metric(value, labels, cache):
    if value.get("source"):
        assert sha(value["source"]) == value["source_sha256"]
        assert json.loads(Path(value["source"]).read_text()) == value["metrics"]
    if value.get("prediction_file"):
        path = value["prediction_file"]
        assert sha(path) == value["prediction_sha256"]
        if path not in cache:
            cache[path] = prediction_metrics(path, labels)
        for field, measured in cache[path].items():
            assert abs(measured-value["metrics"][field]) < 1e-12, (path, field)


def verify_components(z, y, ids, folds, features, ell, cache):
    key_ = (tuple(sorted(features)), folds, ell)
    if key_ not in cache:
        x = z[:, key_[0]]
        stats = []
        for mask in [np.ones(len(y), bool), y == 0, y == 1]:
            stats.append(select_sc_cv(x[mask], [ell], n_folds=folds, n_min=5, fold_id=ids[mask]))
        score = mixed_mi(x, y, ell)
        cache[key_] = dict(cv_score=stats[0]["cv_score"],
            pooled_cv_coverage=stats[0]["cv_coverage"],
            pooled_stable_5=stats[0]["stable_validation_coverage"],
            conditional_cv_coverage=min(s["cv_coverage"] for s in stats[1:]),
            conditional_stable_5=min(s["stable_validation_coverage"] for s in stats[1:]),
            minimum_stable_5=min(s["stable_validation_coverage"] for s in stats),
            class0_stable=stats[1]["stable_validation_coverage"],
            class1_stable=stats[2]["stable_validation_coverage"], score=score["mi"],
            training_coverage=score["coverage_all"], conditional_training_coverage=score["min_conditional_coverage"])
    return cache[key_]


def audit(out):
    protocol = json.loads((out/"protocol.json").read_text())
    for source, checksum in protocol["source_sha256"].items():
        assert sha(ROOT/source) == checksum, source
    for source, checksum in protocol["reference_sha256"].items():
        assert sha(REFERENCE/source) == checksum, source
    assert sha(ROOT/"data/energydata_complete.csv") == protocol["data_sha256"]
    x, target, names = load_data()
    data, diagnostics, choices, developments = [], [], [], []
    prediction_cache = {}
    checked_components = predictions_checked = 0
    methods = [POOLED, GUARDED, MATCHED, "KL tuned", "KL k=1", "Ross tuned",
               "Univariate tuned", "PSS ell=1", "All features"]
    for seed in SEEDS:
        folder = out/f"seed_{seed}"
        split = json.loads((folder/"split.json").read_text())
        tr, te = original.split_indices(np.arange(len(x)), seed)
        it, iv = original.split_indices(tr, seed+1000)
        for label, values in zip(["outer_train", "outer_test", "inner_train", "inner_validation"], [tr, te, it, iv]):
            np.testing.assert_array_equal(split[label], values)
        assert not set(tr) & set(te) and not set(it) & set(iv)
        y, ty, threshold = original.labels(target[tr], target[te])
        _, vy, inner_threshold = original.labels(target[it], target[iv])
        z, active = original.selection_data(x[tr], NOISE, seed+5000)
        # These fold IDs are independently reconstructed from the frozen recipe.
        ids = {}
        for folds in FOLDS:
            ids[folds] = np.empty(len(y), dtype=int)
            rng = np.random.default_rng(seed+6000+91)
            for label in [0, 1]:
                rows = np.flatnonzero(y == label)
                ids[folds][rows] = rng.permutation(np.arange(len(rows)) % folds)
        component_cache = {}
        lock = json.loads((folder/"lock.json").read_text())
        assert lock["locked_before_test"] and lock["protocol_sha256"] == sha(out/"protocol.json")
        records = []
        for filename, checksum in lock["development_sha256"].items():
            p = folder/"development"/filename
            assert sha(p) == checksum
            record = json.loads(p.read_text())
            assert record["path"]["n_selection"] == len(it) == 9669
            assert record["threshold"] == inner_threshold
            assert [v["count"] for v in record["metrics"]] == COUNTS
            for value in record["metrics"]:
                check_metric(value, vy, prediction_cache)
                assert value["features"] == record["path"]["history"][value["count"]-1]["features"]
            records.append(record)
            developments.append(dict(seed=seed, **record["config"],
                inner_accuracy=np.mean([v["metrics"]["accuracy"] for v in record["metrics"]]),
                ell1_fraction=np.mean([r["ell"] == 1 for r in record["path"]["history"]])))
        assert {key(r["config"]) for r in records} == {key(c) for c in configurations("PSS expanded SC-CV")}
        winner = min(records, key=lambda r: (-np.mean([v["metrics"]["accuracy"] for v in r["metrics"]]),
                     -r["config"]["tau"], r["config"]["folds"], key(r["config"])))
        assert winner["config"] == lock["choices"][GUARDED]
        old_lock = json.loads((REFERENCE/f"seed_{seed}"/"lock.json").read_text())
        assert lock["choices"][MATCHED] == old_lock["choices"][CONTROLS[POOLED]]["config"]
        for method in methods:
            record = json.loads((folder/"outer"/(slug(method)+".json")).read_text())
            path, config = record["path"], record["config"]
            assert record["threshold"] == threshold
            if method in [GUARDED, MATCHED]:
                before = json.loads((folder/"outer"/(slug(method)+"_pretest.json")).read_text())
                assert all(record[k] == v for k, v in before.items())
                assert config == lock["choices"][method]
            else:
                source = reference_record(seed, method)
                assert record["source_sha256"] == sha(source)
                old_record = json.loads(source.read_text())
                assert config == old_record["config"]
                for a, b in zip(path["history"], old_record["path"]["history"]):
                    assert a["features"] == b["features"]
            assert path["n_selection"] == (0 if method == "All features" else 13814)
            assert len(path["history"]) == len(record["metrics"]) == (1 if method == "All features" else 20)
            choices.append(dict(seed=seed, method=method, **config))
            previous = []
            for row, value in zip(path["history"], record["metrics"]):
                assert len(row["features"]) == len(set(row["features"])) == value["count"]
                assert row["features"] == value["features"]
                if method != "All features":
                    assert row["features"][:-1] == previous
                previous = row["features"]
                assert value["selected_features"] == [names[j] for j in row["features"]]
                check_metric(value, ty, prediction_cache)
                predictions_checked += len(te)
                data.append(dict(seed=seed, method=method, count=value["count"],
                                 features=",".join(value["selected_features"]), **value["metrics"]))
                if method in [POOLED, GUARDED, MATCHED, "PSS ell=1"]:
                    folds = config.get("folds", 3)
                    exact = verify_components(z, y, ids[folds], folds, row["features"], row["ell"], component_cache)
                    for field, measured in exact.items():
                        assert abs(row[field]-measured) < 1e-9, (seed, method, row["step"], field)
                    if method in [GUARDED, MATCHED] and not row["fallback"]:
                        assert exact["minimum_stable_5"] >= config["tau"]
                    checked_components += 1
                    diagnostics.append(dict(seed=seed, method=method, **row, tau=config.get("tau"), folds=folds))
        print(json.dumps(dict(stage="audit", seed=seed, verified=True)), flush=True)
    frame = pd.DataFrame(data)
    assert len(frame) == len(SEEDS)*(8*20+1)
    assert not frame.duplicated(["seed", "method", "count"]).any()
    save(dict(passed=True, curve_points=len(frame), predictions_verified=predictions_checked,
              components_recomputed=checked_components, source_hashes_verified=True,
              reference_unchanged=True, all_guard_configs_present=True,
              training_only_locks_recomputed=True, full_training_samples_verified=True,
              new_inner_predictions_checked_when_available=True,
              reused_inner_classifier_metrics_verified_against_hashed_source=True), out/"audit.json")
    return frame, pd.DataFrame(diagnostics), pd.DataFrame(choices), pd.DataFrame(developments)


def figures(out, frame, diagnostics):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    destination = out/"plots"
    destination.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5.7))
    for method in COLORS:
        stats = frame[frame.method == method].groupby("count").accuracy.agg(["mean", "std"])
        xx, mean, sd = stats.index.to_numpy(), 100*stats["mean"].to_numpy(), 100*stats["std"].to_numpy()
        ax.plot(xx, mean, label=LABELS[method], color=COLORS[method],
                ls="--" if method in [MATCHED, "PSS ell=1", "Univariate tuned"] else "-",
                marker="o" if method in [POOLED, GUARDED] else None, ms=3,
                lw=2.4 if method in [POOLED, GUARDED] else 1.5)
        ax.fill_between(xx, mean-sd, mean+sd, color=COLORS[method], alpha=.065)
    ax.set(xlabel="Selected features", ylabel="Test accuracy (%)", xticks=[1, 5, 10, 15, 20],
           title="Energy: pooled versus class-aware SC-CV")
    ax.grid(alpha=.18)
    ax.legend(loc="lower right", ncol=2, frameon=False, fontsize=9)
    fig.text(.5, .015, "Same 3 random splits; mean +/- sample SD. All 13,814 training rows used for selection. Classifier and density unchanged.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(destination/"accuracy.png", dpi=210)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    for method in [POOLED, GUARDED, MATCHED]:
        sub = diagnostics[diagnostics.method == method]
        for ax, field, title in zip(axes, ["ell", "minimum_stable_5", "conditional_training_coverage"],
            [r"Selected $\ell$", "Minimum component stable coverage", "Worst-class training skipped fraction"]):
            stats = sub.groupby("step")[field].agg(["mean", "min", "max"])
            if field == "conditional_training_coverage":
                mean, lower, upper = 1-stats["mean"], 1-stats["max"], 1-stats["min"]
            else:
                mean, lower, upper = stats["mean"], stats["min"], stats["max"]
            xx = stats.index.to_numpy()
            ax.plot(xx, mean, color=COLORS[method], label=LABELS[method],
                    ls="--" if method == MATCHED else "-", lw=2)
            ax.fill_between(xx, lower, upper, color=COLORS[method], alpha=.10)
            ax.set(title=title, xlabel="Selected features", xticks=[1, 5, 10, 15, 20])
            ax.grid(alpha=.18)
    axes[0].set_ylim(.8, 5.2)
    axes[0].set_yticks([1, 2, 3, 4, 5])
    axes[1].set_ylim(0, 1.03)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.text(.5, .01, "Coverage = min(pooled, class 0, class 1). Lines are split means; bands show the observed range, not confidence intervals.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .88))
    fig.savefig(destination/"coverage_diagnostics.png", dpi=210)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    out = parser.parse_args().output.resolve()
    frame, diagnostics, choices, development = audit(out)
    for name, value in [("accuracy_curves", frame), ("diagnostics", diagnostics),
                        ("locked_settings", choices), ("development_summary", development)]:
        value.to_csv(out/(name+".csv"), index=False)
    summary = frame.groupby(["method", "count"]).agg(accuracy=("accuracy", "mean"),
        sd=("accuracy", "std"), balanced_accuracy=("balanced_accuracy", "mean"), auc=("auc", "mean")).reset_index()
    summary.to_csv(out/"summary.csv", index=False)
    diagnostic_summary = diagnostics.groupby("method").agg(subsets=("ell", "size"),
        ell1_fraction=("ell", lambda x: np.mean(x == 1)), mean_ell=("ell", "mean"),
        minimum_coverage=("minimum_stable_5", "min"), mean_minimum_coverage=("minimum_stable_5", "mean"),
        max_conditional_training_skip=("conditional_training_coverage", lambda x: 1-x.min()),
        outside_mi_fraction=("outside_mi_bounds", "mean"), fallbacks=("fallback", "sum")).reset_index()
    diagnostic_summary.to_csv(out/"diagnostic_summary.csv", index=False)
    paired = []
    for count in COUNTS:
        pivot = frame[frame["count"] == count].pivot(index="seed", columns="method", values="accuracy")
        for method in [GUARDED, MATCHED]:
            for comparator in [POOLED, "KL tuned", "PSS ell=1"]:
                difference = 100*(pivot[method]-pivot[comparator])
                paired.append(dict(count=count, method=method, comparator=comparator,
                    difference_pp=difference.mean(), sd_pp=difference.std(),
                    positive_splits=int((difference > 1e-12).sum()), tied_splits=int((abs(difference) <= 1e-12).sum())))
    pd.DataFrame(paired).to_csv(out/"paired_differences.csv", index=False)
    save({p: importlib.metadata.version(p) for p in ["numpy", "scipy", "pandas", "scikit-learn", "matplotlib"]},
         out/"environment.json")
    figures(out, frame, diagnostics)
    print(summary[summary["count"].isin(COUNTS)].to_string(index=False))
    print(diagnostic_summary.to_string(index=False))
    print(choices[choices.method.isin([POOLED, GUARDED, MATCHED])].to_string(index=False))


if __name__ == "__main__":
    main()
