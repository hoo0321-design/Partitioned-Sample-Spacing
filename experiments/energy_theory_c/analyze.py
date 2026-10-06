"""Independent audit and descriptive plots for training-selected theory coefficients."""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from experiments.energy_theory_c.core import COEFFICIENTS, original
from experiments.energy_theory_c.run import (REFERENCE, SEEDS, CONTROLS, NEW_METHODS,
    METHODS, key, load_data, record_order, save, sha, slug)
from PSS.pss_v2 import estimate, mixed_mi, select_sc_cv

TUNED = "PSS theory C tuned"
FIXED = "PSS theory C=1"
BASE_PSS = "PSS class-aware SC-CV"
BASE_KL = "KL tuned"
COUNTS = [5, 10, 20]
NOISE = 1e-5
COLORS = {TUNED: "#176ba0", FIXED: "#8077ae", BASE_PSS: "#d27829",
          BASE_KL: "#bb3345", "KL k=1": "#438763", "All features": "#333333"}
LABELS = {TUNED: "PSS theory: tuned C", FIXED: "PSS theory: C = 1",
          BASE_PSS: "PSS: class-aware SC-CV", BASE_KL: "KL: tuned k",
          "KL k=1": "KL: k = 1", "All features": "All 25 features"}
SEED_COLORS = {42: "#13729a", 43: "#ca7929", 44: "#6b63a0"}


def close(a, b, context):
    assert np.isfinite(a) and np.isfinite(b) and abs(a-b) < 1e-9, (context, a, b)


def independent_ell(n, dimension, coefficient):
    value = coefficient * (n/(dimension**6 * math.log(n)**2))**(1/(dimension+8))
    return max(1, math.floor(value+.5))


def verify_metric(value, labels, cache, counts):
    values = value["metrics"]
    for field in ["accuracy", "balanced_accuracy", "auc"]:
        assert np.isfinite(values[field]) and 0 <= values[field] <= 1
    for field in ["fit_seconds", "prediction_seconds"]:
        assert np.isfinite(values[field]) and values[field] >= 0
    if value.get("source"):
        assert sha(value["source"]) == value["source_sha256"]
        stored = json.loads(Path(value["source"]).read_text())
        assert stored.get("metrics", stored) == values
    if value.get("prediction_file"):
        filename = value["prediction_file"]
        assert Path(filename).is_absolute() and sha(filename) == value["prediction_sha256"]
        if filename not in cache:
            with np.load(filename) as prediction:
                if "y" in prediction:
                    np.testing.assert_array_equal(prediction["y"], labels)
                assert len(prediction["prediction"]) == len(labels)
                measured = dict(accuracy=accuracy_score(labels, prediction["prediction"]),
                    balanced_accuracy=balanced_accuracy_score(labels, prediction["prediction"]),
                    auc=roc_auc_score(labels, prediction["decision"]))
            cache[filename] = (labels.copy(), measured)
        np.testing.assert_array_equal(cache[filename][0], labels)
        for field, measured in cache[filename][1].items():
            close(values[field], measured, (filename, field))
        counts["prediction_records_verified"] += 1
        counts["prediction_rows_verified"] += len(labels)
    else:
        assert value.get("reused") and value.get("source"), "Missing new prediction outputs"
        counts["historical_metric_only_records_hash_verified"] += 1


def canonical_score(z, y, features, ell, cache):
    cache_key = (tuple(sorted(features)), ell)
    if cache_key not in cache:
        x = z[:, cache_key[0]]
        measured = mixed_mi(x, y, ell)
        pooled = estimate(x, ell)
        valid = (measured["coverage_all"] > 0 and measured["min_conditional_coverage"] > 0
                 and np.isfinite(measured["mi"]))
        cache[cache_key] = dict(valid=valid, score=measured["mi"] if valid else None,
            training_coverage=measured["coverage_all"],
            conditional_training_coverage=measured["min_conditional_coverage"],
            integrated_mass=pooled["integrated_mass"], label_entropy=measured["label_entropy"],
            outside_mi_bounds=measured["outside_mi_bounds"] if valid else None)
    return cache[cache_key]


def compare_candidate(row, measured, context):
    assert row["valid"] == measured["valid"], context
    if row["valid"]:
        for field in ["score", "training_coverage", "conditional_training_coverage", "integrated_mass"]:
            close(row[field], measured[field], (context, field))
    else:
        assert row["score"] is None and row["error"]
        for field in ["training_coverage", "conditional_training_coverage", "integrated_mass"]:
            if row[field] is not None:
                close(row[field], measured[field], (context, field))


def verify_path(path, coefficient, active, z, y, cache, counts, outer=False):
    """Rebuild all stored greedy decisions and independently check the specified ell."""
    assert path["n_selection"] == len(z)
    assert path["coefficient"] == coefficient
    assert path["status"] in ["complete", "failed"]
    selected = []

    def candidates_at_step(candidates, step):
        ell = independent_ell(len(z), step, coefficient)
        assert len(candidates) == len(active)-len(selected)
        assert {r["feature"] for r in candidates} == set(active)-set(selected)
        for candidate in candidates:
            assert candidate["features"] == sorted(selected+[candidate["feature"]])
            assert candidate["ell"] == ell
            assert isinstance(candidate["valid"], bool)
            if candidate["valid"]:
                assert np.isfinite(candidate["score"])
            else:
                assert candidate["score"] is None and candidate["error"]
            if outer:
                compare_candidate(candidate, canonical_score(z, y, candidate["features"], ell, cache),
                                  ("outer candidate", step, candidate["feature"]))
                counts["outer_candidate_records_recomputed"] += 1
        return [candidate for candidate in candidates if candidate["valid"]]

    for step, row in enumerate(path["history"], 1):
        assert row["step"] == step
        valid = candidates_at_step(row["candidates"], step)
        assert valid
        best = min(valid, key=lambda r: (-r["score"], r["feature"]))
        selected.append(best["feature"])
        assert row["features"] == selected
        for field, value in best.items():
            if field != "features":
                assert row[field] == value, ("selected row", step, field)
        measured = canonical_score(z, y, row["features"], row["ell"], cache)
        compare_candidate(row, measured, ("selected row canonical MI", step))
        counts["outer_selected_rows_recomputed" if outer else "inner_selected_rows_recomputed"] += 1
        if outer and step in COUNTS:
            print(json.dumps(dict(stage="audit_outer_candidates", coefficient=coefficient,
                                  step=step, unique_recomputed=len(cache))), flush=True)
    if path["status"] == "complete":
        assert len(path["history"]) == 20 and not path.get("failure")
    else:
        failure = path["failure"]
        assert len(path["history"]) < 20 and failure["step"] == len(selected)+1
        assert failure["features"] == selected and failure["error"]
        assert failure["ell"] == independent_ell(len(z), failure["step"], coefficient)
        assert not candidates_at_step(failure["candidates"], failure["step"])
    counts["outer_paths_verified" if outer else "development_paths_verified"] += 1


def diagnostic_row(z, y, seed, features, ell, cache):
    cache_key = (tuple(sorted(features)), ell)
    if cache_key not in cache:
        ids = np.empty(len(y), dtype=int)
        rng = np.random.default_rng(seed+6000+91)
        for label in [0, 1]:
            rows = np.flatnonzero(y == label)
            ids[rows] = rng.permutation(np.arange(len(rows)) % 3)
        x = z[:, cache_key[0]]
        stats = [select_sc_cv(x[mask], [ell], n_folds=3, n_min=5, fold_id=ids[mask])
                 for mask in [np.ones(len(y), dtype=bool), y == 0, y == 1]]
        cache[cache_key] = dict(diagnostic_folds=3, diagnostic_n_min=5,
            pooled_stable_5=stats[0]["stable_validation_coverage"],
            class0_stable=stats[1]["stable_validation_coverage"],
            class1_stable=stats[2]["stable_validation_coverage"],
            minimum_stable_5=min(s["stable_validation_coverage"] for s in stats),
            pooled_cv_coverage=stats[0]["cv_coverage"],
            conditional_cv_coverage=min(s["cv_coverage"] for s in stats[1:]))
    return cache[cache_key]


def audit(out):
    protocol_file = out/"protocol.json"
    protocol = json.loads(protocol_file.read_text())
    assert protocol["coefficient_grid"] == COEFFICIENTS == [1, 1.5, 2, 2.5, 3, 4]
    for n in [9669, 13814]:
        for coefficient in COEFFICIENTS:
            expected = [independent_ell(n, dimension, coefficient) for dimension in range(1, 21)]
            assert protocol["ell_schedules"][str(n)][str(coefficient)] == expected
    for source, checksum in protocol["source_sha256"].items():
        assert sha(ROOT/source) == checksum, source
    for source, checksum in protocol["reference_sha256"].items():
        assert Path(source).is_absolute() and sha(source) == checksum, source
    assert sha(ROOT/"data/energydata_complete.csv") == protocol["data_sha256"]
    barrier_file = out/"all_locks.json"
    barrier = json.loads(barrier_file.read_text())
    expected_locks = {str(s): sha(out/f"seed_{s}"/"lock.json") for s in SEEDS}
    assert barrier["locked_before_test"] and barrier["locks_sha256"] == expected_locks
    assert barrier["protocol_sha256"] == sha(protocol_file)
    x, target, names = load_data()
    data, diagnostics, choices, development, outer_status = [], [], [], [], []
    counts = dict(prediction_records_verified=0, prediction_rows_verified=0,
        historical_metric_only_records_hash_verified=0, outer_candidate_records_recomputed=0,
        inner_selected_rows_recomputed=0, outer_selected_rows_recomputed=0,
        outer_paths_verified=0, development_paths_verified=0, unique_inner_score_recomputations=0,
        unique_outer_score_recomputations=0, unique_outer_cv_diagnostics=0)
    prediction_cache, limitations = {}, []
    for seed in SEEDS:
        folder = out/f"seed_{seed}"
        split = json.loads((folder/"split.json").read_text())
        tr, te = original.split_indices(np.arange(len(x)), seed)
        it, iv = original.split_indices(tr, seed+1000)
        for label, indices in zip(["outer_train", "outer_test", "inner_train", "inner_validation"], [tr, te, it, iv]):
            np.testing.assert_array_equal(split[label], indices)
        assert not set(tr) & set(te) and not set(it) & set(iv)
        assert set(it) | set(iv) == set(tr)
        y, ty, threshold = original.labels(target[tr], target[te])
        iy, vy, inner_threshold = original.labels(target[it], target[iv])
        z, active = original.selection_data(x[tr], NOISE, seed+5000)
        iz, inner_active = original.selection_data(x[it], NOISE, seed+3000)
        assert len(tr) == 13814 and len(it) == 9669
        lock = json.loads((folder/"lock.json").read_text())
        assert lock["locked_before_test"] and lock["protocol_sha256"] == sha(protocol_file)
        assert set(lock["choices"]) == set(NEW_METHODS)
        records, inner_cache, outer_cache, cv_cache = [], {}, {}, {}
        assert set(lock["development_sha256"]) == {p.name for p in (folder/"development").glob("*.json")}
        for filename, checksum in lock["development_sha256"].items():
            source = folder/"development"/filename
            assert sha(source) == checksum
            record = json.loads(source.read_text())
            coefficient = record["config"]["coefficient"]
            assert record["method"] == TUNED and record["threshold"] == inner_threshold
            assert record["status"] == record["path"]["status"]
            verify_path(record["path"], coefficient, inner_active, iz, iy, inner_cache, counts)
            if record["status"] == "complete":
                assert [v["count"] for v in record["metrics"]] == COUNTS
                for value in record["metrics"]:
                    verify_metric(value, vy, prediction_cache, counts)
                    assert value["features"] == record["path"]["history"][value["count"]-1]["features"]
            else:
                assert record["metrics"] == []
            history = record["path"]["history"]
            metrics = {v["count"]: v["metrics"]["accuracy"] for v in record["metrics"]}
            development.append(dict(seed=seed, coefficient=coefficient, status=record["status"],
                completed_steps=len(history), failure=json.dumps(record["path"].get("failure"), sort_keys=True),
                inner_accuracy=float(np.mean(list(metrics.values()))) if metrics else np.nan,
                **{f"accuracy_{n}": metrics.get(n, np.nan) for n in COUNTS},
                minimum_training_coverage=min((r["training_coverage"] for r in history), default=np.nan),
                maximum_conditional_training_skip=1-min((r["conditional_training_coverage"] for r in history), default=np.nan),
                ell1_fraction=np.mean([r["ell"] == 1 for r in history]) if history else np.nan))
            records.append(record)
        assert len(records) == len(COEFFICIENTS) == 6
        assert {r["config"]["coefficient"] for r in records} == set(COEFFICIENTS)
        selectable = [r for r in records if r["status"] == "complete"]
        assert selectable, "No complete training configuration exists"
        best = min(selectable, key=lambda r: (-np.mean([v["metrics"]["accuracy"] for v in r["metrics"]]),
                                              r["config"]["coefficient"]))
        assert best == min(selectable, key=record_order)
        assert best["config"] == lock["choices"][TUNED]["config"]
        close(lock["choices"][TUNED]["inner_accuracy"], np.mean([v["metrics"]["accuracy"] for v in best["metrics"]]),
              (seed, "inner objective"))
        assert lock["choices"][FIXED]["config"] == {"coefficient": 1}
        fixed_record = next(r for r in records if r["config"]["coefficient"] == 1)
        assert fixed_record["status"] == "complete"
        close(lock["choices"][FIXED]["inner_accuracy"],
              np.mean([v["metrics"]["accuracy"] for v in fixed_record["metrics"]]), (seed, "fixed-C inner accuracy"))
        for method in METHODS:
            source = folder/"outer"/(slug(method)+".json")
            record = json.loads(source.read_text())
            path, config = record["path"], record["config"]
            assert record["method"] == method and record["threshold"] == threshold
            status = record.get("status", "complete")
            if method in NEW_METHODS:
                before = json.loads(source.with_name(source.stem+"_pretest.json").read_text())
                assert all(record[k] == v for k, v in before.items())
                assert before["locks_sha256"] == expected_locks
                assert before["protocol_sha256"] == sha(protocol_file)
                assert before["lock_barrier_sha256"] == sha(barrier_file)
                assert config == lock["choices"][method]["config"]
                assert status == path["status"]
                verify_path(path, config["coefficient"], active, z, y, outer_cache, counts, outer=True)
                for row in path["history"]:
                    measured = canonical_score(z, y, row["features"], row["ell"], outer_cache)
                    coverage = diagnostic_row(z, y, seed, row["features"], row["ell"], cv_cache)
                    diagnostics.append(dict(seed=seed, method=method, coefficient=config["coefficient"],
                        count=row["step"], ell=row["ell"], features=",".join(map(str, row["features"])),
                        path_status=status, **measured, **coverage))
            else:
                old_path = REFERENCE/f"seed_{seed}"/"outer"/(slug(method)+".json")
                assert Path(record["source_outer_record"]) == old_path
                assert record["source_outer_sha256"] == sha(old_path)
                old_record = json.loads(old_path.read_text())
                for field in ["config", "path", "threshold", "metrics"]:
                    assert record[field] == old_record[field], (seed, method, field)
            assert path["n_selection"] == (0 if method == "All features" else len(tr))
            outer_status.append(dict(seed=seed, method=method, status=status,
                completed_steps=len(path["history"]), evaluated_counts=len(record["metrics"]),
                failure=json.dumps(path.get("failure"), sort_keys=True)))
            choices.append(dict(seed=seed, method=method, **config, outer_status=status,
                inner_accuracy=lock["choices"].get(method, {}).get("inner_accuracy")))
            if status == "failed":
                assert record["metrics"] == []
                limitations.append(f"Seed {seed}: {method} failed at dimension {path['failure']['step']}; no outer metrics reported.")
                continue
            assert len(path["history"]) == len(record["metrics"]) == (1 if method == "All features" else 20)
            previous = []
            for row, value in zip(path["history"], record["metrics"]):
                assert len(row["features"]) == len(set(row["features"])) == value["count"]
                assert row["features"] == value["features"]
                if method != "All features":
                    assert row["features"][:-1] == previous
                previous = row["features"]
                assert value["selected_features"] == [names[j] for j in row["features"]]
                verify_metric(value, ty, prediction_cache, counts)
                data.append(dict(seed=seed, method=method, count=value["count"],
                    features=",".join(value["selected_features"]), **value["metrics"]))
        counts["unique_inner_score_recomputations"] += len(inner_cache)
        counts["unique_outer_score_recomputations"] += len(outer_cache)
        counts["unique_outer_cv_diagnostics"] += len(cv_cache)
        print(json.dumps(dict(stage="audit", seed=seed, verified=True,
                              unique_outer_candidates=len(outer_cache))), flush=True)
    frame = pd.DataFrame(data)
    assert len(frame) == sum((1 if r["method"] == "All features" else 20)
                             for r in outer_status if r["status"] == "complete")
    assert not frame.duplicated(["seed", "method", "count"]).any()
    assert counts["development_paths_verified"] == 18 and counts["outer_paths_verified"] == 6
    counts.update(passed=True, curve_points=len(frame), unique_prediction_files_verified=len(prediction_cache),
        limitations=limitations, complete_outer_comparison=not limitations, source_hashes_verified=True,
        reference_unchanged=True, all_six_coefficients_retained_per_split=True,
        training_only_winners_recomputed=True, full_training_samples_verified=True,
        all_three_locks_verified_before_test_snapshots=True, formula_and_greedy_maxima_verified=True,
        score_recomputation_scope="Every selected development row, and every outer candidate including invalid terminal candidates, independently checked via public PSS MI/entropy APIs; repeated subsets/ell cached within each seed and split.",
        coverage_diagnostic_scope="Selected outer rows only: independent pooled/class coverage on fixed 3 folds with n_min=5; diagnostic only, not used for C or feature selection.",
        historical_metric_only_scope="Historical inner cached metrics without predictions verified against hashed source files; no independent classification recomputation.")
    save(counts, out/"audit.json")
    return frame, pd.DataFrame(diagnostics), pd.DataFrame(choices), pd.DataFrame(development), pd.DataFrame(outer_status)


def summarize_diagnostics(diagnostics):
    return diagnostics.groupby("method").agg(selected_rows=("ell", "size"),
        ell1_fraction=("ell", lambda x: np.mean(x == 1)), mean_ell=("ell", "mean"),
        minimum_stable_coverage=("minimum_stable_5", "min"), mean_minimum_stable_coverage=("minimum_stable_5", "mean"),
        maximum_training_skip=("training_coverage", lambda x: 1-x.min()),
        maximum_conditional_training_skip=("conditional_training_coverage", lambda x: 1-x.min()),
        outside_mi_fraction=("outside_mi_bounds", "mean")).reset_index()


def figures(out, frame, development, choices):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    destination = out/"plots"
    destination.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for method in [TUNED, FIXED, BASE_PSS, BASE_KL, "KL k=1"]:
        subset = frame[frame.method == method]
        if subset.empty:
            continue
        stats = subset.groupby("count").accuracy.agg(["mean", "std", "size"])
        # Avoid presenting a partial set of successful splits as a 3-split result.
        stats.loc[stats["size"] < len(SEEDS), ["mean", "std"]] = np.nan
        xx, mean, sd = stats.index.to_numpy(), 100*stats["mean"].to_numpy(), 100*stats["std"].to_numpy()
        ax.plot(xx, mean, color=COLORS[method], label=LABELS[method],
                ls="--" if method in [FIXED, "KL k=1"] else "-", lw=2.2 if method == TUNED else 1.7)
        ax.fill_between(xx, mean-sd, mean+sd, color=COLORS[method], alpha=.075)
    ax.axhline(100*frame[frame.method == "All features"].accuracy.mean(), color=COLORS["All features"],
               ls=":", lw=1.4, label=LABELS["All features"])
    ax.set(xlabel="Selected features", ylabel="Test accuracy (%)", xticks=[1, 5, 10, 15, 20],
           title="Energy: training-selected theory coefficient C")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.grid(alpha=.18)
    fig.text(.5, .014, "Same 3 inspected splits; mean ± sample SD, not confidence intervals. Curves require all 3 complete paths.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(destination/"accuracy.png", dpi=210)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, field, title in zip(axes.flat, ["inner_accuracy", "accuracy_5", "accuracy_10", "accuracy_20"],
                               ["Selection objective: mean of 5 / 10 / 20", "5 selected features", "10 selected features", "20 selected features"]):
        for seed in SEEDS:
            sub = development[development.seed == seed].sort_values("coefficient")
            ax.plot(sub.coefficient, 100*sub[field], color=SEED_COLORS[seed], marker="o", ms=4,
                    alpha=.7, lw=1.2, label=f"Seed {seed}")
        stats = development.groupby("coefficient")[field].agg(["mean", "std", "count"])
        stats.loc[stats["count"] < len(SEEDS), ["mean", "std"]] = np.nan
        ax.errorbar(stats.index, 100*stats["mean"], yerr=100*stats["std"], fmt="o-", color="#222222",
                    lw=2, ms=4, capsize=3, label="Mean ± SD (3 splits)")
        ax.set(title=title, ylabel="Inner-validation accuracy (%)", xticks=COEFFICIENTS)
        ax.grid(alpha=.18)
    axes[0, 0].legend(fontsize=8, frameon=False)
    for ax in axes[1]:
        ax.set_xlabel("C")
    failed = development[development.status == "failed"]
    note = "Failed configurations are unselectable; no partial-path accuracy is substituted."
    if not failed.empty:
        note += " Failed C: " + "; ".join(f"seed {seed}: " + ", ".join(map(str, g.coefficient)) for seed, g in failed.groupby("seed"))
    fig.suptitle("Training-only coefficient comparison", fontsize=13)
    fig.text(.5, .014, note, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, .965))
    fig.savefig(destination/"inner_validation.png", dpi=210)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), sharey=True)
    dimensions = np.arange(1, 21)
    for ax, n, title in zip(axes, [9669, 13814], ["Inner training (n = 9,669)", "Outer training (n = 13,814)"]):
        for seed in SEEDS:
            coefficient = choices[(choices.seed == seed) & (choices.method == TUNED)].coefficient.iloc[0]
            schedule = [independent_ell(n, int(d), coefficient) for d in dimensions]
            ax.plot(dimensions, schedule, marker="o", ms=3, lw=1.7, color=SEED_COLORS[seed],
                    label=f"Seed {seed}: C = {coefficient:g}")
        ax.plot(dimensions, [independent_ell(n, int(d), 1) for d in dimensions], ls="--", lw=2,
                color="#333333", label="Fixed C = 1")
        ax.set(title=title, xlabel="Subset dimension d", xticks=[1, 5, 10, 15, 20])
        ax.grid(alpha=.18)
    axes[0].set_ylabel(r"Rule-required $\ell$")
    axes[0].legend(frameon=False, fontsize=9)
    fig.text(.5, .015, "C is chosen using inner validation, then held fixed; ell is recomputed using the full outer training size.\nSchedules show the rule at each dimension, including any dimensions beyond a failed path.",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(destination/"ell_schedules.png", dpi=210)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    if args.plots_only:
        assert json.loads((out/"audit.json").read_text())["passed"]
        figures(out, pd.read_csv(out/"metrics.csv"), pd.read_csv(out/"development_summary.csv"),
                pd.read_csv(out/"locked_settings.csv"))
        return
    frame, diagnostics, choices, development, statuses = audit(out)
    for name, value in [("metrics", frame), ("accuracy_curves", frame), ("diagnostics", diagnostics),
                        ("locked_settings", choices), ("development_summary", development), ("outer_status", statuses)]:
        value.to_csv(out/(name+".csv"), index=False)
    summary = frame.groupby(["method", "count"]).agg(accuracy=("accuracy", "mean"), sd=("accuracy", "std"),
        balanced_accuracy=("balanced_accuracy", "mean"), balanced_accuracy_sd=("balanced_accuracy", "std"),
        auc=("auc", "mean"), auc_sd=("auc", "std"), repeats=("seed", "size")).reset_index()
    summary["complete_comparison"] = summary.repeats == len(SEEDS)
    summary.to_csv(out/"summary.csv", index=False)
    summarize_diagnostics(diagnostics).to_csv(out/"diagnostic_summary.csv", index=False)
    paired = []
    for count in COUNTS:
        pivot = frame[frame["count"] == count].pivot(index="seed", columns="method", values="accuracy")
        for comparator in [FIXED, BASE_PSS, BASE_KL, "KL k=1"]:
            if TUNED not in pivot or comparator not in pivot:
                difference = pd.Series(dtype=float)
            else:
                difference = (100*(pivot[TUNED]-pivot[comparator])).dropna()
            paired.append(dict(count=count, method=TUNED, comparator=comparator, difference_pp=difference.mean(),
                sd_pp=difference.std(), paired_splits=len(difference), complete_comparison=len(difference) == len(SEEDS),
                positive_splits=int((difference > 1e-12).sum()), tied_splits=int((abs(difference) <= 1e-12).sum()),
                **{f"seed_{seed}_pp": float(difference.get(seed, np.nan)) for seed in SEEDS}))
    pd.DataFrame(paired).to_csv(out/"paired_differences.csv", index=False)
    save(dict(python=platform.python_version(), platform=platform.platform(),
        packages={p: importlib.metadata.version(p) for p in ["numpy", "scipy", "pandas", "scikit-learn", "matplotlib"]}),
        out/"environment.json")
    figures(out, frame, development, choices)
    print(summary[summary["count"].isin(COUNTS)].to_string(index=False))
    print(choices[choices.method.isin(NEW_METHODS)].to_string(index=False))
    print(summarize_diagnostics(diagnostics).to_string(index=False))
    print(json.dumps(dict(limitations=json.loads((out/"audit.json").read_text())["limitations"])))


if __name__ == "__main__":
    main()
