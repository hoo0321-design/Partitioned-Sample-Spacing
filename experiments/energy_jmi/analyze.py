"""Independent score/path audit and descriptive summaries for the Energy JMI ablation.

No test result is used to alter the protocol, settings, or feature paths. The
recomputation checks every outer JMI component's canonical MI score, and the
selected-ell CV diagnostics for every PSS singleton/pair. It does not repeat the
entire ell search; that selection implementation is covered by the core tests.
"""
import argparse
import importlib.metadata
from itertools import combinations
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from core import ROOT, COUNTS, FOLDS, ELLS, NOISE, PRIMARY, original
from run import (REFERENCE, SEEDS, CONTROLS, NEW_METHODS, METHODS,
                 configurations, key, load_data, record_order, save, sha, slug)
from PSS.pss_v2 import mixed_mi, select_sc_cv

PSS = "PSS class-aware SC-CV"
COLORS = {PSS: "#cf7326", "PSS-JMI": "#146898", "KL tuned": "#ba3444",
          "KL-JMI": "#208365", "PSS ell=1": "#899098", "All features": "#333333"}
LABELS = {PSS: "PSS: class-aware SC-CV", "PSS-JMI": "PSS-JMI", "KL tuned": "KL: tuned k",
          "KL-JMI": "KL-JMI", "PSS ell=1": r"PSS: fixed $\ell=1$", "All features": "All features"}


def close(a, b, context):
    assert np.isfinite(a) and np.isfinite(b) and abs(a-b) < 1e-9, (context, a, b)


def verify_metric(value, labels, cache, counts):
    """Recompute available predictions; separately count historical metric-only reuse."""
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
        assert Path(filename).is_absolute()
        assert sha(filename) == value["prediction_sha256"]
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
        assert value.get("reused") and value.get("source"), "Missing new predictions"
        counts["historical_metric_only_records_hash_verified"] += 1


def component_table(path, active):
    rows = path["score_table"]["singletons"] + path["score_table"]["pairs"]
    table = {}
    for row in rows:
        features = tuple(row["features"])
        assert features == tuple(sorted(set(features))) and len(features) in [1, 2]
        assert features not in table and np.isfinite(row["score"])
        table[features] = row
    expected = {(j,) for j in active} | set(combinations(sorted(active), 2))
    assert set(table) == expected and len(rows) == len(expected)
    assert len(path["score_table"]["singletons"]) == len(active)
    assert all(len(r["features"]) == 1 for r in path["score_table"]["singletons"])
    assert all(len(r["features"]) == 2 for r in path["score_table"]["pairs"])
    return table


def verify_jmi_path(path, active):
    """Reconstruct every candidate and tie break without calling the JMI selector."""
    table = component_table(path, active)
    selected, used = [], set()
    for step, row in enumerate(path["history"], 1):
        candidates = []
        for feature in active:
            if feature in selected:
                continue
            keys = [(feature,)] if not selected else [tuple(sorted([feature, s])) for s in selected]
            score = float(np.mean([table[k]["score"] for k in keys]))
            candidates.append((score, feature, keys))
        score, feature, keys = min(candidates, key=lambda r: (-r[0], r[1]))
        selected.append(feature)
        assert row["step"] == step and row["feature"] == feature
        assert row["features"] == selected
        close(row["score"], score, ("JMI greedy mean", step))
        saved = {tuple(c["features"]): c for c in row["components"]}
        assert len(saved) == len(row["components"]) == len(keys)
        assert set(saved) == set(keys)
        for features in keys:
            assert saved[features] == table[features]
        used.update(keys)
    return table, used


def fold_ids(y, seed, folds):
    result = np.empty(len(y), dtype=int)
    rng = np.random.default_rng(seed+6000+91)
    for label in [0, 1]:
        rows = np.flatnonzero(y == label)
        result[rows] = rng.permutation(np.arange(len(rows)) % folds)
    return result


def canonical_pss(z, y, features, ell, folds, ids):
    """Public PSS API, independent of the experiment selector's caches and CV loop."""
    x = z[:, features]
    stats = [select_sc_cv(x[mask], [ell], n_folds=folds, n_min=5, fold_id=ids[mask])
             for mask in [np.ones(len(y), dtype=bool), y == 0, y == 1]]
    mi = mixed_mi(x, y, ell)
    return dict(score=mi["mi"], cv_score=stats[0]["cv_score"],
        pooled_cv_coverage=stats[0]["cv_coverage"],
        pooled_stable_5=stats[0]["stable_validation_coverage"],
        conditional_cv_coverage=min(r["cv_coverage"] for r in stats[1:]),
        conditional_stable_5=min(r["stable_validation_coverage"] for r in stats[1:]),
        minimum_stable_5=min(r["stable_validation_coverage"] for r in stats),
        class0_stable=stats[1]["stable_validation_coverage"],
        class1_stable=stats[2]["stable_validation_coverage"],
        training_coverage=mi["coverage_all"],
        conditional_training_coverage=mi["min_conditional_coverage"],
        label_entropy=mi["label_entropy"], outside_mi_bounds=mi["outside_mi_bounds"])


def verify_outer_table(table, z, y, seed, method, config, counts):
    """Recompute all 25+300 outer scores and chosen-ell PSS component diagnostics."""
    ids = fold_ids(y, seed, config["folds"]) if method == "PSS-JMI" else None
    for number, (features, row) in enumerate(table.items(), 1):
        if method == "PSS-JMI":
            assert row["ell"] in ELLS
            measured = canonical_pss(z, y, features, row["ell"], config["folds"], ids)
            for field, result in measured.items():
                if field in ["label_entropy", "outside_mi_bounds"] and field not in row:
                    continue
                close(row[field], result, (seed, method, features, field))
            if not row["fallback"]:
                assert row["minimum_stable_5"] >= config["tau"]
            close(row["stable_coverage"], row["minimum_stable_5"], "guard coverage")
            counts["pss_component_scores_recomputed"] += 1
            counts["pss_chosen_ell_cv_diagnostics_recomputed"] += 1
        else:
            x = z[:, features]
            score = original.kl_entropy(x, config["k"]) - sum(
                np.mean(y == label)*original.kl_entropy(x[y == label], config["k"])
                for label in [0, 1])
            close(row["score"], score, (seed, method, features, "score"))
            counts["kl_component_scores_recomputed"] += 1
        if number % 100 == 0:
            print(json.dumps(dict(stage="audit_components", seed=seed, method=method,
                                  checked=number, total=len(table))), flush=True)


def audit(out):
    protocol_file = out/"protocol.json"
    protocol = json.loads(protocol_file.read_text())
    for source, checksum in protocol["source_sha256"].items():
        assert sha(ROOT/source) == checksum, source
    for source, checksum in protocol["reference_sha256"].items():
        assert Path(source).is_absolute()
        assert sha(source) == checksum, source
    assert sha(ROOT/"data/energydata_complete.csv") == protocol["data_sha256"]
    barrier_file = out/"all_locks.json"
    barrier = json.loads(barrier_file.read_text())
    expected_locks = {str(s): sha(out/f"seed_{s}"/"lock.json") for s in SEEDS}
    assert barrier["locked_before_test"] and barrier["locks_sha256"] == expected_locks
    assert barrier["protocol_sha256"] == sha(protocol_file)
    x, target, names = load_data()
    data, diagnostics, choices, developments, step_diagnostics = [], [], [], [], []
    counts = dict(prediction_records_verified=0, prediction_rows_verified=0,
        historical_metric_only_records_hash_verified=0, pss_component_scores_recomputed=0,
        pss_chosen_ell_cv_diagnostics_recomputed=0, kl_component_scores_recomputed=0,
        development_paths_reconstructed=0, outer_paths_reconstructed=0)
    prediction_cache = {}
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
        _, vy, inner_threshold = original.labels(target[it], target[iv])
        z, active = original.selection_data(x[tr], NOISE, seed+5000)
        _, inner_active = original.selection_data(x[it], NOISE, seed+3000)
        assert len(tr) == 13814 and len(it) == 9669
        lock = json.loads((folder/"lock.json").read_text())
        assert lock["locked_before_test"] and lock["protocol_sha256"] == sha(protocol_file)
        assert set(lock["choices"]) == set(NEW_METHODS)
        records = []
        assert set(lock["development_sha256"]) == {p.name for p in (folder/"development").glob("*.json")}
        for filename, checksum in lock["development_sha256"].items():
            source = folder/"development"/filename
            assert sha(source) == checksum
            record = json.loads(source.read_text())
            assert record["path"]["n_selection"] == len(it)
            assert record["threshold"] == inner_threshold
            assert [v["count"] for v in record["metrics"]] == COUNTS
            table, used = verify_jmi_path(record["path"], inner_active)
            counts["development_paths_reconstructed"] += 1
            for value in record["metrics"]:
                verify_metric(value, vy, prediction_cache, counts)
                assert value["features"] == record["path"]["history"][value["count"]-1]["features"]
            records.append(record)
            developments.append(dict(seed=seed, method=record["method"], **record["config"],
                inner_accuracy=np.mean([v["metrics"]["accuracy"] for v in record["metrics"]]),
                unique_selected_components=len(used), component_candidates=len(table)))
        assert len(records) == sum(len(configurations(m)) for m in NEW_METHODS) == 25
        for method in NEW_METHODS:
            subset = [r for r in records if r["method"] == method]
            assert {key(r["config"]) for r in subset} == {key(c) for c in configurations(method)}
            def independent_order(record):
                config = record["config"]
                return (-float(np.mean([v["metrics"]["accuracy"] for v in record["metrics"]])),
                        -config.get("tau", 0), config.get("folds", 0), config.get("k", 0),
                        json.dumps(config, sort_keys=True, separators=(",", ":")))
            best = min(subset, key=independent_order)
            assert best == min(subset, key=record_order)
            choice = lock["choices"][method]
            assert best["config"] == choice["config"]
            close(choice["inner_accuracy"], np.mean([v["metrics"]["accuracy"] for v in best["metrics"]]),
                  (seed, method, "training-only winner"))
        for method in METHODS:
            source = folder/"outer"/(slug(method)+".json")
            record = json.loads(source.read_text())
            path, config = record["path"], record["config"]
            assert record["method"] == method and record["threshold"] == threshold
            if method in NEW_METHODS:
                before = json.loads(source.with_name(source.stem+"_pretest.json").read_text())
                assert all(record[k] == v for k, v in before.items())
                assert before["locks_sha256"] == expected_locks
                assert before["protocol_sha256"] == sha(protocol_file)
                assert before["lock_barrier_sha256"] == sha(barrier_file)
                assert config == lock["choices"][method]["config"]
                table, used = verify_jmi_path(path, active)
                counts["outer_paths_reconstructed"] += 1
                verify_outer_table(table, z, y, seed, method, config, counts)
                for features, component in table.items():
                    values = {k: v for k, v in component.items() if k != "features"}
                    probabilities = np.bincount(y, minlength=2)/len(y)
                    bound = -float(np.dot(probabilities, np.log(probabilities)))
                    values.setdefault("label_entropy", bound)
                    values.setdefault("outside_mi_bounds", not 0 <= component["score"] <= bound)
                    diagnostics.append(dict(seed=seed, method=method, component_kind="singleton" if len(features) == 1 else "pair",
                        selected=features in used, features=",".join(map(str, features)),
                        **values,
                        tau=config.get("tau"), folds=config.get("folds"), k=config.get("k")))
            else:
                old_path = REFERENCE/f"seed_{seed}"/"outer"/(slug(method)+".json")
                assert Path(record["source_outer_record"]) == old_path
                assert record["source_outer_sha256"] == sha(old_path)
                old_record = json.loads(old_path.read_text())
                for field in ["config", "path", "threshold", "metrics"]:
                    assert record[field] == old_record[field], (seed, method, field)
            assert path["n_selection"] == (0 if method == "All features" else len(tr))
            assert len(path["history"]) == len(record["metrics"]) == (1 if method == "All features" else 20)
            choices.append(dict(seed=seed, method=method, **config,
                inner_accuracy=lock["choices"].get(method, {}).get("inner_accuracy")))
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
                if method in [PSS, "PSS ell=1"]:
                    diagnostics.append(dict(seed=seed, method=method, component_kind="subset", selected=True,
                        **{k: v for k, v in row.items() if k != "features"},
                        features=",".join(map(str, row["features"])), tau=config.get("tau"), folds=config.get("folds", 3)))
                if method in [PSS, "PSS ell=1", "PSS-JMI"]:
                    components = row["components"] if method == "PSS-JMI" else [row]
                    step_diagnostics.append(dict(seed=seed, method=method, step=row["step"],
                        mean_ell=np.mean([c["ell"] for c in components]),
                        minimum_stable_5=min(c["minimum_stable_5"] for c in components),
                        conditional_training_coverage=min(c["conditional_training_coverage"] for c in components)))
        print(json.dumps(dict(stage="audit", seed=seed, verified=True)), flush=True)
    frame = pd.DataFrame(data)
    assert len(frame) == len(SEEDS)*((len(METHODS)-1)*20+1)
    assert not frame.duplicated(["seed", "method", "count"]).any()
    assert counts["pss_component_scores_recomputed"] == counts["kl_component_scores_recomputed"] == 975
    counts.update(passed=True, curve_points=len(frame), unique_prediction_files_verified=len(prediction_cache),
        analysis_source_sha256=sha(Path(__file__)),
        source_hashes_verified=True, reference_unchanged=True, all_configurations_present=True,
        training_only_winners_recomputed=True, all_three_locks_verified_before_test_snapshots=True,
        full_training_samples_verified=True, full_jmi_tables_and_all_candidate_greedy_maxima_verified=True,
        independent_pss_recomputation_scope="All 325 outer singleton/pair scores and chosen-ell pooled/class CV diagnostics per seed. Entire ell grid not reselected.",
        historical_metric_only_scope="Historical inner cached metrics are checked against hashed source files, not independently recalculated without prediction files.")
    save(counts, out/"audit.json")
    return frame, pd.DataFrame(diagnostics), pd.DataFrame(choices), pd.DataFrame(developments), pd.DataFrame(step_diagnostics)


def diagnostic_summary(diagnostics):
    rows = []
    for (method, kind), group in diagnostics.groupby(["method", "component_kind"]):
        for scope, data in [("all_candidates", group), ("selected_unique_components", group[group.selected])]:
            assert not data.duplicated(["seed", "features"]).any()
            row = dict(method=method, component_kind=kind, scope=scope, components=len(data))
            for field in ["score", "ell", "minimum_stable_5", "conditional_training_coverage"]:
                values = data[field].dropna() if field in data else pd.Series(dtype=float)
                if len(values):
                    row["mean_"+field] = values.mean()
                    row["min_"+field] = values.min()
            if "ell" in data and data.ell.notna().any():
                row["ell1_fraction"] = float(np.mean(data.ell.dropna() == 1))
                row["fallbacks"] = int(data.fallback.fillna(False).astype(bool).sum())
                row["maximum_conditional_training_skip"] = 1-data.conditional_training_coverage.min()
            if "outside_mi_bounds" in data and data.outside_mi_bounds.notna().any():
                row["outside_mi_fraction"] = data.outside_mi_bounds.dropna().astype(bool).mean()
            rows.append(row)
    return pd.DataFrame(rows)


def redundancy_diagnostics(frame):
    """Descriptive dependence of selected features, using outer training rows only."""
    x, _, names = load_data()
    indices = {name: j for j, name in enumerate(names)}
    rows = []
    anchors = frame[frame.method.isin(PRIMARY) & frame["count"].isin(COUNTS)]
    for seed in SEEDS:
        train, _ = original.split_indices(np.arange(len(x)), seed)
        for record in anchors[anchors.seed == seed].itertuples(index=False):
            features = [indices[name] for name in record.features.split(",")]
            selected = pd.DataFrame(x[train][:, features])
            lower = np.tril_indices(len(features), -1)
            pearson = np.abs(selected.corr(method="pearson").to_numpy()[lower])
            spearman = np.abs(selected.corr(method="spearman").to_numpy()[lower])
            assert np.isfinite(pearson).all() and np.isfinite(spearman).all()
            assert len(features) == record.count and len(pearson) == record.count*(record.count-1)//2
            rows.append(dict(seed=seed, method=record.method, count=record.count,
                n_training=len(train), unique_pairs=len(pearson),
                mean_abs_pearson=float(pearson.mean()), pairs_abs_pearson_gt_08=int((pearson > .8).sum()),
                mean_abs_spearman=float(spearman.mean()), max_abs_pearson=float(pearson.max())))
    assert len(rows) == len(SEEDS)*len(PRIMARY)*len(COUNTS)
    return pd.DataFrame(rows)


def figures(out, frame, diagnostics, steps):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    destination = out/"plots"
    destination.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.7, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
    for method in [*PRIMARY, "PSS ell=1"]:
        stats = frame[frame.method == method].groupby("count").accuracy.agg(["mean", "std"])
        xx, mean, sd = stats.index.to_numpy(), 100*stats["mean"].to_numpy(), 100*stats["std"].to_numpy()
        axes[0].plot(xx, mean, color=COLORS[method], label=LABELS[method],
                     ls="--" if method == "PSS ell=1" else "-", lw=2)
        axes[0].fill_between(xx, mean-sd, mean+sd, color=COLORS[method], alpha=.065)
    all_accuracy = 100*frame[frame.method == "All features"].accuracy.mean()
    axes[0].axhline(all_accuracy, color=COLORS["All features"], ls=":", lw=1.4, label="All 25 features")
    axes[0].set(xlabel="Selected features", ylabel="Test accuracy (%)", xticks=[1, 5, 10, 15, 20],
                title="Subset MI versus pairwise JMI")
    axes[0].legend(fontsize=8.5, loc="lower right", frameon=False)
    for index, method in enumerate(PRIMARY):
        stats = frame[(frame.method == method) & frame["count"].isin(COUNTS)].groupby("count").accuracy.agg(["mean", "std"])
        axes[1].errorbar(np.arange(3)+(index-1.5)*.16, 100*stats["mean"], yerr=100*stats["std"],
                         fmt="o", ms=5, capsize=3, color=COLORS[method], label=LABELS[method])
    axes[1].set(xticks=np.arange(3), xticklabels=COUNTS, xlabel="Selected features", ylabel="Test accuracy (%)",
                title="Prespecified comparison counts")
    for ax in axes:
        ax.grid(alpha=.18)
    fig.text(.5, .012, "Same 3 previously inspected splits; mean ± sample SD, not confidence intervals. Selection uses all 13,814 training rows.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(destination/"accuracy.png", dpi=210)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    pairs = diagnostics[(diagnostics.method == "PSS-JMI") & (diagnostics.component_kind == "pair") & diagnostics.selected]
    fractions = pd.crosstab(pairs.seed, pairs.ell, normalize="index").reindex(index=SEEDS, columns=ELLS, fill_value=0)
    bottom = np.zeros(len(SEEDS))
    for ell, color in zip(ELLS, plt.cm.Blues(np.linspace(.28, .9, len(ELLS)))):
        values = fractions[ell].to_numpy()
        axes[0].bar(np.arange(len(SEEDS)), values, bottom=bottom, color=color, label=str(ell))
        bottom += values
    axes[0].set(xticks=np.arange(len(SEEDS)), xticklabels=SEEDS, ylim=(0, 1.03),
                xlabel="Split seed", ylabel="Fraction of unique selected pairs", title=r"PSS-JMI: selected pair $\ell$")
    axes[0].legend(title=r"$\ell$", ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(.5, -.16), fontsize=8)
    for method in [PSS, "PSS-JMI"]:
        sub = steps[steps.method == method]
        for ax, field in zip(axes[1:], ["minimum_stable_5", "conditional_training_coverage"]):
            stats = sub.groupby("step")[field].agg(["mean", "min", "max"])
            xx = stats.index.to_numpy()
            if field == "conditional_training_coverage":
                mean, low, high = 100*(1-stats["mean"]), 100*(1-stats["max"]), 100*(1-stats["min"])
            else:
                mean, low, high = 100*stats["mean"], 100*stats["min"], 100*stats["max"]
            ax.plot(xx, mean, color=COLORS[method], lw=2, label=LABELS[method])
            ax.fill_between(xx, low, high, color=COLORS[method], alpha=.12)
            ax.set(xlabel="Selected features", xticks=[1, 5, 10, 15, 20])
            ax.grid(alpha=.18)
    minimum = steps[steps.method.isin([PSS, "PSS-JMI"])].minimum_stable_5.min()
    axes[1].set(title="Minimum component stable coverage", ylabel="Coverage (%)", ylim=(max(0, 100*minimum-3), 103))
    axes[2].set(title="Worst-class training skipped fraction", ylabel="Skipped (%)")
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    fig.text(.66, .03, "Lines: split means; bands: observed range.\nJMI: minimum across components entering the selected candidate score.",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, .11, 1, 1))
    fig.savefig(destination/"coverage_diagnostics.png", dpi=210)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--plots-only", action="store_true", help="Redraw already audited CSV outputs without repeating recomputation")
    args = parser.parse_args()
    out = args.output.resolve()
    if args.plots_only:
        assert json.loads((out/"audit.json").read_text())["passed"]
        figures(out, pd.read_csv(out/"metrics.csv"), pd.read_csv(out/"diagnostics.csv"), pd.read_csv(out/"step_diagnostics.csv"))
        return
    frame, diagnostics, choices, development, steps = audit(out)
    for name, value in [("metrics", frame), ("accuracy_curves", frame), ("diagnostics", diagnostics),
                        ("locked_settings", choices), ("development_summary", development), ("step_diagnostics", steps)]:
        value.to_csv(out/(name+".csv"), index=False)
    summary = frame.groupby(["method", "count"]).agg(accuracy=("accuracy", "mean"), sd=("accuracy", "std"),
        balanced_accuracy=("balanced_accuracy", "mean"), balanced_accuracy_sd=("balanced_accuracy", "std"),
        auc=("auc", "mean"), auc_sd=("auc", "std"), repeats=("seed", "size")).reset_index()
    summary.to_csv(out/"summary.csv", index=False)
    diagnostic_summary(diagnostics).to_csv(out/"diagnostic_summary.csv", index=False)
    redundancy_diagnostics(frame).to_csv(out/"redundancy_diagnostics.csv", index=False)
    paired = []
    for count in COUNTS:
        pivot = frame[frame["count"] == count].pivot(index="seed", columns="method", values="accuracy")
        for method, comparator in [("PSS-JMI", PSS), ("PSS-JMI", "KL tuned"), ("PSS-JMI", "KL-JMI"),
                                   ("KL-JMI", "KL tuned")]:
            difference = 100*(pivot[method]-pivot[comparator])
            paired.append(dict(count=count, method=method, comparator=comparator, difference_pp=difference.mean(),
                sd_pp=difference.std(), positive_splits=int((difference > 1e-12).sum()),
                tied_splits=int((abs(difference) <= 1e-12).sum()),
                **{f"seed_{seed}_pp": float(difference.loc[seed]) for seed in SEEDS}))
    pd.DataFrame(paired).to_csv(out/"paired_differences.csv", index=False)
    save(dict(python=platform.python_version(), platform=platform.platform(),
              packages={p: importlib.metadata.version(p) for p in ["numpy", "scipy", "pandas", "scikit-learn", "matplotlib"]}),
         out/"environment.json")
    figures(out, frame, diagnostics, steps)
    print(summary[summary["count"].isin(COUNTS)].to_string(index=False))
    print(pd.DataFrame(paired).to_string(index=False))
    print(diagnostic_summary(diagnostics).to_string(index=False))


if __name__ == "__main__":
    main()
