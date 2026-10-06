"""Verify all locked selections/predictions and summarize every declared policy."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from core import ROOT, PRIMARY, COUNTS, FOLDS, TAUS, configurations, original
from run import ABLATIONS, SEEDS, key, load_data, save_csv, save_json, sha, splits

COLORS = {"PSS expanded SC-CV": "#164DDB", "KL tuned": "#D23F2A", "Ross tuned": "#8863B8",
          "Univariate tuned": "#24876D", "PSS 3-fold": "#A07835", "PSS 5-fold": "#24876D",
          "PSS 10-fold": "#B54E87", "PSS matched-budget": "#727985", "PSS tau=0.99": "#8863B8",
          "PSS ell=1": "#666666", "PSS ell=2": "#D88D1D", "KL k=1": "#D23F2A"}
LABELS = {"PSS expanded SC-CV": "PSS + expanded SC-CV", "KL tuned": "kNN entropy difference (tuned)",
          "Ross tuned": "Ross kNN-MI (tuned)", "Univariate tuned": "Univariate kNN ranking",
          "PSS 3-fold": "3-fold SC-CV", "PSS 5-fold": "5-fold SC-CV", "PSS 10-fold": "10-fold SC-CV",
          "PSS matched-budget": "SC-CV, 10 configurations", "PSS tau=0.99": r"3-fold, $\tau=0.99$",
          "PSS ell=1": r"Fixed $\ell=1$", "PSS ell=2": r"Fixed $\ell=2$", "KL k=1": "kNN entropy difference, k=1"}


def audit(out, frame):
    _, target, names = load_data()
    protocol = json.loads((out/"protocol.json").read_text())
    assert sha(ROOT/"data/energydata_complete.csv") == protocol["data_sha256"]
    for source, checksum in protocol["code_sha256"].items():
        assert sha(ROOT/source) == checksum, source
    assert len(frame) == 5*(20*len(PRIMARY+ABLATIONS)+1)
    assert not frame.duplicated(["seed", "method", "features"]).any()
    for (_, method), group in frame.groupby(["seed", "method"]):
        assert set(group.features) == ({25} if method == "All features" else set(range(1, 21)))
    checked, predictions = 0, 0
    prediction_cache = {}
    development_rows, choices = [], []
    for seed in SEEDS:
        folder = out/f"seed_{seed}"
        tr, te, it, iv = splits(seed)
        saved = json.loads((folder/"split.json").read_text())
        for field, values in zip(["outer_train", "outer_test", "inner_train", "inner_validation"], [tr, te, it, iv]):
            np.testing.assert_array_equal(saved[field], values)
        y, ty, threshold = original.labels(target[tr], target[te])
        lock = json.loads((folder/"lock.json").read_text())
        assert lock["locked_before_test"] and lock["protocol_sha256"] == sha(out/"protocol.json")
        seen_configs = set()
        for filename, checksum in lock["development_sha256"].items():
            p = folder/"development"/filename
            assert sha(p) == checksum
            dev = json.loads(p.read_text())
            c = dev["config"]
            seen_configs.add(key(c))
            assert dev["path"]["n_selection"] == len(it)
            assert dev["threshold"] == float(np.median(target[it]))
            development_rows.append(dict(seed=seed, method=dev["method"], config=key(c), **c,
                mean_accuracy=np.mean([r["accuracy"] for r in dev["metrics"]]),
                ell1_fraction=np.mean([r.get("ell") == 1 for r in dev["path"]["history"]])))
        expected = {key(c) for m in PRIMARY for c in configurations(m)}
        assert seen_configs == expected
        for method, choice in lock["choices"].items():
            choices.append(dict(seed=seed, method=method, **choice["config"],
                                inner_accuracy=choice.get("inner_accuracy"), reporting_count=choice["reporting_count"]))
            if "inner_accuracy" in choice:
                candidates = [r for r in development_rows if r["seed"] == seed]
                if method in PRIMARY:
                    candidates = [r for r in candidates if r["method"] == method]
                elif method.startswith("PSS ") and method.endswith("-fold"):
                    folds = int(method.split()[1].split("-")[0])
                    candidates = [r for r in candidates if r.get("estimator") == "pss" and r.get("folds") == folds]
                elif method == "PSS matched-budget":
                    candidates = [r for r in candidates if r.get("estimator") == "pss" and r.get("folds") in [3, 5]]
                elif method == "PSS tau=0.99":
                    candidates = [r for r in candidates if r.get("estimator") == "pss" and r.get("folds") == 3 and r.get("tau") == .99]
                else:
                    raise AssertionError(method)
                best = min(candidates, key=lambda r: (-r["mean_accuracy"], -r.get("tau", 0),
                                                      r.get("folds", 0), r.get("k", 0), r["config"]))
                assert best["config"] == key(choice["config"])
                assert abs(best["mean_accuracy"]-choice["inner_accuracy"]) < 1e-12
        for method in PRIMARY+ABLATIONS+["All features"]:
            slug = method.replace(" ", "_").replace("=", "")
            record = json.loads((folder/"outer"/(slug+".json")).read_text())
            path_record = json.loads((folder/"outer"/(slug+"_path.tmp.json")).read_text())
            assert record["path"] == path_record["path"]
            assert record["threshold"] == threshold
            assert record["path"]["n_selection"] == (0 if method == "All features" else len(tr))
            if method != "All features":
                assert record["config"] == lock["choices"][method]["config"]
            previous = []
            assert len(record["path"]["history"]) == len(record["metrics"]) == (1 if method == "All features" else 20)
            for step, metric in zip(record["path"]["history"], record["metrics"]):
                assert len(step["features"]) == len(set(step["features"])) == step["step"]
                if method != "All features":
                    assert step["features"][:-1] == previous
                previous = step["features"]
                assert metric["selected_features"] == ",".join(names[j] for j in step["features"])
                pred_path = out/metric["prediction_file"]
                if str(pred_path) not in prediction_cache:
                    with np.load(pred_path) as pred:
                        prediction_cache[str(pred_path)] = dict(
                            accuracy=accuracy_score(ty, pred["prediction"]),
                            balanced_accuracy=balanced_accuracy_score(ty, pred["prediction"]),
                            auc=roc_auc_score(ty, pred["decision"]))
                row = frame[(frame.seed == seed) & (frame.method == method) & (frame.features == step["step"])].iloc[0]
                for field, value in prediction_cache[str(pred_path)].items():
                    assert abs(value-row[field]) < 1e-12
                checked += 1
                predictions += len(te)
    matched_paths = 0
    reference = ROOT/"results/energy_sameperiod_20261005/checkpoints"
    if reference.exists():
        for seed in SEEDS:
            old = json.loads((reference/f"dev_{seed}_1e-05_paths.json").read_text())
            for p in (out/f"seed_{seed}"/"development").glob("*.json"):
                new = json.loads(p.read_text())
                c = new["config"]
                if c["estimator"] != "pss" or c["folds"] != 3 or c["tau"] not in [.9, .95, .99]:
                    continue
                previous = next(r for r in old if r["config"]["method"] == "PSS SC-CV" and
                                r["config"]["n_min"] == 5 and r["config"]["tau"] == c["tau"])
                for a, b in zip(previous["history"], new["path"]["history"]):
                    assert a["features"] == b["features"] and a["ell"] == b["ell"]
                    assert abs(a["score"]-b["score"]) < 1e-10
                    assert abs(a["cv_score"]-b["cv_score"]) < 1e-10
                matched_paths += 1
        assert matched_paths == 15
    return (dict(passed=True, curve_points=checked, predictions_verified=predictions,
                 unique_classifier_fits=len(prediction_cache), data_and_code_hashes_verified=True,
                 complete_candidate_grids_verified=True, training_only_locks_recomputed=True,
                 historical_same_configuration_paths_matched=matched_paths),
            pd.DataFrame(development_rows), pd.DataFrame(choices))


def curve(ax, frame, methods, field="accuracy", factor=100, band="sd"):
    for method in methods:
        stats = frame[frame.method == method].groupby("features")[field].agg(["mean", "std", "min", "max"])
        x = stats.index.to_numpy()
        y = factor*stats["mean"].to_numpy()
        lower = factor*stats["min"].to_numpy() if band == "range" else y-factor*stats["std"].to_numpy()
        upper = factor*stats["max"].to_numpy() if band == "range" else y+factor*stats["std"].to_numpy()
        ax.plot(x, y, color=COLORS[method], lw=2 if method == PRIMARY[0] else 1.4,
                ls="--" if method in ["Univariate tuned", "PSS ell=1", "PSS matched-budget"] else "-",
                marker="o" if method == PRIMARY[0] else None, markevery=3, ms=3, label=LABELS[method])
        ax.fill_between(x, lower, upper, color=COLORS[method], alpha=.08, linewidth=0)
    ax.set_xlabel("Selected features")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlim(.6, 20.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.20)


def save_figure(fig, out, name):
    (out/"plots").mkdir(exist_ok=True)
    pdf = ROOT/"output/pdf"/out.name
    pdf.mkdir(parents=True, exist_ok=True)
    fig.savefig(out/"plots"/(name+".png"), dpi=240, bbox_inches="tight")
    fig.savefig(pdf/(name+".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    frame = pd.read_csv(out/"evaluation_metrics.csv")
    verified, development, choices = audit(out, frame)
    save_json(verified, out/"audit.json")
    save_json(dict(python=platform.python_version(), platform=platform.platform(),
                   packages={name: importlib.metadata.version(name)
                             for name in ["numpy", "scipy", "pandas", "scikit-learn", "matplotlib"]}),
              out/"environment.json")
    save_csv(development, out/"development_summary.csv")
    save_csv(choices, out/"locked_settings.csv")
    stats = frame.groupby(["method", "features"], as_index=False).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"), auc_mean=("auc", "mean"))
    save_csv(stats, out/"accuracy_curves.csv")
    save_csv(stats[stats.features.isin([5, 10, 20, 25])], out/"summary_at_5_10_20.csv")
    selected = []
    for _, c in choices.iterrows():
        selected.append(frame[(frame.seed == c.seed) & (frame.method == c.method) &
                              (frame.features == c.reporting_count)].iloc[0])
    save_csv(pd.DataFrame(selected), out/"inner_selected_count_metrics.csv")
    comparisons = []
    for method in PRIMARY[1:]+ABLATIONS:
        for count in COUNTS:
            a = frame[(frame.method == PRIMARY[0]) & (frame.features == count)].set_index("seed")
            b = frame[(frame.method == method) & (frame.features == count)].set_index("seed")
            delta = 100*(a.accuracy-b.accuracy)
            comparisons.append(dict(method=method, features=count, mean_difference_pp=delta.mean(),
                                    sd_difference_pp=delta.std(), positive_splits=int((delta > 0).sum()),
                                    ties=int((delta == 0).sum())))
    save_csv(pd.DataFrame(comparisons), out/"paired_accuracy_differences.csv")
    diag = frame[frame.method.str.startswith("PSS")].copy()
    diag["training_skipped"] = 1-diag.training_coverage
    diag["conditional_training_skipped"] = 1-diag.conditional_training_coverage
    p = diag.train_positive
    diag["label_entropy"] = -p*np.log(p)-(1-p)*np.log(1-p)
    diag["score_outside_mi_bounds"] = (diag.score < 0) | (diag.score > diag.label_entropy)
    save_csv(diag, out/"pss_diagnostics.csv")
    diag_summary = diag.groupby("method", as_index=False).agg(
        ell1_fraction=("ell", lambda s: s.eq(1).mean()), mean_ell=("ell", "mean"),
        ell5_fraction=("ell", lambda s: s.eq(5).mean()),
        minimum_pooled_stable=("pooled_stable_5", "min"),
        minimum_component_stable=("minimum_stable_5", "min"),
        maximum_conditional_training_skip=("conditional_training_skipped", "max"),
        score_outside_mi_bounds=("score_outside_mi_bounds", "sum"),
        fallback_count=("fallback", lambda s: s.fillna(False).sum()))
    save_csv(diag_summary, out/"diagnostic_summary.csv")
    timing = frame.groupby(["seed", "method"], as_index=False).first().groupby("method", as_index=False).agg(
        mean_seconds=("selection_seconds", "mean"), sd_seconds=("selection_seconds", "std"))
    save_csv(timing, out/"selection_runtime.csv")
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(5.2, 3.8), layout="constrained")
    curve(ax, frame, PRIMARY)
    ax.axhline(100*frame[frame.method == "All features"].accuracy.mean(), color="#666666", ls=":", lw=1.2,
               label="All 25 features")
    ax.set_ylabel("Test accuracy (%)")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    save_figure(fig, out, "energy_expanded_accuracy")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained", sharey=True)
    curve(axes[0], frame, [PRIMARY[0], "PSS 3-fold", "PSS 5-fold", "PSS 10-fold", "PSS matched-budget"])
    curve(axes[1], frame, [PRIMARY[0], "PSS tau=0.99", "PSS ell=1", "PSS ell=2", "KL k=1"])
    for ax, title in zip(axes, ["Fold count and search budget", "Constraints and fixed partitions"]):
        ax.set_title(title, fontsize=11)
        ax.legend(loc="lower right", frameon=False, fontsize=8)
    axes[0].set_ylabel("Test accuracy (%)")
    save_figure(fig, out, "energy_expanded_ablation")
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0), layout="constrained")
    curve(axes[0, 0], frame, [PRIMARY[0], "PSS 3-fold", "PSS 5-fold", "PSS 10-fold"], "ell", 1, "range")
    axes[0, 0].set_ylabel(r"Selected $\ell$")
    axes[0, 0].set_ylim(.8, 5.2)
    axes[0, 0].set_yticks([1, 2, 3, 4, 5])
    axes[0, 0].legend(frameon=False, fontsize=8)
    part = diag[diag.method == PRIMARY[0]]
    for ax, fields, ylabel in [(axes[0, 1], ["pooled_stable_5", "minimum_stable_5"], "Stable validation coverage (%)"),
                               (axes[1, 0], ["training_skipped", "conditional_training_skipped"], "Skipped training points (%)")]:
        for field, color, label in zip(fields, ["#164DDB", "#D23F2A"], ["Pooled", "Worst component"]):
            s = part.groupby("features")[field].agg(["mean", "min", "max"])
            x = s.index.to_numpy()
            ax.plot(x, 100*s["mean"].to_numpy(), color=color, label=label, lw=1.6)
            ax.fill_between(x, 100*s["min"].to_numpy(), 100*s["max"].to_numpy(), color=color, alpha=.1)
        ax.set_xlabel("Selected features")
        ax.set_ylabel(ylabel)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_xlim(.6, 20.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.2)
        ax.legend(frameon=False, fontsize=8)
    axes[0, 1].set_ylim(0, 101)
    axes[1, 0].set_ylim(bottom=-.02)
    grid = development[development.method == PRIMARY[0]].pivot_table(index="tau", columns="folds", values="mean_accuracy")
    ax = axes[1, 1]
    heat = ax.imshow(100*grid.to_numpy(), aspect="auto", cmap="YlGnBu", origin="lower")
    ax.set_xticks(range(len(grid.columns)), [str(int(v)) for v in grid.columns])
    ax.set_yticks(range(len(grid.index)), [f"{v:.2f}" for v in grid.index])
    ax.set_xlabel("CV folds")
    ax.set_ylabel(r"Coverage threshold $\tau$")
    ax.set_title("Inner-validation accuracy (%)", fontsize=10)
    for i in range(len(grid)):
        for j in range(len(grid.columns)):
            value = 100*grid.iloc[i, j]
            rgb = np.array(heat.cmap(heat.norm(value))[:3])
            linear = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
            luminance = float(linear @ np.array([.2126, .7152, .0722]))
            color = "white" if luminance < .179 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=9)
    save_figure(fig, out, "energy_expanded_diagnostics")
    report = ["# Energy coverage and fold expansion", "",
        "Exploratory nested evaluation of the same five previously inspected random holdouts. All feature-selection paths were rebuilt using training data, with fixed noise=1e-5 and n_min=5. This is not a new independent confirmation dataset.", "",
        "## Protocol", "",
        "19,735 observations and 25 predictors. Outer train/test: 13,814/5,921; inner train/validation: 9,669/4,145. All selection rows retained. Training-only median target and range normalization; unjittered predictors for the fixed RBF SVM. Canonical PSS v2 and N_eff unchanged. Common ell for pooled and conditional entropy scores; no forced ell>1.", "",
        "PSS: tau in {0.80,0.85,0.90,0.95,0.99}, K in {3,5,10}: 15 configurations. KL entropy difference, Ross joint MI, and univariate Ross: k in {1,2,3,5,7,10,15,20,30,50}: 10 configurations each. PSS matched-budget restricts K to {3,5}, also 10 configurations. All choices use mean inner-validation accuracy at 5,10,20 features and are locked before new outer evaluation. There is no separate classifier tuning.", "",
        "## Accuracy", "", "Mean +/- sample SD (%). Overlapping splits, not independent-sample confidence intervals.", "",
        "| Method | 5 features | 10 features | 20 features |", "|---|---:|---:|---:|"]
    for method in PRIMARY+ABLATIONS:
        values = []
        for count in COUNTS:
            r = stats[(stats.method == method) & (stats.features == count)].iloc[0]
            values.append(f"{100*r.accuracy_mean:.2f} +/- {100*r.accuracy_sd:.2f}")
        report.append("| "+method+" | "+" | ".join(values)+" |")
    r = stats[stats.method == "All features"].iloc[0]
    displayed_choices = choices.loc[choices.method.isin(PRIMARY),
                                   ["seed", "method", "folds", "tau", "k", "inner_accuracy", "reporting_count"]]
    report += ["", f"All 25 features: {100*r.accuracy_mean:.2f} +/- {100*r.accuracy_sd:.2f}%.", "",
               "## Locked settings", "", displayed_choices.fillna("-").to_markdown(index=False), "",
               "## Coverage diagnostics", "", diag_summary.to_markdown(index=False), "",
               "## Selection timing", "", timing.to_markdown(index=False), "",
               "Cold forward-selection seconds per unique outer configuration, excluding post-selection diagnostics and classifier fits. Identical policies reuse the same selected path. Three concurrent workers; not isolated hardware benchmarks. Inner searches reuse exact subset calculations and are separately checkpointed.", "",
               "## Limits", "",
               "- Increasing ell is not itself evidence of better prediction or MI estimation. Covered NLL averages different subsets at different ell; relaxing coverage can improve this objective by excluding difficult points.",
               "- These are same-period interpolation results in one household, not future-time or new-household performance. Tests were previously inspected during method development, although current settings never use outer outcomes in their selection objective.",
               "- Training jitter and high coverage do not establish the continuous-density theorem assumptions. Raw entropy differences are selection scores, not calibrated MI.",
               "- Comparisons with the previous experiment also change its tuned jitter/n_min to common fixed values. The new fold-specific policies provide the controlled comparison within this protocol.",
               "- Fixed ell and k=1 are ablations. All tuned baseline results, unfavorable settings, and search-budget control are retained.", "",
               "- Search is limited to the declared grids. A choice of k=50 or ell=5 is at its grid boundary, not evidence of global optimality.", "",
               "## Audit", "", json.dumps(verified, indent=2), "",
               "## Figures", "",
               "Accuracy and ablation bands show +/- one sample SD. Diagnostic bands show observed minimum-to-maximum, with the mean line. The heatmap uses inner validation only. Vector PDFs are saved under output/pdf with the same experiment directory name."]
    (out/"REPORT.md").write_text("\n".join(report)+"\n")
    print(json.dumps(verified, indent=2))
    print(stats[stats.features.isin(COUNTS)].to_string(index=False))


if __name__ == "__main__":
    main()
