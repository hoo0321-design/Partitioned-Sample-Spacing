"""Audit saved predictions and produce descriptive, non-cherry-picked summaries."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from core import MAIN, ABLATIONS, ROOT, labels
from run import SEEDS, load_data, sha, save_json, save_csv, splits

COLORS = {"PSS SC-CV": "#164DDB", "Joint Ross MI": "#DF4F22",
          "Univariate Ross MI": "#27866C", "All features": "#676B73",
          "PSS default SC-CV": "#9771BD", "PSS component guard": "#BA528D",
          "PSS ell=1": "#555555", "PSS ell=2": "#D88D1D",
          "PSS global SC-CV": "#27866C", "KL difference k=1": "#DF4F22"}
LABELS = {"PSS SC-CV": "PSS + SC-CV", "Joint Ross MI": "Joint kNN-MI",
          "Univariate Ross MI": "Univariate kNN ranking", "All features": "All 25 features",
          "PSS default SC-CV": "Default SC-CV", "PSS component guard": "All-component guard",
          "PSS ell=1": r"PSS, $\ell=1$", "PSS ell=2": r"PSS, $\ell=2$",
          "PSS global SC-CV": "Global SC-CV", "KL difference k=1": "KL difference, k=1"}


def audit(out, frame):
    _, target, names = load_data()
    protocol = json.loads((out/"protocol.json").read_text())
    assert sha(ROOT/"data/energydata_complete.csv") == protocol["data_sha256"]
    for name, checksum in protocol["code_sha256"].items():
        assert sha(ROOT/name) == checksum, name
    assert len(frame) == len(SEEDS) * (len(MAIN+ABLATIONS)*20+1)
    checked_predictions = 0
    checked_curves = 0
    for seed in SEEDS:
        lock = json.loads((out/"checkpoints"/f"lock_{seed}.json").read_text())
        assert lock["locked_before_test"]
        assert lock["protocol_sha256"] == sha(out/"protocol.json")
        for noise, checksum in lock["development_sha256"].items():
            assert sha(out/"checkpoints"/f"dev_{seed}_{float(noise):g}.csv") == checksum
        tr, te, it, iv = splits(seed)
        saved = json.loads((out/"checkpoints"/f"split_{seed}.json").read_text())
        for key, values in [("outer_train", tr), ("outer_test", te), ("inner_train", it), ("inner_validation", iv)]:
            np.testing.assert_array_equal(saved[key], values)
        y, ty, threshold = labels(target[tr], target[te])
        for p in sorted((out/"checkpoints").glob(f"outer_{seed}_*_predictions.csv")):
            pred = pd.read_csv(p, float_precision="round_trip")
            path = json.loads(Path(str(p).replace("_predictions.csv", "_path.json")).read_text())
            assert path["threshold"] == threshold
            assert path["n_selection"] == (0 if path["method"] == "All features" else len(tr))
            previous = []
            for h in path["history"]:
                if path["method"] != "All features":
                    assert h["features"][:-1] == previous
                assert len(h["features"]) == len(set(h["features"])) == h["step"]
                previous = h["features"]
            for (method, count), group in pred.groupby(["method", "features"]):
                group = group.sort_values("row_id")
                np.testing.assert_array_equal(group.row_id, te)
                np.testing.assert_array_equal(group.truth, ty)
                row = frame[(frame.seed == seed) & (frame.method == method) & (frame.features == count)].iloc[0]
                for name, value in [("accuracy", accuracy_score(ty, group.prediction)),
                                    ("balanced_accuracy", balanced_accuracy_score(ty, group.prediction)),
                                    ("auc", roc_auc_score(ty, group.decision))]:
                    assert abs(row[name]-value) < 1e-12, (seed, method, count, name)
                checked_curves += 1
                checked_predictions += len(group)
    return dict(passed=True, seeds=SEEDS, curve_points=checked_curves,
                predictions_verified=checked_predictions, data_and_code_hashes_verified=True,
                locked_hyperparameters_verified=True, training_only_labels_and_indices_verified=True)


def curve(ax, frame, methods, field="accuracy", percent=True, bands=True, band="sd"):
    for method in methods:
        part = frame[frame.method == method]
        stats = part.groupby("features")[field].agg(["mean", "std", "min", "max"])
        factor = 100 if percent else 1
        x = stats.index.to_numpy()
        y = stats["mean"].to_numpy()*factor
        s = stats["std"].to_numpy()*factor
        style = "--" if method in ["PSS ell=1", "PSS global SC-CV", "Univariate Ross MI"] else "-"
        ax.plot(x, y, style, color=COLORS[method], lw=2 if method == "PSS SC-CV" else 1.5,
                marker="o" if method == "PSS SC-CV" else None, ms=3, markevery=3, label=LABELS[method])
        if bands:
            lower = stats["min"].to_numpy()*factor if band == "range" else y-s
            upper = stats["max"].to_numpy()*factor if band == "range" else y+s
            ax.fill_between(x, lower, upper, color=COLORS[method], alpha=.10, linewidth=0)
    ax.set_xlim(.6, 20.4)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("Selected features")
    ax.grid(alpha=.20)
    ax.spines[["top", "right"]].set_visible(False)


def save_figure(fig, out, name):
    fig.savefig(out/"plots"/(name+".png"), dpi=240, bbox_inches="tight")
    pdf_dir = ROOT/"output/pdf"/out.name
    pdf_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_dir/(name+".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    frame = pd.read_csv(out/"evaluation_metrics.csv")
    audit_result = audit(out, frame)
    save_json(audit_result, out/"audit.json")
    summary = frame.groupby(["method", "features"], as_index=False).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"), balanced_accuracy_sd=("balanced_accuracy", "std"),
        auc_mean=("auc", "mean"), auc_sd=("auc", "std"), repeats=("seed", "nunique"))
    save_csv(summary, out/"accuracy_curves.csv")
    save_csv(summary[summary.features.isin([5, 10, 20, 25])], out/"summary_at_5_10_20.csv")
    choices = []
    selected_count_rows = []
    for seed in SEEDS:
        lock = json.loads((out/"checkpoints"/f"lock_{seed}.json").read_text())
        for method in MAIN:
            choice = lock["choices"][method]
            choices.append(dict(seed=seed, **choice["config"],
                                inner_accuracy=choice["inner_accuracy"], reporting_count=choice["reporting_count"]))
            selected_count_rows.append(frame[(frame.seed == seed) & (frame.method == method) &
                                             (frame.features == choice["reporting_count"])].iloc[0])
    choices = pd.DataFrame(choices)
    save_csv(choices, out/"locked_settings.csv")
    development = pd.read_csv(out/"development_scores.csv")
    development_summary = development.groupby(["seed", "method", "config"], as_index=False).agg(
        mean_validation_accuracy=("accuracy", "mean"), mean_validation_balanced_accuracy=("balanced_accuracy", "mean"),
        counts_evaluated=("features", "size"))
    save_csv(development_summary, out/"development_configuration_summary.csv")
    save_csv(pd.DataFrame(selected_count_rows), out/"inner_selected_count_metrics.csv")
    timing = frame.groupby(["seed", "method"], as_index=False).first()
    runtime = timing.groupby("method", as_index=False).agg(mean_seconds=("selection_seconds", "mean"),
                                                            sd_seconds=("selection_seconds", "std"))
    save_csv(runtime, out/"selection_runtime.csv")
    diag = frame[frame.method.str.startswith("PSS")].copy()
    diag["training_skipped_fraction"] = 1-diag.training_coverage
    diag["conditional_training_skipped_fraction"] = 1-diag.conditional_training_coverage
    p = diag.train_positive
    diag["label_entropy_nats"] = -p*np.log(p)-(1-p)*np.log(1-p)
    diag["score_outside_binary_mi_bounds"] = (diag.score < 0) | (diag.score > diag.label_entropy_nats)
    save_csv(diag, out/"pss_diagnostics.csv")
    overlaps = []
    for seed in SEEDS:
        for count in range(1, 21):
            a = frame[(frame.seed == seed) & (frame.method == "PSS SC-CV") & (frame.features == count)].iloc[0]
            b = frame[(frame.seed == seed) & (frame.method == "PSS ell=1") & (frame.features == count)].iloc[0]
            left, right = set(a.selected_features.split(",")), set(b.selected_features.split(","))
            overlaps.append(dict(seed=seed, features=count, overlap_fraction=len(left & right)/count,
                                 identical_set=left == right))
    overlaps = pd.DataFrame(overlaps)
    save_csv(overlaps, out/"sc_vs_ell1_feature_overlap.csv")
    comparisons = []
    for method in MAIN[1:]+ABLATIONS:
        for count in [5, 10, 20]:
            primary = frame[(frame.method == "PSS SC-CV") & (frame.features == count)].set_index("seed")
            other = frame[(frame.method == method) & (frame.features == count)].set_index("seed")
            difference = primary.accuracy-other.accuracy
            comparisons.append(dict(comparator=method, features=count, mean_difference_pp=100*difference.mean(),
                                    sd_difference_pp=100*difference.std(), positive_splits=int((difference > 0).sum()),
                                    ties=int((difference == 0).sum())))
    save_csv(pd.DataFrame(comparisons), out/"paired_accuracy_differences.csv")

    (out/"plots").mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(4.6, 3.4), layout="constrained")
    curve(ax, frame, MAIN)
    all_accuracy = frame[frame.method == "All features"].accuracy.mean()*100
    ax.axhline(all_accuracy, color=COLORS["All features"], ls=":", lw=1.3, label=LABELS["All features"])
    ax.set_ylabel("Test accuracy (%)")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    save_figure(fig, out, "energy_accuracy")

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), layout="constrained", sharey=True)
    groups = [["PSS SC-CV", "PSS default SC-CV", "PSS ell=1", "PSS ell=2"],
              ["PSS SC-CV", "PSS component guard", "PSS global SC-CV", "KL difference k=1"]]
    for ax, methods, title in zip(axes, groups, ["Partition selection", "Coverage guard and selection scope"]):
        curve(ax, frame, methods)
        ax.set_title(title, fontsize=11)
        ax.legend(frameon=False, loc="lower right", fontsize=8)
    axes[0].set_ylabel("Test accuracy (%)")
    save_figure(fig, out, "energy_ablation")

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.3), layout="constrained")
    curve(axes[0], frame, ["PSS SC-CV", "PSS component guard", "PSS default SC-CV"], "ell", False, band="range")
    axes[0].set_ylabel(r"Selected $\ell$")
    axes[0].set_yticks([1, 2, 3, 4, 5])
    axes[0].set_ylim(.8, 5.2)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    selected = frame[frame.method == "PSS SC-CV"].copy()
    minima = choices[choices.method == "PSS SC-CV"].set_index("seed").n_min.to_dict()
    selected["pooled_stable"] = [r[f"pooled_stable_{int(minima[r.seed])}"] for _, r in selected.iterrows()]
    selected["minimum_stable"] = [r[f"minimum_stable_{int(minima[r.seed])}"] for _, r in selected.iterrows()]
    for ax, fields, ylabel in [(axes[1], ["pooled_stable", "minimum_stable"], "Stable validation coverage (%)"),
                               (axes[2], ["training_coverage", "conditional_training_coverage"], "Skipped training points (%)")]:
        for field, color, label in zip(fields, ["#164DDB", "#DF4F22"], ["Pooled", "Minimum across components"]):
            values = selected.groupby("features")[field].agg(["mean", "min", "max"])
            center = 100*values["mean"].to_numpy()
            lower = 100*values["min"].to_numpy()
            upper = 100*values["max"].to_numpy()
            if ax is axes[2]:
                center = 100-center
                lower, upper = 100-upper, 100-lower
                label = "Pooled" if field == fields[0] else "Worst conditional model"
            x = values.index.to_numpy()
            ax.plot(x, center, color=color, label=label, lw=1.7)
            ax.fill_between(x, lower, upper, color=color, alpha=.1)
        ax.set_xlabel("Selected features")
        ax.set_ylabel(ylabel)
        ax.set_xlim(.6, 20.4)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.2)
        ax.legend(frameon=False, fontsize=8, loc="lower right" if ax is axes[1] else "upper right")
    axes[1].set_ylim(top=100.5)
    axes[2].set_ylim(bottom=-.02)
    save_csv(selected, out/"primary_pss_diagnostics.csv")
    save_figure(fig, out, "energy_sc_diagnostics")

    report = ["# Energy same-period experiment", "", "## Status", "",
              f"Completed five prespecified random holdouts, seeds {SEEDS}. All {audit_result['curve_points']} curve points and {audit_result['predictions_verified']:,} saved predictions passed the audit.", "",
              "## Protocol", "",
              "19,735 observations, 25 predictors. Outer train/test = 13,814/5,921; inner train/validation = 9,669/4,145. Unstratified random splits; targets use only the current training median. All rows used for feature selection. Train-only range scaling and Gaussian tie smoothing. Fixed SVM: C=1, gamma=1/subset size, sample-SD scaling, no class weights.", "",
              "Each primary selector receives 12 configurations evaluated on the same inner holdout. Settings are locked before each outer test. All 1..20 feature counts are reported; no test-selected count or seed. SC-CV is pooled, subsetwise, common ell for all entropy components. The all-component guard is an ablation, not the main definition.", "",
              "## Accuracy", "", "Mean +/- sample SD (%); five overlapping splits, not independent-replicate confidence intervals.", "",
              "| Method | 5 features | 10 features | 20 features |", "|---|---:|---:|---:|"]
    for method in MAIN+ABLATIONS:
        values = []
        for count in [5, 10, 20]:
            row = summary[(summary.method == method) & (summary.features == count)].iloc[0]
            values.append(f"{100*row.accuracy_mean:.2f} +/- {100*row.accuracy_sd:.2f}")
        report.append("| "+method+" | "+" | ".join(values)+" |")
    all_row = summary[summary.method == "All features"].iloc[0]
    report.extend(["", f"All 25 features: {100*all_row.accuracy_mean:.2f} +/- {100*all_row.accuracy_sd:.2f}%.", "",
                   "## Selected Settings", "", choices.to_markdown(index=False), "",
                   "## SC-CV Diagnostics", ""])
    primary = selected
    report.extend([f"ell=1 at {int(primary.ell.eq(1).sum())}/100 selected subsets; ell>1 at {int(primary.ell.gt(1).sum())}/100.",
                   f"Upper grid endpoint ell=5 at {int(primary.ell.eq(5).sum())}/100 subsets; optimality beyond the declared 1..5 grid is not assessed.",
                   f"SC-CV fallback at {int(primary.fallback.fillna(False).sum())}/100 subsets.",
                   f"Pooled stable coverage range: {primary.pooled_stable.min():.6f} to {primary.pooled_stable.max():.6f}.",
                   f"Minimum component stable coverage range: {primary.minimum_stable.min():.6f} to {primary.minimum_stable.max():.6f}.",
                   f"Primary raw selection score is outside [0, empirical label entropy] at {int(diag[diag.method.eq('PSS SC-CV')].score_outside_binary_mi_bounds.sum())}/100 subsets; this score is not calibrated MI.",
                   f"Primary SC-CV and fixed ell=1 choose identical 20-feature sets in {int(overlaps[overlaps.features.eq(20)].identical_set.sum())}/5 splits.",
                   "", "## Timing", "", runtime.to_markdown(index=False), "",
                   "Seconds for cold forward selection (including candidate-level SC-CV and conditional coverage diagnostics), excluding post-selection diagnostics and classifier fitting. Inner search is separately recorded in checkpoint time files. Three concurrent workers; not an isolated speed benchmark.", "",
                   "## Interpretation Limits", "",
                   "- These evaluate same-period interpolation for one household, not prediction in a future period or a new home. Adjacent time-series observations can occur on both sides of random splits.",
                   "- Prior temporal evaluation is preserved in the old repository. The altered split, sample budget, preprocessing and SVM settings mean old/new performance differences cannot be attributed to SC-CV alone.",
                   "- The public dataset was studied previously. Held-out means not used in this run's parameter selection, not never inspected historically.",
                   "- Scores are entropy differences, not calibrated MI. No MI-growth or theoretical-consistency figure is produced. Neither high coverage nor accuracy proves consistency.",
                   "- Fixed ell=2 is not SC-CV. Historical KL difference at k=1 is not the tuned mixed-type baseline. All ablations are reported, including unfavorable results.",
                   "- Five overlapping holdouts provide descriptive sensitivity, not statistical significance, equivalence, or evidence across independent datasets.", "",
                   "## Figures", "",
                   "Main: energy_accuracy. Supplement: energy_ablation and energy_sc_diagnostics. Accuracy-plot shading denotes +/- one sample SD across splits. Diagnostic-plot shading denotes the observed minimum-to-maximum across splits, with the mean as the line. All-feature horizontal line is a 25-predictor reference, not a same-budget subset curve.", "",
                   f"Vector PDFs: `{ROOT/'output/pdf'/out.name}`. PNG previews: `{out/'plots'}`.", "",
                   "## Reproduction", "",
                   "Protocol, code/data hashes, split indices, lock files, complete feature paths, validation scores, test predictions, diagnostics and audit.json are retained alongside these outputs.", ""])
    (out/"REPORT.md").write_text("\n".join(report))
    print(summary[summary.features.isin([5, 10, 20, 25])].to_string(index=False))
    print(json.dumps(audit_result, indent=2))


if __name__ == "__main__":
    main()
