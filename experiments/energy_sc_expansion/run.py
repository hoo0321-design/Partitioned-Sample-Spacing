"""Prespecified exploratory nested holdouts with restartable per-config checkpoints."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from core import ROOT, PRIMARY, TAUS, FOLDS, KS, NOISE, N_MIN, COUNTS, ELLS, Selector, configurations, original

SEEDS = [42, 43, 44, 45, 46]
ABLATIONS = ["PSS 3-fold", "PSS 5-fold", "PSS 10-fold", "PSS matched-budget",
             "PSS tau=0.99", "PSS ell=1", "PSS ell=2", "KL k=1"]


def key(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(value, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def save_csv(frame, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def load_data():
    frame = pd.read_csv(ROOT/"data/energydata_complete.csv")
    names = [c for c in frame if c not in ["date", "Appliances", "rv1", "rv2"]]
    assert len(frame) == 19735 and len(names) == 25
    return frame[names].to_numpy(float), frame.Appliances.to_numpy(float), names


def splits(seed):
    tr, te = original.split_indices(np.arange(19735), seed)
    it, iv = original.split_indices(tr, seed+1000)
    return tr, te, it, iv


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    sources = ["PSS/pss_v2.py", "PSS/pss_v2.cpp", "experiments/energy_sameperiod/core.py",
               "experiments/energy_v2/core.py", "experiments/energy_sc_expansion/core.py",
               "experiments/energy_sc_expansion/run.py"]
    protocol = dict(seeds=SEEDS, data_sha256=sha(ROOT/"data/energydata_complete.csv"),
        code_sha256={p: sha(ROOT/p) for p in sources}, platform=platform.platform(),
        purpose="Exploratory SC-CV coverage/fold expansion and tuned entropy-difference kNN comparison; NOT confirmation on never-inspected tests.",
        split="Same five random 70:30 holdouts and nested 70:30 training/validation holdouts as previous same-period study. Repeated/overlapping test sets. No seed selected or removed.",
        target="Appliances > training median, ties class 0; threshold independently fit in inner and outer training",
        selection="All training rows; training min/max scaling, fixed Gaussian jitter SD=1e-5, RNG seeds unchanged. No test preprocessing information.",
        classifier="Identical fixed RBF SVM for all: C=1, gamma=1/feature_count, training sample-SD scaling; raw unjittered predictors. No class weights.",
        density="Canonical v2 smoothed subgrid/N_eff, no normalization or extrapolation, no change to estimator",
        sc_cv="For each candidate feature subset: pooled covered NLL, constrained pooled stable validation coverage >= tau, fixed n_min=5. Same ell for pooled and conditional entropy scores. Candidate ell=1..5. No forced ell>1.",
        folds="Stratified training-only folds, K=3,5,10. Conditional density coverage diagnosed but not an additional constraint.",
        fallback="No feasible ell: maximize stable coverage, then minimize covered NLL, then smallest ell. Report all fallbacks.",
        primary_configs={m: configurations(m) for m in PRIMARY}, noise=NOISE, n_min=N_MIN,
        ell_grid=ELLS, tau_grid=TAUS, fold_grid=FOLDS, k_grid=KS,
        selection_objective="Mean inner-validation accuracy at 5,10,20 features. Lock ALL policies before any new test evaluation.",
        tie_break="Prefer higher tau, fewer folds, smaller k, then lexical config; never outer scores",
        search_budget="15 PSS configs; 10 each for KL, Ross, univariate. Also report PSS matched-budget using only K=3,5 (10 configs). Equal counts are not equal compute.",
        ablations="PSS K=3,5,10 each tunes tau separately; matched-budget K=3,5; tau=.99,K=3; fixed ell=1/2; historical KL k=1; all25 SVM. All fixed noise and n_min; fresh forward selection.",
        checkpoint="Paths are selected using only respective training data; classifiers cached by identical subset only, per seed and inner/outer split separately. Cold selector timings at outer fit; diagnostic cost separately excluded.",
        interpretation="MI-like scores are not calibrated MI. Random same-period interpolation, not forecasting or new-household generalization. Five-split SD is descriptive, not independent-dataset uncertainty.")
    destination = out/"protocol.json"
    if destination.exists():
        assert json.loads(destination.read_text()) == protocol, "Frozen protocol changed"
    else:
        save_json(protocol, destination)
    for seed in SEEDS:
        folder = out/f"seed_{seed}"
        folder.mkdir(exist_ok=True)
        for name in ["development", "outer", "inner_classifier", "outer_classifier"]:
            (folder/name).mkdir(exist_ok=True)
        tr, te, it, iv = splits(seed)
        record = dict(outer_train=tr.tolist(), outer_test=te.tolist(), inner_train=it.tolist(), inner_validation=iv.tolist())
        p = folder/"split.json"
        if p.exists():
            assert json.loads(p.read_text()) == record
        else:
            save_json(record, p)


def classify_cached(folder, train_x, y, query_x, query_y, features, predictions=False):
    stem = "_".join(map(str, sorted(features)))
    p = folder/(stem+".json")
    if p.exists():
        return json.loads(p.read_text()), folder/(stem+".npz")
    result = original.classify(train_x, y, query_x, query_y, features)
    metrics = {k: v for k, v in result.items() if k not in ["prediction", "decision"]}
    if predictions:
        destination = folder/(stem+".npz")
        temporary = destination.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, prediction=result["prediction"], decision=result["decision"])
        temporary.replace(destination)
    save_json(metrics, p)
    return metrics, folder/(stem+".npz")


def develop(out_string, seed):
    out = Path(out_string)
    folder = out/f"seed_{seed}"
    x, target, _ = load_data()
    _, _, train, valid = splits(seed)
    y, vy, threshold = original.labels(target[train], target[valid])
    z, active = original.selection_data(x[train], NOISE, seed+3000)
    selector = Selector(z, y, active, seed+4000)
    start = time.perf_counter()
    for method in PRIMARY:
        for i, config in enumerate(configurations(method)):
            tag = hashlib.sha256(key(config).encode()).hexdigest()[:16]
            p = folder/"development"/(tag+".json")
            if p.exists():
                assert json.loads(p.read_text())["config"] == config
                continue
            path = selector.path(config)
            metrics = []
            for count in COUNTS:
                values, _ = classify_cached(folder/"inner_classifier", x[train], y, x[valid], vy,
                                             path["history"][count-1]["features"])
                metrics.append(dict(features=count, **values))
            save_json(dict(method=method, config=config, threshold=threshold, path=path, metrics=metrics), p)
            print(json.dumps(dict(stage="development", seed=seed, method=method, config=i+1,
                                  of=len(configurations(method)), seconds=round(time.perf_counter()-start))), flush=True)
    records = [json.loads(p.read_text()) for p in sorted((folder/"development").glob("*.json"))]
    assert len(records) == sum(len(configurations(m)) for m in PRIMARY)
    def best(options):
        def order(r):
            c = r["config"]
            return (-np.mean([v["accuracy"] for v in r["metrics"]]), -c.get("tau", 0),
                    c.get("folds", 0), c.get("k", 0), key(c))
        chosen = min(options, key=order)
        return dict(config=chosen["config"], inner_accuracy=float(np.mean([v["accuracy"] for v in chosen["metrics"]])),
                    reporting_count=min(chosen["metrics"], key=lambda r: (-r["accuracy"], r["features"]))["features"])
    choices = {method: best([r for r in records if r["method"] == method]) for method in PRIMARY}
    pss = [r for r in records if r["method"] == PRIMARY[0]]
    for folds in FOLDS:
        choices[f"PSS {folds}-fold"] = best([r for r in pss if r["config"]["folds"] == folds])
    choices["PSS matched-budget"] = best([r for r in pss if r["config"]["folds"] in [3, 5]])
    choices["PSS tau=0.99"] = best([r for r in pss if r["config"]["folds"] == 3 and r["config"]["tau"] == .99])
    for method, config in [("PSS ell=1", dict(estimator="fixed", ell=1)),
                           ("PSS ell=2", dict(estimator="fixed", ell=2)), ("KL k=1", dict(estimator="kl", k=1))]:
        choices[method] = dict(config=config, reporting_count=choices[PRIMARY[0]]["reporting_count"],
                               note="Fixed ablation; reporting count inherited, not test-selected")
    lock = dict(seed=seed, choices=choices, locked_before_test=True, protocol_sha256=sha(out/"protocol.json"),
                development_sha256={p.name: sha(p) for p in sorted((folder/"development").glob("*.json"))})
    p = folder/"lock.json"
    if p.exists():
        assert json.loads(p.read_text()) == lock
    else:
        save_json(lock, p)
    print(json.dumps(dict(stage="locked", seed=seed, choices=choices)), flush=True)


def outer(out_string, seed):
    out = Path(out_string)
    folder = out/f"seed_{seed}"
    lock = json.loads((folder/"lock.json").read_text())
    assert lock["protocol_sha256"] == sha(out/"protocol.json")
    x, target, names = load_data()
    train, test, _, _ = splits(seed)
    y, ty, threshold = original.labels(target[train], target[test])
    z, active = original.selection_data(x[train], NOISE, seed+5000)
    path_cache = {}
    for method in PRIMARY+ABLATIONS+["All features"]:
        slug = method.replace(" ", "_").replace("=", "")
        p = folder/"outer"/(slug+".json")
        if p.exists():
            continue
        config = lock["choices"][method]["config"] if method != "All features" else {}
        if method == "All features":
            path = dict(history=[dict(step=25, features=list(range(25)))], n_selection=0, selection_seconds=0.)
        elif key(config) in path_cache:
            path = path_cache[key(config)]
        else:
            selector = Selector(z, y, active, seed+6000)
            path = selector.path(config)
            selector.diagnostics(path, config)
            path_cache[key(config)] = path
        # Persist train-only selections before reading classifier test outcomes.
        save_json(dict(method=method, config=config, threshold=threshold, path=path), p.with_name(slug+"_path.tmp.json"))
        metrics = []
        for row in path["history"]:
            values, pred_path = classify_cached(folder/"outer_classifier", x[train], y, x[test], ty, row["features"], True)
            metrics.append(dict(seed=seed, method=method, features=row["step"], threshold=threshold,
                config=key(config), selected_features=",".join(names[j] for j in row["features"]),
                prediction_file=str(pred_path.relative_to(out)), selection_seconds=path["selection_seconds"],
                train_n=len(train), test_n=len(test), train_positive=float(y.mean()), **values,
                **{k: v for k, v in row.items() if k not in ["step", "feature", "features"]}))
        save_json(dict(method=method, config=config, threshold=threshold, path=path, metrics=metrics), p)
        print(json.dumps(dict(stage="test", seed=seed, method=method,
                              selection_seconds=round(path["selection_seconds"], 2))), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "develop", "evaluate", "all"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    out = args.output.resolve()
    prepare(out)
    if args.stage in ["develop", "all"]:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(develop, str(out), seed) for seed in SEEDS]
            for future in as_completed(futures):
                future.result()
    if args.stage in ["evaluate", "all"]:
        assert all((out/f"seed_{seed}"/"lock.json").exists() for seed in SEEDS)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(outer, str(out), seed) for seed in SEEDS]
            for future in as_completed(futures):
                future.result()
        rows = []
        for seed in SEEDS:
            for p in sorted((out/f"seed_{seed}"/"outer").glob("*.json")):
                record = json.loads(p.read_text())
                rows.extend(record.get("metrics", []))
        save_csv(pd.DataFrame(rows), out/"evaluation_metrics.csv")
        print("ENERGY_SC_EXPANSION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
