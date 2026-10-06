"""Nested random holdouts. Freeze protocol before development and choices before test."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import scipy
import sklearn

from core import (ROOT, MAIN, ABLATIONS, COUNTS, ELLS, KS, MINIMUMS, NOISES, TAUS,
                  Selector, classify, configs, labels, selection_data, split_indices)

SEEDS = [42, 43, 44, 45, 46]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


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
    frame = pd.read_csv(ROOT / "data/energydata_complete.csv")
    names = [c for c in frame if c not in ["date", "Appliances", "rv1", "rv2"]]
    assert len(names) == 25 and len(frame) == 19735
    assert pd.to_datetime(frame.date).is_monotonic_increasing
    x = frame[names].to_numpy(float)
    target = frame.Appliances.to_numpy(float)
    assert np.isfinite(x).all() and np.isfinite(target).all()
    return x, target, names


def splits(seed, n=19735):
    train, test = split_indices(np.arange(n), seed)
    inner_train, inner_valid = split_indices(train, seed+1000)
    return train, test, inner_train, inner_valid


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    sources = [ROOT / p for p in ["PSS/pss_v2.cpp", "PSS/pss_v2.py",
               "experiments/energy_v2/core.py", "experiments/energy_sameperiod/core.py",
               "experiments/energy_sameperiod/run.py"]]
    record = dict(
        version=1, seeds=SEEDS, data_sha256=sha(ROOT / "data/energydata_complete.csv"),
        code_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources},
        data="19735 rows; 25 predictors, excluding date, Appliances, rv1, rv2; retain lights",
        target="Appliances > training median; ties kept as class 0; median fit independently inside inner training and outer training",
        split="5 prespecified unstratified random 70:30 outer holdouts. Within each outer train, one independently seeded 70:30 inner holdout. Indices sampled without labels.",
        estimand="Same-period classification of contemporaneous consumption in one household; not future-time or new-household generalization",
        history="Public dataset and earlier results were used in designing this protocol. These are protocol-held-out tests, not a never-seen external dataset.",
        preprocessing="All selection rows retained. Min/max fit on respective training sample, then Gaussian jitter SD 1e-5 or 1e-4 in range-scaled units. No test inputs used. Classifier uses unjittered predictors.",
        jitter="Explicit smoothing for quantized coordinates, not a theorem guarantee or identified measurement-noise model",
        svm="Fixed for ALL methods: RBF, C=1, gamma=1/subset_size, training sample-SD standardization, no class weighting. Comparable to historical e1071 defaults. Not classifier-tuned.",
        candidates={m: configs(m) for m in MAIN}, ell_grid=ELLS,
        inner_objective="Mean validation accuracy at 5,10,20 features; both noise levels and all selector settings fit only inside each outer training sample",
        tie_break="Smaller noise, larger tau, larger n_min, smaller k, then serialized config; no outer metric used",
        feature_count="Report all 1..20 counts, and separately a count selected by inner accuracy among 5,10,20. Never select count on outer test.",
        sc_cv="For each candidate subset, choose common ell minimizing POOLED covered NLL subject to POOLED stable validation coverage >= tau; same ell used in pooled and class-conditional entropy terms. Three stratified random internal folds.",
        fallback="If no feasible candidate: max pooled stable coverage, then minimum pooled covered NLL, then smallest ell; explicitly flag. No penalty, no forced lower ell bound >1.",
        density="Canonical v2 smoothed-subgrid / N_eff. No extrapolation or normalization. MI-like entropy-difference score not clipped; not claimed calibrated MI.",
        ablations={
            "PSS default SC-CV": "noise=1e-5, tau=.99, n_min=10, otherwise pooled subsetwise SC-CV",
            "PSS component guard": "Inherits primary chosen noise/tau/n_min; constrain minimum stable coverage across pooled and BOTH conditional models, same pooled NLL objective",
            "PSS ell=1": "Inherits primary noise; additive univariate PSS ranking, no partition tuning",
            "PSS ell=2": "Inherits primary noise; fixed partition count at every subset, no SC-CV; diagnostic, never relabeled as SC-CV",
            "PSS global SC-CV": "Inherits primary noise/tau/n_min; choose pooled ell on all 25 predictors once, then reuse in forward selection (historical selection scope)",
            "KL difference k=1": "Historical baseline form H_KL(X)-sum p H_KL(X|Y), fixed historical k=1, noise=1e-5. Not the primary tuned mixed-MI baseline"},
        all_features="Same fixed SVM with all 25 predictors",
        uncertainty="Mean and sample SD over five overlapping random holdouts; NOT a 95% CI or five independent households",
        runtime="Cold full-data forward selection+SC-CV timing; diagnostics excluded. Inner search recorded separately. Concurrent workers mean these are workflow times, not isolated benchmarks.",
        seed_policy="No seed discarded or replaced based on results; all failures must be reported",
        prior_temporal_results="Preserved in old repository results/energy_v2_forward_20261005; no claim current random split supersedes future-time evaluation",
        environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                         sklearn=sklearn.__version__, platform=platform.platform()))
    path = out / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError("Frozen protocol/source changed: use a new output directory")
    save_json(record, path)
    for seed in SEEDS:
        tr, te, it, iv = splits(seed)
        value = dict(outer_train=tr.tolist(), outer_test=te.tolist(),
                     inner_train=it.tolist(), inner_validation=iv.tolist())
        target = out / "checkpoints" / f"split_{seed}.json"
        if target.exists() and json.loads(target.read_text()) != value:
            raise ValueError("Frozen split changed")
        save_json(value, target)


def develop(out_string, seed, noise):
    out = Path(out_string)
    tag = f"dev_{seed}_{noise:g}"
    destination = out / "checkpoints" / f"{tag}.csv"
    if destination.exists():
        return dict(job=tag, cached=True)
    x, target, _ = load_data()
    _, _, tr, va = splits(seed)
    y, vy, threshold = labels(target[tr], target[va])
    z, active = selection_data(x[tr], noise, seed+3000)
    selector = Selector(z, y, active, seed+4000)
    rows, paths = [], []
    classifier_cache = {}
    start = time.perf_counter()
    for method in MAIN:
        configurations = [c for c in configs(method) if c["noise"] == noise]
        for i, config in enumerate(configurations):
            path = selector.path(config)
            paths.append(dict(config=config, **path))
            for count in COUNTS:
                features = path["history"][count-1]["features"]
                cache_key = tuple(sorted(features))
                if cache_key not in classifier_cache:
                    values = classify(x[tr], y, x[va], vy, features)
                    classifier_cache[cache_key] = {k: v for k, v in values.items()
                                                  if k not in ["prediction", "decision"]}
                rows.append(dict(seed=seed, method=method, config=key(config), features=count,
                                 threshold=threshold, train_n=len(tr), validation_n=len(va),
                                 selection_seconds=path["selection_seconds"],
                                 **classifier_cache[cache_key]))
            print(json.dumps(dict(stage="development", seed=seed, noise=noise, method=method,
                                  config=i+1, of=len(configurations), seconds=round(time.perf_counter()-start, 1))), flush=True)
    save_json(paths, out / "checkpoints" / f"{tag}_paths.json")
    save_json(dict(seconds=time.perf_counter()-start), out / "checkpoints" / f"{tag}_time.json")
    save_csv(pd.DataFrame(rows), destination)
    return dict(job=tag, seconds=round(time.perf_counter()-start, 1))


def lock_choices(out, seed):
    frame = pd.concat([pd.read_csv(out / "checkpoints" / f"dev_{seed}_{noise:g}.csv") for noise in NOISES])
    groups = frame.groupby(["method", "config"], as_index=False).agg(accuracy=("accuracy", "mean"),
                                                                      n=("accuracy", "size"))
    assert groups.n.eq(len(COUNTS)).all()
    def order(row):
        c = json.loads(row["config"])
        return (-row["accuracy"], c["noise"], -c.get("tau", 0), -c.get("n_min", 0), c.get("k", 0), row["config"])
    choices = {}
    for method in MAIN:
        best = min(groups[groups.method == method].to_dict("records"), key=order)
        selected = frame[frame.config == best["config"]].sort_values(["accuracy", "features"], ascending=[False, True]).iloc[0]
        choices[method] = dict(config=json.loads(best["config"]), inner_accuracy=best["accuracy"],
                               reporting_count=int(selected.features))
    pss = choices["PSS SC-CV"]
    for method in ABLATIONS:
        config = dict(**pss["config"])
        config["method"] = method
        if method == "PSS default SC-CV":
            config.update(noise=1e-5, tau=.99, n_min=10)
        elif method == "KL difference k=1":
            config = dict(method=method, noise=1e-5, k=1)
        choices[method] = dict(config=config, reporting_count=pss["reporting_count"],
                               note="Ablation reporting count inherited from primary PSS, not separately optimized")
    record = dict(seed=seed, choices=choices, locked_before_test=True,
                  protocol_sha256=sha(out / "protocol.json"),
                  development_sha256={str(n): sha(out / "checkpoints" / f"dev_{seed}_{n:g}.csv") for n in NOISES})
    destination = out / "checkpoints" / f"lock_{seed}.json"
    if destination.exists() and json.loads(destination.read_text()) != record:
        raise ValueError("Refusing to overwrite locked settings")
    save_json(record, destination)
    return record


def outer(out_string, seed):
    out = Path(out_string)
    lock = lock_choices(out, seed)
    x, target, names = load_data()
    tr, te, _, _ = splits(seed)
    y, ty, threshold = labels(target[tr], target[te])
    begin = time.perf_counter()
    for method in MAIN + ABLATIONS + ["All features"]:
        slug = method.replace(" ", "_").replace("=", "")
        prefix = out / "checkpoints" / f"outer_{seed}_{slug}"
        metrics_file = Path(str(prefix)+"_metrics.csv")
        if metrics_file.exists():
            continue
        if method == "All features":
            config = {}
            path = dict(history=[dict(step=25, features=list(range(25)))], n_selection=0, selection_seconds=0.)
        else:
            config = lock["choices"][method]["config"]
            z, active = selection_data(x[tr], config["noise"], seed+5000)
            selector = Selector(z, y, active, seed+6000)
            path = selector.path(config)
        save_json(dict(method=method, config=config, threshold=threshold, **path), Path(str(prefix)+"_path.json"))
        metrics, predictions = [], []
        for row in path["history"]:
            count = row["step"]
            features = row["features"]
            values = classify(x[tr], y, x[te], ty, features)
            metrics.append(dict(seed=seed, method=method, features=count, threshold=threshold,
                                train_n=len(tr), test_n=len(te), selection_n=path["n_selection"],
                                train_positive=float(y.mean()), test_positive=float(ty.mean()),
                                majority_accuracy=float(np.mean(ty == int(y.mean() > .5))),
                                config=key(config), selected_features=",".join(names[j] for j in features),
                                selection_seconds=path["selection_seconds"],
                                **{k: v for k, v in values.items() if k not in ["prediction", "decision"]},
                                **{k: v for k, v in row.items() if k not in ["step", "features", "feature"]}))
            # Retain every point on every reported curve, not just selected counts.
            predictions.append(pd.DataFrame(dict(seed=seed, method=method, features=count, row_id=te,
                                                 truth=ty, prediction=values["prediction"], decision=values["decision"])))
        save_csv(pd.concat(predictions, ignore_index=True), Path(str(prefix)+"_predictions.csv"))
        save_csv(pd.DataFrame(metrics), metrics_file)
        print(json.dumps(dict(stage="test", seed=seed, method=method,
                              selection_seconds=round(path["selection_seconds"], 1),
                              elapsed=round(time.perf_counter()-begin, 1))), flush=True)
    return dict(seed=seed, stage="complete", seconds=round(time.perf_counter()-begin, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["develop", "evaluate", "all"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    out = args.output.resolve()
    prepare(out)
    if args.stage in ["develop", "all"]:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(develop, str(out), seed, noise) for seed in SEEDS for noise in NOISES]
            for future in as_completed(futures):
                print(json.dumps(future.result()), flush=True)
        for seed in SEEDS:
            print("LOCKED", json.dumps(lock_choices(out, seed)), flush=True)
    if args.stage in ["evaluate", "all"]:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(outer, str(out), seed) for seed in SEEDS]
            for future in as_completed(futures):
                print(json.dumps(future.result()), flush=True)
        save_csv(pd.concat([pd.read_csv(p) for p in sorted((out / "checkpoints").glob("outer_*_metrics.csv"))],
                           ignore_index=True), out / "evaluation_metrics.csv")
        save_csv(pd.concat([pd.read_csv(p) for p in sorted((out / "checkpoints").glob("dev_*.csv"))],
                           ignore_index=True), out / "development_scores.csv")
        print("ENERGY_SAMEPERIOD_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
