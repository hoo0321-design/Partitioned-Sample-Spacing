"""Frozen, restartable Energy JMI comparison with training-only model selection."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from core import (ROOT, Selector, PSS_JMI, KL_JMI, BASE_PSS, BASE_KL,
                  PRIMARY, COUNTS, NOISE, FOLDS, TAUS, ELLS, KS, N_MIN,
                  configurations, original)

REFERENCE = ROOT / "results/energy_component_guard_20261005"
SECONDARY_REFERENCE = ROOT / "results/energy_sc_expansion_20261005"
SEEDS = [42, 43, 44]
NEW_METHODS = [PSS_JMI, KL_JMI]
CONTROLS = [BASE_PSS, BASE_KL, "PSS ell=1", "All features"]
METHODS = [*PRIMARY, "PSS ell=1", "All features"]


def key(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(value, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def slug(method):
    return method.replace(" ", "_").replace("=", "")


def load_data():
    frame = pd.read_csv(ROOT / "data/energydata_complete.csv")
    names = [c for c in frame if c not in ["date", "Appliances", "rv1", "rv2"]]
    assert len(names) == 25 and len(frame) == 19735
    return frame[names].to_numpy(float), frame.Appliances.to_numpy(float), names


def record_order(record):
    config = record["config"]
    return (-float(np.mean([v["metrics"]["accuracy"] for v in record["metrics"]])),
            -config.get("tau", 0), config.get("folds", 0), config.get("k", 0), key(config))


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    previous = json.loads((REFERENCE / "protocol.json").read_text())
    assert sha(ROOT / "data/energydata_complete.csv") == previous["data_sha256"]
    for source, checksum in previous["source_sha256"].items():
        assert sha(ROOT / source) == checksum, source
    for source, checksum in previous["reference_sha256"].items():
        assert sha(SECONDARY_REFERENCE / source) == checksum, source
    sources = sorted(set([*previous["source_sha256"],
                          "experiments/energy_jmi/core.py", "experiments/energy_jmi/run.py"]))
    reused = {REFERENCE / "protocol.json", SECONDARY_REFERENCE / "protocol.json"}
    for seed in SEEDS:
        for reference in [REFERENCE, SECONDARY_REFERENCE]:
            folder = reference / f"seed_{seed}"
            reused.update([folder / "split.json", folder / "lock.json"])
            # Freeze all available reusable classifier outputs before development.
            for stage in ["inner_classifier", "outer_classifier"]:
                reused.update((folder / stage).glob("*.json"))
                reused.update((folder / stage).glob("*.npz"))
        reused.update(REFERENCE / f"seed_{seed}/outer/{slug(method)}.json" for method in CONTROLS)
    protocol = dict(
        purpose="Exploratory paired Energy PSS-JMI and KL-JMI score ablation",
        seeds=SEEDS, data_sha256=previous["data_sha256"],
        source_sha256={p: sha(ROOT / p) for p in sources},
        reference_sha256={str(p): sha(p) for p in sorted(reused)},
        held_fixed=previous["held_fixed"],
        methods=METHODS, primary=PRIMARY, new_methods=NEW_METHODS,
        configurations={method: configurations(method) for method in NEW_METHODS},
        ell_grid=ELLS, tau_grid=TAUS, fold_grid=FOLDS, k_grid=KS,
        noise=NOISE, n_min=N_MIN, counts=COUNTS,
        jmi="First feature: univariate entropy-difference MI score. Later: arithmetic mean of pair-label MI over every previously selected feature. Only continuous singleton/2D estimates; no additional relevance term, clipping, correlation penalty or forced ell>1.",
        pss="For each singleton/pair, common ell for pooled/class entropy components is chosen by covered pooled NLL subject to min pooled/class0/class1 stable CV coverage >= tau. Canonical unnormalized PSS density and full training samples unchanged.",
        kl="Same KL entropy-difference estimator and k grid; one k chosen per complete path, separately for KL-JMI. No label shuffling or estimator substitution.",
        sample_sizes=dict(inner_train=9669, inner_validation=4145, outer_train=13814, outer_test=5921),
        selection_objective="Mean inner-validation SVM accuracy at 5,10,20 features. All three split choices are locked before any new outer evaluation.",
        config_tie_break="Higher tau, fewer folds, smaller k, lexical JSON after inner mean accuracy; feature tie: smallest index.",
        search_budget="15 PSS-JMI and 10 KL-JMI configurations, identical to each respective baseline grid. Unequal counts disclosed; not equal compute.",
        cache="Only exact same feature subset, seed, data split, preprocessing and fixed SVM. Reference sources and outputs hashed before development. Local output caches separated by inner/outer split. All new fits retain predictions, decision values and true labels.",
        timing="Cache-assisted workflow time, not an unbiased estimator speed benchmark.",
        interpretation="Same three previously inspected random-row holdouts, exploratory only. Split SD is descriptive. No future-time/new-household generalization or independent confirmation claimed.")
    destination = out / "protocol.json"
    if destination.exists():
        assert json.loads(destination.read_text()) == protocol, "Frozen protocol changed"
    else:
        save(protocol, destination)
    for seed in SEEDS:
        folder = out / f"seed_{seed}"
        folder.mkdir(exist_ok=True)
        for stage in ["development", "inner_classifier", "outer_classifier", "outer"]:
            (folder / stage).mkdir(exist_ok=True)
        split = json.loads((REFERENCE / f"seed_{seed}/split.json").read_text())
        assert split == json.loads((SECONDARY_REFERENCE / f"seed_{seed}/split.json").read_text())
        p = folder / "split.json"
        if p.exists():
            assert json.loads(p.read_text()) == split
        else:
            save(split, p)


def setup(out, seed, outer=False):
    folder = Path(out) / f"seed_{seed}"
    x, target, names = load_data()
    split = json.loads((folder / "split.json").read_text())
    train = np.asarray(split["outer_train" if outer else "inner_train"])
    query = np.asarray(split["outer_test" if outer else "inner_validation"])
    y, qy, threshold = original.labels(target[train], target[query])
    z, active = original.selection_data(x[train], NOISE, seed + (5000 if outer else 3000))
    selector = Selector(z, y, active, seed + (6000 if outer else 4000))
    return folder, x, names, train, query, y, qy, threshold, selector


def classify_cached(folder, references, x, y, query, qy, features, frozen_hashes, predictions=False):
    stem = "_".join(map(str, sorted(features)))
    p = folder / (stem + ".json")
    if p.exists():
        result = json.loads(p.read_text())
        assert not predictions or result.get("prediction_file")
        if result.get("prediction_file"):
            assert sha(result["prediction_file"]) == result["prediction_sha256"]
        return result
    for reference in references:
        old = reference / (stem + ".json")
        if str(old) not in frozen_hashes:
            continue
        assert sha(old) == frozen_hashes[str(old)]
        cached = json.loads(old.read_text())
        metrics = cached.get("metrics", cached)
        prediction_file = Path(cached.get("prediction_file", old.with_suffix(".npz")))
        if predictions and not prediction_file.exists():
            continue
        if cached.get("source"):
            assert sha(cached["source"]) == cached["source_sha256"] == frozen_hashes[cached["source"]]
        result = dict(metrics=metrics, reused=True, source=str(old), source_sha256=sha(old))
        if prediction_file.exists():
            assert str(prediction_file) in frozen_hashes
            assert sha(prediction_file) == frozen_hashes[str(prediction_file)]
            result.update(prediction_file=str(prediction_file), prediction_sha256=sha(prediction_file))
        save(result, p)
        return result
    result = original.classify(x, y, query, qy, features)
    prediction = result.pop("prediction")
    decision = result.pop("decision")
    prediction_file = p.with_suffix(".npz")
    np.savez_compressed(prediction_file, prediction=prediction, decision=decision, y=qy)
    value = dict(metrics=result, reused=False, prediction_file=str(prediction_file),
                 prediction_sha256=sha(prediction_file))
    save(value, p)
    return value


def develop(out, seed):
    start = time.perf_counter()
    folder, x, _, train, valid, y, vy, threshold, selector = setup(out, seed)
    protocol = json.loads((Path(out) / "protocol.json").read_text())
    references = [r / f"seed_{seed}/inner_classifier" for r in [REFERENCE, SECONDARY_REFERENCE]]
    records = []
    for method in NEW_METHODS:
        configs = configurations(method)
        for index, config in enumerate(configs):
            tag = hashlib.sha256(key(config).encode()).hexdigest()[:16]
            p = folder / "development" / (tag + ".json")
            if p.exists():
                record = json.loads(p.read_text())
                assert record["config"] == config and record["method"] == method
            else:
                path = selector.path_jmi(config)
                values = []
                for count in COUNTS:
                    subset = path["history"][count - 1]["features"]
                    value = classify_cached(folder / "inner_classifier", references,
                        x[train], y, x[valid], vy, subset, protocol["reference_sha256"])
                    values.append(dict(count=count, features=subset, **value))
                record = dict(method=method, config=config, path=path, metrics=values, threshold=threshold)
                save(record, p)
            records.append(record)
            print(json.dumps(dict(stage="development", seed=seed, method=method, config=index + 1,
                                  total=len(configs), seconds=round(time.perf_counter() - start))), flush=True)
    choices = {}
    for method in NEW_METHODS:
        best = min([r for r in records if r["method"] == method], key=record_order)
        choices[method] = dict(config=best["config"], inner_accuracy=-record_order(best)[0])
    lock = dict(choices=choices, locked_before_test=True, protocol_sha256=sha(Path(out) / "protocol.json"),
        development_sha256={p.name: sha(p) for p in sorted((folder / "development").glob("*.json"))})
    p = folder / "lock.json"
    if p.exists():
        assert json.loads(p.read_text()) == lock
    else:
        save(lock, p)
    print(json.dumps(dict(stage="locked", seed=seed, choices=choices)), flush=True)
    return dict(seed=seed, seconds=time.perf_counter() - start)


def lock_barrier(out):
    locks = {}
    for seed in SEEDS:
        p = out / f"seed_{seed}/lock.json"
        record = json.loads(p.read_text())
        assert record["locked_before_test"] and record["protocol_sha256"] == sha(out / "protocol.json")
        for filename, checksum in record["development_sha256"].items():
            assert sha(p.parent / "development" / filename) == checksum
        locks[str(seed)] = sha(p)
    barrier = dict(locked_before_test=True, protocol_sha256=sha(out / "protocol.json"), locks_sha256=locks)
    p = out / "all_locks.json"
    if p.exists():
        assert json.loads(p.read_text()) == barrier
    else:
        save(barrier, p)
    return barrier


def evaluate(out, seed):
    start = time.perf_counter()
    out = Path(out)
    barrier = json.loads((out / "all_locks.json").read_text())
    assert barrier["protocol_sha256"] == sha(out / "protocol.json")
    for other_seed, checksum in barrier["locks_sha256"].items():
        assert sha(out / f"seed_{other_seed}/lock.json") == checksum
    folder, x, names, train, test, y, ty, threshold, selector = setup(out, seed, True)
    protocol = json.loads((out / "protocol.json").read_text())
    lock = json.loads((folder / "lock.json").read_text())
    references = [r / f"seed_{seed}/outer_classifier" for r in [REFERENCE, SECONDARY_REFERENCE]]
    # Both new paths are selected/saved before evaluating this split's outcomes.
    for method in NEW_METHODS:
        p = folder / "outer" / (slug(method) + "_pretest.json")
        config = lock["choices"][method]["config"]
        if p.exists():
            snapshot = json.loads(p.read_text())
            assert snapshot["config"] == config and snapshot["locks_sha256"] == barrier["locks_sha256"]
        else:
            path = selector.path_jmi(config)
            snapshot = dict(method=method, config=config, threshold=threshold, path=path,
                protocol_sha256=sha(out / "protocol.json"), locks_sha256=barrier["locks_sha256"],
                lock_barrier_sha256=sha(out / "all_locks.json"))
            save(snapshot, p)
        print(json.dumps(dict(stage="outer_path_saved", seed=seed, method=method)), flush=True)
    for method in NEW_METHODS:
        p = folder / "outer" / (slug(method) + ".json")
        if p.exists():
            continue
        snapshot = json.loads(p.with_name(p.stem + "_pretest.json").read_text())
        values = []
        for row in snapshot["path"]["history"]:
            result = classify_cached(folder / "outer_classifier", references,
                x[train], y, x[test], ty, row["features"], protocol["reference_sha256"], predictions=True)
            values.append(dict(seed=seed, method=method, count=row["step"], features=row["features"],
                selected_features=[names[j] for j in row["features"]], **result))
            if row["step"] in COUNTS:
                print(json.dumps(dict(stage="test_curve", seed=seed, method=method, count=row["step"],
                                      seconds=round(time.perf_counter() - start))), flush=True)
        save(dict(**snapshot, metrics=values), p)
    for method in CONTROLS:
        p = folder / "outer" / (slug(method) + ".json")
        if p.exists():
            continue
        source = REFERENCE / f"seed_{seed}/outer/{slug(method)}.json"
        assert sha(source) == protocol["reference_sha256"][str(source)]
        record = json.loads(source.read_text())
        assert record["threshold"] == threshold
        save(dict(**record, source_outer_record=str(source), source_outer_sha256=sha(source)), p)
    return dict(seed=seed, seconds=time.perf_counter() - start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "develop", "evaluate", "all"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    out = args.output.resolve()
    prepare(out)
    for stage, function in [("develop", develop), ("evaluate", evaluate)]:
        if args.stage not in [stage, "all"]:
            continue
        if stage == "evaluate":
            lock_barrier(out)
        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(function, str(out), seed) for seed in SEEDS]
            jobs = [f.result() for f in as_completed(futures)]
        save(dict(wall_seconds=time.perf_counter() - start, jobs=jobs), out / (stage + "_timing.json"))
    print("ENERGY_JMI_STAGE_COMPLETE", args.stage, flush=True)


if __name__ == "__main__":
    main()
