"""Small, frozen Energy experiment selecting one theory-rate coefficient in training."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.energy_theory_c.core import (
    Selector, COEFFICIENTS, TUNED, FIXED, BASE_PSS, BASE_KL,
    COUNTS, NOISE, original, theory_ell,
)

REFERENCE = ROOT / "results/energy_component_guard_20261005"
SECONDARY_REFERENCE = ROOT / "results/energy_sc_expansion_20261005"
JMI_REFERENCE = ROOT / "results/energy_jmi_20261006"
REFERENCES = [REFERENCE, SECONDARY_REFERENCE, JMI_REFERENCE]
SEEDS = [42, 43, 44]
NEW_METHODS = [TUNED, FIXED]
CONTROLS = [BASE_PSS, BASE_KL, "KL k=1", "All features"]
METHODS = [TUNED, FIXED, BASE_PSS, BASE_KL, "KL k=1", "All features"]


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


def configurations():
    return [dict(coefficient=c) for c in COEFFICIENTS]


def record_order(record):
    if record["status"] != "complete":
        return (float("inf"), record["config"]["coefficient"])
    assert [v["count"] for v in record["metrics"]] == COUNTS
    return (-float(np.mean([v["metrics"]["accuracy"] for v in record["metrics"]])),
            record["config"]["coefficient"])


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    previous = json.loads((REFERENCE / "protocol.json").read_text())
    assert sha(ROOT / "data/energydata_complete.csv") == previous["data_sha256"]
    sources = set(previous["source_sha256"])
    for source, checksum in previous["source_sha256"].items():
        assert sha(ROOT / source) == checksum, source
    for source, checksum in previous["reference_sha256"].items():
        assert sha(SECONDARY_REFERENCE / source) == checksum, source
    jmi = json.loads((JMI_REFERENCE / "protocol.json").read_text())
    assert jmi["data_sha256"] == previous["data_sha256"]
    for source, checksum in jmi["source_sha256"].items():
        assert sha(ROOT / source) == checksum, source
    sources.update(jmi["source_sha256"])
    sources.update(["experiments/energy_theory_c/core.py", "experiments/energy_theory_c/run.py"])
    reused = {r / "protocol.json" for r in REFERENCES}
    for seed in SEEDS:
        for reference in REFERENCES:
            folder = reference / f"seed_{seed}"
            reused.update([folder / "split.json", folder / "lock.json"])
            for stage in ["inner_classifier", "outer_classifier"]:
                reused.update((folder / stage).glob("*.json"))
                reused.update((folder / stage).glob("*.npz"))
        reused.update(REFERENCE / f"seed_{seed}/outer/{slug(m)}.json" for m in CONTROLS)
    protocol = dict(
        purpose="Exploratory inner-validation calibration of one shared theory-rate coefficient C",
        seeds=SEEDS, data_sha256=previous["data_sha256"],
        source_sha256={p: sha(ROOT / p) for p in sorted(sources)},
        reference_sha256={str(p): sha(p) for p in sorted(reused)},
        held_fixed=previous["held_fixed"], methods=METHODS, new_methods=NEW_METHODS,
        coefficient_grid=COEFFICIENTS, configurations=configurations(), counts=COUNTS, noise=NOISE,
        formula="ell=max(1,floor(C*(n/(d**6*log(n)**2))**(1/(d+8))+0.5)); natural log, C OUTSIDE root, half-up rounding",
        n_definition="Full pooled sample available for feature selection: 9669 inner training or 13814 outer training. Common ell used in pooled and both class entropy components, as in the previous Energy score. No component-specific n rule.",
        d_definition="Number of variables in the entire candidate subset at that forward-selection step. This is full-subset MI, not pairwise JMI.",
        coefficient_scope="One C per split, held constant over the whole 1..20 feature path. No per-dimension, per-feature, per-class or per-count C tuning. Recompute ell with outer training n after C is locked.",
        ell_schedules={str(n): {str(c): [theory_ell(n, d, c) for d in range(1, 21)]
                               for c in COEFFICIENTS} for n in [9669, 13814]},
        score="Unchanged canonical PSS entropy difference, common ell. No score clipping, density normalization, JMI, added penalty, coverage gate or fallback to another ell.",
        selection_objective="For each C, rebuild path using all inner training rows; fit identical fixed SVM on that inner training set and score the held-out inner validation set at 5,10,20 features. Select maximum arithmetic mean accuracy; tie chooses smaller C.",
        sample_sizes=dict(inner_train=9669, inner_validation=4145, outer_train=13814, outer_test=5921),
        failure_policy="A candidate with zero valid observations in any entropy component is inadmissible and retained with null score/error. If no valid candidate remains before completing 20 steps, retain the failed C but exclude it from selection. Never average partial-count accuracies. An outer failure is reported without changing C, ell or silently substituting another method.",
        evaluation="Lock all three C choices before any new test evaluation. Save both new outer paths before reading classifier outcomes. Evaluate only tuned C and prespecified C=1, not every C on test data.",
        search_budget="Six new C settings; baseline PSS used 15 fold/tau settings and KL ten k settings. These are disclosed, not equal tuning or compute budgets.",
        cache="Reuse only exact same sorted feature subset, seed, split, training-only preprocessing and fixed SVM after frozen-source/output checks. All new fits retain predictions/decisions/labels. Historical inner metrics without predictions are identified.",
        timing="Cache-assisted workflow time; not an unbiased method runtime benchmark.",
        interpretation="Previously inspected Energy same-period random splits. Exploratory development, not independent confirmation. Split SD is descriptive. C selection makes this a theory-guided TUNED method, not a tuning-free optimality result.")
    p = out / "protocol.json"
    if p.exists():
        assert json.loads(p.read_text()) == protocol, "Frozen protocol changed"
    else:
        save(protocol, p)
    for seed in SEEDS:
        folder = out / f"seed_{seed}"
        folder.mkdir(exist_ok=True)
        for stage in ["development", "inner_classifier", "outer_classifier", "outer"]:
            (folder / stage).mkdir(exist_ok=True)
        split = json.loads((REFERENCE / f"seed_{seed}/split.json").read_text())
        for reference in REFERENCES:
            assert split == json.loads((reference / f"seed_{seed}/split.json").read_text())
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
    assert len(train) == (13814 if outer else 9669) and len(query) == (5921 if outer else 4145)
    y, qy, threshold = original.labels(target[train], target[query])
    z, active = original.selection_data(x[train], NOISE, seed + (5000 if outer else 3000))
    selector = Selector(z, y, active, seed + (6000 if outer else 4000))
    return folder, x, names, train, query, y, qy, threshold, selector


def classify_cached(folder, references, x, y, query, qy, features, frozen_hashes, predictions=False):
    stem = "_".join(map(str, sorted(features)))
    p = folder / (stem + ".json")
    if p.exists():
        value = json.loads(p.read_text())
        assert not predictions or value.get("prediction_file")
        if value.get("prediction_file"):
            assert sha(value["prediction_file"]) == value["prediction_sha256"]
        return value
    for reference in references:
        old = reference / (stem + ".json")
        if str(old) not in frozen_hashes:
            continue
        assert sha(old) == frozen_hashes[str(old)]
        cached = json.loads(old.read_text())
        prediction_file = Path(cached.get("prediction_file", old.with_suffix(".npz")))
        if predictions and not prediction_file.exists():
            continue
        if cached.get("source"):
            assert sha(cached["source"]) == cached["source_sha256"] == frozen_hashes[cached["source"]]
        value = dict(metrics=cached.get("metrics", cached), reused=True, source=str(old), source_sha256=sha(old))
        if prediction_file.exists():
            assert sha(prediction_file) == frozen_hashes[str(prediction_file)]
            value.update(prediction_file=str(prediction_file), prediction_sha256=sha(prediction_file))
        save(value, p)
        return value
    result = original.classify(x, y, query, qy, features)
    pred, decision = result.pop("prediction"), result.pop("decision")
    prediction_file = p.with_suffix(".npz")
    np.savez_compressed(prediction_file, prediction=pred, decision=decision, y=qy)
    value = dict(metrics=result, reused=False, prediction_file=str(prediction_file),
                 prediction_sha256=sha(prediction_file))
    save(value, p)
    return value


def develop(out, seed):
    start = time.perf_counter()
    folder, x, _, train, valid, y, vy, threshold, selector = setup(out, seed)
    hashes = json.loads((Path(out) / "protocol.json").read_text())["reference_sha256"]
    references = [r / f"seed_{seed}/inner_classifier" for r in REFERENCES]
    records = []
    for c in COEFFICIENTS:
        config = dict(coefficient=c)
        p = folder / "development" / f"C_{c:g}.json"
        if p.exists():
            record = json.loads(p.read_text())
            assert record["config"] == config
        else:
            path = selector.path_coefficient(c)
            values = []
            if path["status"] == "complete":
                for count in COUNTS:
                    subset = path["history"][count - 1]["features"]
                    result = classify_cached(folder / "inner_classifier", references,
                        x[train], y, x[valid], vy, subset, hashes)
                    values.append(dict(count=count, features=subset, **result))
            record = dict(method=TUNED, config=config, path=path, status=path["status"],
                          metrics=values, threshold=threshold)
            save(record, p)
        records.append(record)
        print(json.dumps(dict(stage="development", seed=seed, coefficient=c, status=record["status"],
                              seconds=round(time.perf_counter() - start))), flush=True)
    complete = [r for r in records if r["status"] == "complete"]
    assert complete, "All C settings failed; no coefficient can be selected"
    best = min(complete, key=record_order)
    fixed = next(r for r in records if r["config"]["coefficient"] == 1)
    assert fixed["status"] == "complete", "The fixed C=1 control unexpectedly failed"
    lock = dict(choices={TUNED: dict(config=best["config"], inner_accuracy=-record_order(best)[0]),
                         FIXED: dict(config=fixed["config"], inner_accuracy=-record_order(fixed)[0])},
        locked_before_test=True, protocol_sha256=sha(Path(out) / "protocol.json"),
        development_sha256={p.name: sha(p) for p in sorted((folder / "development").glob("*.json"))})
    p = folder / "lock.json"
    if p.exists():
        assert json.loads(p.read_text()) == lock
    else:
        save(lock, p)
    print(json.dumps(dict(stage="locked", seed=seed, choices=lock["choices"])), flush=True)
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
    value = dict(locked_before_test=True, protocol_sha256=sha(out / "protocol.json"), locks_sha256=locks)
    p = out / "all_locks.json"
    if p.exists():
        assert json.loads(p.read_text()) == value
    else:
        save(value, p)


def evaluate(out, seed):
    start = time.perf_counter()
    out = Path(out)
    barrier = json.loads((out / "all_locks.json").read_text())
    assert barrier["protocol_sha256"] == sha(out / "protocol.json")
    for s, checksum in barrier["locks_sha256"].items():
        assert sha(out / f"seed_{s}/lock.json") == checksum
    folder, x, names, train, test, y, ty, threshold, selector = setup(out, seed, True)
    protocol = json.loads((out / "protocol.json").read_text())
    lock = json.loads((folder / "lock.json").read_text())
    references = [r / f"seed_{seed}/outer_classifier" for r in REFERENCES]
    path_cache = {}
    for method in NEW_METHODS:
        p = folder / "outer" / (slug(method) + "_pretest.json")
        config = lock["choices"][method]["config"]
        if p.exists():
            value = json.loads(p.read_text())
            assert value["config"] == config and value["locks_sha256"] == barrier["locks_sha256"]
        else:
            c = config["coefficient"]
            if c not in path_cache:
                path_cache[c] = selector.path_coefficient(c)
            path = path_cache[c]
            value = dict(method=method, config=config, threshold=threshold, path=path, status=path["status"],
                protocol_sha256=sha(out / "protocol.json"), locks_sha256=barrier["locks_sha256"],
                lock_barrier_sha256=sha(out / "all_locks.json"))
            save(value, p)
        print(json.dumps(dict(stage="outer_path_saved", seed=seed, method=method, status=value["status"])), flush=True)
    for method in NEW_METHODS:
        p = folder / "outer" / (slug(method) + ".json")
        if p.exists():
            continue
        snapshot = json.loads(p.with_name(p.stem + "_pretest.json").read_text())
        metrics = []
        if snapshot["status"] == "complete":
            for row in snapshot["path"]["history"]:
                result = classify_cached(folder / "outer_classifier", references,
                    x[train], y, x[test], ty, row["features"], protocol["reference_sha256"], predictions=True)
                metrics.append(dict(seed=seed, method=method, count=row["step"], features=row["features"],
                    selected_features=[names[j] for j in row["features"]], **result))
                if row["step"] in COUNTS:
                    print(json.dumps(dict(stage="test_curve", seed=seed, method=method, count=row["step"],
                                          seconds=round(time.perf_counter() - start))), flush=True)
        save(dict(**snapshot, metrics=metrics), p)
    for method in CONTROLS:
        p = folder / "outer" / (slug(method) + ".json")
        if p.exists():
            continue
        source = REFERENCE / f"seed_{seed}/outer/{slug(method)}.json"
        assert sha(source) == protocol["reference_sha256"][str(source)]
        record = json.loads(source.read_text())
        assert record["threshold"] == threshold
        save(dict(**record, status="complete", source_outer_record=str(source), source_outer_sha256=sha(source)), p)
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
    print("ENERGY_THEORY_C_STAGE_COMPLETE", args.stage, flush=True)


if __name__ == "__main__":
    main()
