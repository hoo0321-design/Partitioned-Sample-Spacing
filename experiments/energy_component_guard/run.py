"""Paired coverage-guard ablation; verified old controls are reused read-only."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from core import (ROOT, POOLED, GUARDED, MATCHED, Selector, FOLDS, TAUS, ELLS,
                  N_MIN, NOISE, COUNTS, configurations, original)

REFERENCE = ROOT/"results/energy_sc_expansion_20261005"
SEEDS = [42, 43, 44]
CONTROLS = {POOLED: "PSS expanded SC-CV", "KL tuned": "KL tuned", "KL k=1": "KL k=1",
            "Ross tuned": "Ross tuned", "Univariate tuned": "Univariate tuned",
            "PSS ell=1": "PSS ell=1", "All features": "All features"}


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


def reference_record(seed, method):
    return REFERENCE/f"seed_{seed}"/"outer"/(slug(CONTROLS[method])+".json")


def load_data():
    frame = pd.read_csv(ROOT/"data/energydata_complete.csv")
    names = [c for c in frame if c not in ["date", "Appliances", "rv1", "rv2"]]
    assert len(names) == 25 and len(frame) == 19735
    return frame[names].to_numpy(float), frame.Appliances.to_numpy(float), names


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    old = json.loads((REFERENCE/"protocol.json").read_text())
    assert sha(ROOT/"data/energydata_complete.csv") == old["data_sha256"]
    for source, checksum in old["code_sha256"].items():
        assert sha(ROOT/source) == checksum, source
    reused = [REFERENCE/"protocol.json"]
    for seed in SEEDS:
        ref = REFERENCE/f"seed_{seed}"
        reused += [ref/"split.json", ref/"lock.json"]
        reused += list((ref/"development").glob("*.json"))
        reused += [reference_record(seed, m) for m in CONTROLS]
    sources = ["experiments/energy_component_guard/core.py", "experiments/energy_component_guard/run.py",
               *old["code_sha256"].keys(), "PSS/libpss_v2.dylib"]
    protocol = dict(seeds=SEEDS, purpose="Exploratory paired class-aware coverage ablation; only Step 2, no redundancy-score change",
        reference=str(REFERENCE), reference_sha256={str(p.relative_to(REFERENCE)): sha(p) for p in reused},
        data_sha256=old["data_sha256"], source_sha256={p: sha(ROOT/p) for p in sources},
        scope="First three previously inspected Energy random splits, fixed in advance; not fresh independent confirmation",
        held_fixed={k: old[k] for k in ["split", "target", "selection", "classifier", "density",
                    "selection_objective", "tie_break", "interpretation"]},
        change="Feasibility requires min(S_pooled,S_class0,S_class1)>=tau instead of S_pooled>=tau. Covered pooled NLL objective, N_eff density, entropy-difference score, ell grid, training folds and seeds are unchanged.",
        ell_grid=ELLS, tau_grid=TAUS, fold_grid=FOLDS, n_min=N_MIN, noise=NOISE,
        guard_configs=configurations("PSS expanded SC-CV"),
        methods=[GUARDED, MATCHED, *CONTROLS],
        matched="Additional guard path with tau and fold count frozen to the old pooled winner, to isolate the gate from outer selector retuning",
        invalid="An ell with zero effective samples in an entropy component is inadmissible. The canonical density is unchanged.",
        fallback="If no feasible finite candidate, maximize minimum coverage then minimize pooled covered NLL then ell; report every fallback",
        compute="Unchanged controls and exact subset classifier fits may be reused after hashes and identical splits are verified; new guard feature paths rebuilt from all training rows. No test outcomes used to choose a path or setting.",
        timing="Cache-assisted wall time only; not an unbiased algorithm runtime comparison")
    p = out/"protocol.json"
    if p.exists():
        assert json.loads(p.read_text()) == protocol
    else:
        save(protocol, p)
    for seed in SEEDS:
        folder = out/f"seed_{seed}"
        folder.mkdir(exist_ok=True)
        for name in ["development", "inner_classifier", "outer_classifier", "outer"]:
            (folder/name).mkdir(exist_ok=True)
        split = json.loads((REFERENCE/f"seed_{seed}"/"split.json").read_text())
        if (folder/"split.json").exists():
            assert json.loads((folder/"split.json").read_text()) == split
        else:
            save(split, folder/"split.json")


def setup(out, seed, outer=False):
    folder = Path(out)/f"seed_{seed}"
    x, target, names = load_data()
    split = json.loads((folder/"split.json").read_text())
    train = np.array(split["outer_train" if outer else "inner_train"])
    query = np.array(split["outer_test" if outer else "inner_validation"])
    y, qy, threshold = original.labels(target[train], target[query])
    z, active = original.selection_data(x[train], NOISE, seed+(5000 if outer else 3000))
    selector = Selector(z, y, active, seed+(6000 if outer else 4000))
    return folder, x, names, train, query, y, qy, threshold, selector


def classify_cached(folder, ref_folder, x, y, query, qy, features, predictions=False):
    stem = "_".join(map(str, sorted(features)))
    p, old = folder/(stem+".json"), ref_folder/(stem+".json")
    if p.exists():
        return json.loads(p.read_text())
    if old.exists() and (not predictions or old.with_suffix(".npz").exists()):
        result = dict(metrics=json.loads(old.read_text()), reused=True, source=str(old), source_sha256=sha(old))
        if predictions:
            result.update(prediction_file=str(old.with_suffix(".npz")), prediction_sha256=sha(old.with_suffix(".npz")))
    else:
        values = original.classify(x, y, query, qy, features)
        pred, decision = values.pop("prediction"), values.pop("decision")
        prediction_file = p.with_suffix(".npz")
        np.savez_compressed(prediction_file, prediction=pred, decision=decision, y=qy)
        result = dict(metrics=values, reused=False, prediction_file=str(prediction_file),
                      prediction_sha256=sha(prediction_file))
    save(result, p)
    return result


def develop(out, seed):
    start = time.perf_counter()
    folder, x, _, train, valid, y, vy, threshold, selector = setup(out, seed)
    records = []
    for index, config in enumerate(configurations("PSS expanded SC-CV")):
        tag = hashlib.sha256(key(config).encode()).hexdigest()[:16]
        p = folder/"development"/(tag+".json")
        if p.exists():
            record = json.loads(p.read_text())
            assert record["config"] == config
        else:
            path = selector.path_guard(config)
            values = []
            for count in COUNTS:
                subset = path["history"][count-1]["features"]
                result = classify_cached(folder/"inner_classifier", REFERENCE/f"seed_{seed}"/"inner_classifier",
                    x[train], y, x[valid], vy, subset)
                values.append(dict(count=count, features=subset, **result))
            record = dict(config=config, path=path, metrics=values, threshold=threshold)
            save(record, p)
        records.append(record)
        print(json.dumps(dict(stage="development", seed=seed, config=index+1, total=15,
                              seconds=round(time.perf_counter()-start))), flush=True)
    best = min(records, key=lambda r: (-np.mean([v["metrics"]["accuracy"] for v in r["metrics"]]),
        -r["config"]["tau"], r["config"]["folds"], key(r["config"])))
    pooled = json.loads((REFERENCE/f"seed_{seed}"/"lock.json").read_text())["choices"][CONTROLS[POOLED]]
    lock = dict(choices={GUARDED: best["config"], MATCHED: pooled["config"]},
        inner_accuracy=float(np.mean([v["metrics"]["accuracy"] for v in best["metrics"]])),
        locked_before_test=True, protocol_sha256=sha(Path(out)/"protocol.json"),
        development_sha256={p.name: sha(p) for p in sorted((folder/"development").glob("*.json"))})
    p = folder/"lock.json"
    if p.exists():
        assert json.loads(p.read_text()) == lock
    else:
        save(lock, p)
    return dict(seed=seed, seconds=time.perf_counter()-start)


def evaluate(out, seed):
    start = time.perf_counter()
    folder, x, names, train, test, y, ty, threshold, selector = setup(out, seed, True)
    lock = json.loads((folder/"lock.json").read_text())
    assert lock["protocol_sha256"] == sha(Path(out)/"protocol.json")
    cache = {}
    for method in [GUARDED, MATCHED]:
        p = folder/"outer"/(slug(method)+".json")
        if p.exists():
            continue
        config = lock["choices"][method]
        if key(config) not in cache:
            path = selector.path_guard(config)
            selector.diagnose(path, config)
            cache[key(config)] = path
        path = cache[key(config)]
        snapshot = dict(method=method, config=config, threshold=threshold, path=path)
        save(snapshot, p.with_name(p.stem+"_pretest.json"))
        values = []
        for row in path["history"]:
            result = classify_cached(folder/"outer_classifier", REFERENCE/f"seed_{seed}"/"outer_classifier",
                x[train], y, x[test], ty, row["features"], predictions=True)
            values.append(dict(seed=seed, method=method, count=row["step"], features=row["features"],
                selected_features=[names[j] for j in row["features"]], **result))
        save(dict(**snapshot, metrics=values), p)
        print(json.dumps(dict(stage="test", seed=seed, method=method,
                              seconds=round(time.perf_counter()-start))), flush=True)
    for method in CONTROLS:
        p = folder/"outer"/(slug(method)+".json")
        if p.exists():
            continue
        source = reference_record(seed, method)
        record = json.loads(source.read_text())
        config, path = record["config"], record["path"]
        if method in [POOLED, "PSS ell=1"]:
            diagnostic_config = dict(config, folds=config.get("folds", 3))
            selector.diagnose(path, diagnostic_config)
        values = []
        for row, old in zip(path["history"], record["metrics"]):
            fields = {k: old[k] for k in ["accuracy", "balanced_accuracy", "auc", "fit_seconds", "prediction_seconds"]}
            pred = REFERENCE/old["prediction_file"]
            values.append(dict(seed=seed, method=method, count=row["step"], features=row["features"],
                selected_features=[names[j] for j in row["features"]], metrics=fields,
                reused=True, prediction_file=str(pred), prediction_sha256=sha(pred)))
        save(dict(method=method, config=config, path=path, threshold=threshold, metrics=values,
                  source=str(source), source_sha256=sha(source)), p)
    return dict(seed=seed, seconds=time.perf_counter()-start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "develop", "evaluate", "all"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    out = args.output.resolve()
    prepare(out)
    for stage, function in [("develop", develop), ("evaluate", evaluate)]:
        if args.stage not in [stage, "all"]:
            continue
        if stage == "evaluate":
            assert all((out/f"seed_{s}"/"lock.json").exists() for s in SEEDS)
        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(function, str(out), seed) for seed in SEEDS]
            jobs = [f.result() for f in as_completed(futures)]
        save(dict(wall_seconds=time.perf_counter()-start, jobs=jobs), out/(stage+"_timing.json"))


if __name__ == "__main__":
    main()
