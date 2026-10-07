"""Maintainer-only export from the full original Energy workspace (no model fits)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def subset_csv(source, destination, methods, columns):
    with source.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["method"] in methods]
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=HERE / "data")
    args = parser.parse_args()
    root = args.archive_root.resolve()
    original = root / "experiments/energy_theory_c/plot_four_methods.py"
    spec = importlib.util.spec_from_file_location("original_energy_figure", original)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = root / "results/energy_theory_c_20261006"
    # This checks the original audits, dataset, split labels, outer records,
    # prediction hashes, every accuracy and the saved mean/sample SD first.
    stats, inputs = module.load_verified(source)
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    methods = [method[0] for method in module.METHODS]
    seeds = module.SEEDS
    predictions = np.empty((4, 3, 20, 5921), dtype=np.uint8)
    train, test, records = [], [], []
    for seed in seeds:
        split = json.loads((source / f"seed_{seed}/split.json").read_text())
        train.append(split["outer_train"])
        test.append(split["outer_test"])
    for method_index, (method, _, filename, *_) in enumerate(module.METHODS):
        for seed_index, seed in enumerate(seeds):
            record = json.loads((source / f"seed_{seed}/outer/{filename}").read_text())
            if record.get("source_outer_record"):
                path = Path(record["source_outer_record"])
                inputs[str(path)] = sha(path)
            for stored in record["metrics"]:
                path = Path(stored["prediction_file"])
                with np.load(path, allow_pickle=False) as value:
                    prediction = value["prediction"]
                    if not np.isin(prediction, [0, 1]).all():
                        raise ValueError("Nonbinary predictions cannot be exported as uint8")
                    predictions[method_index, seed_index, stored["count"] - 1] = prediction
                records.append(dict(method=method, seed=seed, count=stored["count"],
                                    source=path.relative_to(root).as_posix()))
    targets = pd.read_csv(root / "data/energydata_complete.csv").Appliances.to_numpy()
    np.savez_compressed(destination / "predictions.npz", predictions=predictions,
                        outer_train=np.asarray(train, dtype=np.int32),
                        outer_test=np.asarray(test, dtype=np.int32), targets=targets,
                        methods=np.asarray(methods), seeds=np.asarray(seeds, dtype=np.int32),
                        counts=np.arange(1, 21, dtype=np.int32))
    subset_csv(source / "metrics.csv", destination / "metrics.csv", methods,
               ["seed", "method", "count", "accuracy"])
    subset_csv(source / "summary.csv", destination / "summary.csv", methods,
               ["method", "count", "accuracy", "sd", "repeats"])
    inputs[str(original)] = sha(original)
    logical_hashes = {Path(path).relative_to(root).as_posix(): value
                      for path, value in inputs.items()}
    provenance = dict(
        schema_version=1, methods=methods, seeds=seeds, counts=list(range(1, 21)),
        original_verifier="experiments/energy_theory_c/plot_four_methods.py:load_verified",
        original_verifier_passed=True, prediction_metrics_verified=240,
        summary_points_verified=len(stats), source_file_sha256=logical_hashes,
        prediction_records=records,
        export_source_sha256=sha(__file__),
        artifact_sha256={name: sha(destination / name)
                         for name in ["predictions.npz", "metrics.csv", "summary.csv"]},
        scope="Saved binary predictions, split indices and targets; no training, new predictions or experiments.",
    )
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"original_verifier": "PASS", "prediction_metrics": 240,
                      "summary_points": len(stats), "data_bytes": sum(
                          p.stat().st_size for p in destination.iterdir())}, indent=2))


if __name__ == "__main__":
    main()
