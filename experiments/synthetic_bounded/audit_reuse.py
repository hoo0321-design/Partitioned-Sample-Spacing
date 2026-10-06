"""Audit reuse of archived bounded-ridge results against the canonical estimator.

This is a numerical spotcheck, not a full-replicate recomputation. Original
archives are read-only. Run with PYTHONDONTWRITEBYTECODE=1 to avoid cache writes.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path("/Users/hojeongwoo/Documents/Codex/2026-05-03/"
                      "d-dimensional-partitioned-sample-spacing-pss/Partitioned-Sample-Spacing")
FAMILIES = {"ridge_medium": (5, 1), "ridge_frequency2": (7, 2)}
GROUPS = [
    ("theory_selection_20261003", 20261003, 100, [1000, 3000, 10000, 30000, 100000]),
    ("theory_selection_large_n_20261003", 20261004, 30, [300000, 1000000]),
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bound_objective(n, d, ell):
    return ell ** -2 + math.sqrt(2*d*math.log(2*n+1) + math.log(96/.05)) * (ell**d/n)**.25


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=WORKSPACE/"results/synthetic_bounded_20261006")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    source_code = args.source / "experiments/theory_selection"
    errors, provenance, raw_checks, spotchecks, settings = [], [], [], [], []
    source_manifests = {}

    for group, seed, reps, ns in GROUPS:
        directory = args.source / "results" / group
        manifest_path = directory / "design.json"
        design = json.loads(manifest_path.read_text())
        source_manifests[group] = dict(path=str(manifest_path), sha256=digest(manifest_path))
        if design["seed"] != seed or design["eval_reps"] != reps:
            errors.append(f"{group}: seed or replicate count differs from expected design")
        for filename, expected in design.get("sources", design.get("source_hashes", {})).items():
            observed = digest(source_code / filename)
            ok = expected == observed
            provenance.append(dict(group=group, file=filename, expected_sha256=expected,
                                   observed_sha256=observed, passed=ok))
            if not ok:
                errors.append(f"{group}: archived source hash mismatch for {filename}")
        by_key = {(s["family"], s["n"], s["d"]): s for s in design["settings"]}
        for family, (family_id, frequency) in FAMILIES.items():
            for d in [2, 5]:
                for n in ns:
                    config = by_key[(family, n, d)]
                    if config["family_id"] != family_id:
                        errors.append(f"{family}, n={n}, d={d}: family identifier mismatch")
                    settings.append(dict(group=group, directory=directory, seed=seed, reps=reps,
                                         family=family, family_id=family_id, frequency=frequency,
                                         n=n, d=d, config=config))

    # Hash verification precedes importing the sampler used by the original run.
    if errors:
        payload = dict(status="FAIL", errors=errors, provenance=provenance,
                       scope="Source provenance precheck; no numerical rerun performed.")
        (args.out/"audit.json").write_text(json.dumps(payload, indent=2)+"\n")
        raise SystemExit("Audit failed source precheck: " + "; ".join(errors))
    sampler = module_at("archived_bounded_sampler", source_code/"pss_theory.py")
    canonical = module_at("canonical_pss_v2", WORKSPACE/"PSS/pss_v2.py")

    a = .7
    analytic_truth = -(1-math.sqrt(1-a*a)+math.log((1+math.sqrt(1-a*a))/2))
    nodes, weights = np.polynomial.legendre.leggauss(256)
    nodes, weights = (nodes+1)/2, weights/2
    entropy_checks = []
    for family, (_, frequency) in FAMILIES.items():
        density = 1+a*np.cos(2*np.pi*frequency*(nodes[:, None]-nodes[None, :]))
        quadrature = -float(weights @ (density*np.log(density)) @ weights)
        difference = abs(quadrature-analytic_truth)
        entropy_checks.append(dict(family=family, analytic_entropy=analytic_truth,
                                   tensor_gauss_legendre_entropy=quadrature, order_per_axis=256,
                                   absolute_difference=difference, passed=difference <= 1e-12))
        if difference > 1e-12:
            errors.append(f"{family}: numerical entropy quadrature disagreement")

    selected_rules = []
    for index, setting in enumerate(settings, 1):
        family, n, d = setting["family"], setting["n"], setting["d"]
        name = f"{family}_n{n}_d{d}"
        file = setting["directory"] / "evaluation" / f"{name}.csv"
        with file.open() as stream:
            rows = list(csv.DictReader(stream))
        grid = list(map(int, setting["config"]["ells"]))
        reps = setting["reps"]
        lookup = {(int(r["rep"]), int(r["ell"])): r for r in rows}
        expected_keys = {(rep, ell) for rep in range(reps) for ell in grid}
        valid_keys = set(lookup) == expected_keys and len(rows) == len(expected_keys)
        truth_error = max(abs(float(r["true_entropy"])-analytic_truth) for r in rows)
        error_field_difference = max(abs(float(r["error"])-(float(r["estimate"])-analytic_truth)) for r in rows)
        valid_metadata = all(r["family"] == family and int(r["n"]) == n and int(r["d"]) == d
                             and r["phase"] == "evaluation" for r in rows)
        valid_coverage = all(float(r["coverage"]) == float(r["n_valid"])/n for r in rows)
        finite = all(np.isfinite(float(r[field])) for r in rows
                     for field in ["estimate", "coverage", "integrated_mass", "error"])
        raw_ok = (valid_keys and valid_metadata and valid_coverage and finite
                  and truth_error <= 1e-12 and error_field_difference <= 1e-12)
        raw_checks.append(dict(setting=name, group=setting["group"], csv_path=str(file), sha256=digest(file),
                               rows=len(rows), expected_rows=reps*len(grid), replicates=reps,
                               candidates=len(grid), unique_complete_rep_ell_keys=valid_keys,
                               metadata_valid=valid_metadata, coverage_matches_n_valid=valid_coverage,
                               all_metrics_finite=finite, max_truth_difference=truth_error,
                               max_error_field_difference=error_field_difference, passed=raw_ok))
        if not raw_ok:
            errors.append(f"{name}: saved raw validation failed")
            continue

        full = [ell for ell in grid if min(float(lookup[rep, ell]["coverage"]) for rep in range(reps)) == 1]
        if not full:
            errors.append(f"{name}: no full-coverage candidate for oracle")
            continue
        oracle = min(full, key=lambda ell: (np.mean([float(lookup[rep, ell]["error"])**2
                                                    for rep in range(reps)]), ell))
        bound = min(grid, key=lambda ell: (bound_objective(n, d, ell), ell))
        selected_rules.append(dict(family=family, n=n, d=d, no_partition=1,
                                   fixed_delta_bound=bound, full_coverage_rmse_oracle=oracle,
                                   saved_grid=grid, full_coverage_grid=full))
        for rep in [0, reps-1]:
            sequence = np.random.SeedSequence([setting["seed"], 1, setting["family_id"], n, d, rep])
            x = sampler.sample(np.random.default_rng(sequence), n, d, family)
            candidates = set([1, bound, oracle])
            sensitivity = rep == 0 and n in [10000, 100000]
            if sensitivity:
                candidates.update(grid)
            for ell in sorted(candidates):
                result = canonical.estimate(x, ell)
                saved = lookup[rep, ell]
                estimate_diff = abs(result["estimate"]-float(saved["estimate"]))
                mass_ref = float(saved["integrated_mass"])
                mass_diff = abs(result["integrated_mass"]-mass_ref)
                mass_relative_diff = mass_diff/abs(mass_ref) if mass_ref else mass_diff
                coverage_match = result["coverage"] == float(saved["coverage"])
                count_fields = ["n_valid", "occupied_cells", "min_cell_size", "singleton_cells"]
                counts_match = all(result[field] == float(saved[field]) for field in count_fields)
                ok = estimate_diff <= 1e-9 and coverage_match and counts_match and mass_relative_diff <= 1e-8
                roles = []
                if ell == 1: roles.append("no_partition")
                if ell == bound: roles.append("fixed_delta_bound")
                if ell == oracle: roles.append("full_coverage_rmse_oracle")
                if sensitivity: roles.append("sensitivity_all_saved_candidates")
                spotchecks.append(dict(setting=name, family=family, n=n, d=d, rep=rep, ell=ell,
                                       roles=";".join(roles), archived_estimate=float(saved["estimate"]),
                                       canonical_estimate=result["estimate"], estimate_abs_difference=estimate_diff,
                                       coverage_exact_match=coverage_match, cell_count_fields_match=counts_match,
                                       mass_abs_difference=mass_diff, mass_relative_difference=mass_relative_diff,
                                       passed=ok))
                if not ok:
                    errors.append(f"{name}: canonical mismatch at rep={rep}, ell={ell}")
        print(f"Audit {index}/{len(settings)}: {name}; bound={bound}, oracle={oracle}", flush=True)

    write_csv(args.out/"audit_raw_validation.csv", raw_checks)
    write_csv(args.out/"audit_spotchecks.csv", spotchecks)
    binary = WORKSPACE/"PSS"/("libpss_v2.dylib" if sys.platform == "darwin" else "libpss_v2.so")
    payload = dict(
        status="PASS" if not errors else "FAIL", completed_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-start,
        scope="Numerical spotcheck of archived estimates against current canonical PSS, not full-replicate recalculation.",
        numerical_scope="First and last evaluation replicates in every setting at ell=1, fixed-delta bound minimizer, and full-coverage aggregate RMSE oracle; all saved candidate ell for rep=0 at n=10000 and 100000.",
        limitations=["Unrecomputed replicates are reused on the basis of unchanged source provenance and the stated numerical spotchecks.",
                     "Oracle is selected and evaluated on the same archived replicates; it is a truth-tuned diagnostic.",
                     "The fixed-delta rule uses unit coefficients and delta=0.05 over each SAVED candidate grid; no finite-sample theorem certificate is claimed."],
        source_root=str(args.source), source_manifests=source_manifests, source_provenance=provenance,
        canonical_definition=canonical.VERSION,
        current_implementation_hashes={str(p.relative_to(WORKSPACE)): digest(p)
                                       for p in [WORKSPACE/"PSS/pss_v2.py", WORKSPACE/"PSS/pss_v2.cpp", binary]},
        sampler_seed_sequence="[archive_seed, 1, archived_family_id, n, d, rep]",
        fixed_delta_bound_objective="ell^-2 + sqrt(2*d*log(2*n+1)+log(96/0.05))*(ell^d/n)^0.25",
        tolerances=dict(estimate_absolute=1e-9, coverage="exact", cell_counts="exact", integrated_mass_relative=1e-8,
                        true_entropy_absolute=1e-12),
        entropy_checks=entropy_checks, setting_count=len(settings),
        saved_raw_row_count=sum(r["rows"] for r in raw_checks),
        regenerated_dataset_count=2*len(raw_checks), numerical_comparison_count=len(spotchecks),
        max_estimate_absolute_difference=max((r["estimate_abs_difference"] for r in spotchecks), default=None),
        max_mass_relative_difference=max((r["mass_relative_difference"] for r in spotchecks), default=None),
        all_coverage_exact=all(r["coverage_exact_match"] for r in spotchecks),
        selected_rules=selected_rules, errors=errors,
        audit_script_sha256=digest(__file__), python=sys.version.split()[0], numpy=np.__version__)
    (args.out/"audit.json").write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps({k: payload[k] for k in ["status", "setting_count", "saved_raw_row_count",
                                              "numerical_comparison_count", "max_estimate_absolute_difference",
                                              "max_mass_relative_difference", "elapsed_seconds", "errors"]}), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
