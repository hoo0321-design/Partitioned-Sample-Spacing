"""Read-only verification of the bundled synthetic results (standard library).

This validates saved estimates, summaries, provenance and recorded checks. It
does not regenerate samples, refit estimators, or reproduce bootstrap intervals.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
RESULTS = PACKAGE / "results"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def close(actual, expected, message):
    require(math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-12),
            f"{message}: {actual} != {expected}")


def key(row, columns):
    return tuple(float(row[column]) for column in columns)


def validate_candidates(rows, expected_rows, columns, repetitions, first_rep=1):
    require(len(rows) == expected_rows, f"Expected {expected_rows} candidates, got {len(rows)}")
    grouped = defaultdict(list)
    for row in rows:
        require(row.get("status", "") in ("", "ok"), "Failed candidate retained in input")
        require(not row.get("failure"), "Nonempty candidate failure")
        estimate = float(row["estimate"])
        truth = float(row.get("truth") or row["true_entropy"])
        error = float(row["error"])
        coverage = float(row["coverage"])
        require(all(math.isfinite(v) for v in (estimate, truth, error, coverage)),
                "Nonfinite estimate, truth, error or coverage")
        require(0 <= coverage <= 1, "Coverage outside [0,1]")
        close(error, estimate - truth, "Candidate error")
        close(coverage, float(row["n_valid"]) / float(row["n"]), "Candidate coverage")
        n, d = int(float(row["n"])), int(float(row["d"]))
        if row["family"] == "Normal":
            rho = float(row["rho"])
            expected_truth = .5 * (d * math.log(2 * math.pi * math.e)
                                  + (d - 1) * math.log1p(-rho) + math.log1p((d - 1) * rho))
        else:
            require(row["family"] == "ridge_medium", "Unexpected bounded family")
            expected_truth = -(1 - math.sqrt(1 - .7 ** 2)
                               + math.log((1 + math.sqrt(1 - .7 ** 2)) / 2))
        close(truth, expected_truth, f"Exact entropy at n={n}, d={d}")
        grouped[key(row, columns)].append(row)
    for group_key, group in grouped.items():
        reps = {int(float(row.get("replicate") or row["rep"])) for row in group}
        count = repetitions(group_key) if callable(repetitions) else repetitions
        require(len(group) == count and reps == set(range(first_rep, first_rep + count)),
                f"Missing/duplicate replicate in {group_key}")
    return grouped


def compare_summary(groups, summary, columns):
    require(len(summary) == len(groups), "Summary group count mismatch")
    require(len({key(row, columns) for row in summary}) == len(summary), "Duplicate summary")
    require({key(row, columns) for row in summary} == set(groups), "Summary setting mismatch")
    for row in summary:
        group = groups[key(row, columns)]
        errors = [float(r["error"]) for r in group]
        coverage = [float(r["coverage"]) for r in group]
        rmse = math.sqrt(statistics.mean(e * e for e in errors))
        close(row["rmse"], rmse, "RMSE")
        close(row["bias"], statistics.mean(errors), "Bias")
        close(row["coverage_min"], min(coverage), "Minimum coverage")
        if "coverage_mean" in row:
            close(row["coverage_mean"], statistics.mean(coverage), "Mean coverage")
        require(int(row["reps"]) == len(group), "Summary repetition count")
        if row.get("rmse_low") and row.get("rmse_high"):
            require(float(row["rmse_low"]) - 1e-12 <= rmse <= float(row["rmse_high"]) + 1e-12,
                    "Stored bootstrap interval does not contain the saved RMSE")
    return len(summary)


def verify_checkpoints(directory, rows, expected_datasets, columns):
    checkpoints = sorted((directory / "checkpoints").glob("c*_r*.csv"))
    audits = sorted((directory / "checkpoints").glob("c*_r*.json"))
    require(len(checkpoints) == len(audits) == expected_datasets, "Missing dataset checkpoint")
    values = {key(r, columns): r for r in rows}
    require(len(values) == len(rows), "Duplicate consolidated candidate")
    seen = set()
    for path in checkpoints:
        for row in read_csv(path):
            row_key = key(row, columns)
            require(row_key in values and row_key not in seen, "Unexpected checkpoint candidate")
            seen.add(row_key)
            for field in ("estimate", "truth", "error", "coverage", "n_valid"):
                close(row[field], values[row_key][field], "Checkpoint " + field)
    require(seen == set(values), "Incomplete checkpoints")
    hashes = set()
    for path in audits:
        audit = read_json(path)
        require(audit["status"] == "ok" and audit["all_finite"] is True, "Failed dataset audit")
        require(not audit["failure"], "Failed dataset")
        if "in_support" in audit:
            require(audit["in_support"] is True, "Dataset outside support")
        hashes.add(audit["array_sha256"])
    require(len(hashes) == expected_datasets, "Repeated sample hash")


def verify_recorded_audits():
    positive = {"passed", "array_sha256_match", "full_expected_counts", "all_coverage_exact",
                "all_generated_datasets_finite", "all_generated_datasets_in_support",
                "archived_sampler_hash_verified", "canonical_hashes_verified",
                "code_and_binary_unchanged", "core_source_unchanged", "binary_unchanged",
                "baseline_reproduced", "source_hashes_match", "full_archive_minima_preserved",
                "previous_intervals_preserved"}
    checks = 0

    def walk(obj):
        nonlocal checks
        if isinstance(obj, dict):
            for name, value in obj.items():
                if name in positive:
                    require(value is True, f"Saved audit {name} is not true")
                    checks += 1
                elif name in {"failed_fits", "failed_datasets", "failures"}:
                    require(value == 0, f"Saved audit {name} is nonzero")
                    checks += 1
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)

    for path in sorted(RESULTS.glob("*/*audit.json")):
        audit = read_json(path)
        if "status" in audit:
            require(audit["status"] == "PASS", f"Audit status: {path}")
            checks += 1
        walk(audit)
        for field, filename in [("protocol_sha256", "protocol.json"),
                                ("candidates_sha256", "candidates.csv"),
                                ("raw_sha256", "candidates.csv")]:
            if field in audit:
                require(digest(path.parent / filename) == audit[field], f"Audit hash: {filename}")
                checks += 1
    spotchecks = read_json(RESULTS / "synthetic_gaussian_rho_20261006/checkpoints/baseline_spotchecks.json")
    require(len(spotchecks) == 8, "Missing baseline spotchecks")
    walk(spotchecks)
    return checks


def verify_legacy():
    directory = RESULTS / "five_family_legacy_20261006"
    rows = read_csv(directory / "selected_replicates.csv")
    summaries = read_csv(directory / "summary.csv")
    columns = ["Experiment", "Distribution", "Dimensions", "N_Samples", "Correlation", "Method"]
    group_key = lambda row: tuple(row[c] for c in columns)
    groups = defaultdict(list)
    require(len(rows) == 9900 and len(summaries) == 330, "Five-family row counts")
    for row in rows:
        groups[group_key(row)].append(row)
    require(set(groups) == {group_key(r) for r in summaries}, "Five-family setting mismatch")
    for summary in summaries:
        group = groups[group_key(summary)]
        require(len(group) == 30 and {int(r["Replicate"]) for r in group} == set(range(1, 31)),
                "Five-family repetitions")
        errors = [float(r["Estimate"]) - float(r["True_Entropy"]) for r in group]
        squared = [e * e for e in errors]
        times = [float(r["Eval_Time_s"]) + float(r["Train_Time_s"]) for r in group]
        rmse = math.sqrt(statistics.mean(squared))
        close(summary["RMSE"], rmse, "Legacy RMSE")
        close(summary["RMSE_SE"], statistics.stdev(squared) / math.sqrt(30) / (2 * rmse), "Legacy RMSE SE")
        close(summary["Bias"], statistics.mean(errors), "Legacy bias")
        close(summary["Mean_Time_s"], statistics.mean(times), "Legacy runtime")
        close(summary["Time_SE_s"], statistics.stdev(times) / math.sqrt(30), "Legacy runtime SE")
    return {"selected_replicates": len(rows), "recomputed_summaries": len(summaries)}


def main():
    manifest = read_json(PACKAGE / "manifest.json")
    for record in manifest["files"]:
        path = ROOT / record["path"]
        require(path.stat().st_size == record["bytes"] and digest(path) == record["sha256"],
                f"Publication copy changed: {record['path']}")
    expected_core = read_json(RESULTS / "synthetic_gaussian_lowdim_20261006/protocol.json")["code_sha256"]
    for path in ("PSS/pss_v2.py", "PSS/pss_v2.cpp"):
        require(digest(ROOT / path) == expected_core[path], "Canonical source changed: " + path)

    rho_dir = RESULTS / "synthetic_gaussian_rho_20261006"
    rho_paths = sorted((rho_dir / "checkpoints").glob("candidates_*.csv"))
    require(len(rho_paths) == 48, "Expected 48 Gaussian correlation tables")
    rho_rows = [r for p in rho_paths for r in read_csv(p)]
    rho_groups = validate_candidates(rho_rows, 46800, ["rho", "n", "d", "ell"], 30)
    rho_summary = compare_summary(rho_groups, read_csv(rho_dir / "per_ell_summary.csv"), ["rho", "n", "d", "ell"])
    base_groups = validate_candidates([r for r in rho_rows if float(r["rho"]) == .5],
                                      7800, ["n", "d", "ell"], 30)
    base_summary = compare_summary(base_groups, read_csv(RESULTS / "synthetic_gaussian_20261006/per_ell_summary.csv"), ["n", "d", "ell"])

    low_dir = RESULTS / "synthetic_gaussian_lowdim_20261006"
    low_rows = read_csv(low_dir / "candidates.csv")
    low_groups = validate_candidates(low_rows, 1200, ["n", "d", "ell"], 30)
    low_summary = compare_summary(low_groups, read_csv(low_dir / "per_ell_summary.csv"), ["n", "d", "ell"])
    verify_checkpoints(low_dir, low_rows, 150, ["n", "d", "replicate", "ell"])

    ext_dir = RESULTS / "synthetic_bounded_fig2_extension_20261006"
    ext_rows = read_csv(ext_dir / "candidates.csv")
    ext_groups = validate_candidates(ext_rows, 1400, ["n", "d", "ell"], 100, 0)
    verify_checkpoints(ext_dir, ext_rows, 200, ["n", "d", "replicate", "ell"])
    compact = read_csv(RESULTS / "synthetic_bounded_fig2_compact_20261006/per_ell_summary.csv")
    ext_summary = compare_summary(ext_groups, [r for r in compact if int(r["d"]) in (3, 4)], ["n", "d", "ell"])

    bounded_figures = {}
    for run, count in [("synthetic_bounded_fig2_20261006", 2310),
                       ("synthetic_bounded_fig2_extended_20261006", 4410),
                       ("synthetic_bounded_fig2_compact_20261006", 3500)]:
        directory = RESULTS / run
        rows = read_csv(directory / "plotted_replicates.csv")
        groups = validate_candidates(rows, count, ["n", "d", "ell"],
                                     lambda k: 30 if k[0] == 1000000 else 100, 0)
        bounded_figures[run] = {"plotted_candidate_rows": count,
                               "recomputed_summaries": compare_summary(groups, read_csv(directory / "per_ell_summary.csv"), ["n", "d", "ell"])}

    print(json.dumps({"status": "PASS", "verified_manifest_files": len(manifest["files"]),
                      "recorded_audit_checks": verify_recorded_audits(),
                      "gaussian_rho": {"candidate_rows": len(rho_rows), "recomputed_summaries": rho_summary},
                      "gaussian_baseline": {"candidate_rows": 7800, "recomputed_summaries": base_summary},
                      "gaussian_lowdim": {"candidate_rows": len(low_rows), "recomputed_summaries": low_summary},
                      "bounded_extension": {"candidate_rows": len(ext_rows), "recomputed_summaries": ext_summary},
                      "bounded_figures": bounded_figures, "five_family_legacy": verify_legacy(),
                      "new_estimator_fits": 0, "bootstrap_intervals_recomputed": False}, indent=2))


if __name__ == "__main__":
    main()
