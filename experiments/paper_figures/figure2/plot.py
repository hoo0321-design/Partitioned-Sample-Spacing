#!/usr/bin/env python3
"""Verify and plot Figure 2's historical saved results, without rerunning PSS."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
SIZES = (1000, 3000, 10000, 30000)
METHODS = ("Oracle", "Stable CV")
REPS = 30


class VerificationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def close(actual, expected, label):
    require(math.isfinite(actual) and math.isfinite(expected), f"Nonfinite value: {label}")
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12),
            f"Value mismatch for {label}: recomputed {actual:.16g}, saved {expected:.16g}")


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def se(values):
    return statistics.stdev(values) / math.sqrt(len(values))


def row_key(row):
    return int(row["N_Samples"]), row["Tuning"], int(row["Replicate"])


def verify(bundle):
    """Return recomputed rows and a report; use no third-party dependencies."""
    manifest = json.loads((HERE / "provenance.json").read_text(encoding="utf-8"))
    paths = {}
    for role, item in manifest["files"].items():
        path = bundle / item["path"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == item["sha256"], f"SHA-256 mismatch: {item['path']}")
        paths[role] = path

    raw = read_rows(paths["replicates"])
    saved = read_rows(paths["summary"])
    evidence = read_rows(paths["condition_evidence"])
    expected = {(n, method, rep) for n in SIZES for method in METHODS
                for rep in range(1, REPS + 1)}
    keys = [row_key(row) for row in raw]
    require(len(raw) == 240 and len(set(keys)) == 240 and set(keys) == expected,
            "Replicates must contain exactly 30 unique rows per method and sample size")
    require(len(saved) == 8, "Expected eight summary rows")
    saved_map = {(int(row["N_Samples"]), row["Tuning"]): row for row in saved}
    require(set(saved_map) == {(n, method) for n in SIZES for method in METHODS},
            "Invalid or duplicate summary groups")

    groups = {(n, method): [] for n in SIZES for method in METHODS}
    for row in raw:
        n, method, _ = row_key(row)
        for name in ("ell", "cv_raw_nll", "cv_coverage", "stable_coverage",
                     "Estimate", "True_Entropy", "Error"):
            row[name] = float(row[name])
            require(math.isfinite(row[name]), f"Nonfinite replicate field: {name}")
        require(row["ell"] >= 1 and row["ell"].is_integer(), "Invalid selected ell")
        require(0 <= row["stable_coverage"] <= row["cv_coverage"] <= 1,
                "Invalid coverage ordering")
        close(row["Estimate"] - row["True_Entropy"], row["Error"], "replicate error")
        expected_rule = "oracle_rmse" if method == "Oracle" else "stable_coverage"
        require(row["selection_rule"] == expected_rule, "Unexpected archived selection rule")
        groups[n, method].append(row)

    # This companion table establishes d=5/rho=0 and shares the Oracle estimates.
    # Its other method is historical Constrained CV, not the plotted Stable CV.
    require(len(evidence) == 240, "Expected 240 companion condition-evidence rows")
    require(all(row["Distribution"] == "Gamma" and int(row["Dimensions"]) == 5
                and float(row["Correlation"]) == 0 for row in evidence),
            "Condition evidence does not identify Gamma, d=5, rho=0")
    source_oracle = {(int(row["N_Samples"]), int(row["Replicate"])): row
                     for row in evidence if row["Tuning"] == "Oracle"}
    require(len(source_oracle) == 120, "Expected 120 companion Oracle records")
    for n in SIZES:
        for row in groups[n, "Oracle"]:
            old = source_oracle[n, int(row["Replicate"])]
            for field in ("Estimate", "True_Entropy", "Error"):
                close(row[field], float(old[field]), f"condition-evidence {field}")
            close(row["ell"], float(old["Selected_Ell"]), "condition-evidence selected ell")

    computed = []
    for n in SIZES:
        oracle = groups[n, "Oracle"]
        require(len({row["ell"] for row in oracle}) == 1, "Oracle ell is not fixed across repeats")
        oracle_rmse = math.sqrt(statistics.mean(row["Error"] ** 2 for row in oracle))
        for method in METHODS:
            rows = groups[n, method]
            errors = [row["Error"] for row in rows]
            squares = [value ** 2 for value in errors]
            ell = [row["ell"] for row in rows]
            coverage = [row["cv_coverage"] for row in rows]
            stable = [row["stable_coverage"] for row in rows]
            rmse = math.sqrt(statistics.mean(squares))
            mse_se = se(squares)
            values = dict(
                N_Samples=n, Tuning=method,
                Selected_Ell_Mean=statistics.mean(ell), Selected_Ell_SE=se(ell),
                RMSE=rmse, MSE_SE=mse_se, RMSE_SE=mse_se / (2 * rmse),
                Bias=statistics.mean(errors),
                CV_Coverage_Mean=statistics.mean(coverage), CV_Coverage_SE=se(coverage),
                Stable_Coverage_Mean=statistics.mean(stable), Stable_Coverage_SE=se(stable),
                Fallback_Rate=0, N_Reps=len(rows), oracle_ell=oracle[0]["ell"],
                oracle_rmse_grid=oracle_rmse, RMSE_Ratio_vs_Oracle=rmse / oracle_rmse,
            )
            for field, value in values.items():
                if field not in ("N_Samples", "Tuning"):
                    close(value, float(saved_map[n, method][field]), f"{n}/{method}/{field}")
            computed.append(values)

    # In K-fold validation, the global minimum and maximum in any tie-free
    # coordinate are outside their own fold's training empirical range.
    # Consequently the v2 rule implies S <= 1 - 2/n for each data set.
    first = next(row for row in computed if row["N_Samples"] == 1000
                 and row["Tuning"] == "Stable CV")
    ceiling = 1 - 2 / 1000
    require(first["Stable_Coverage_Mean"] > ceiling,
            "Expected historical/v2 coverage discrepancy is absent")
    report = dict(
        status="verified_historical_saved_results", replicate_rows=len(raw),
        summary_rows=len(computed), matching_companion_oracle_rows=len(source_oracle),
        verified_input_sha256={key: item["sha256"] for key, item in manifest["files"].items()},
        simulation_rerun=False, original_generator_recovered=False,
        current_v2_coverage_rule_matches=False,
        coverage_evidence=dict(n=1000, historical_mean_stable_coverage=first["Stable_Coverage_Mean"],
                               v2_upper_bound=ceiling),
    )
    return computed, report


def render(rows, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.1))
    fields = (("Selected_Ell_Mean", "Selected_Ell_SE", r"$\ell$"),
              ("RMSE", "RMSE_SE", "RMSE"),
              ("Stable_Coverage_Mean", "Stable_Coverage_SE", r"$S(\ell)$"))
    for method, color, offset in (("Oracle", "#171717", -.045),
                                   ("Stable CV", "#ce651f", .045)):
        selected = [next(row for row in rows if row["N_Samples"] == n and row["Tuning"] == method)
                    for n in SIZES]
        for ax, (mean_field, se_field, label) in zip(axes, fields):
            ax.errorbar([i + offset for i in range(4)],
                        [row[mean_field] for row in selected],
                        yerr=[row[se_field] for row in selected],
                        color=color, marker="o", linewidth=1.8, markersize=5,
                        capsize=2, elinewidth=1, label=method)
            ax.set_ylabel(label)
    for ax in axes:
        ax.set_xticks(range(4), [str(n) for n in SIZES])
        ax.set_xlabel("n")
        ax.grid(alpha=.25)
        ax.set_axisbelow(True)
    axes[0].set_ylim(.8, 5.25)
    axes[0].set_yticks(range(1, 6))
    axes[1].set_ylim(.01, .225)
    axes[2].axhline(.99, color="#777777", linestyle="--", linewidth=1)
    axes[2].set_ylim(.974, 1.002)
    axes[2].set_yticks([.98, .99, 1.])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.subplots_adjust(left=.06, right=.99, bottom=.19, top=.84, wspace=.32)
    for suffix in ("png", "pdf"):
        fig.savefig(destination / f"figure2_historical.{suffix}", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true",
                        help="Verify hashes and saved statistics; write no files; no matplotlib needed")
    parser.add_argument("--output-dir", type=Path, default=HERE / "generated",
                        help="Destination for regenerated figure, summary and verification report")
    parser.add_argument("--input-dir", type=Path, default=HERE,
                        help="Bundle root containing data/ and archive/ (default: beside this script)")
    args = parser.parse_args()
    try:
        rows, report = verify(args.input_dir)
        if not args.verify_only:
            # Verify all inputs before creating output or importing plotting dependencies.
            args.output_dir.mkdir(parents=True, exist_ok=True)
            render(rows, args.output_dir)
            with (args.output_dir / "recomputed_summary.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (args.output_dir / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    except (OSError, ValueError, KeyError, csv.Error, ImportError) as error:
        print(f"Figure 2 verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
