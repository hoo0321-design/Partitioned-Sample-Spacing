"""Post-hoc, training-only diagnostics on frozen feature paths, not a new evaluation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import core


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((args.source / "protocol.json").read_text())
    data_path = core.ROOT / "data/energydata_complete.csv"
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == protocol["data_sha256"]
    data = pd.read_csv(data_path)
    names = [c for c in data if c not in ["date", "Appliances", "rv1", "rv2"]]
    x = data[names].to_numpy(float)
    target = data.Appliances.to_numpy(float)
    core.MINIMUMS = [2, 5, 10]
    candidates, decisions, denominator = [], [], []
    causes = {"coverage_excludes_better_nll": 0, "nll_prefers_ell1": 0}
    for seed in protocol["seeds"]:
        checkpoint = args.source / "checkpoints"
        split = json.loads((checkpoint / f"split_{seed}.json").read_text())
        train = np.asarray(split["outer_train"])
        saved = json.loads((checkpoint / f"outer_{seed}_PSS_SC-CV_path.json").read_text())
        config = saved["config"]
        y = (target[train] > np.median(target[train])).astype(int)
        z, active = core.selection_data(x[train], config["noise"], seed+5000)
        selector = core.Selector(z, y, active, seed+6000)
        for row in saved["history"]:
            features = row["features"]
            table = selector.table(features)
            choice = core.choose(table, config)
            assert choice["ell"] == row["ell"]
            assert np.isclose(choice["cv_score"], row["cv_score"], rtol=0, atol=1e-10)
            unrestricted = min(table, key=lambda r: (r["cv_score"], r["ell"]))
            if choice["ell"] == 1:
                reason = ("coverage_excludes_better_nll" if unrestricted["ell"] > 1
                          else "nll_prefers_ell1")
                causes[reason] += 1
            for candidate in table:
                candidates.append(dict(seed=seed, step=row["step"], **candidate))
            for tau in [.80, .85, .90, .95, .99]:
                for minimum in core.MINIMUMS:
                    result = core.choose(table, dict(tau=tau, n_min=minimum))
                    decisions.append(dict(seed=seed, step=row["step"], tau=tau,
                                          n_min=minimum, ell=result["ell"],
                                          stable_coverage=result["stable_coverage"],
                                          fallback=result["fallback"]))
            for label, ell in [("original_selected_ell", row["ell"]), ("fixed_ell2", 2)]:
                subset = z[:, features]
                entropy = [core.estimate(subset, ell)] + [
                    core.estimate(subset[y == c], ell) for c in [0, 1]]
                weights = np.array([1., -np.mean(y == 0), -np.mean(y == 1)])
                h = np.array([r["estimate"] for r in entropy])
                coverage = np.array([r["coverage"] for r in entropy])
                # Same log densities and valid points; isolate ONLY the denominator.
                score_neff = float(weights @ h)
                score_n = float(weights @ (coverage*h))
                denominator.append(dict(seed=seed, step=row["step"], rule=label, ell=ell,
                                        pooled_coverage=coverage[0],
                                        min_conditional_coverage=float(coverage[1:].min()),
                                        score_neff=score_neff, score_n=score_n,
                                        absolute_difference=abs(score_neff-score_n)))
        print(json.dumps(dict(seed=seed, status="replayed")), flush=True)
    pd.DataFrame(candidates).to_csv(args.output / "candidate_tables.csv", index=False)
    decisions = pd.DataFrame(decisions)
    decisions.to_csv(args.output / "fixed_path_sensitivity.csv", index=False)
    denominator = pd.DataFrame(denominator)
    denominator.to_csv(args.output / "denominator_only.csv", index=False)
    summary = dict(
        scope="Post-hoc diagnostics on 100 frozen selected subsets. No new feature selection, classifier fits, or test evaluation. Different thresholds can change feature paths; these are not rerun performance estimates.",
        ell1_causes=causes,
        threshold_sensitivity=[dict(tau=float(tau), n_min=int(minimum), ell1_count=int(g.ell.eq(1).sum()),
                                    fallback_count=int(g.fallback.sum()))
                               for (tau, minimum), g in decisions.groupby(["tau", "n_min"])],
        denominator_summary=[dict(rule=rule, steps=int(step), mean_abs_difference=float(g.absolute_difference.mean()),
                                  max_abs_difference=float(g.absolute_difference.max()),
                                  min_pooled_coverage=float(g.pooled_coverage.min()),
                                  min_conditional_coverage=float(g.min_conditional_coverage.min()))
                             for (rule, step), g in denominator.groupby(["rule", "step"])
                             if step in [5, 10, 20]],
        denominator_caveat="H_n = coverage * H_Neff only for identical density values and valid sets. This isolates the denominator and is NOT an old-versus-new estimator comparison.")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
