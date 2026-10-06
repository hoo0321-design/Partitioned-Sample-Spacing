"""Publication Fig. 2: audited bounded dependent-density settings.

The common display grid is 1..7; full archived grids are summarized as a
check against truncating an empirical minimum. No coverage filter is applied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("/Users/hojeongwoo/Documents/Codex/2026-05-03/"
              "d-dimensional-partitioned-sample-spacing-pss/Partitioned-Sample-Spacing")
OUT = ROOT / "results/synthetic_bounded_fig2_20261006"
PDFDIR = ROOT / "output/pdf/synthetic_bounded_fig2_20261006"
PAIRS = [(100000, 2), (100000, 5), (10000, 2), (1000000, 2)]
GRID = list(range(1, 8))
AMPLITUDE = .7
TRUTH = np.sqrt(1-AMPLITUDE**2)-1-np.log((1+np.sqrt(1-AMPLITUDE**2))/2)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def theory_objective(n, d, ell):
    lam = 2*d*np.log(2*n+1)+np.log(96/.05)
    return ell**-2 + np.sqrt(lam)*(ell**d/n)**.25


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extended", action="store_true", help="Add d=3,4 at n=100000 and n=1000 at d=2")
    parser.add_argument("--compact", action="store_true", help="Show only d=2,3,4 and n=1000,10000,100000 from the extended results")
    args = parser.parse_args()
    if args.compact:
        args.extended = True
    out = ROOT / "results/synthetic_bounded_fig2_extended_20261006" if args.extended else OUT
    pdfdir = ROOT / "output/pdf/synthetic_bounded_fig2_extended_20261006" if args.extended else PDFDIR
    pairs = [(100000, d) for d in [2, 3, 4, 5]] + [(n, 2) for n in [1000, 10000, 1000000]] if args.extended else PAIRS
    if args.compact:
        out = ROOT / "results/synthetic_bounded_fig2_compact_20261006"
        pdfdir = ROOT / "output/pdf/synthetic_bounded_fig2_compact_20261006"
        pairs = [(100000, d) for d in [2, 3, 4]] + [(n, 2) for n in [1000, 10000]]
    out.mkdir(parents=True, exist_ok=True)
    pdfdir.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    PDFDIR.mkdir(parents=True, exist_ok=True)
    audit_path = ROOT / "results/synthetic_bounded_20261006/audit.json"
    prior_audit = json.loads(audit_path.read_text())
    prior_manifest = json.loads((ROOT / "results/synthetic_bounded_20261006/analysis_manifest.json").read_text())
    assert prior_audit["status"] == "PASS"
    for name, digest in prior_audit["current_implementation_hashes"].items():
        assert sha(ROOT/name) == digest, f"Canonical implementation changed: {name}"
    files, frames, settings = {}, [], []
    extension = None
    if args.extended:
        extension_dir = ROOT / "results/synthetic_bounded_fig2_extension_20261006"
        extension = pd.read_csv(extension_dir / "candidates.csv")
        extension_audit = json.loads((extension_dir / "audit.json").read_text())
        assert extension_audit["failed_fits"] == 0 and extension_audit["failed_datasets"] == 0
        assert extension_audit["datasets"] == 200 and extension_audit["candidate_rows"] == 1400
        assert extension_audit["code_and_binary_unchanged"] and extension_audit["canonical_hashes_verified"]
        assert extension_audit["protocol_sha256"] == sha(extension_dir / "protocol.json")
        assert extension_audit["candidates_sha256"] == sha(extension_dir / "candidates.csv")
        assert extension_audit["all_generated_datasets_in_support"]
        assert len(extension) == 1400 and extension.groupby(["n", "d", "ell"]).size().eq(100).all()
        assert set(map(tuple, extension[["n", "d"]].drop_duplicates().to_numpy())) == {(100000, 3), (100000, 4)}
        for name in ["candidates.csv", "protocol.json", "audit.json"]:
            files[str(extension_dir / name)] = sha(extension_dir / name)
        base_per = pd.read_csv(OUT / "per_ell_summary.csv")
    for n, d in pairs:
        if args.extended and (n, d) in [(100000, 3), (100000, 4)]:
            raw = extension[(extension.n == n) & (extension.d == d)].copy()
            archive_grid, reps, origin = GRID, 100, "new canonical simulations"
            assert not raw.duplicated(["rep", "ell"]).any()
            assert set(raw.ell) == set(GRID) and set(raw.rep) == set(range(100))
            assert raw.family.eq("ridge_medium").all()
            assert np.isfinite(raw[["estimate", "error", "coverage"]]).all().all()
            np.testing.assert_allclose(raw.true_entropy, TRUTH, rtol=0, atol=1e-14)
            np.testing.assert_allclose(raw.error, raw.estimate-TRUTH, rtol=0, atol=1e-13)
            np.testing.assert_allclose(raw.coverage, raw.n_valid/n, rtol=0, atol=1e-14)
            frames.append(raw)
            settings.append(dict(n=n, d=d, reps=reps, archive_grid=archive_grid, origin=origin))
            continue
        run = "theory_selection_20261003" if n <= 100000 else "theory_selection_large_n_20261003"
        directory = SOURCE / "results" / run
        design_path = directory / "design.json"
        assert sha(design_path) == prior_manifest["source_files"][str(design_path)]
        design = json.loads(design_path.read_text())
        cfg = next(s for s in design["settings"] if (s["family"], s["n"], s["d"]) == ("ridge_medium", n, d))
        path = directory / "evaluation" / f"ridge_medium_n{n}_d{d}.csv"
        assert sha(path) == prior_manifest["source_files"][str(path)]
        for p in (path, design_path):
            files[str(p)] = sha(p)
        raw = pd.read_csv(path)
        assert not raw.duplicated(["rep", "ell"]).any()
        assert raw.n.eq(n).all() and raw.d.eq(d).all() and raw.family.eq("ridge_medium").all()
        assert set(raw.ell) == set(cfg["ells"]) and set(GRID) <= set(raw.ell)
        assert raw.groupby("ell").size().eq(design["eval_reps"]).all()
        assert set(raw.rep) == set(range(design["eval_reps"]))
        assert np.isfinite(raw[["estimate", "error", "coverage"]]).all().all()
        np.testing.assert_allclose(raw.true_entropy, TRUTH, rtol=0, atol=1e-14)
        np.testing.assert_allclose(raw.error, raw.estimate-TRUTH, rtol=0, atol=1e-13)
        np.testing.assert_allclose(raw.coverage, raw.n_valid/n, rtol=0, atol=1e-14)
        frames.append(raw)
        settings.append(dict(n=n, d=d, reps=design["eval_reps"], archive_grid=cfg["ells"], origin="archived simulations"))
    raw = pd.concat(frames, ignore_index=True)
    selected_raw = raw[raw.ell.isin(GRID)].copy()
    # Saved repetitions are the independent units; the same data underlie all ell in a repetition.
    rng = np.random.default_rng(2026100606)
    rows = []
    for (n, d, ell), group in raw.groupby(["n", "d", "ell"], sort=True):
        errors = group.sort_values("rep").error.to_numpy()
        row = dict(n=int(n), d=int(d), ell=int(ell), reps=len(errors),
                   rmse=float(np.sqrt(np.mean(errors**2))), bias=float(errors.mean()),
                   coverage_min=float(group.coverage.min()), coverage_mean=float(group.coverage.mean()))
        if ell in GRID:
            if args.extended and (n, d) in PAIRS:
                old = base_per[(base_per.n == n) & (base_per.d == d) & (base_per.ell == ell)].iloc[0]
                np.testing.assert_allclose(row["rmse"], old.rmse, rtol=0, atol=1e-14)
                row["rmse_low"], row["rmse_high"] = float(old.rmse_low), float(old.rmse_high)
            else:
                bootstrap_rng = np.random.default_rng(np.random.SeedSequence([2026100608, int(n), int(d), int(ell)])) if args.extended else rng
                boot = errors[bootstrap_rng.integers(len(errors), size=(2000, len(errors)))]
                row["rmse_low"], row["rmse_high"] = map(float, np.quantile(np.sqrt(np.mean(boot**2, axis=1)), [.025, .975]))
        rows.append(row)
    full = pd.DataFrame(rows)
    per = full[full.ell.isin(GRID)].copy()
    comparisons = []
    for n, d in pairs:
        group = per[(per.n == n) & (per.d == d)].sort_values("ell")
        archived = full[(full.n == n) & (full.d == d)].sort_values("ell")
        best = group.sort_values(["rmse", "ell"]).iloc[0]
        full_best = archived.sort_values(["rmse", "ell"]).iloc[0]
        theory_ell = min(GRID, key=lambda ell: (theory_objective(n, d, ell), ell))
        full_theory = min(archived.ell, key=lambda ell: (theory_objective(n, d, float(ell)), ell))
        theory = group[group.ell == theory_ell].iloc[0]
        comparisons.append(dict(n=n, d=d, reps=int(best.reps), empirical_ell=int(best.ell),
                                empirical_rmse=float(best.rmse), theory_ell=theory_ell,
                                theory_rmse=float(theory.rmse), theory_rmse_ratio=float(theory.rmse/best.rmse),
                                match=theory_ell == int(best.ell), full_archive_empirical_ell=int(full_best.ell),
                                full_archive_theory_ell=int(full_theory), archive_ell_max=int(archived.ell.max()),
                                empirical_coverage_min=float(best.coverage_min),
                                empirical_at_display_upper=int(best.ell) == max(GRID)))
    comp = pd.DataFrame(comparisons)
    assert comp.empirical_ell.eq(comp.full_archive_empirical_ell).all(), "Display grid omits an archived RMSE minimum"
    assert comp.theory_ell.eq(comp.full_archive_theory_ell).all(), "Display grid changes theoretical selection"
    assert not comp.empirical_at_display_upper.any(), "An upper-boundary empirical minimum requires review"
    selected_raw.to_csv(out / "plotted_replicates.csv", index=False)
    full.to_csv(out / "full_archive_summary.csv", index=False)
    per.to_csv(out / "per_ell_summary.csv", index=False)
    comp.to_csv(out / "optimal_ell_comparison.csv", index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.titlesize": 8.8, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 300})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15))
    panels = [[(100000, 2), (100000, 5)], [(10000, 2), (100000, 2), (1000000, 2)]]
    palette = [["#D55E00", "#0072B2"], ["#0072B2", "#D55E00", "#009E73"]]
    if args.extended:
        panels = [[(100000, d) for d in [2, 3, 4, 5]], [(n, 2) for n in [1000, 10000, 100000, 1000000]]]
        palette = [["#D55E00", "#009E73", "#AA4499", "#0072B2"], ["#AA4499", "#0072B2", "#D55E00", "#009E73"]]
    if args.compact:
        panels = [[(100000, d) for d in [2, 3, 4]], [(n, 2) for n in [1000, 10000, 100000]]]
        palette = [["#D55E00", "#009E73", "#AA4499"], ["#AA4499", "#0072B2", "#D55E00"]]
    for index, pairs in enumerate(panels):
        ax = axes[index]
        for (n, d), color in zip(pairs, palette[index]):
            group = per[(per.n == n) & (per.d == d)].sort_values("ell")
            choice = comp[(comp.n == n) & (comp.d == d)].iloc[0]
            label = f"$d={d}$" if index == 0 else f"$n=10^{{{int(np.log10(n))}}}$"
            ax.plot(group.ell, group.rmse, color=color, lw=1.4, marker="o", markersize=2.6, label=label)
            ax.fill_between(group.ell.to_numpy(), group.rmse_low.to_numpy(), group.rmse_high.to_numpy(), color=color, alpha=.14, lw=0)
            sparse = group[group.coverage_min < 1]
            ax.scatter(sparse.ell, sparse.rmse, marker="o", s=13, facecolors="white", edgecolors=color, linewidths=.75, zorder=4)
            ax.scatter([choice.empirical_ell], [choice.empirical_rmse], color=color, marker="*", s=90, edgecolors="black", linewidths=.5, zorder=6)
            ax.scatter([choice.theory_ell], [choice.theory_rmse], facecolors="none", edgecolors=color, marker="s", s=76, linewidths=1.2, zorder=5)
        ax.set(yscale="log", xticks=GRID, xlabel="Partitions per coordinate $\\ell$", ylabel="Entropy RMSE (nats)")
        ax.set_title("(a) Fixed $n=100,000$; varying dimension" if index == 0 else "(b) Fixed $d=2$; varying sample size")
        ax.grid(alpha=.22)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="upper left" if index == 0 or args.compact else "lower left", fontsize=8, handlelength=1.8,
                  handletextpad=.55, labelspacing=.35, borderaxespad=.6)
    handles = [Line2D([], [], color="black", marker="*", linestyle="none", markersize=9, label="RMSE-minimizing $\\ell$"),
               Line2D([], [], color="black", marker="s", markerfacecolor="none", linestyle="none", markersize=7, label="Theory-selected $\\ell$ ($\\delta=0.05$)")]
    fig.legend(handles=handles, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.0), fontsize=9)
    fig.subplots_adjust(left=.085, right=.988, top=.79, bottom=.17, wspace=.29)
    fig.savefig(pdfdir / "bounded_ell_rmse.pdf")
    fig.savefig(out / "bounded_ell_rmse.png")
    plt.close(fig)

    # Analytic entropy is independently checked by tensor Gauss-Legendre quadrature.
    x, w = np.polynomial.legendre.leggauss(128)
    x, w = (x+1)/2, w/2
    density = 1+AMPLITUDE*np.cos(2*np.pi*(x[:, None]-x[None, :]))
    quadrature = -float(w @ (density*np.log(density)) @ w)
    assert abs(quadrature-TRUTH) < 1e-12
    assert all(sha(path) == digest for path, digest in files.items())
    audit = dict(status="PASS", study="Bounded-dependent Fig. 2 with requested dimension/sample-size extension" if args.extended else "Posthoc presentation of existing bounded-dependent simulations",
                 density="1+0.7*cos(2*pi*(x1-x2)) on [0,1]^d", density_lower=.3, density_upper=1.7,
                 lipschitz_constant=float(2*np.pi*AMPLITUDE*np.sqrt(2)), exact_entropy=float(TRUTH),
                 quadrature_entropy=quadrature, settings=settings, display_grid=GRID,
                 unique_datasets=len(raw[["n", "d", "rep"]].drop_duplicates()),
                 saved_candidate_rows=len(raw), archived_candidate_rows=len(raw) - (1400 if args.extended else 0),
                 new_candidate_rows=1400 if args.extended else 0, plotted_candidate_rows=len(selected_raw),
                 theory_matches=int(comp.match.sum()), settings_count=len(comp),
                 full_archive_minima_preserved=True, coverage_filter=False,
                 minimum_display_coverage=float(selected_raw.coverage.min()),
                 selected_empirical_minimum_coverage=float(comp.empirical_coverage_min.min()),
                 bootstrap_seed=2026100606, bootstrap_extension_seed_namespace=2026100608 if args.extended else None,
                 previous_intervals_preserved=bool(args.extended), bootstrap_resamples=2000, source_files=files,
                 prior_canonical_audit_path=str(audit_path), prior_canonical_audit_sha256=sha(audit_path),
                 canonical_validation="Archived results verified by prior spotchecks; extended d=3,4 computed directly with unchanged canonical estimator" if args.extended else "Reuse of prior source and numerical spotchecks, not a full-replicate rerun",
                 selector="Minimize ell^-2+sqrt(2*d*log(2*n+1)+log(96/.05))*(ell^d/n)^.25; common K*d cancels",
                 theorem_scope="Distributional regularity holds; finite-sample admissibility involving unspecified K0 is not certified",
                 script_sha256=sha(__file__))
    if args.compact:
        audit["study"] = "Requested five-setting presentation subset of the seven-setting extended Fig. 2"
        audit["new_simulations_for_this_figure"] = 0
        audit["full_expanded_results"] = str(ROOT / "results/synthetic_bounded_fig2_extended_20261006")
    (out / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    notes = ["Bounded-dependent Fig. 2", "",
             ("Requested subset of existing seven-setting results; three archived settings and two already-computed extended dimensions. No new simulation or coefficient fitting."
              if args.compact else "Five archived settings plus two new dimensions; no coefficient fitting." if args.extended else "Reuses four existing settings; no new simulation and no coefficient fitting."),
             "Main figure: ell=1..7, with unrestricted same-repeat RMSE minima (no coverage filter).",
             "All displayed empirical and theory-selected minima equal those over every available saved candidate grid. New d=3,4 grids are ell=1..7.",
             "Stars are descriptive RMSE minima, not independently evaluated tuned selectors.",
             f"Empty circles indicate at least one repetition with coverage below 1. Minimum displayed coverage is {selected_raw.coverage.min():.8f}.",
             f"Minimum per-repeat coverage of empirical-minimum candidates: {comp.empirical_coverage_min.min():.8f}.",
             "For d>2, d-2 independent uniform coordinates are added to the same dependent pair.",
             "Lipschitz regularity holds on the cube; no continuity of the zero extension is asserted.",
             "The density satisfies the theorem's distributional assumptions with dimension-independent c,C,L.",
             "The bound's common positive factor K*d cancels from the ell minimization; it need not be estimated.",
             "The theorem's sample-size requirements involving unspecified K0 are not certified.",
             ("100 repeats for every displayed setting. Pointwise 95% bootstrap intervals, 2,000 resamples."
              if args.compact else "100 repeats for n<=100,000; 30 for n=1,000,000. Pointwise 95% bootstrap intervals, 2,000 resamples."),
             f"Shared setting n=100,000,d=2 appears in both panels. {len(comp)} unique settings.", "",
             comp.to_string(index=False)]
    (out / "notes.txt").write_text("\n".join(notes)+"\n")
    print(comp.to_string(index=False))
    print(json.dumps({k: audit[k] for k in ["status", "unique_datasets", "archived_candidate_rows", "plotted_candidate_rows", "minimum_display_coverage"]}))


if __name__ == "__main__":
    main()
