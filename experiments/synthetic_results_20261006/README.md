# Synthetic experiment archive, 2026-10-06

This package publishes the saved bounded-density, Gaussian and historical
five-family experiments together with their raw estimate tables, protocols,
audits, figures and manuscript fragments. The files were copied without
changing estimates, seeds, intervals or selected partition counts. Simulation
datasets are represented by estimate records, fixed seed schedules and sample
hashes; the generated observation arrays are not bundled.

The portable entry points are the read-only verifier below and the existing
[combined figure package](../synthetic_ell_comparison/README.md). The historical
experiment scripts are preserved for provenance and require the original
execution environment to rerun unchanged; see the limitations below.

## Verify saved results

From the repository root, with Python 3.9 or newer:

```sh
python experiments/synthetic_results_20261006/verify.py
```

The verifier uses only the Python standard library. It reads paths relative to
its own location, writes no files and performs no estimator fits. It checks:

- SHA-256 hashes and byte counts for all 903 copied files in `manifest.json`,
  and the canonical `PSS/pss_v2.py` and `PSS/pss_v2.cpp` source hashes.
- All 46,800 Gaussian correlation candidates and 1,560 RMSE/bias/coverage
  summaries, including the 7,800 rho=0.5 candidates underlying the earlier
  Gaussian figure's 260 summaries.
- All 1,200 low-dimensional Gaussian candidates and 40 summaries, with all
  150 per-dataset CSV and JSON checkpoints.
- All 1,400 bounded-extension candidates and 14 summaries, with all 200
  per-dataset CSV and JSON checkpoints.
- All 2,310, 4,410 and 3,500 plotted candidate rows for the initial, extended
  and compact bounded figures, respectively; these intentionally overlap.
- The five-family figure's 9,900 selected replicate records, recomputing all
  330 RMSE, Monte Carlo SE, bias and runtime summaries.
- Recorded audit success flags, available protocol/candidate hashes and eight
  archived Gaussian baseline spotchecks. These checks validate the saved audit
  records; they do not repeat the historical estimator spotchecks.

Stored bootstrap intervals are preserved and checked for consistency with the
point estimates, with floating-point tolerance. Bootstrap resampling and
simulation generation are not repeated. The wider bounded study has only its
saved summaries and spotcheck records here, so its full historical candidate
archive is outside the verifier's scope.

## Reproduce the current four-panel figure

```sh
python -m pip install -r experiments/synthetic_ell_comparison/requirements.txt
python experiments/synthetic_ell_comparison/plot.py --output-dir /tmp/pss-synthetic-figure
```

That plotter is self-contained: it validates the eight frozen table hashes,
repetitions, candidate grids, empirical minima and displayed theoretical rule,
then exports the vector PDF, PNG and `figure_checks.json`. It requires NumPy,
pandas and Matplotlib, and does not require this larger archive or a compiled
PSS library. See the [figure README](../synthetic_ell_comparison/README.md) for
the full statistical interpretation and figure-specific provenance.

## Result index

All directories below are under [`results/`](results/). Run names retain their
original dates. Exact counts describe saved records, not independent validation
datasets, and must not be added across overlapping analyses.

| Run directory | Saved material and scope |
| --- | --- |
| [`synthetic_bounded_20261006`](results/synthetic_bounded_20261006/) | Wider bounded ridge/frequency comparison: per-ell summaries, selected summaries, numerical spotchecks, source provenance, plots and notes. The 40,480 original candidate records cited by its audit are not all bundled. |
| [`synthetic_bounded_fig2_20261006`](results/synthetic_bounded_fig2_20261006/) | Initial four-setting figure: 330 datasets, 2,310 plotted candidate records, full-grid summaries and selection table. |
| [`synthetic_bounded_fig2_extension_20261006`](results/synthetic_bounded_fig2_extension_20261006/) | New d=3,4 bounded simulations: 200 datasets, 1,400 candidate records, all per-dataset checkpoints, seed schedule and frozen protocol. |
| [`synthetic_bounded_fig2_extended_20261006`](results/synthetic_bounded_fig2_extended_20261006/) | Seven-setting figure: 630 datasets and 4,410 plotted candidate records; includes d=5 and n=1,000,000. |
| [`synthetic_bounded_fig2_compact_20261006`](results/synthetic_bounded_fig2_compact_20261006/) | Publication subset: five settings, 500 datasets and 3,500 plotted candidate records; full available-grid summaries retained. |
| [`synthetic_gaussian_20261006`](results/synthetic_gaussian_20261006/) | Earlier rho=0.5 figure: eight settings, 240 datasets, summaries and audit. Its 7,800 candidate estimates are also retained as the baseline in the next run. |
| [`synthetic_gaussian_rho_20261006`](results/synthetic_gaussian_rho_20261006/) | All rho=0,0.1,...,0.5 results: 48 settings, 1,440 datasets, 46,800 candidates in 48 CSVs, dataset audits, protocol and agreement summaries. |
| [`synthetic_gaussian_lowdim_20261006`](results/synthetic_gaussian_lowdim_20261006/) | New-seed rho=0.2 extension: five settings, 150 datasets, 1,200 candidate records, all checkpoints, summaries and protocol. |
| [`five_family_legacy_20261006`](results/five_family_legacy_20261006/) | Original five-family benchmark: 55 settings, six methods, 30 repetitions; 9,900 selected replicates and 330 summaries, caption and audit. Candidate-search tables and raw observation arrays from the original experiment are not included. |

Vector PDFs are under [`output/pdf/`](output/pdf/), including the
[combined bounded/Gaussian figure](output/pdf/synthetic_ell_comparison_20261006/bounded_gaussian_ell_rmse.pdf)
and the [historical five-family figure](output/pdf/five_family_legacy_20261006/five_family_benchmark.pdf).
The matching PNGs, figure QA records and earlier figure versions are retained.
The [`output/`](output/) directory also contains the original manuscript
insertion fragments; these are fragments for a parent manuscript, not
standalone LaTeX documents. Their figure paths refer to the original workspace
layout and may need adjustment in a manuscript project.

## Source index and historical reproduction limits

The byte-identical study sources are kept at their original repository-relative
locations:

- [`../synthetic_bounded/`](../synthetic_bounded/): source audit, wider analysis,
  bounded figure variants and d=3,4 extension runner.
- [`../synthetic_gaussian/`](../synthetic_gaussian/): original rho=0.5 analysis.
- [`../synthetic_gaussian_rho/`](../synthetic_gaussian_rho/): correlation runner
  and analysis.
- [`../synthetic_gaussian_lowdim/`](../synthetic_gaussian_lowdim/): low-dimensional
  extension runner and analysis.
- [`../five_family_legacy/`](../five_family_legacy/): original five-family figure
  reconstruction.

[`archive_sources/`](archive_sources/) contains nine original supporting source
files, retaining their historical relative paths: `pss_core.cpp`,
`pss_theory.py`, `run_study.py`, `run_large_n.py`, `run_dependent_five.py`,
`reanalyze_anchor_results.py`, `make_anchor_grid_datasets.py`,
`plot_anchor_grid_results.py` and `rebuild_saved_scaling.py`. These snapshots
document the generators, archived estimator and audit helpers; they are not
installed as importable replacements or rewritten to use the publication paths.

The historical files intentionally retain absolute machine paths and frozen
hashes. Many runners import the original repository's `experiments/theory_selection`
directory, and analyses expect `results/<run>` at the repository root rather
than inside this publication package. Merely running those scripts in a fresh
clone therefore does not reproduce the archived experiment.

Exact historical reruns additionally require the following original result
archives, which are not fully bundled here:

| Historical directory under `results/` | Dependency |
| --- | --- |
| `theory_selection_20261003` and `theory_selection_large_n_20261003` | Design manifests and complete bounded evaluation CSVs; the former study's saved frozen selections are also used by the historical large-n runner. |
| `sc_cv_v2_20261005` | Original candidate table and protocol used to select the Gaussian rho=0.5 baseline. |
| `dependent_five_pdf_20261003` | Earlier Gaussian summaries, settings and protocol referenced by the source-validation chain. |
| `anchor_grid_all_estimators_20260504_172951` | Original combined summary, PSS/CADEE and kNN/UM candidate tables, and data-generation configuration. Selected replicate records sufficient to verify the published figure are bundled here. |

The low-dimensional and bounded-extension runners also record the original
macOS `libpss_v2.dylib`; their unchanged historical code hardcodes that filename.
Compiled libraries, caches and logs are excluded from this package. The
canonical builder supports `.dylib` on macOS and `.so` on Linux, but a newly
built binary is not expected to reproduce the original binary hash.

Frozen runner protocols compare source hashes, paths, binary hashes and/or the
recorded environment before resuming an existing output directory. Moving or
editing a runner can correctly trigger a provenance failure. A future portable
rerun should use a new output directory and an explicitly new protocol,
preserving these published records. Some earlier figure audits record an
earlier presentation script revision; the publication manifest identifies the
current copied source bytes, and the verifier does not claim to reproduce all
historical script hashes.

The new density estimator uses canonical sub-grid spacing and the effective
sample-size denominator. The five-family legacy figure retains historical
rank-spacing divided by n, same-repetition oracle choices and timing excluding
parameter search. They are different experiments. Bounded-density regularity
assumptions do not certify finite-sample conditions with unspecified constants;
the Gaussian studies are outside the bounded-support assumptions. Empirical
RMSE minima are descriptive oracle diagnostics, and the Gaussian extension
followed an exploratory correlation comparison.
