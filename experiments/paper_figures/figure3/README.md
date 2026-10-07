# Figure 3: historical five-family benchmark

This portable entry point rebuilds the paper's four-by-five benchmark layout
using the existing [saved results](../../synthetic_results_20261006/results/five_family_legacy_20261006).
It does not copy the tables or require the author's original archive directory.

**The PSS curves use the historical rank-spacing / n estimator, not the
canonical v2 smoothed-subgrid / N_eff estimator.** This distinction is retained
in the figure footer. The figure is historical empirical evidence, not a new
benchmark of the canonical estimator used in the current theory and Figure 4.

## Reproduce or verify

From the repository root, with Python 3.9 or newer:

```sh
python experiments/paper_figures/figure3/plot.py --verify-only
python -m pip install -r experiments/paper_figures/figure3/requirements.txt
python experiments/paper_figures/figure3/plot.py --output-dir /tmp/pss-figure3
```

The script resolves its inputs from its own location, so it also works from
another working directory when invoked by its absolute path. Without
`--output-dir`, it writes PDF, PNG, and `figure_checks.json` to `figures/` beside
the script. `--verify-only` uses only Python's standard library and writes
nothing; Matplotlib is imported only for plotting.

Before plotting, the script checks the existing synthetic export manifest's
SHA-256 hashes and file sizes for `summary.csv`, `selected_replicates.csv`, and
`audit.json`. It calls the bundled verifier's `verify_legacy()` to independently
recompute all 330 summaries from 9,900 selected replicate records: RMSE,
delta-method RMSE standard error, bias, mean runtime, and runtime standard error.
The output check record includes the input and verification-source hashes.

## Interpretation

Columns are Normal, Gamma, Beta, Lognormal, and Laplace marginals. The six
methods are PSS, CADEE, KL, KSG, UM-tKL, and UM-tKSG. Every condition has 30
repetitions. Rows show:

| Row | Varied quantity | Fixed quantities |
| --- | --- | --- |
| Sample-size RMSE | n = 1,000; 3,000; 10,000; 30,000 | d = 5, rho = 0 |
| Runtime | Same sample-size grid | d = 5, rho = 0 |
| Dimension RMSE | d = 2, 5, 10, 20 | n = 20,000, rho = 0 |
| Dependence RMSE | rho = 0, 0.5, 0.8 | n = 20,000, d = 5 |

Error bars are plus/minus one Monte Carlo standard error, conditional on the
saved oracle parameter; they are not 95% intervals and exclude oracle-selection
uncertainty. PSS retains its coverage-filtered same-repetition RMSE oracle
choice. The kNN-based methods also retain same-repetition RMSE oracle choices;
CADEE has no candidate parameter in this run. Runtime includes evaluation at
the selected parameter and flow training where applicable, and excludes
hyperparameter search. The two flow curves each include their shared fit plus
their own evaluation time. No plotted values are clipped or floored.

This command performs no new simulations, estimator fits, timing measurements,
or tuning. Recomputing summaries validates the saved records, not the original
data generation or estimator fits. See the [synthetic archive README](../../synthetic_results_20261006/README.md)
for full-run provenance and reproduction limits.
