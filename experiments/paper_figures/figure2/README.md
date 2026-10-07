# Figure 2: historical saved SC-CV results

This bundle preserves the numerical series displayed in Figure 2 of
`AISTATS2027_submission.pdf` (page 5, inspected 2026-10-07). It verifies the
archived records and regenerates the three panels: selected partition level,
entropy RMSE, and stable validation coverage. **These are historical saved
results, not results of the current `PSS/pss_v2.py` implementation.**

The exact original simulation and figure generator has not been recovered.
Running this script does not rerun synthetic sampling, oracle search, or SC-CV.
The archived PDF has the same numerical series with different panel headings
and typography; the regenerated plot reproduces the values, not identical pixels.

## Verify or plot

Verification uses only the Python standard library and works from any directory:

```sh
python experiments/paper_figures/figure2/plot.py --verify-only
```

Plotting additionally requires matplotlib:

```sh
python experiments/paper_figures/figure2/plot.py --output-dir figure2_render
```

The plot command writes `figure2_historical.pdf`, `figure2_historical.png`,
`recomputed_summary.csv`, and `verification.json`. Default output is the ignored
`generated/` directory beside the script. `--verify-only` writes no files and
does not import matplotlib. Missing or modified inputs fail verification with a
nonzero exit status before any plot is written. `--input-dir` can point to a
relocated copy of the bundle containing its `data/` and `archive/` directories.

## What is checked

- SHA-256 hashes for the original, byte-preserved CSVs and archived PDF.
- Exactly 240 unique selected-result records: 30 repetitions for two methods
  and four sample sizes, with no missing or duplicate repetitions.
- Every stored error against `Estimate - True_Entropy` and valid coverage ranges.
- All eight summary rows, including selected-ell mean/SE, RMSE, bias, coverage
  mean/SE, RMSE SE, fallback rate, and ratios to the stored Oracle reference.
- All 120 Oracle estimates against the companion archive table that records the
  distribution, dimension and correlation.

SE uses the sample standard deviation divided by the square root of 30.
RMSE SE uses the archived delta-method formula `SE(error^2) / (2 * RMSE)`;
error bars are one SE, not confidence intervals. All archived selection-rule
labels are `oracle_rmse` or `stable_coverage`, so the saved fallback rate is zero.
The fixed Oracle partition level and RMSE can be verified from its selected
records; the optimality of that choice over the original full grid cannot be
verified without the missing candidate results and generator.

## Conditions and provenance

The companion records identify Gamma data, dimension 5, copula correlation 0,
sample sizes 1,000 / 3,000 / 10,000 / 30,000, and 30 repetitions. Correlation 0
is the independent Gaussian-copula case. Gamma shape/scale, the complete tuning
grid and original random seeds are not established by the bundled records.

`provenance.json` names the original archive and files and freezes their hashes.
The two `gamma_stable_cv_smin099_rep30_*.csv` files are the plotted records and
summary. The companion `gamma_n_scaling_tuning_replicates_rep30.csv` supplies
condition evidence; its additional `Constrained CV` rows are not the plotted
`Stable CV` series. No machine-specific paths are needed.

## Known mismatch with the manuscript's current SC-CV definition

The current density is zero outside each fold's training empirical box. For any
tie-free continuous sample, the global minimum and maximum of one coordinate
are each outside the training range when assigned to validation. Consequently,
stable validation coverage must satisfy `S <= 1 - 2/n` in every repetition,
regardless of the selected partition level.

At `n=1,000`, that upper bound is **0.998**, but the archived Figure 2 Stable CV
mean is **0.999766666666667**. This proves that these saved coverage values do
not follow the current training-box convention. The historical implementation
clamps queries to boundary cells/ranks and is consistent with this discrepancy;
this is not proof of the cause, because the exact original generating script is
unavailable. The verifier deliberately
reports `current_v2_coverage_rule_matches: false` while preserving the original
numbers. Reproducing these saved panels does not resolve that estimator mismatch
or establish reproduction of the current manuscript method.
