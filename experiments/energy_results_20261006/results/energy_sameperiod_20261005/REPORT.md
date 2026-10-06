# Energy same-period experiment

## Status

Completed five prespecified random holdouts, seeds [42, 43, 44, 45, 46]. All 905 curve points and 5,358,505 saved predictions passed the audit.

## Protocol

19,735 observations, 25 predictors. Outer train/test = 13,814/5,921; inner train/validation = 9,669/4,145. Unstratified random splits; targets use only the current training median. All rows used for feature selection. Train-only range scaling and Gaussian tie smoothing. Fixed SVM: C=1, gamma=1/subset size, sample-SD scaling, no class weights.

Each primary selector receives 12 configurations evaluated on the same inner holdout. Settings are locked before each outer test. All 1..20 feature counts are reported; no test-selected count or seed. SC-CV is pooled, subsetwise, common ell for all entropy components. The all-component guard is an ablation, not the main definition.

## Accuracy

Mean +/- sample SD (%); five overlapping splits, not independent-replicate confidence intervals.

| Method | 5 features | 10 features | 20 features |
|---|---:|---:|---:|
| PSS SC-CV | 72.78 +/- 1.47 | 78.55 +/- 1.05 | 83.51 +/- 0.47 |
| Joint Ross MI | 72.43 +/- 0.58 | 80.62 +/- 1.04 | 83.28 +/- 0.68 |
| Univariate Ross MI | 74.40 +/- 2.03 | 78.69 +/- 0.81 | 83.40 +/- 0.64 |
| PSS default SC-CV | 71.89 +/- 1.60 | 77.49 +/- 1.03 | 83.61 +/- 0.45 |
| PSS component guard | 73.07 +/- 0.55 | 78.65 +/- 0.82 | 83.61 +/- 0.45 |
| PSS ell=1 | 72.44 +/- 1.32 | 77.25 +/- 0.26 | 83.61 +/- 0.45 |
| PSS ell=2 | 74.18 +/- 0.98 | 79.04 +/- 0.85 | 83.59 +/- 0.22 |
| PSS global SC-CV | 72.44 +/- 1.32 | 77.25 +/- 0.26 | 83.61 +/- 0.45 |
| KL difference k=1 | 69.05 +/- 0.50 | 75.76 +/- 0.66 | 82.40 +/- 0.65 |

All 25 features: 84.49 +/- 0.61%.

## Selected Settings

|   seed | method             |   n_min |   noise |    tau |   inner_accuracy |   reporting_count |   k |
|-------:|:-------------------|--------:|--------:|-------:|-----------------:|------------------:|----:|
|     42 | PSS SC-CV          |      10 |  1e-05  |   0.9  |         0.767591 |                20 | nan |
|     42 | Joint Ross MI      |     nan |  1e-05  | nan    |         0.788179 |                20 |  15 |
|     42 | Univariate Ross MI |     nan |  1e-05  | nan    |         0.778448 |                20 |  10 |
|     43 | PSS SC-CV          |      10 |  1e-05  |   0.99 |         0.782067 |                20 | nan |
|     43 | Joint Ross MI      |     nan |  1e-05  | nan    |         0.795094 |                20 |  15 |
|     43 | Univariate Ross MI |     nan |  0.0001 | nan    |         0.788259 |                20 |   1 |
|     44 | PSS SC-CV          |      10 |  0.0001 |   0.95 |         0.786409 |                20 | nan |
|     44 | Joint Ross MI      |     nan |  1e-05  | nan    |         0.793567 |                20 |  20 |
|     44 | Univariate Ross MI |     nan |  0.0001 | nan    |         0.789304 |                20 |  10 |
|     45 | PSS SC-CV          |       5 |  0.0001 |   0.95 |         0.794692 |                20 | nan |
|     45 | Joint Ross MI      |     nan |  1e-05  | nan    |         0.789867 |                20 |  20 |
|     45 | Univariate Ross MI |     nan |  0.0001 | nan    |         0.781504 |                20 |  20 |
|     46 | PSS SC-CV          |       5 |  0.0001 |   0.95 |         0.788018 |                20 | nan |
|     46 | Joint Ross MI      |     nan |  1e-05  | nan    |         0.791878 |                20 |  15 |
|     46 | Univariate Ross MI |     nan |  0.0001 | nan    |         0.78657  |                20 |  10 |

## SC-CV Diagnostics

ell=1 at 70/100 selected subsets; ell>1 at 30/100.
Upper grid endpoint ell=5 at 14/100 subsets; optimality beyond the declared 1..5 grid is not assessed.
SC-CV fallback at 0/100 subsets.
Pooled stable coverage range: 0.902201 to 0.999348.
Minimum component stable coverage range: 0.795788 to 0.997926.
Primary raw selection score is outside [0, empirical label entropy] at 19/100 subsets; this score is not calibrated MI.
Primary SC-CV and fixed ell=1 choose identical 20-feature sets in 4/5 splits.

## Timing

| method              |   mean_seconds |   sd_seconds |
|:--------------------|---------------:|-------------:|
| All features        |      0         |   0          |
| Joint Ross MI       |     58.9973    |  13.2541     |
| KL difference k=1   |      9.49329   |   0.945511   |
| PSS SC-CV           |     80.7984    |   3.54703    |
| PSS component guard |     83.6135    |   5.13467    |
| PSS default SC-CV   |     83.8599    |   5.02188    |
| PSS ell=1           |      0.0623453 |   0.00413002 |
| PSS ell=2           |      5.71156   |   0.538577   |
| PSS global SC-CV    |      7.82572   |   0.327145   |
| Univariate Ross MI  |      0.500049  |   0.0838312  |

Seconds for cold forward selection (including candidate-level SC-CV and conditional coverage diagnostics), excluding post-selection diagnostics and classifier fitting. Inner search is separately recorded in checkpoint time files. Three concurrent workers; not an isolated speed benchmark.

## Interpretation Limits

- These evaluate same-period interpolation for one household, not prediction in a future period or a new home. Adjacent time-series observations can occur on both sides of random splits.
- Prior temporal evaluation is preserved in the old repository. The altered split, sample budget, preprocessing and SVM settings mean old/new performance differences cannot be attributed to SC-CV alone.
- The public dataset was studied previously. Held-out means not used in this run's parameter selection, not never inspected historically.
- Scores are entropy differences, not calibrated MI. No MI-growth or theoretical-consistency figure is produced. Neither high coverage nor accuracy proves consistency.
- Fixed ell=2 is not SC-CV. Historical KL difference at k=1 is not the tuned mixed-type baseline. All ablations are reported, including unfavorable results.
- Five overlapping holdouts provide descriptive sensitivity, not statistical significance, equivalence, or evidence across independent datasets.

## Figures

Main: energy_accuracy. Supplement: energy_ablation and energy_sc_diagnostics. Accuracy-plot shading denotes +/- one sample SD across splits. Diagnostic-plot shading denotes the observed minimum-to-maximum across splits, with the mean as the line. All-feature horizontal line is a 25-predictor reference, not a same-budget subset curve.

Vector PDFs: `/Users/hojeongwoo/Documents/ChatGPT/pss/output/pdf/energy_sameperiod_20261005`. PNG previews: `/Users/hojeongwoo/Documents/ChatGPT/pss/results/energy_sameperiod_20261005/plots`.

## Reproduction

Protocol, code/data hashes, split indices, lock files, complete feature paths, validation scores, test predictions, diagnostics and audit.json are retained alongside these outputs.
