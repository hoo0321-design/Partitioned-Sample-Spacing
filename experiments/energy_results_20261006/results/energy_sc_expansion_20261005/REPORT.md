# Energy coverage and fold expansion

Exploratory nested evaluation of the same five previously inspected random holdouts. All feature-selection paths were rebuilt using training data, with fixed noise=1e-5 and n_min=5. This is not a new independent confirmation dataset.

## Protocol

19,735 observations and 25 predictors. Outer train/test: 13,814/5,921; inner train/validation: 9,669/4,145. All selection rows retained. Training-only median target and range normalization; unjittered predictors for the fixed RBF SVM. Canonical PSS v2 and N_eff unchanged. Common ell for pooled and conditional entropy scores; no forced ell>1.

PSS: tau in {0.80,0.85,0.90,0.95,0.99}, K in {3,5,10}: 15 configurations. KL entropy difference, Ross joint MI, and univariate Ross: k in {1,2,3,5,7,10,15,20,30,50}: 10 configurations each. PSS matched-budget restricts K to {3,5}, also 10 configurations. All choices use mean inner-validation accuracy at 5,10,20 features and are locked before new outer evaluation. There is no separate classifier tuning.

## Accuracy

Mean +/- sample SD (%). Overlapping splits, not independent-sample confidence intervals.

| Method | 5 features | 10 features | 20 features |
|---|---:|---:|---:|
| PSS expanded SC-CV | 72.78 +/- 1.05 | 79.32 +/- 1.02 | 83.39 +/- 0.77 |
| KL tuned | 74.14 +/- 1.58 | 80.75 +/- 0.94 | 84.22 +/- 0.41 |
| Ross tuned | 73.75 +/- 1.59 | 81.02 +/- 0.91 | 83.37 +/- 0.63 |
| Univariate tuned | 74.31 +/- 1.33 | 79.08 +/- 0.66 | 83.83 +/- 0.33 |
| PSS 3-fold | 71.91 +/- 1.81 | 78.61 +/- 1.19 | 83.35 +/- 0.40 |
| PSS 5-fold | 72.37 +/- 2.03 | 79.64 +/- 1.33 | 83.62 +/- 0.54 |
| PSS 10-fold | 72.19 +/- 0.97 | 79.42 +/- 1.14 | 83.35 +/- 0.72 |
| PSS matched-budget | 71.79 +/- 2.24 | 78.93 +/- 1.21 | 83.43 +/- 0.60 |
| PSS tau=0.99 | 72.51 +/- 0.91 | 77.49 +/- 1.03 | 83.61 +/- 0.45 |
| PSS ell=1 | 73.15 +/- 0.46 | 77.37 +/- 0.18 | 83.61 +/- 0.45 |
| PSS ell=2 | 73.91 +/- 1.13 | 79.31 +/- 0.72 | 83.10 +/- 0.28 |
| KL k=1 | 69.05 +/- 0.50 | 75.76 +/- 0.66 | 82.40 +/- 0.65 |

All 25 features: 84.49 +/- 0.61%.

## Locked settings

|   seed | method             | folds   | tau   | k    |   inner_accuracy |   reporting_count |
|-------:|:-------------------|:--------|:------|:-----|-----------------:|------------------:|
|     42 | PSS expanded SC-CV | 10.0    | 0.9   | -    |         0.781423 |                20 |
|     42 | KL tuned           | -       | -     | 50.0 |         0.78271  |                20 |
|     42 | Ross tuned         | -       | -     | 15.0 |         0.788179 |                20 |
|     42 | Univariate tuned   | -       | -     | 10.0 |         0.778448 |                20 |
|     43 | PSS expanded SC-CV | 10.0    | 0.85  | -    |         0.783112 |                20 |
|     43 | KL tuned           | -       | -     | 30.0 |         0.793888 |                20 |
|     43 | Ross tuned         | -       | -     | 15.0 |         0.795094 |                20 |
|     43 | Univariate tuned   | -       | -     | 10.0 |         0.784238 |                20 |
|     44 | PSS expanded SC-CV | 10.0    | 0.8   | -    |         0.788661 |                20 |
|     44 | KL tuned           | -       | -     | 50.0 |         0.795979 |                20 |
|     44 | Ross tuned         | -       | -     | 50.0 |         0.798713 |                20 |
|     44 | Univariate tuned   | -       | -     | 50.0 |         0.790028 |                20 |
|     45 | PSS expanded SC-CV | 5.0     | 0.95  | -    |         0.789626 |                20 |
|     45 | KL tuned           | -       | -     | 50.0 |         0.793325 |                20 |
|     45 | Ross tuned         | -       | -     | 30.0 |         0.796622 |                20 |
|     45 | Univariate tuned   | -       | -     | 7.0  |         0.780941 |                20 |
|     46 | PSS expanded SC-CV | 3.0     | 0.95  | -    |         0.787696 |                20 |
|     46 | KL tuned           | -       | -     | 30.0 |         0.799035 |                20 |
|     46 | Ross tuned         | -       | -     | 30.0 |         0.794853 |                20 |
|     46 | Univariate tuned   | -       | -     | 20.0 |         0.786088 |                20 |

## Coverage diagnostics

| method             |   ell1_fraction |   mean_ell |   ell5_fraction |   minimum_pooled_stable |   minimum_component_stable |   maximum_conditional_training_skip |   score_outside_mi_bounds |   fallback_count |
|:-------------------|----------------:|-----------:|----------------:|------------------------:|---------------------------:|------------------------------------:|--------------------------:|-----------------:|
| PSS 10-fold        |            0.57 |       2.16 |            0.19 |                0.805487 |                   0.648838 |                         0.0246737   |                        57 |                0 |
| PSS 3-fold         |            0.57 |       2.14 |            0.19 |                0.800999 |                   0.63545  |                         0.0231334   |                        64 |                0 |
| PSS 5-fold         |            0.63 |       1.97 |            0.16 |                0.855002 |                   0.74801  |                         0.00971028  |                        49 |                0 |
| PSS ell=1          |            1    |       1    |            0    |                0.994933 |                   0.98951  |                         0           |                         8 |                0 |
| PSS ell=2          |            0    |       2    |            0    |                0.316346 |                   0.15925  |                         0.130801    |                        70 |                0 |
| PSS expanded SC-CV |            0.58 |       2.13 |            0.19 |                0.805487 |                   0.648838 |                         0.0246737   |                        61 |                0 |
| PSS matched-budget |            0.6  |       2.05 |            0.17 |                0.800999 |                   0.638423 |                         0.022128    |                        48 |                0 |
| PSS tau=0.99       |            0.82 |       1.49 |            0.08 |                0.99001  |                   0.976478 |                         0.000788022 |                        10 |                0 |

## Selection timing

| method             |   mean_seconds |   sd_seconds |
|:-------------------|---------------:|-------------:|
| All features       |       0        |     0        |
| KL k=1             |      10.0535   |     1.12989  |
| KL tuned           |     157.758    |    24.2161   |
| PSS 10-fold        |     125.954    |     1.07198  |
| PSS 3-fold         |      56.8716   |     1.67651  |
| PSS 5-fold         |      77.0289   |     1.19993  |
| PSS ell=1          |       6.36656  |     0.130473 |
| PSS ell=2          |       5.3695   |     0.108502 |
| PSS expanded SC-CV |     102.412    |    31.9065   |
| PSS matched-budget |      69.2254   |     9.89729  |
| PSS tau=0.99       |      55.7842   |     0.715693 |
| Ross tuned         |      91.2891   |    39.446    |
| Univariate tuned   |       0.635523 |     0.247924 |

Cold forward-selection seconds per unique outer configuration, excluding post-selection diagnostics and classifier fits. Identical policies reuse the same selected path. Three concurrent workers; not isolated hardware benchmarks. Inner searches reuse exact subset calculations and are separately checkpointed.

## Limits

- Increasing ell is not itself evidence of better prediction or MI estimation. Covered NLL averages different subsets at different ell; relaxing coverage can improve this objective by excluding difficult points.
- These are same-period interpolation results in one household, not future-time or new-household performance. Tests were previously inspected during method development, although current settings never use outer outcomes in their selection objective.
- Training jitter and high coverage do not establish the continuous-density theorem assumptions. Raw entropy differences are selection scores, not calibrated MI.
- Comparisons with the previous experiment also change its tuned jitter/n_min to common fixed values. The new fold-specific policies provide the controlled comparison within this protocol.
- Fixed ell and k=1 are ablations. All tuned baseline results, unfavorable settings, and search-budget control are retained.

- Search is limited to the declared grids. A choice of k=50 or ell=5 is at its grid boundary, not evidence of global optimality.

## Audit

{
  "passed": true,
  "curve_points": 1205,
  "predictions_verified": 7134805,
  "unique_classifier_fits": 911,
  "data_and_code_hashes_verified": true,
  "complete_candidate_grids_verified": true,
  "training_only_locks_recomputed": true,
  "historical_same_configuration_paths_matched": 15
}

## Figures

Accuracy and ablation bands show +/- one sample SD. Diagnostic bands show observed minimum-to-maximum, with the mean line. The heatmap uses inner validation only. Vector PDFs are saved under output/pdf with the same experiment directory name.
