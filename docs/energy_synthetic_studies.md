# October 2026 Energy and synthetic study notes

These study notes preserve the original protocol descriptions and exploratory
findings. For the published result locations and supported reproduction commands,
start with the [Energy bundle](../experiments/energy_results_20261006/README.md)
and [synthetic bundle](../experiments/synthetic_results_20261006/README.md).
Historical commands below refer to the original workspace layout: `results/`
and `output/` paths are archival names, not the locations of the publication
bundles. Source scripts are retained unchanged so their frozen hashes remain
verifiable. The synthetic archive workflows require additional historical input
files and retain absolute paths; they are not turnkey fresh-clone commands.

The canonical PSS v2 source and the cached Energy CSV are included in this
repository. Existing historical R workflows and results remain separate.

## Protocol

Five prespecified random 70:30 splits (seeds 42--46), with one 70:30 inner
training/validation split per outer training sample. Every feature selection
uses all available training rows, with training-only median labels, range
scaling, and explicitly specified Gaussian tie smoothing. There are 25 predictors.
All methods use the same fixed RBF SVM settings corresponding to historical
e1071 defaults. No feature count, seed, or hyperparameter is picked on test data.

Main selectors are pooled subsetwise PSS SC-CV, joint continuous/discrete Ross MI,
and univariate Ross ranking. Each has 12 inner-validation configurations (two
smoothing magnitudes times six selector settings). SC-CV uses ell=1..5 and the
canonical covered-NLL objective/coverage constraint, with no extra penalty.
The chosen common ell is used for all three entropy components. Conditional
coverage is reported, but constraining all components is a separate ablation.

Additional ablations: fixed default SC-CV, all-component guard, ell=1, ell=2,
global once-only SC-CV, and historical KL entropy difference at fixed k=1.
Raw scores are not clipped or presented as calibrated mutual information.

The fixed ell=2 and KL k=1 results are diagnostics, not tuned competitors or an
exact replay of the published figure. Historical code used full-data range
scaling/jitter and a different random-number stream for its stratified split.
The new classifier intentionally sees unjittered predictors. These distinctions
must be disclosed when comparing to old results.

Random-split results target same-period classification in this one household,
not forecasting or new-household generalization. Previous temporal results remain
in the old repository and should not be replaced by these results. Mean +/- SD
across overlapping splits is descriptive split variability, not a confidence
interval from independent subjects. Neither smoothing nor high coverage proves
the Energy data satisfy the continuous-density theorem assumptions.

## Reproduction

```sh
python PSS/build_pss_v2.py
python PSS/test_density.py
python -m unittest discover -s experiments/energy_sameperiod -p 'test_*.py'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/energy_sameperiod/run.py all --output results/energy_sameperiod_20261005 --workers 3
python experiments/energy_sameperiod/analyze.py results/energy_sameperiod_20261005
```

Jobs checkpoint per inner split/noise and per outer split/method. Frozen protocol
and source hashes prevent resuming with silently changed definitions. Saved
paths precede test evaluation; all curve predictions are retained for audit.

References: [UCI dataset](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction),
[Ross (2014)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0087357),
[Cawley and Talbot (2010)](https://jmlr.org/papers/v11/cawley10a.html).

## Coverage and fold expansion

`experiments/energy_sc_expansion/` preserves the earlier experiment and evaluates
SC-CV with tau=0.80,0.85,0.90,0.95,0.99 and 3,5,10 folds. Noise=1e-5 and n_min=5
are fixed to isolate this comparison. The density definition, N_eff denominator,
ell=1..5 candidates, preprocessing, and SVM are unchanged. No ell>1 requirement is
introduced. Each configuration rebuilds the forward-selection path from training
data; exact shared subset calculations are cached without using test metrics.

KL entropy-difference kNN is tuned over k=1,2,3,5,7,10,15,20,30,50. Joint Ross MI
and univariate Ross ranking use the same k grid. There are 15 PSS configurations
and 10 per kNN method; a separate PSS control restricts folds to 3 or 5, giving
10 configurations. Fold-specific policies, tau=0.99, fixed ell=1/2, and KL k=1
are also reported. All policies are locked before new test evaluation.

The same five previously inspected random splits are reused. This is an
exploratory extension, not confirmation on a never-inspected dataset. Changes in
downstream accuracy and selected ell must be distinguished: a larger ell is not
automatically better, and covered NLL can favor candidates that exclude more
validation points. Training coverage and held-out coverage are separate diagnostics.

```sh
python -m unittest discover -s experiments/energy_sc_expansion -p 'test_*.py'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/energy_sc_expansion/run.py all --output results/energy_sc_expansion_20261005 --workers 3
python experiments/energy_sc_expansion/analyze.py results/energy_sc_expansion_20261005
```

Per-configuration development records, locked settings, training-only selected
paths, all test-curve predictions, and data/source hashes are retained. The audit
recomputes configuration choices and every saved accuracy, balanced accuracy,
and AUROC value. Confidence intervals from independent datasets are not inferred
from five overlapping random holdouts.

## Energy class-aware coverage ablation

`experiments/energy_component_guard/` compares pooled SC-CV with feasibility based
on the minimum stable validation coverage across the pooled model and both class
conditional models. The covered pooled negative log-likelihood, canonical PSS
density/N_eff, and entropy-difference feature score are unchanged. No redundancy
correction is introduced in this experiment.

This first comparison uses the first three previously inspected Energy splits
(seeds 42--44) and all 9,669 inner-training / 13,814 outer-training observations.
The candidate grid is unchanged: ell=1..5, n_min=5, tau=0.80,0.85,0.90,0.95,0.99,
and 3,5,10 folds. Each rule tunes on mean inner accuracy at 5,10,20 features.
An additional class-aware path freezes tau and fold count to the old pooled
winner, separating the constraint change from selector retuning. All new paths
are selected afresh, and all settings are locked before new test evaluation.

Unchanged controls and exact-subset classifier fits are reused read-only from
`results/energy_sc_expansion_20261005/` after checking data, source, split and
result hashes. Comparisons use the same three splits, not old five-split means.
Plots include descriptive SD/ranges, not independent-dataset confidence intervals.
The new audit recomputes test metrics and selected PSS component coverage using
the canonical evaluator. Cache-assisted times are not algorithm timing benchmarks.

```sh
python -m unittest discover -s experiments/energy_component_guard -p 'test_*.py'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/energy_component_guard/run.py all --output results/energy_component_guard_20261005 --workers 3
python experiments/energy_component_guard/analyze.py results/energy_component_guard_20261005
```

## Energy pairwise JMI score comparison

`experiments/energy_jmi/` compares the existing class-aware PSS and tuned KL
selectors with PSS-JMI and KL-JMI. The first feature maximizes univariate
entropy-difference MI. Each subsequent candidate maximizes the mean of
`I((candidate, selected_feature); label)` over the features already selected.
Every component is one- or two-dimensional. No additional relevance term,
clipping, correlation penalty, or forced ell>1 is introduced. The canonical
PSS density, class-aware coverage gate, preprocessing, and fixed SVM are unchanged.

Each new selector is tuned separately on mean inner accuracy at 5,10,20 features:
PSS-JMI uses the existing 15 fold/tau configurations, and KL-JMI the existing
10 k values. These are the same respective grids as the baselines; their counts
and computational budgets are not equal. All 9,669 inner-training and 13,814
outer-training rows are used, with the same three seeds 42--44. Every split's
settings are locked before any new test evaluation, and both new training-only
feature paths are saved before evaluating that split's test curves.

The protocol hashes the data, estimator/runner sources, and reusable baseline
outputs. Unchanged controls and exact-subset fixed-SVM fits may be reused after
checking provenance; reported workflow times therefore are not estimator speed
benchmarks. Full singleton/pair score tables retain every greedy candidate for
audit. Pair-level ell and coverage diagnostics are summarized over unique
components, rather than repeatedly counting a pair at later path steps.

```sh
python -m unittest discover -s experiments/energy_jmi -p 'test_*.py'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/energy_jmi/run.py all --output results/energy_jmi_20261006 --workers 3
python experiments/energy_jmi/analyze.py results/energy_jmi_20261006
```

This remains an exploratory comparison on previously inspected random-row
holdouts. Split SD is descriptive, and the results do not establish forecasting
or new-household generalization. JMI is a feature-selection wrapper, so comparing
both PSS and KL within the same wrapper is necessary to distinguish the wrapper
effect from the density estimator effect. The criterion follows
[Brown et al. (2012)](https://www.jmlr.org/papers/v13/brown12a.html).

The completed exploratory run did not support replacing the existing selector
with pairwise JMI. Test accuracy (percent, mean +/- sample SD over three splits):

| Selector | 5 features | 10 features | 20 features |
| --- | ---: | ---: | ---: |
| PSS class-aware SC-CV | 73.24 +/- 1.07 | 79.06 +/- 0.41 | 83.50 +/- 0.53 |
| PSS-JMI | 71.96 +/- 0.09 | 78.34 +/- 0.58 | 83.82 +/- 0.48 |
| KL tuned | 74.60 +/- 0.50 | 80.58 +/- 0.85 | 84.05 +/- 0.45 |
| KL-JMI | 72.63 +/- 0.52 | 79.28 +/- 0.38 | 83.75 +/- 0.37 |

PSS-JMI decreased accuracy at 5 and 10 features in all three splits, while
increasing it at 20 features in all three (mean +0.32 percentage points). Its
20-feature mean remained 0.23 points below tuned KL. None of the 570 unique
selected pair components chose ell=1, but the number of highly correlated
predictor pairs (absolute Pearson r>0.8) at 10 features increased from 10/8/5
to 11/13/11. Thus the pairwise formulation avoided ell=1 collapse without
empirically reducing this correlation diagnostic. Neither this result nor the
small 20-feature gain establishes a causal explanation or a general advantage.

`summary.csv`, `paired_differences.csv`, `redundancy_diagnostics.csv`, and
`audit.json` in the result directory retain the values and verification scope.
The audit replayed all 75 development paths and six new outer paths, recomputed
975 PSS and 975 KL outer singleton/pair scores, and verified every test curve
metric from saved predictions. Historical inner cached metrics without stored
predictions were checked against frozen source hashes and identified separately.

## Energy theory-rate coefficient calibration

`experiments/energy_theory_c/` selects one coefficient from
`C = 1, 1.5, 2, 2.5, 3, 4` using the existing inner validation split. It uses
the dimension-aware entropy-rate rule
`ell = max(1, floor(C * (n / (d^6 * log(n)^2))^(1/(d+8)) + 0.5))`.
The coefficient multiplies the root, the logarithm is natural, and ties round
upward. Here `n` is the complete pooled selection-training sample size and `d`
is the full candidate subset dimension. A common ell is used for all three
entropy terms. This explicitly preserves the common-ell convention; it is not
a component-specific sample-size rule or the separate fixed-d rate without d^6.

For each C, all 9,669 inner-training rows produce a fresh forward-selection
path. The unchanged SVM is fit on those rows and evaluated on the 4,145 inner
validation rows at 5,10,20 features. The largest mean accuracy selects one C
for the whole path; ties prefer smaller C. All three split choices are locked
before any new test evaluation. Then the path is rebuilt using all 13,814
outer-training rows, keeping C fixed but recomputing ell with the larger n.
For example C=4 at d=20 gives ell=2 during inner selection and ell=3 in the
outer fit; the outer ell is not silently clamped to the inner value.

There is no SC-CV, coverage constraint, alternate-ell fallback, JMI score,
normalization, or clipping in this selector. An entropy component with zero
valid observations makes that candidate inadmissible. A C unable to complete
the entire path is retained as a failure and cannot win by averaging a partial
set of validation accuracies. Outer failures, if any, are reported without
changing the chosen rule. Candidate scores and selected-component coverage
diagnostics are retained for audit.

Only the selected C and prespecified C=1 receive new test evaluation. Existing
class-aware SC-CV, tuned KL, KL k=1, and all-feature controls are reused after
source/output verification. The new C search has six configurations, compared
with 15 settings for prior PSS SC-CV and ten for prior KL tuning. This remains
an exploratory study on the same three previously inspected splits. Choosing C
on validation data makes it a theory-guided tuned method, not a tuning-free
finite-sample optimum. Cached workflow times are not method runtime benchmarks.

```sh
python -m unittest discover -s experiments/energy_theory_c -p 'test_*.py'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/energy_theory_c/run.py all --output results/energy_theory_c_20261006 --workers 3
python experiments/energy_theory_c/analyze.py results/energy_theory_c_20261006
```

The completed run selected C=2.5,3,2.5 for seeds 42,43,44. All 18 development
paths and six new outer paths completed. Each selected C required ell=2 at
5,10,20 features, whereas C=1 required ell=1 at those dimensions. Accuracy
(percent, mean +/- sample SD over the same three splits) was:

| Selector | 5 features | 10 features | 20 features |
| --- | ---: | ---: | ---: |
| PSS theory, tuned C | 72.23 +/- 0.98 | 77.44 +/- 1.57 | 83.04 +/- 1.47 |
| PSS theory, C=1 | 73.21 +/- 0.59 | 77.37 +/- 0.10 | 83.54 +/- 0.45 |
| PSS class-aware SC-CV | 73.24 +/- 1.07 | 79.06 +/- 0.41 | 83.50 +/- 0.53 |
| KL tuned | 74.60 +/- 0.50 | 80.58 +/- 0.85 | 84.05 +/- 0.45 |
| KL k=1 | 68.97 +/- 0.58 | 75.92 +/- 0.28 | 82.26 +/- 0.57 |

Tuning C changed accuracy versus C=1 by -0.98,+0.07,-0.51 percentage points;
the small 10-feature increase occurred in only one of three splits and came
with greater split variability. This does not support adopting this calibration
as an improvement. It avoids ell=1 without resolving finite-sample instability.
In the common 3-fold, n_min=5 diagnostic, the minimum component stable coverage
across the 60 selected subsets was 15.47% for tuned C versus 99.03% for C=1.
At 20 features, tuned-C class-conditional training coverage was 85.74--87.90%,
pooled integrated mass was 191.95--229.91, and unmodified MI estimates were
5.79--6.12 nats versus label entropy about 0.689 nats. These are estimator
diagnostics, not a proven causal explanation of the accuracy differences.
C=1 also produced negative MI estimates at 20 features. The prior class-aware
SC-CV coverage figure used selected 5/10-fold settings and is not directly
comparable to this common 3-fold diagnostic.

The independent audit passed: it replayed all 18 development paths, recomputed
every selected development score and all 1,860 outer candidate scores, checked
the lock barrier, and verified all test metrics from saved predictions.
Twelve historical inner metric-only records were checked against frozen source
hashes and explicitly identified rather than described as prediction-verified.
`development_summary.csv`, `summary.csv`, `paired_differences.csv`,
`diagnostic_summary.csv`, and `audit.json` retain the results and verification
scope in `results/energy_theory_c_20261006/`.

## Bounded dependent synthetic experiment

`experiments/synthetic_bounded/` reuses the archived sub-grid / N_eff results
for `f(x)=1+0.7*cos(2*pi*k*(x1-x2))` on `[0,1]^d`, with k=1,2. These densities
have uniform marginals, bounds 0.3 and 1.7, and known entropy -0.1316231322 nats.
The main figures use d=2; a second page in each PDF uses d=5, with three
additional independent uniform coordinates. The densities are Lipschitz on
their support. Finite-sample admissibility constants are not certified.

The 28 settings reuse 2,240 datasets and 40,480 candidate estimates: 100
repetitions at n=1,000,3,000,10,000,30,000,100,000, and 30 repetitions at
n=300,000,1,000,000. The original five-family benchmark is not changed.
Source hashes and raw counts are checked. A numerical audit regenerated 56
datasets and compared 278 canonical estimates, with maximum difference
2.22e-14 nats and exact coverage/cell-count agreement. This is a deterministic
spot check, not a canonical rerun of every archived repetition.

The sample-size figure compares ell=1, the saved bound-minimizing rule at
delta=.05, and the same-repeat aggregate-RMSE oracle restricted to candidates
with full coverage in every repetition. The oracle is an optimistic reference.
The sensitivity figure shows every saved ell at n=10,000 and 100,000, with
RMSE and signed bias. Hollow markers identify incomplete coverage; the
canonical estimator averages covered observations. Pointwise 95% bootstrap
intervals use 2,000 resamples. No coefficient or selector was retuned.

The fixed ell=1 error approaches the nonzero dependence-approximation bias.
The bound rule improves some settings but retains substantial bias for the
higher-frequency density; its errors are nonmonotone over sample size.
These figures illustrate partition sensitivity, not an exact empirical
convergence exponent or finite-sample optimality of the theoretical rule.

```sh
PYTHONDONTWRITEBYTECODE=1 python experiments/synthetic_bounded/audit_reuse.py
MPLCONFIGDIR=/private/tmp/pss-bounded-mpl XDG_CACHE_HOME=/private/tmp/pss-bounded-cache python experiments/synthetic_bounded/analyze.py
```

The source archive path is recorded in the scripts and result manifests.
Final two-page figures are in `output/pdf/synthetic_bounded_20261006/`.
`results/synthetic_bounded_20261006/` contains PNG previews, recomputed
summaries, audit records, provenance hashes, and `notes.txt` with captions.

## Gaussian ell / RMSE comparison

`experiments/synthetic_gaussian/analyze.py` extracts the canonical Normal,
rho=0.5 results from the previous SC-CV v2 archive. It compares ell-RMSE curves
at fixed n=20,000 with d=2,5,10,20 and at fixed d=5 with
n=1,000,3,000,10,000,30,000. All 7,800 saved candidate estimates are retained,
with 30 repetitions per condition. Current Python/C++ source hashes match
the archived protocol; all 260 recomputed RMSEs agree with the previous
PDF-definition aggregate table within 1.69e-14 nats.

Stars mark unrestricted same-repetition grid RMSE minima; open squares mark
the integer bound minimizer at delta=.05. These agree in 3/8 conditions.
The rounded dimension-aware rate with C=1 selects ell=1 in all eight cases.
Neither formula is assumed to minimize actual finite-sample RMSE. Gaussian
data are outside the bounded-support theorem assumptions; sparse-grid results
also retain the canonical covered-observation averaging convention.

```sh
MPLCONFIGDIR=/private/tmp/pss-bounded-mpl XDG_CACHE_HOME=/private/tmp/pss-bounded-cache python experiments/synthetic_gaussian/analyze.py
```

The two-panel figure is `output/pdf/synthetic_gaussian_20261006/gaussian_ell_rmse.pdf`.
Tables, a PNG preview, captions, and provenance are in
`results/synthetic_gaussian_20261006/`. No estimator or coefficient is refit.

## Exploratory lower-correlation Gaussian sweep

`experiments/synthetic_gaussian_rho/` fixes rho=0,.1,.2,.3,.4 before evaluating
new outcomes and retains the previous rho=.5 baseline. The same eight n/d
settings, 30 repetitions, full ell grids, estimator, and theoretical rules
are used. Seeds from the archived baseline are reused at each correlation;
this is a paired exploratory extension prompted by the rho=.5 results, not
an independent confirmation. All correlations are reported, with rho=0
explicitly labeled as the independence control.

The run completed 39,000 new fits and copied 7,800 baseline estimates, with
no failures or clipping/ties. All 7,037 zero-valid outcomes are preserved
under the existing estimate=0 convention. Eight regenerated baseline samples
matched archived array hashes and estimates to 1.78e-15 nats.

| rho | Bound-minimum matches / 8 | C=1 rate matches / 8 | Worst bound RMSE / empirical minimum |
| --- | ---: | ---: | ---: |
| 0.0 | 7 | 8 | 2.762 |
| 0.1 | 7 | 8 | 2.228 |
| 0.2 | 7 | 6 | 1.238 |
| 0.3 | 5 | 4 | 3.119 |
| 0.4 | 4 | 3 | 7.619 |
| 0.5 | 3 | 3 | 28.673 |

At rho=.1 every empirical minimum is ell=1. At rho=.2 the n=20,000,d=2
minimum is ell=2 and matches the bound; n=30,000,d=5 is the sole bound
mismatch. Smaller correlation reduces the population product-approximation
bias, so more ell=1 matches do not establish finite-sample optimality of the
theoretical rule. The empirical minima and ratios are descriptive, selected
and evaluated on the same repetitions. Gaussian densities remain outside
the theorem's distributional assumptions.

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/synthetic_gaussian_rho/run.py
MPLCONFIGDIR=/private/tmp/pss-bounded-mpl XDG_CACHE_HOME=/private/tmp/pss-bounded-cache python experiments/synthetic_gaussian_rho/analyze.py
```

The two-page comparison is in `output/pdf/synthetic_gaussian_rho_20261006/`;
the full candidate results, optimal-ell table, agreement summary, protocol,
and audits are in `results/synthetic_gaussian_rho_20261006/`.

## Low-dimensional Gaussian extension

`experiments/synthetic_gaussian_lowdim/` freezes rho=.2, d=3 with
n=30,000,100,000,300,000 and n=100,000 with d=2,3,4 (five unique settings),
30 new-seed repetitions and ell=1..8 before computation. The current canonical
estimator and theoretical rules are unchanged. All 1,200 candidate fits
completed; no clipping, ties, failed fits, or zero-valid outcomes occurred.
The minimum coverage over every candidate/repetition was 0.99459; every
empirical-minimum candidate had full coverage in every repetition.

| n | d | Empirical minimum ell | Bound minimum ell | Minimum RMSE |
| ---: | ---: | ---: | ---: | ---: |
| 30,000 | 3 | 2 | 2 | 0.027702 |
| 100,000 | 2 | 2 | 3 | 0.004789 |
| 100,000 | 3 | 2 | 2 | 0.009886 |
| 100,000 | 4 | 2 | 2 | 0.022737 |
| 300,000 | 3 | 2 | 2 | 0.002264 |

Bound minimization agrees in 4/5 settings, all at ell=2. The distinct
dimension-aware rate with C=1 continues to choose ell=1 in every setting.
All conditions, including the d=2 mismatch, are shown. Empirical minima are
descriptive same-repetition grid minima. This is a targeted exploratory
extension after earlier findings, with Gaussian densities outside the theorem
assumptions; it is not a proof of finite-sample optimality or a fitted rate.

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/synthetic_gaussian_lowdim/run.py
MPLCONFIGDIR=/private/tmp/pss-bounded-mpl XDG_CACHE_HOME=/private/tmp/pss-bounded-cache python experiments/synthetic_gaussian_lowdim/analyze.py
```

The two-panel figure is in `output/pdf/synthetic_gaussian_lowdim_20261006/`;
raw results, protocol, summaries and audits are in
`results/synthetic_gaussian_lowdim_20261006/`.
The publication version uses the legend labels "RMSE-minimizing ell" and
"Theory-selected ell (delta=0.05)" at a 7.2-inch figure width, with explanatory
details moved into the manuscript. `output/fig2_gaussian_manuscript.tex` contains the
insertion text, the explicit unit-coefficient selection criterion, and the
figure environment/caption. It is a fragment for the parent manuscript.

## Original five-family figure reuse

`experiments/five_family_legacy/analyze.py` rebuilds the original five-family
comparison from saved data without new estimator fits or parameter selection.
All 330 RMSE/SE values are recomputed and checked against 9,900 selected
replicate records (55 settings, six methods, 30 repetitions each). The figure
retains sample-size RMSE, sample-size runtime, dimension RMSE, and copula
correlation RMSE. Error bars are +/- one Monte Carlo SE, conditional on the
saved oracle choices; runtime excludes parameter search.

This figure explicitly retains historical rank-spacing/n PSS. It is not
relabeled as the newer sub-grid/N_eff estimator. Original files are read-only;
recomputed summaries, source hashes and the caption are retained in
`results/five_family_legacy_20261006/`. The four-row PDF is
`output/pdf/five_family_legacy_20261006/five_family_benchmark.pdf`.

```sh
MPLCONFIGDIR=/private/tmp/pss-bounded-mpl XDG_CACHE_HOME=/private/tmp/pss-bounded-cache PYTHONDONTWRITEBYTECODE=1 python experiments/five_family_legacy/analyze.py
```

A separate occupancy figure may be omitted from the concise presentation:
the low-dimensional Gaussian ell-RMSE figure already illustrates partition
sensitivity, and coverage remains recorded in the corresponding summaries.
All five selected empirical-minimum candidates in that extension had full
training coverage in every repetition; this does not certify entropy accuracy
or theorem admissibility.

## Bounded dependent-density Fig. 2

`experiments/synthetic_bounded/fig2.py` reuses the audited cosine-ridge density
`f(x)=1+0.7*cos(2*pi*(x1-x2))` on `[0,1]^d`. It satisfies the bounded-support,
positive-density-lower-bound and Lipschitz assumptions with dimension-independent
constants `c=.3`, `C=1.7`, and `L=1.4*pi*sqrt(2)`.

The initial figure compares d=2,5 at n=100,000 and n=10,000,100,000,1,000,000
at d=2. These are four unique settings: 330 existing datasets and 9,400 archived
candidate estimates. The common displayed grid ell=1..7 preserves all four
empirical minima and theory selections from the full archived grids.
There is no coverage filter; hollow circles identify incomplete coverage.
All four empirical minima have full coverage in every repetition.

The theoretical rule agrees with the empirical minimum in two of four settings.
At d=2 the empirical ell grows 2,3,4 with sample size, while the theoretical
selection is 2,3,3. At n=100,000,d=5 they are 2 and 1, respectively. The density
regularity assumptions hold, but the sample-size conditions involving unknown
K0 are not certified. The common K*d multiplier in the displayed bound cancels
from its minimization; no coefficient is fitted.

```sh
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/pss-bounded-mpl XDG_CACHE_HOME=/private/tmp/pss-bounded-cache python experiments/synthetic_bounded/fig2.py
```

The vector figure is `output/pdf/synthetic_bounded_fig2_20261006/bounded_ell_rmse.pdf`.
Updated manuscript text, the explicit selection formula and caption are in
`output/fig2_bounded_initial_manuscript.tex`. The Gaussian figure and its manuscript fragment
are preserved separately. Data summaries, provenance hashes and the audit are
in `results/synthetic_bounded_fig2_20261006/`.

### Requested dimension and small-sample extension

The updated figure adds d=3,4 at n=100,000 (100 new fixed-seed repetitions
each, ell=1..7) and reuses the archived n=1,000,d=2 results (100 repetitions).
`experiments/synthetic_bounded/run_fig2_extension.py` freezes the new conditions,
seeds, sampler and estimator hashes before evaluation and retains all 1,400
new candidate estimates. The corrected request does not include n=10,000,000.
The sampler and canonical estimator are unchanged; no candidates or coefficients
are tuned for agreement.

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/synthetic_bounded/run_fig2_extension.py
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/pss-bounded-mpl XDG_CACHE_HOME=/private/tmp/pss-bounded-cache python experiments/synthetic_bounded/fig2.py --extended
```

The extended figure has seven unique settings and 630 datasets. At n=100,000,
the empirical ell for d=2,3,4,5 is 3,3,2,2 and the theoretical ell is 3,2,2,1.
At d=2, the empirical ell for n=1,000,10,000,100,000,1,000,000 is 1,2,3,4 and
the theoretical ell is 2,2,3,3. They agree in three of seven unique settings.
Existing central values and bootstrap intervals are preserved. All empirical
minima have full coverage, with no coverage filter applied.

The seven-setting comparison PDF is
`output/pdf/synthetic_bounded_fig2_extended_20261006/bounded_ell_rmse.pdf`, and
its manuscript insertion is `output/fig2_bounded_extended_manuscript.tex`.
New raw simulations and their frozen protocol are in
`results/synthetic_bounded_fig2_extension_20261006/`; the combined figure's
summaries and audit are in `results/synthetic_bounded_fig2_extended_20261006/`.

### Compact publication Fig. 2

The current publication figure presents a requested subset of the seven-setting
comparison: n=100,000 with d=2,3,4 in panel (a), and d=2 with
n=1,000,10,000,100,000 in panel (b). These are five distinct displayed settings,
each with 100 repetitions (500 saved datasets), using the common grid ell=1..7.
This is a presentation change using saved results; no additional simulations
are run. The expanded seven-setting figure and results remain preserved above.

In panel (a), the empirical ell choices are 3,3,2 and the theoretical choices
are 3,2,2. In panel (b), they are 1,2,3 and 2,2,3, respectively. They agree in
three of the five displayed settings. The other two displayed settings have
theory-selected RMSE about 7.8%--7.9% above the empirical grid minimum.
These counts describe the displayed subset, not every tested setting.
All displayed empirical minima have full coverage; no coverage filter is used.

```sh
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/pss-bounded-mpl XDG_CACHE_HOME=/private/tmp/pss-bounded-cache python experiments/synthetic_bounded/fig2.py --compact
```

The current vector figure is
`output/pdf/synthetic_bounded_fig2_compact_20261006/bounded_ell_rmse.pdf`.
The compact bounded-only manuscript insertion and caption were saved in
`output/fig2_bounded_compact_manuscript.tex`;
the compact summaries, provenance and audit are in
`results/synthetic_bounded_fig2_compact_20261006/`.
