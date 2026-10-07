# Reproducing the October 2026 manuscript figures

This guide maps the manuscript **Nonparametric Estimation of Joint Entropy via
Partitioned Sample-Spacing** to the included source and saved results. Its
numbering refers to the October 2026 manuscript: SC-CV is Figure 2, the
five-family benchmark is Figure 3, partition sensitivity is Figure 4, and Energy
feature selection is Figure 5.

The portable commands here read saved results. They do not fit new estimators,
choose new hyperparameters, train classifiers, or establish that every historical
experiment can be rerun from scratch. The original frozen study sources and
hash records are retained separately.

## Figure and implementation map

| Figure | Included entry point | What is verified | PSS implementation |
| --- | --- | --- | --- |
| 1: bivariate geometry | No verified reproduction entry point in this bundle | Illustration only; this guide makes no reproduction claim | Not assigned from the image alone |
| 2: Gamma SC-CV versus oracle | [figure2/plot.py](../experiments/paper_figures/figure2/plot.py) | Historical saved replicate/summary records and their plotted values | Historical coverage convention; differs from v2's training-box support rule |
| 3: five marginal families | [figure3/plot.py](../experiments/paper_figures/figure3/plot.py) | Saved selected repetitions and RMSE/runtime summaries | Historical rank-spacing with denominator `n` |
| 4: bounded/Gaussian partition sensitivity | [synthetic_ell_comparison/plot.py](../experiments/synthetic_ell_comparison/plot.py) | Frozen tables, grids, repetitions, empirical and theoretical choices | Canonical Python/C++ v2, sub-grid spacing and denominator `N_eff` |
| 5: Energy, four methods | [figure5/plot.py](../experiments/paper_figures/figure5/plot.py) | Saved predicted labels, train-median targets, 240 accuracies, 80 means and sample SDs | Canonical v2; class-aware SC-CV and fixed theory coefficient `C=1` |

### Setup and commands

Run from the repository root, using Python 3.11. The pinned dependencies reproduce
the recorded Python environment for these checks:

```sh
python -m pip install -r experiments/requirements-energy-synthetic.txt
python experiments/paper_figures/figure2/plot.py --verify-only
python experiments/paper_figures/figure3/plot.py --verify-only
python experiments/paper_figures/figure5/plot.py --verify-only
python experiments/paper_figures/figure2/plot.py --output-dir /tmp/pss-figure2
python experiments/paper_figures/figure3/plot.py --output-dir /tmp/pss-figure3
python experiments/synthetic_ell_comparison/plot.py --output-dir /tmp/pss-figure4
python experiments/paper_figures/figure5/plot.py --output-dir /tmp/pss-figure5
```

Every entry point resolves inputs relative to its source file. To run from a
different working directory, give the absolute path to the script. Output paths
are user-selected and do not replace the archived inputs. Plot styling or PDF
metadata may differ across plotting-library versions; saved numeric values are
the reproduction target.

The [synthetic archive verifier](../experiments/synthetic_results_20261006/verify.py)
also verifies the larger saved-result bundle:

```sh
python experiments/synthetic_results_20261006/verify.py
```

## Historical Figure 2 and Figure 3

Figure 2's archived Gamma setting has `d=5`, latent copula correlation `rho=0`,
sample sizes `1000, 3000, 10000, 30000`, and 30 repetitions. Its saved means,
including SC-CV stable coverage `0.999766666666667` at `n=1000`, are preserved.
The exact simulation generator has not been recovered, so the new entry point
reproduces the saved figure and verifies its tables only.

That coverage value cannot be produced by the current manuscript's complete
K-fold rule with density zero outside each training sample's bounding box. For
continuous samples, the global minimum and maximum in any one coordinate are
outside their respective training boxes when held out. Thus stable coverage is
at most `1 - 2/n`, or `0.998` at `n=1000`, in every repetition. This establishes
a difference in the historical support/coverage convention, without identifying
an unverified original generator. Replotting these values does not resolve that
difference or turn them into v2 results.

Figure 3 is explicitly the **historical rank-spacing / n** experiment. Current
v2 instead evaluates the smoothed sub-grid density at each observation and
averages over `N_eff` valid observations. The old results should not be relabeled
as estimates from the new algorithm. Each displayed point uses 30 repetitions.
PSS and kNN candidates were chosen using aggregate RMSE on the same repetitions
used for reporting. PSS also used its recorded coverage/occupancy eligibility
filter. Error bars show one Monte Carlo SE conditional on that choice. Runtime
includes selected-parameter evaluation and flow training where applicable,
excluding parameter search. See the
[historical figure bundle](../experiments/paper_figures/figure3/README.md).

## Figure 4: partition sensitivity

The bounded density is `f(x)=1+0.7*cos(2*pi*(x1-x2))` on `[0,1]^d`. The Gaussian
model has unit marginal variances and common correlation `rho=.2`.

- Bounded: five unique settings, 100 repetitions, `ell=1..7`.
- Gaussian: five unique settings, 30 repetitions, `ell=1..8`.
- Theoretical choice: minimize
  `ell^-2 + sqrt(2*d*log(2*n+1)+log(96/.05))*(ell^d/n)^.25`
  over the displayed grid, preferring smaller `ell` in ties.
- Empirical choice: minimize aggregate RMSE over the same repetitions, without
  a coverage filter. Agreement is 3/5 bounded settings and 4/5 Gaussian settings.

The current plot uses the manuscript's horizontal four-panel arrangement. The
[figure package](../experiments/synthetic_ell_comparison/README.md) preserves
pointwise bootstrap intervals and the exploratory design details.

## Figure 5: exact Energy protocol

The four curves are **PSS class-aware SC-CV**, **KL tuned**, **PSS theory C=1**,
and **KL k=1**, over 1--20 selected features. The separately explored
`PSS theory C tuned` method is not the tuned-PSS curve in this figure.

The Appliances Energy dataset has 19,735 rows and 25 candidate predictors.
Outer 70:30 splits use seeds `42,43,44` with 13,814 training and 5,921 test
observations. Each outer training set has a further 70:30 inner split. Labels
are `Appliances > training median`, with equality assigned to class 0.

For feature selection, predictors receive training-only min--max scaling and
seeded independent Gaussian jitter with SD `1e-5`. The classifier receives
unjittered predictors and uses training-only mean/sample-SD scaling. Every
classifier is an RBF SVM with penalty `C_SVM=1` and `gamma=1/d`, where `d` is
the candidate subset size.

Forward selection maximizes the full-subset entropy difference
`H(X_S) - sum_y (n_y/n)*H(X_S | Y=y)`. Scores are not clipped. All entropy
components use one common `ell` or `k`. A PSS component with no valid observations
makes that candidate inadmissible. Exact feature-score ties prefer the smaller
original feature index.

Class-aware SC-CV evaluates `ell=1..5` separately for each candidate subset.
It minimizes pooled held-out NLL over covered observations, subject to stable
coverage in the pooled sample and both class-conditional samples. Energy fixes
`n_min=5` and tunes the 15 combinations of folds `{3,5,10}` and thresholds
`{.80,.85,.90,.95,.99}`. These are Energy-specific settings; the standalone
`select_sc_cv` defaults remain `n_min=10`, `tau=.99`, and 3 folds. Fallback
maximizes minimum component stable coverage, then minimizes pooled covered NLL,
then prefers smaller `ell`.

KL tunes `k` over `{1,2,3,5,7,10,15,20,30,50}`. Each configuration builds an
inner-training feature path and is scored by mean validation accuracy at 5, 10,
and 20 features. Equal scores prefer larger `tau`, then fewer folds for PSS,
and smaller `k` for KL. The chosen configuration is locked before outer test
evaluation and the path is rebuilt on all outer-training observations.

| Seed | PSS folds | PSS tau | KL k |
| ---: | ---: | ---: | ---: |
| 42 | 5 | .90 | 50 |
| 43 | 5 | .85 | 30 |
| 44 | 10 | .85 | 50 |

The fixed theory-guided curve uses

```text
ell = max(1, floor(C*(n/(d^6*log(n)^2))^(1/(d+8)) + .5)), C=1
```

Here `n` is the full pooled selection-training sample size, `d` is subset
dimension, `log` is natural, and halves round upward. The same `ell` is used
in all three entropy components. It gives `ell=2` at `d=1` and `ell=1` for
`d=2..20` at both recorded training sizes. This rule differs from Figure 4's
finite-grid minimization. `C` here is distinct from the SVM penalty.

Shading is one sample SD across three overlapping random splits, not a confidence
interval. These exploratory results concern same-period classification in one
household. The saved prediction checks do not establish generalization to new
households or accuracy of the MI estimates.

## Estimator checks and full-rerun limits

The canonical source is [PSS/pss_v2.cpp](../PSS/pss_v2.cpp), with the
[Python entry point](../PSS/pss_v2.py). A C++17 compiler and NumPy are needed
for estimator execution, but no compiled library is needed for the saved plots.

```sh
python PSS/build_pss_v2.py
python PSS/test_density.py
python -m unittest discover -s experiments/energy_sameperiod -p 'test_*.py'
python -m unittest discover -s experiments/energy_sc_expansion -p 'test_*.py'
python -m unittest discover -s experiments/energy_component_guard -p 'test_*.py'
python -m unittest discover -s experiments/energy_jmi -p 'test_*.py'
python -m unittest discover -s experiments/energy_theory_c -p 'test_*.py'
```

The portable figure bundles add saved-result verification without rewriting
historical estimator sources, cached protocols, or frozen audit hashes. Full
reruns still have additional dependencies: original synthetic archive inputs,
the historical UM/Theano environment, Energy prerequisite study outputs, and
platform/path adaptations. Figure 5 bundles the predictions needed for its
accuracy curves, not every original selector candidate, classifier model, or
training cache. See the existing
[Energy limitations](../experiments/energy_results_20261006/README.md#saved-audits-and-reproducibility-limits)
and [synthetic limitations](../experiments/synthetic_results_20261006/README.md#source-index-and-historical-reproduction-limits).

Repository documentation makes these differences inspectable; it does not
change the estimator used to generate an existing manuscript figure.
