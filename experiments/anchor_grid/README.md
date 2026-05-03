# Anchor-Grid Synthetic Experiments

This folder contains the NeurIPS-revision synthetic experiment driver used for
compact all-estimator comparisons across
Normal, Gamma, Beta, Lognormal, and Laplace Gaussian-copula distributions.

The experiment uses a common anchor and varies one axis at a time:

- `N scaling`: vary `n`, fix `d=5`, `rho=0`
- `d scaling`: vary `d`, fix `n=10000`, `rho=0`
- `rho scaling`: vary `rho`, fix `n=10000`, `d=5`

Estimators:

- PSS
- CADEE
- KL
- KSG
- trained UM-tKL
- trained UM-tKSG

## Run

From the repository root:

```bash
python experiments/anchor_grid/make_anchor_grid_datasets.py
```

The command prints the created output directory, for example
`results/anchor_grid_all_estimators_YYYYMMDD_HHMMSS`.

Then run:

```bash
Rscript experiments/anchor_grid/run_pss_cadee.R --base-dir=results/anchor_grid_all_estimators_YYYYMMDD_HHMMSS
python experiments/anchor_grid/run_knn_um.py --base-dir results/anchor_grid_all_estimators_YYYYMMDD_HHMMSS
python experiments/anchor_grid/plot_anchor_grid_results.py --base-dir results/anchor_grid_all_estimators_YYYYMMDD_HHMMSS
```

The Python UM baseline uses the original normalizing-flow code under `KNN/`.
Because that code is Python-2-era Theano code, `run_knn_um.py` prepares a
temporary Python 3 copy under the result directory and patches only that
temporary copy. KL/KSG are implemented directly in the driver and do not require
Theano.

For a smoke test without normalizing-flow training:

```bash
python experiments/anchor_grid/run_knn_um.py --base-dir results/anchor_grid_all_estimators_YYYYMMDD_HHMMSS --skip-um
```

## Outputs

- `settings.csv`
- `r_pss_cadee_summary.csv`
- `knn_um_summary.csv`
- `combined_summary.csv`
- `plots/fig_n_scaling.{png,pdf}`
- `plots/fig_d_scaling.{png,pdf}`
- `plots/fig_rho_scaling.{png,pdf}`

