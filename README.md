# PSS: Partitioned Sample-Spacing Estimator
*A nonparametric estimator for multivariate joint entropy and mutual information*

This repository contains the full implementation of the **Partitioned Sample-Spacing (PSS) estimator**, along with comparison baselines (KNN, CADEE) and real-world experiments (ICA, UCI Energy feature selection).
This code accompanies the manuscript:

**"Nonparametric Estimation of Joint Entropy via Partitioned Sample-Spacing"**
See manuscript for full theoretical details.
[![arXiv](https://img.shields.io/badge/arXiv-2511.13602-b31b1b.svg)](https://arxiv.org/abs/2511.13602)

## Start here: October 2026 manuscript figures

Use the [paper reproduction guide](docs/paper_reproduction.md) for the current
manuscript's Figure 2--5 commands, exact Energy settings, and implementation
mapping. Each command below uses files included in this repository and can run
from a fresh checkout; it does not require the original author's archive paths.

```sh
# Tested with Python 3.11 and the archived package versions.
python -m pip install -r experiments/requirements-energy-synthetic.txt
python experiments/paper_figures/figure2/plot.py --output-dir /tmp/pss-figure2
python experiments/paper_figures/figure3/plot.py --output-dir /tmp/pss-figure3
python experiments/synthetic_ell_comparison/plot.py --output-dir /tmp/pss-figure4
python experiments/paper_figures/figure5/plot.py --output-dir /tmp/pss-figure5
```

These commands reproduce saved-result plots. Figure 5 additionally verifies
accuracy from the bundled predictions. They do not rerun simulation, tuning,
feature selection, or classifier training.

**Implementation mapping matters.** Figure 3 uses historical rank-spacing PSS
with denominator `n`. Figure 2 preserves historical SC-CV results whose coverage
convention differs from the training-box support rule in the current manuscript;
its original simulation generator has not been recovered. Neither is certified
as a result of the current Python/C++ v2 estimator. Figure 4 and the PSS methods
in Figure 5 use v2. The [guide](docs/paper_reproduction.md) gives the evidence and
the limits of each reproduction command.

## October 2026 Energy and synthetic experiments

The updated studies include the canonical Python/C++ PSS v2 estimator,
[the Energy dataset and attribution](data/README.md), experiment source,
frozen result tables, protocols, saved audits, and publication figures.

| Study bundle | Contents |
| --- | --- |
| [Energy experiments](experiments/energy_results_20261006/README.md) | Same-period classification, coverage/fold expansion, class-aware coverage, JMI, and theory-rate coefficient calibration |
| [Synthetic experiments](experiments/synthetic_results_20261006/README.md) | Bounded dependent densities, Gaussian correlation and low-dimension extensions, and the historical five-family comparison |
| [Study notes](docs/energy_synthetic_studies.md) | Protocols, interpretation, limitations, and original workflow commands |

Reproduce the published summary figures and verify the synthetic bundle:

```sh
python -m pip install -r experiments/requirements-energy-synthetic.txt
python experiments/energy_results_20261006/plot.py --output-dir /tmp/pss-energy-figures
python experiments/synthetic_results_20261006/verify.py
python experiments/synthetic_ell_comparison/plot.py --output-dir /tmp/pss-synthetic-figures
```

To build and check the canonical estimator, use `python PSS/build_pss_v2.py`
and `python PSS/test_density.py`. A C++17 compiler is required for estimator
runs, but not for plotting saved tables. The study bundles distinguish portable
plot/verification commands from historical simulation workflows: Energy caches
and some older synthetic inputs are omitted, and frozen historical scripts
retain their original source hashes and archive paths. No new simulation or
selection of parameters was performed for this publication.

## Partition-resolution comparison: bounded and Gaussian densities

The combined four-panel figure compares the RMSE-minimizing partition count
with the theory-selected count for bounded dependent and correlated Gaussian
densities. It reuses saved estimates and pointwise bootstrap intervals.

See [the figure and reproduction instructions](experiments/synthetic_ell_comparison/README.md),
[the vector PDF](experiments/synthetic_ell_comparison/figures/bounded_gaussian_ell_rmse.pdf),
and [the manuscript text and caption](experiments/synthetic_ell_comparison/manuscript.tex).

```sh
python -m pip install -r experiments/synthetic_ell_comparison/requirements.txt
python experiments/synthetic_ell_comparison/plot.py
```

💻 Installation & Usage
1. Prerequisites

Ensure the following software and libraries are installed.

R Environment (Tested on R ≥ 4.0)
Install required R packages
install.packages(c("tidyverse", "ggrepel", "scales", "stringr", "mvtnorm", "FNN", "fastICA", "RWeka", "e1071"))

Python Environment (For kNN baselines)
Install required Python libraries
pip install numpy scipy pandas matplotlib scikit-learn


2. Historical simulation workflows (earlier manuscript versions)

The commands in this section retain the original R and baseline workflows.
Their old figure numbering does not map to the current manuscript. For current
Figure 2--5 saved-result reproduction, use the guide above. Full historical UM
experiments also require their original Theano environment; the Python
requirements above cover the saved plots and newer Energy experiments.

The reproduction process consists of two steps: Simulation and Visualization.

Step 1: Run Simulations

Execute the simulation scripts in each directory to generate the result data (.csv).
PSS (Proposed):
Rscript PSS/run_pss_mvn.R
Rscript PSS/run_pss_gamma.R

NeurIPS revision diagnostics:
Rscript PSS/run_pss_regime_diagnostics.R --quick
Rscript PSS/run_pss_regime_diagnostics.R --full
Rscript PSS/run_pss_regime_diagnostics.R --full --stable-coverage-min=0.99
Rscript PSS/plot_pss_regime_diagnostics.R --mode=full --rho=0.5

NeurIPS anchor-grid all-estimator experiments (Normal/Gamma/Beta/Lognormal/Laplace):
python experiments/anchor_grid/make_anchor_grid_datasets.py
Rscript experiments/anchor_grid/run_pss_cadee.R --base-dir=results/anchor_grid_all_estimators_YYYYMMDD_HHMMSS
python experiments/anchor_grid/run_knn_um.py --base-dir results/anchor_grid_all_estimators_YYYYMMDD_HHMMSS
python experiments/anchor_grid/plot_anchor_grid_results.py --base-dir results/anchor_grid_all_estimators_YYYYMMDD_HHMMSS

Or run the full pipeline:
bash experiments/anchor_grid/run_all.sh

kNN Baselines:
python KNN/run_knn_mvn_n.py
python KNN/run_knn_gamma_corr.py
... (Run other experiment scripts in KNN/ folder as needed)

CADEE:
Rscript CADEE/run_cadee_mvn.R
...

Step 2: Generate Figures (Important!)

After the simulations are complete, use the master plotting script run_plots.R.

Collect Data: Move ALL .csv output files generated from the simulations (PSS, KNN, CADEE) into the root directory (the same directory where run_plots.R is located).

Run Script: Execute the plotting script.

Rscript run_plots.R



3. Historical real-data workflows

To reproduce the application results:

Historical feature selection: Rscript uci_energy.R

The current manuscript's Figure 5 uses the Python Energy study and the portable
four-method plot command above; it is not the output of this historical R script.

ICA Experiment: Refer to the ICA/ directory and run ica_pss.R.


## 📂 Repository Structure

```text
.
├── CADEE/          # R implementation of CADEE estimator (Ariel & Louzoun, 2020)
├── ICA/            # ICA experiment code (UCI EEG Eye-State dataset)
├── KNN/            # Python implementations of kNN baselines (KL, KSG, UM-kNN)
├── PSS/            # Core source code for the proposed PSS estimator
├── experiments/    # NeurIPS-revision synthetic experiment drivers
├── run_plots.R     # R script to reproduce simulation plots (Figures 2, 3, and 4)
├── uci_energy.R    # Feature selection experiment script (UCI Appliances Energy)
└── README.md       # Project documentation


## 📂 Folder Details

### **PSS/**
Contains the core implementation of the Partitioned Sample-Spacing (PSS) estimator.
**Main files:**
- **pss_entropy.R**: Main function for the PSS joint entropy estimator.
- **run_pss_gamma.R**: Computes RMSE and runtime under the multivariate Gamma distribution with varying sample size ($N$), dimension ($d$), and correlation ($\rho$), using the optimal $\ell$ that minimizes RMSE.
- **run_pss_mvn.R**: Computes RMSE and runtime under the multivariate Normal distribution with varying sample size ($N$), dimension ($d$), and correlation ($\rho$), using the optimal $\ell$ that minimizes RMSE.
- **run_pss_regime_diagnostics.R**: NeurIPS-oriented simulation driver for oracle-vs-SC-CV tuning, Normal/Gamma/Beta/Lognormal/Laplace Gaussian-copula families, and occupancy/skipped-point diagnostics. SC-CV selects \(\ell\) by validation negative log-likelihood subject to stable validation coverage \(S(\ell) \ge 0.99\), where a validation point is stable if it has finite PSS density and falls in a training cell with at least `occupancy_min` observations. Results are written to `results/pss_diagnostics/`.
- **plot_pss_regime_diagnostics.R**: Generates paper-ready plots from the diagnostic CSVs, including SC-CV-vs-oracle RMSE gap, empirical convergence, occupancy/skipped-point diagnostics, selected partition counts, and CV objective curves.

### **experiments/anchor_grid/**
Contains the compact all-estimator synthetic comparison used for the NeurIPS revision. It generates shared datasets from Normal, Gamma, Beta, Lognormal, and Laplace Gaussian-copula models; runs PSS, CADEE, KL, KSG, trained UM-tKL, and trained UM-tKSG; and writes the paper-style `N scaling`, `d scaling`, and `rho scaling` plots to `plots/`.

### **KNN/**
Implements $k$-Nearest-Neighbor ($k$NN) based baseline entropy estimators (KL, KSG, etc.).
**Main files:**
- **run_knn_gamma_corr.py**: Computes RMSE/runtime for Gamma distribution with varying correlations ($\rho$).
- **run_knn_gamma_d.py**: Computes RMSE/runtime for Gamma distribution with varying dimensions ($d$).
- **run_knn_gamma_n.py**: Computes RMSE/runtime for Gamma distribution with varying sample sizes ($N$).
- **run_knn_mvn_corr.py**: Computes RMSE/runtime for Normal distribution with varying correlations ($\rho$).
- **run_knn_mvn_d.py**: Computes RMSE/runtime for Normal distribution with varying dimensions ($d$).
- **run_knn_mvn_n.py**: Computes RMSE/runtime for Normal distribution with varying sample sizes ($N$).
- **ica_knn.py**: Python implementation of $k$NN estimators for the ICA experiment.
- **util/temp_data/**: Directory storing the generated `.csv` result files.

### **CADEE/**
Implementation of the Copula Decomposition Entropy Estimator (CADEE) by Ariel & Louzoun (2020).
**Main files:**
- **CADEE.R**: Main function for CADEE entropy estimation.
- **run_cadee_gamma.R**: Computes RMSE and runtime under the multivariate Gamma distribution with varying parameters.
- **run_cadee_mvn.R**: Computes RMSE and runtime under the multivariate Normal distribution with varying parameters.

### **ICA/**
Contains the full pipeline for the Independent Component Analysis (ICA) experiment using the UCI EEG Eye-State dataset.
**Main files:**
- **ica_pss.R**: Executes FastICA and calculates Total Correlation (TC) using the PSS method.
- **EEG Eye State.arff**: The UCI EEG Eye-State dataset file.

### **Analysis Scripts**
- **uci_energy.R**: Script for the feature selection experiment using the UCI Appliances Energy dataset.
  - Mutual Information (MI) based greedy forward selection.
  - Comparison of PSS vs. $k$NN estimators.
  - SVM (RBF kernel) accuracy evaluation.
  - Generates **Figure 6(a)** and **6(b)**.
- **run_plots.R**: Generates all simulation plots (**Figure 2, 3, and 4**) by aggregating `.csv` results from the PSS, KNN, and CADEE experiments.


### **run_plots.R**
## 📊 Visualization

To generate the plots (Figures 2, 3, and 4), use the provided R script `run_plots.R`.`
 How to Run
1. **Collect Data**: Move **all** `.csv` files generated by the Python simulations into the **same directory** as `run_plots.R`.
   > *Note: The script searches for `.csv` files in the current directory only.*
2. **Run Script**: Execute the script in RStudio or via terminal:
   ```R
   source("run_plots.R")

---

### **README.md**
Project documentation, structure overview, usage instructions, and references.
