# PSS: Partitioned Sample-Spacing Estimator  
*A nonparametric estimator for multivariate joint entropy and mutual information*

This repository contains the full implementation of the **Partitioned Sample-Spacing (PSS) estimator**, along with comparison baselines (KNN, CADEE) and real-world experiments (ICA, UCI Energy feature selection).  
This code accompanies the manuscript:

**"Nonparametric Estimation of Joint Entropy via Partitioned Sample-Spacing"**    
See manuscript for full theoretical details. 
[![arXiv](https://img.shields.io/badge/arXiv-2511.13602-b31b1b.svg)](https://arxiv.org/abs/2511.13602)

💻 Installation & Usage
1. Prerequisites

Ensure the following software and libraries are installed.

R Environment (Tested on R ≥ 4.0)
Install required R packages
install.packages(c("tidyverse", "ggrepel", "scales", "stringr", "mvtnorm", "FNN", "fastICA", "RWeka", "e1071"))

Python Environment (For kNN baselines)
Install required Python libraries
pip install numpy scipy pandas matplotlib scikit-learn


2. How to Reproduce Simulation Results (Figures 2, 3, 4)

The reproduction process consists of two steps: Simulation and Visualization.

Step 1: Run Simulations

Execute the simulation scripts in each directory to generate the result data (.csv).
PSS (Proposed):
Rscript PSS/run_pss_mvn.R
Rscript PSS/run_pss_gamma.R

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



3. Real Data Experiments (Figure 6)

To reproduce the application results:

Feature Selection (Figure 6): Rscript uci_energy.R

ICA Experiment: Refer to the ICA/ directory and run ica_pss.R.


## 📂 Repository Structure

```text
.
├── CADEE/          # R implementation of CADEE estimator (Ariel & Louzoun, 2020)
├── ICA/            # ICA experiment code (UCI EEG Eye-State dataset)
├── KNN/            # Python implementations of kNN baselines (KL, KSG, UM-kNN)
├── PSS/            # Core source code for the proposed PSS estimator
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


