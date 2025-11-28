#==============================================================================
#  CONSOLIDATED PSS SIMULATION SCRIPT (Multivariate Gamma, Gaussian Copula)
#------------------------------------------------------------------------------
#  • Experiment 1: Varying Sample Size (n) with fixed dimension d = 5
#  • Experiment 2: Varying Dimension (d) with fixed sample size n = 30000
#. • Experiment 3: Varying correlation(rho) with fixed sample size n = 50000, d=7
#  Output Columns (unified for plotting):
#    Distribution, Dimensions, N_Samples, Correlation, Method,
#    Optimal_Param, Eval_Time_s, RMSE
#==============================================================================

# --- 1) Load required libraries ----------------------------------------------
pkgs <- c("MASS", "dplyr", "knitr")
to_install <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]
if (length(to_install)) install.packages(to_install, repos = "https://cloud.r-project.org")
library(MASS)
library(dplyr)
library(knitr)

# --- 2) Ensure the required entropy function exists --------------------------
if (!exists("calculate_entropy_pss_d")) {
  stop("Missing function: calculate_entropy_pss_d(data, n_partitions). Please define it first.")
}

# --- 3) Gamma helper functions ------------------------------------------------

# Differential entropy of Gamma(shape, scale), vectorized version
gamma_entropy <- function(shape, scale) {
  # Both shape and scale may be scalars or vectors of the same length
  shape + log(scale) + lgamma(shape) + (1 - shape) * digamma(shape)
}

# True entropy of a Gaussian-copula Gamma with equicorrelation rho
true_entropy_gamma_gausscop <- function(d, rho, shape, scale) {
  # Recycle scalars to vectors of length d
  shape_vec <- rep_len(shape, d)
  scale_vec <- rep_len(scale, d)
  
  # Marginal contribution
  H_marginal <- sum(gamma_entropy(shape_vec, scale_vec))
  
  # Copula contribution: (1/2) * log(det(R))
  R <- matrix(rho, d, d); diag(R) <- 1
  H_copula <- 0.5 * as.numeric(determinant(R, logarithm = TRUE)$modulus)
  
  H_marginal + H_copula
}

# Generate random samples from a Gaussian-copula multivariate Gamma(shape, scale)
rmgamma_gausscop <- function(n, d, rho, shape, scale, u_clip = 1e-12) {
  # Recycle scalars to vectors of length d
  shape_vec <- rep_len(shape, d)
  scale_vec <- rep_len(scale, d)
  
  # Basic validity checks
  if (any(!is.finite(shape_vec)) || any(!is.finite(scale_vec)) ||
      any(shape_vec <= 0) || any(scale_vec <= 0)) {
    stop("shape/scale must be positive and finite (scalars or length-d vectors).")
  }
  
  # Correlation matrix for the Gaussian copula
  R <- matrix(rho, d, d); diag(R) <- 1
  
  # Step 1: Generate latent Gaussian samples
  Z <- MASS::mvrnorm(n, mu = rep(0, d), Sigma = R)
  
  # Step 2: Transform to uniforms via Φ
  U <- pnorm(Z)
  
  # Step 3: Avoid exact 0/1 values (prevents Inf/0 in qgamma)
  if (!is.null(u_clip) && u_clip > 0) {
    U <- pmin(pmax(U, u_clip), 1 - u_clip)
  }
  
  # Step 4: Apply inverse CDF of Gamma to obtain target samples
  X <- matrix(NA_real_, nrow = n, ncol = d)
  for (j in seq_len(d)) {
    X[, j] <- qgamma(U[, j], shape = shape_vec[j], scale = scale_vec[j])
  }
  X
}

# --- 4) Core routine for a single (n, d, rho) configuration -------------------
run_setting_pss_gamma <- function(n, d, rho, l_range, n_reps, shape, scale) {
  # Pre-generate datasets once (shared across ℓ values)
  datasets <- lapply(seq_len(n_reps), function(i) {
    rmgamma_gausscop(n, d, rho, shape, scale)
  })
  
  # True entropy (analytical)
  H_true <- true_entropy_gamma_gausscop(d, rho, shape, scale)
  
  # Squared error matrix across ℓ
  pss_se_matrix <- sapply(l_range, function(l) {
    vapply(datasets, function(X) {
      est <- calculate_entropy_pss_d(X, n_partitions = l)
      (est - H_true)^2
    }, numeric(1L))
  })
  
  # Ensure columns correspond to ℓ
  if (nrow(pss_se_matrix) == length(l_range)) pss_se_matrix <- t(pss_se_matrix)
  
  # Select ℓ* (minimizing MSE) and compute RMSE
  mse_per_l <- colMeans(pss_se_matrix, na.rm = TRUE)
  opt_idx   <- which.min(mse_per_l)
  opt_l     <- l_range[opt_idx]
  opt_rmse  <- sqrt(mse_per_l[opt_idx])
  
  # Measure evaluation time for a single call at ℓ* (search not timed)
  opt_eval_time <- system.time({
    invisible(calculate_entropy_pss_d(datasets[[1]], n_partitions = opt_l))
  })[["elapsed"]]
  
  # Unified tibble output
  tibble::tibble(
    Distribution   = "Gamma",
    Dimensions     = d,
    N_Samples      = n,
    Correlation    = rho,
    Method         = "PSS",
    Optimal_Param  = opt_l,
    Eval_Time_s    = as.numeric(opt_eval_time),
    RMSE           = as.numeric(opt_rmse)
  )
}

# --- 5) Global settings -------------------------------------------------------
set.seed(42)
l_range <- 1:23
n_reps  <- 10

#==============================================================================
#  Experiment 1: Varying Sample Size (n) with fixed d = 5
#==============================================================================
cat("============================================================\n")
cat("PSS Gamma — Experiment 1: Varying n (fixed d = 5)\n")
cat("============================================================\n")

sample_sizes_n <- c(1000, 2000, 4000, 8000)
d_n            <- 5
rho_n          <- 0
shape_n        <- 0.4
scale_n        <- 0.3

sim_grid_n <- expand.grid(n = sample_sizes_n, d = d_n, rho = rho_n)

results_list_n <- lapply(seq_len(nrow(sim_grid_n)), function(i) {
  params <- sim_grid_n[i, ]
  cat(sprintf("Running PSS Gamma (n=%d, d=%d, rho=%.1f) ...\n", params$n, params$d, params$rho))
  run_setting_pss_gamma(
    n = params$n, d = params$d, rho = params$rho,
    l_range = l_range, n_reps = n_reps,
    shape = shape_n, scale = scale_n
  )
})
results_n <- bind_rows(results_list_n)

cat("\nExperiment 1 complete.\n")
cat("Saving results to pss_gamma_results_n.csv...\n\n")
write.csv(results_n, "pss_gamma_results_n.csv", row.names = FALSE)
print(knitr::kable(
  results_n,
  caption = "PSS (Gamma, Gaussian Copula): varying n with fixed d = 5",
  digits  = 5,
  align   = "c"
))

#==============================================================================
#  Experiment 2: Varying Dimension (d) with fixed n = 30000
#==============================================================================
cat("\n\n============================================================\n")
cat("PSS Gamma — Experiment 2: Varying d (fixed n = 30000)\n")
cat("============================================================\n")

d_values_d     <- c(2, 5, 10, 15, 20)
sample_sizes_d <- 30000
rho_d          <- 0
shape_d        <- 0.4
scale_d        <- 0.3

sim_grid_d <- expand.grid(d = d_values_d, n = sample_sizes_d, rho = rho_d)
cat("Simulation grid (fixed n = 30000):\n"); print(sim_grid_d)

results_list_d <- lapply(seq_len(nrow(sim_grid_d)), function(i) {
  params <- sim_grid_d[i, ]
  cat(sprintf("\nRunning PSS Gamma (d=%d, n=%d, rho=%.1f) ...\n", params$d, params$n, params$rho))
  run_setting_pss_gamma(
    n = params$n, d = params$d, rho = params$rho,
    l_range = l_range, n_reps = n_reps,
    shape = shape_d, scale = scale_d
  )
})
results_d <- bind_rows(results_list_d)

cat("\nExperiment 2 complete.\n")
cat("Saving results to pss_gamma_results_d.csv...\n\n")
write.csv(results_d, "pss_gamma_results_d.csv", row.names = FALSE)
print(knitr::kable(
  results_d,
  caption = "PSS (Gamma, Gaussian Copula): varying d with fixed n = 30000",
  digits  = 5,
  align   = "c"
))

#==============================================================================
#  Experiment 3: Varying Correlation (rho) with fixed d = 7, n = 50000
#  (Robustness Check for High Correlation)
#==============================================================================
cat("\n\n============================================================\n")
cat("PSS Gamma — Experiment 3: Varying rho (fixed d=7, n=50000)\n")
cat("============================================================\n")

d_rho        <- 7
n_rho        <- 50000
rho_values   <- c(0.0,0,1, 0.3, 0.8)
shape_rho    <- 0.4
scale_rho    <- 0.3

sim_grid_rho <- expand.grid(rho = rho_values, d = d_rho, n = n_rho)
cat("Simulation grid (Robustness Check):\n"); print(sim_grid_rho)

results_list_rho <- lapply(seq_len(nrow(sim_grid_rho)), function(i) {
  params <- sim_grid_rho[i, ]
  cat(sprintf("\nRunning PSS Gamma (rho=%.1f, d=%d, n=%d) ...\n", params$rho, params$d, params$n))
  
  run_setting_pss_gamma(
    n = params$n, d = params$d, rho = params$rho,
    l_range = l_range, n_reps = n_reps,
    shape = shape_rho, scale = scale_rho
  )
})

results_rho <- bind_rows(results_list_rho)

cat("\nExperiment 3 complete.\n")
cat("Saving results to pss_gamma_results_rho.csv...\n\n")
write.csv(results_rho, "pss_gamma_results_rho.csv", row.names = FALSE)

print(knitr::kable(
  results_rho,
  caption = "PSS (Gamma, Gaussian Copula): varying rho with fixed d=7, n=50000",
  digits  = 5,
  align   = "c"
))
