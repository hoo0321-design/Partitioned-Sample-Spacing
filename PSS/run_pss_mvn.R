#==============================================================================
#  CONSOLIDATED PSS SIMULATION SCRIPT (MVN, PSS only)
#  - Experiment 1: Varying Sample Size (n) with fixed d = 10
#  - Experiment 2: Varying Dimension (d) with fixed n = 3000
#. - Experiment 3: Varying correlation (rho) with fixed n = 20000, d=5
#  Notes:
#    • For each (n, d, rho): generate n_reps datasets once (shared across ℓ)
#    • Sweep ℓ ∈ l_range, compute MSE across reps, pick ℓ* = argmin_l MSE_l
#    • Time exactly ONE evaluation at ℓ* on the first dataset (search not timed)
#    • Report RMSE = sqrt(MSE)
#    • Output columns are unified for plotting:
#      Distribution, Dimensions, N_Samples, Correlation, Method,
#      Optimal_Param, Eval_Time_s, RMSE
#==============================================================================

# --- 1) Libraries ---
pkgs <- c("MASS", "dplyr", "knitr")
to_install <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]
if (length(to_install)) install.packages(to_install, repos = "https://cloud.r-project.org")
library(MASS)
library(dplyr)
library(knitr)

# --- 2) Guard: required function must exist ----------------------------------
if (!exists("calculate_entropy_pss_d")) {
  stop("Missing function: calculate_entropy_pss_d(data, n_partitions). Please define it first.")
}

# --- 3) True entropy for MVN with equicorrelation ρ --------------------------
# H = (1/2) * log( (2πe)^d * det(Σ) ), Σ has unit variances and off-diag ρ
true_entropy_mvn <- function(d, rho, variance = 1) {
  Sigma <- matrix(rho, nrow = d, ncol = d)
  diag(Sigma) <- variance
  0.5 * log((2 * pi * exp(1))^d * det(Sigma))
}

# --- 4) Core routine for one (n, d, rho) -------------------------------------
run_setting_pss <- function(n, d, rho, l_range, n_reps, jitter_sd = 0) {
  # Pre-generate datasets once (shared across ℓ)
  Sigma <- matrix(rho, nrow = d, ncol = d); diag(Sigma) <- 1
  datasets <- lapply(seq_len(n_reps), function(i) {
    X <- MASS::mvrnorm(n, mu = rep(0, d), Sigma = Sigma)
    if (jitter_sd > 0) X <- X + matrix(rnorm(n * d, 0, jitter_sd), n, d)
    X
  })
  
  # True entropy
  H_true <- true_entropy_mvn(d, rho)
  
  # SE matrix: rows = reps, cols = ℓ
  pss_se_matrix <- sapply(l_range, function(l) {
    vapply(datasets, function(X) {
      (calculate_entropy_pss_d(X, n_partitions = l) - H_true)^2
    }, numeric(1L))
  })
  if (nrow(pss_se_matrix) == length(l_range)) pss_se_matrix <- t(pss_se_matrix)
  
  pss_mse_per_l <- colMeans(pss_se_matrix, na.rm = TRUE)
  opt_idx       <- which.min(pss_mse_per_l)
  opt_l         <- l_range[opt_idx]
  opt_rmse      <- sqrt(pss_mse_per_l[opt_idx])
  
  # Time exactly one evaluation at ℓ* (search excluded)
  opt_eval_time <- system.time({
    invisible(calculate_entropy_pss_d(datasets[[1]], n_partitions = opt_l))
  })[["elapsed"]]
  
  # Unified output columns (compatible with plotting pipeline)
  tibble::tibble(
    Distribution   = "Normal",
    Dimensions     = d,
    N_Samples      = n,
    Correlation    = rho,
    Method         = "PSS",
    Optimal_Param  = opt_l,                 # ℓ*
    Eval_Time_s    = as.numeric(opt_eval_time),
    RMSE           = as.numeric(opt_rmse)
  )
}

# --- 5) Global settings shared by both experiments ----------------------------
set.seed(42)
l_range   <- 1:8
n_reps    <- 10
jitter_sd <- 0

#==============================================================================
#  EXPERIMENT 1: VARYING SAMPLE SIZE (n), FIXED d = 10
#==============================================================================
cat("============================================================\n")
cat("Starting PSS Experiment 1: Varying Sample Size (n), fixed d = 10\n")
cat("============================================================\n")

sample_sizes_n <- c(1000, 2000, 4000, 6000)
correlations_n <- c(0)   # add more if needed, e.g., c(0, 0.5)

d_n <- 10  # <- FIXED dimension for Experiment 1

results_n <- bind_rows(lapply(correlations_n, function(rho) {
  bind_rows(lapply(sample_sizes_n, function(n) {
    cat(sprintf("Running PSS (n=%d, d=%d, rho=%.1f) ...\n", n, d_n, rho))
    run_setting_pss(
      n = n, d = d_n, rho = rho,
      l_range = l_range, n_reps = n_reps, jitter_sd = jitter_sd
    )
  }))
}))

cat("\nPSS simulation for varying n complete.\n")
cat("Saving results to pss_mvn_results_n.csv...\n\n")
write.csv(results_n, "pss_mvn_results_n.csv", row.names = FALSE)

print(knitr::kable(
  results_n,
  caption = "PSS results for varying sample sizes (n) with fixed d = 10",
  digits  = 5,
  align   = "c"
))

#==============================================================================
#  EXPERIMENT 2: VARYING DIMENSION (d), FIXED n = 3000
#==============================================================================
cat("\n\n============================================================\n")
cat("Starting PSS Experiment 2: Varying Dimension (d), fixed n = 3000\n")
cat("============================================================\n")

d_values_d    <- c(2, 5, 10, 20, 30, 40)
sample_sizes_d <- 3000      # <- FIXED sample size for Experiment 2
correlations_d <- c(0)

sim_grid_d <- expand.grid(
  d   = d_values_d,
  n   = sample_sizes_d,
  rho = correlations_d
)

cat("Simulation grid for varying dimensions (fixed n = 3000):\n")
print(sim_grid_d)

results_list_d <- lapply(seq_len(nrow(sim_grid_d)), function(i) {
  current_d   <- sim_grid_d$d[i]
  current_n   <- sim_grid_d$n[i]     # always 3000 here
  current_rho <- sim_grid_d$rho[i]
  
  cat(sprintf("\nRunning PSS (d=%d, n=%d, rho=%.1f) ...\n", current_d, current_n, current_rho))
  
  run_setting_pss(
    n = current_n,
    d = current_d,
    rho = current_rho,
    l_range = l_range,
    n_reps = n_reps,
    jitter_sd = jitter_sd
  )
})

results_d <- bind_rows(results_list_d)

cat("\nPSS simulation for varying d complete.\n")
cat("Saving results to pss_mvn_results_d.csv...\n\n")
write.csv(results_d, "pss_mvn_results_d.csv", row.names = FALSE)

print(knitr::kable(
  results_d,
  caption = "PSS results for varying dimensions (d) with fixed n = 3000",
  digits  = 5,
  align   = "c"
))


#==============================================================================
#  EXPERIMENT 3: VARYING CORRELATION (rho), FIXED d = 5, n = 20000
#  (Robustness Check for High Correlation in MVN)
#==============================================================================
cat("\n\n============================================================\n")
cat("Starting PSS Experiment 3: Varying rho (fixed d=5, n=20000)\n")
cat("============================================================\n")

# --- Experiment 3 configuration ----------------------------------------------
d_rho      <- 5
n_rho      <- 20000
rho_values <- c(0.0, 0.3, 0.8)

sim_grid_rho <- expand.grid(
  rho = rho_values,
  d   = d_rho,
  n   = n_rho
)

cat("Simulation grid for varying correlation (Robustness Check):\n")
print(sim_grid_rho)

# --- Run Experiment 3 --------------------------------------------------------
results_list_rho <- lapply(seq_len(nrow(sim_grid_rho)), function(i) {
  params <- sim_grid_rho[i, ]
  cat(sprintf("\nRunning PSS (rho=%.1f, d=%d, n=%d) ...\n", params$rho, params$d, params$n))
  
  run_setting_pss(
    n = params$n,
    d = params$d,
    rho = params$rho,
    l_range = l_range,
    n_reps = n_reps,
    jitter_sd = jitter_sd
  )
})

results_rho <- bind_rows(results_list_rho)

# --- Save and display results ------------------------------------------------
cat("\nPSS simulation for varying rho complete.\n")
cat("Saving results to pss_mvn_results_rho.csv...\n\n")
write.csv(results_rho, "pss_mvn_results_rho.csv", row.names = FALSE)

print(knitr::kable(
  results_rho,
  caption = "PSS results for varying correlation (rho) with fixed d=5, n=20000",
  digits  = 5,
  align   = "c"
))
