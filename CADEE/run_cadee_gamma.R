#==============================================================================
#  CONSOLIDATED CADEE SIMULATION SCRIPT (Multivariate Gamma)
#------------------------------------------------------------------------------
#  • Experiment 1: Varying Sample Size (n)
#  • Experiment 2: Varying Dimension (d)
#  • Experiment 3: Varying correlation (rho)
#==============================================================================

# --- 1) Load required libraries ----------------------------------------------
pkgs <- c("MASS", "dplyr", "knitr")
to_install <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]
if (length(to_install)) install.packages(to_install, repos = "https://cloud.r-project.org")
library(MASS)
library(dplyr)
library(knitr)

# --- 2) Load the CADEE function ----------------------------------------------
# This script requires the `copulasH_R` function to be defined beforehand.
# You may paste the full function code below or load it from an external file:
#   source("copulasH_R_function.R")
if (!exists("copulasH_R")) {
  stop("Missing function: `copulasH_R`. Please define it or load it via source().")
}

# --- 3) Shared simulation utilities ------------------------------------------

#' Compute the true entropy of a multivariate Gamma distribution
#' under a Gaussian copula with equicorrelation ρ.
true_entropy_gamma_gausscop <- function(d, rho, shape, scale) {
  H_marginal_gamma <- shape + log(scale) + lgamma(shape) + (1 - shape) * digamma(shape)
  R <- matrix(rho, d, d); diag(R) <- 1
  H_copula_gaussian <- 0.5 * as.numeric(determinant(R, logarithm = TRUE)$modulus)
  d * H_marginal_gamma + H_copula_gaussian
}

#' Generate multivariate Gamma samples with dependence structure
#' induced by a Gaussian copula.
rmgamma_gausscop <- function(n, d, rho, shape, scale) {
  R <- matrix(rho, d, d); diag(R) <- 1
  Z <- MASS::mvrnorm(n, mu = rep(0, d), Sigma = R)
  U <- pnorm(Z)
  sapply(seq_len(d), \(j) qgamma(U[, j], shape = shape, scale = scale))
}

#' Run CADEE entropy estimation for a single (n, d, ρ) configuration (Gamma case).
#' @param n      Sample size (integer)
#' @param d      Dimension (integer)
#' @param rho    Equicorrelation coefficient
#' @param n_reps Number of Monte Carlo repetitions
#' @param shape  Shape parameter of the Gamma distribution
#' @param scale  Scale parameter of the Gamma distribution
#' @return       Data frame with averaged evaluation time and RMSE
run_setting_cadee_gamma <- function(n, d, rho, n_reps, shape, scale) {
  H_true <- true_entropy_gamma_gausscop(d, rho, shape, scale)
  
  estimates <- numeric(n_reps)
  times <- numeric(n_reps)
  
  for (i in seq_len(n_reps)) {
    X <- rmgamma_gausscop(n, d, rho, shape, scale)
    t0 <- proc.time()
    estimates[i] <- copulasH_R(X)
    times[i] <- (proc.time() - t0)[["elapsed"]]
  }
  
  rmse <- sqrt(mean((estimates - H_true)^2))
  
  data.frame(
    Distribution     = "Gamma",
    Dimensions       = d,
    N_Samples        = n,
    Correlation      = rho,
    Method           = "CADEE",
    Optimal_Param    = NA,  # CADEE has no tuning parameter
    Eval_Time_s      = mean(times),
    RMSE             = rmse
  )
}

#==============================================================================
#  Experiment 1: Varying Sample Size (n) — Fixed Dimension (d = 5)
#==============================================================================
cat("============================================================\n")
cat("Starting CADEE Gamma Experiment 1: Varying Sample Size (n)\n")
cat("============================================================\n")

# --- Experiment 1 configuration ----------------------------------------------
set.seed(42)
sim_params_n <- list(
  n       = c(1000, 2000, 4000, 8000),
  d       = 5,
  rho     = 0,
  n_reps  = 10,
  shape   = 0.4,
  scale   = 0.3
)

# Generate simulation grid
sim_grid_n <- expand.grid(
  n   = sim_params_n$n,
  d   = sim_params_n$d,
  rho = sim_params_n$rho
)

# --- Run Experiment 1 --------------------------------------------------------
results_list_n <- lapply(seq_len(nrow(sim_grid_n)), function(i) {
  params <- sim_grid_n[i, ]
  cat(sprintf("Running CADEE Gamma (n=%d, d=%d, rho=%.1f) ...\n", params$n, params$d, params$rho))
  run_setting_cadee_gamma(
    n = params$n, d = params$d, rho = params$rho, 
    n_reps = sim_params_n$n_reps, 
    shape  = sim_params_n$shape, 
    scale  = sim_params_n$scale
  )
})

results_n <- bind_rows(results_list_n)

# --- Save and display results ------------------------------------------------
cat("\nCADEE Gamma simulation for varying n complete.\n")
cat("Saving results to cadee_gamma_results_n.csv...\n\n")
write.csv(results_n, "cadee_gamma_results_n.csv", row.names = FALSE)
print(knitr::kable(
  results_n,
  caption = "CADEE Gamma results for varying sample sizes (n)",
  digits  = 5,
  align   = "c"
))

#==============================================================================
#  Experiment 2: Varying Dimension (d) — Fixed Sample Size (n = 30000)
#==============================================================================
cat("\n\n============================================================\n")
cat("Starting CADEE Gamma Experiment 2: Varying Dimension (d)\n")
cat("============================================================\n")

# --- Experiment 2 configuration ----------------------------------------------
set.seed(42)
sim_params_d <- list(
  n       = 30000,
  d       = c(2, 5, 10, 15, 20),
  rho     = 0,
  n_reps  = 10,
  shape   = 0.4,
  scale   = 0.3
)

# Generate simulation grid
sim_grid_d <- expand.grid(
  n   = sim_params_d$n,
  d   = sim_params_d$d,
  rho = sim_params_d$rho
)

# --- Run Experiment 2 --------------------------------------------------------
results_list_d <- lapply(seq_len(nrow(sim_grid_d)), function(i) {
  params <- sim_grid_d[i, ]
  cat(sprintf("\nRunning CADEE Gamma (d=%d, n=%d, rho=%.1f) ...\n", params$d, params$n, params$rho))
  run_setting_cadee_gamma(
    n = params$n, d = params$d, rho = params$rho, 
    n_reps = sim_params_d$n_reps,
    shape  = sim_params_d$shape,
    scale  = sim_params_d$scale
  )
})

results_d <- bind_rows(results_list_d)

# --- Save and display results ------------------------------------------------
cat("\nCADEE Gamma simulation for varying d complete.\n")
cat("Saving results to cadee_gamma_results_d.csv...\n\n")
write.csv(results_d, "cadee_gamma_results_d.csv", row.names = FALSE)
print(knitr::kable(
  results_d,
  caption = "CADEE Gamma results for varying dimensions (d)",
  digits  = 5,
  align   = "c"
))

#==============================================================================
#  EXPERIMENT 3: VARYING CORRELATION (rho), FIXED d = 7, n = 50000
#  (Robustness Check for High Correlation in CADEE Gamma)
#==============================================================================
cat("\n\n============================================================\n")
cat("Starting CADEE Gamma Experiment 3: Varying rho (fixed d=7, n=50000)\n")
cat("============================================================\n")

# --- Experiment 3 configuration ----------------------------------------------
set.seed(42)
sim_params_rho <- list(
  n       = 50000,
  d       = 7,
  rho     = c(0.0, 0.1, 0.3,0.8),
  n_reps  = 100,  
  shape   = 0.4,
  scale   = 0.3
)

sim_grid_rho <- expand.grid(
  n   = sim_params_rho$n,
  d   = sim_params_rho$d,
  rho = sim_params_rho$rho
)

cat("Simulation grid for varying correlation (Robustness Check):\n")
print(sim_grid_rho)

# --- Run Experiment 3 --------------------------------------------------------
results_list_rho <- lapply(seq_len(nrow(sim_grid_rho)), function(i) {
  params <- sim_grid_rho[i, ]
  cat(sprintf("\nRunning CADEE Gamma (rho=%.1f, d=%d, n=%d) ...\n", params$rho, params$d, params$n))
  
  run_setting_cadee_gamma(
    n = params$n, 
    d = params$d, 
    rho = params$rho, 
    n_reps = sim_params_rho$n_reps,
    shape = sim_params_rho$shape,
    scale = sim_params_rho$scale
  )
})

results_rho <- bind_rows(results_list_rho)

# --- Save and display results ------------------------------------------------
cat("\nCADEE Gamma simulation for varying rho complete.\n")
cat("Saving results to cadee_gamma_results_rho.csv...\n\n")
write.csv(results_rho, "cadee_gamma_results_rho.csv", row.names = FALSE)

print(knitr::kable(
  results_rho,
  caption = "CADEE Gamma results for varying correlation (rho) with fixed d=7, n=50000",
  digits  = 5,
  align   = "c"
))
