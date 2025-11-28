#==============================================================================
#  CONSOLIDATED CADEE SIMULATION SCRIPT (Multivariate Normal)
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
# You may paste the full function definition below, or load it from a separate file:
#   source("copulasH_R_function.R")
if (!exists("copulasH_R")) {
  stop("Missing function: `copulasH_R`. Please define it or load it via source().")
}

# --- 3) Shared simulation utilities ------------------------------------------

#' Compute the true differential entropy of a multivariate normal distribution.
true_entropy_mvn <- function(Sigma) {
  d <- ncol(Sigma)
  0.5 * log((2 * pi * exp(1))^d * det(Sigma))
}

#' Run CADEE entropy estimation for a single configuration (n, d, rho).
#' @param n Sample size (integer)
#' @param d Dimension (integer)
#' @param rho Equicorrelation coefficient (real number)
#' @param n_reps Number of Monte Carlo repetitions (integer)
#' @return A data frame containing mean evaluation time and RMSE.
run_setting_cadee_mvn <- function(n, d, rho, n_reps) {
  # Construct equicorrelated covariance matrix
  Sigma <- matrix(rho, d, d); diag(Sigma) <- 1
  
  # Analytical true entropy
  H_true <- true_entropy_mvn(Sigma)
  
  # Initialize storage
  estimates <- numeric(n_reps)
  times <- numeric(n_reps)
  
  # Monte Carlo repetitions
  for (i in seq_len(n_reps)) {
    X <- MASS::mvrnorm(n, mu = rep(0, d), Sigma = Sigma)
    t0 <- proc.time()
    estimates[i] <- copulasH_R(X)
    times[i] <- (proc.time() - t0)[["elapsed"]]
  }
  
  # Compute RMSE over repetitions
  rmse <- sqrt(mean((estimates - H_true)^2))
  
  # Return unified result format
  data.frame(
    Distribution     = "Normal",
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
#  Experiment 1: Varying Sample Size (n)
#==============================================================================
cat("============================================================\n")
cat("Starting CADEE Experiment 1: Varying Sample Size (n)\n")
cat("============================================================\n")

# --- Experiment 1 configuration ----------------------------------------------
set.seed(42)
sim_params_n <- list(
  n       = c(1000, 2000, 4000, 6000),
  d       = 10,
  rho     = 0,
  n_reps  = 100
)

# Generate full simulation grid
sim_grid_n <- expand.grid(
  n   = sim_params_n$n,
  d   = sim_params_n$d,
  rho = sim_params_n$rho
)

# --- Run Experiment 1 --------------------------------------------------------
results_list_n <- lapply(seq_len(nrow(sim_grid_n)), function(i) {
  params <- sim_grid_n[i, ]
  cat(sprintf("Running CADEE (n=%d, d=%d, rho=%.1f) ...\n", params$n, params$d, params$rho))
  run_setting_cadee_mvn(
    n = params$n, d = params$d, rho = params$rho, n_reps = sim_params_n$n_reps
  )
})

results_n <- bind_rows(results_list_n)

# --- Save and display results ------------------------------------------------
cat("\nCADEE simulation for varying n complete.\n")
cat("Saving results to cadee_mvn_results_n.csv...\n\n")
write.csv(results_n, "cadee_mvn_results_n.csv", row.names = FALSE)
print(knitr::kable(
  results_n,
  caption = "CADEE results for varying sample sizes (n)",
  digits  = 5,
  align   = "c"
))

#==============================================================================
#  Experiment 2: Varying Dimension (d)
#==============================================================================
cat("\n\n============================================================\n")
cat("Starting CADEE Experiment 2: Varying Dimension (d)\n")
cat("============================================================\n")

# --- Experiment 2 configuration ----------------------------------------------
set.seed(42)
sim_params_d <- list(
  n       = 3000,
  d       = c(2, 5, 10, 20, 30, 40),
  rho     = 0,
  n_reps  = 100
)

# Generate full simulation grid
sim_grid_d <- expand.grid(
  n   = sim_params_d$n,
  d   = sim_params_d$d,
  rho = sim_params_d$rho
)

# --- Run Experiment 2 --------------------------------------------------------
results_list_d <- lapply(seq_len(nrow(sim_grid_d)), function(i) {
  params <- sim_grid_d[i, ]
  cat(sprintf("\nRunning CADEE (d=%d, n=%d, rho=%.1f) ...\n", params$d, params$n, params$rho))
  run_setting_cadee_mvn(
    n = params$n, d = params$d, rho = params$rho, n_reps = sim_params_d$n_reps
  )
})

results_d <- bind_rows(results_list_d)

# --- Save and display results ------------------------------------------------
cat("\nCADEE simulation for varying d complete.\n")
cat("Saving results to cadee_mvn_results_d.csv...\n\n")
write.csv(results_d, "cadee_mvn_results_d.csv", row.names = FALSE)
print(knitr::kable(
  results_d,
  caption = "CADEE results for varying dimensions (d)",
  digits  = 5,
  align   = "c"
))


#==============================================================================
#  EXPERIMENT 3: VARYING CORRELATION (rho), FIXED d = 5, n = 20000
#  (Robustness Check for High Correlation in CADEE)
#==============================================================================
cat("\n\n============================================================\n")
cat("Starting CADEE Experiment 3: Varying rho (fixed d=5, n=20000)\n")
cat("============================================================\n")

# --- Configuration matching Python/PSS experiments ---
d_rho      <- 5
n_rho      <- 20000
rho_values <- c(0.0, 0.3, 0.8)
n_reps     <- 50  

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
  cat(sprintf("\nRunning CADEE (rho=%.1f, d=%d, n=%d) ...\n", params$rho, params$d, params$n))
  
  run_setting_cadee_mvn(
    n = params$n, 
    d = params$d, 
    rho = params$rho, 
    n_reps = n_reps
  )
})

results_rho <- bind_rows(results_list_rho)

# --- Save and display results ------------------------------------------------
cat("\nCADEE simulation for varying rho complete.\n")
cat("Saving results to cadee_mvn_results_rho.csv...\n\n")
write.csv(results_rho, "cadee_mvn_results_rho.csv", row.names = FALSE)

print(knitr::kable(
  results_rho,
  caption = "CADEE results for varying correlation (rho) with fixed d=5, n=20000",
  digits  = 5,
  align   = "c"
))
