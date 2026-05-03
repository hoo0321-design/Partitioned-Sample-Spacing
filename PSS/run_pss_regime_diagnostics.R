#==============================================================================
# PSS regime diagnostics: oracle vs CV, distribution expansion, occupancy stats
#
# Usage from repository root:
#   Rscript PSS/run_pss_regime_diagnostics.R --quick
#   Rscript PSS/run_pss_regime_diagnostics.R --full
#
# Outputs:
#   results/pss_diagnostics/pss_regime_summary_*.csv
#   results/pss_diagnostics/pss_oracle_grid_*.csv
#   results/pss_diagnostics/pss_cv_selected_ell_*.csv
#   results/pss_diagnostics/pss_cv_tables_*.csv
#==============================================================================

pkgs <- c("MASS", "dplyr", "knitr")
to_install <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]
if (length(to_install)) install.packages(to_install, repos = "https://cloud.r-project.org")
invisible(lapply(pkgs, require, character.only = TRUE))

source_file <- if (file.exists("PSS/PSS_entropy.R")) {
  "PSS/PSS_entropy.R"
} else {
  "PSS_entropy.R"
}
source(source_file)

args <- commandArgs(trailingOnly = TRUE)
mode <- if ("--full" %in% args) "full" else "quick"

get_arg_value <- function(name, default) {
  prefix <- paste0("--", name, "=")
  hit <- args[startsWith(args, prefix)]
  if (!length(hit)) return(default)
  sub(prefix, "", hit[[1]], fixed = TRUE)
}

timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
out_dir <- file.path("results", "pss_diagnostics")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

family_label <- function(family) {
  switch(
    tolower(family),
    normal = "Normal",
    gamma = "Gamma",
    beta = "Beta",
    lognormal = "Lognormal",
    laplace = "Laplace",
    family
  )
}

l_grid_for_d <- function(d, mode) {
  if (mode == "quick") return(if (d <= 5) 1:4 else 1:3)
  if (d <= 5) return(1:12)
  if (d <= 10) return(1:8)
  1:4
}

make_config_grid <- function(mode) {
  families <- c("normal", "gamma", "beta", "lognormal", "laplace")
  if (mode == "quick") {
    return(expand.grid(
      family = families,
      n = c(300, 600),
      d = c(2, 5),
      rho = c(0.3),
      stringsAsFactors = FALSE
    ))
  }

  expand.grid(
    family = families,
    n = c(1000, 3000, 6000),
    d = c(2, 5, 10, 20),
    rho = c(0.0, 0.5),
    stringsAsFactors = FALSE
  )
}

run_single_setting <- function(family, n, d, rho, n_reps, n_folds, seed, mode,
                               coverage_min = 0.95,
                               occupancy_min = 10,
                               occupancy_fraction_min = 0.95) {
  l_range <- l_grid_for_d(d, mode)
  H_true <- true_entropy_gaussian_copula(d, rho, family)

  datasets <- lapply(seq_len(n_reps), function(rep_id) {
    set.seed(seed + 1000L * rep_id)
    simulate_gaussian_copula(n, d, rho, family)
  })

  oracle_grid <- do.call(rbind, lapply(l_range, function(ell) {
    summarize_pss_for_l(datasets, H_true, ell)
  }))
  oracle_l <- oracle_grid$ell[which.min(oracle_grid$RMSE)]
  oracle_row <- oracle_grid[oracle_grid$ell == oracle_l, , drop = FALSE][1, ]

  cv_rows <- vector("list", n_reps)
  cv_tables <- vector("list", n_reps)
  estimates <- numeric(n_reps)
  biases <- numeric(n_reps)
  eval_times <- numeric(n_reps)
  tuning_times <- numeric(n_reps)
  diag_rows <- vector("list", n_reps)

  for (rep_id in seq_len(n_reps)) {
    X <- datasets[[rep_id]]
    tune_time <- system.time({
      cv <- select_l_via_cv_pss(
        X,
        l_range = l_range,
        n_folds = n_folds,
        coverage_penalty = 3,
        coverage_min = coverage_min,
        one_se_rule = TRUE,
        occupancy_min = occupancy_min,
        occupancy_fraction_min = occupancy_fraction_min,
        seed = seed + rep_id
      )
    })[["elapsed"]]

    eval_time <- system.time({
      out <- calculate_entropy_pss_d(
        X,
        n_partitions = cv$ell_star,
        return_diagnostics = TRUE
      )
    })[["elapsed"]]

    estimates[rep_id] <- out$entropy
    biases[rep_id] <- out$entropy - H_true
    eval_times[rep_id] <- eval_time
    tuning_times[rep_id] <- tune_time
    diag_rows[[rep_id]] <- out$diagnostics

    cv_rows[[rep_id]] <- data.frame(
      Replicate = rep_id,
      Selected_Ell = cv$ell_star,
      Estimate = out$entropy,
      Error = out$entropy - H_true,
      Eval_Time_s = eval_time,
      Tuning_Time_s = tune_time,
      Selection_Rule = cv$selection_rule,
      Coverage_Min = cv$coverage_min,
      Occupancy_Min = cv$occupancy_min,
      Occupancy_Fraction_Min = cv$occupancy_fraction_min,
      stringsAsFactors = FALSE
    )

    cv_tables[[rep_id]] <- transform(
      cv$cv_table,
      Replicate = rep_id
    )
  }

  cv_diag <- do.call(rbind, diag_rows)
  cv_selected <- do.call(rbind, cv_rows)
  cv_table <- do.call(rbind, cv_tables)
  cv_squared_errors <- biases^2
  cv_rmse <- sqrt(mean(cv_squared_errors, na.rm = TRUE))
  cv_mse_se <- stats::sd(cv_squared_errors, na.rm = TRUE) / sqrt(length(cv_squared_errors))
  cv_rmse_se <- if (is.finite(cv_rmse) && cv_rmse > 0) {
    cv_mse_se / (2 * cv_rmse)
  } else {
    NA_real_
  }

  add_metadata <- function(df, tuning) {
    transform(
      df,
      Distribution = family_label(family),
      Dimensions = d,
      N_Samples = n,
      Correlation = rho,
      Tuning = tuning
    )
  }

  summary <- rbind(
    data.frame(
      Distribution = family_label(family),
      Dimensions = d,
      N_Samples = n,
      Correlation = rho,
      Method = "PSS",
      Tuning = "Oracle",
      Selection_Rule = "oracle_rmse",
      Coverage_Min = NA_real_,
      Occupancy_Min = NA_real_,
      Occupancy_Fraction_Min = NA_real_,
      Selected_Ell_Mean = oracle_l,
      Selected_Ell_Median = oracle_l,
      Selected_Ell_SD = 0,
      N_Reps = n_reps,
      RMSE = oracle_row$RMSE,
      RMSE_SE = oracle_row$RMSE_SE,
      RMSE_Lower = oracle_row$RMSE_Lower,
      RMSE_Upper = oracle_row$RMSE_Upper,
      Bias = oracle_row$Bias,
      Estimate_SD = oracle_row$Estimate_SD,
      Eval_Time_s = oracle_row$Eval_Time_s,
      Tuning_Time_s = NA_real_,
      Occupied_Cell_Fraction = oracle_row$Occupied_Cell_Fraction,
      Usable_Cell_Fraction = oracle_row$Usable_Cell_Fraction,
      Skipped_Occupied_Cell_Fraction = oracle_row$Skipped_Occupied_Cell_Fraction,
      Skipped_Point_Fraction = oracle_row$Skipped_Point_Fraction,
      Coverage_Fraction = oracle_row$Coverage_Fraction,
      Mean_Occupied_Cell_Size = oracle_row$Mean_Occupied_Cell_Size,
      Median_Occupied_Cell_Size = oracle_row$Median_Occupied_Cell_Size,
      Max_Occupied_Cell_Size = oracle_row$Max_Occupied_Cell_Size,
      Sorting_Cost_Proxy = oracle_row$Sorting_Cost_Proxy,
      stringsAsFactors = FALSE
    ),
    data.frame(
      Distribution = family_label(family),
      Dimensions = d,
      N_Samples = n,
      Correlation = rho,
      Method = "PSS",
      Tuning = "CV",
      Selection_Rule = paste(unique(cv_selected$Selection_Rule), collapse = ";"),
      Coverage_Min = coverage_min,
      Occupancy_Min = occupancy_min,
      Occupancy_Fraction_Min = occupancy_fraction_min,
      Selected_Ell_Mean = mean(cv_selected$Selected_Ell),
      Selected_Ell_Median = stats::median(cv_selected$Selected_Ell),
      Selected_Ell_SD = stats::sd(cv_selected$Selected_Ell),
      N_Reps = n_reps,
      RMSE = cv_rmse,
      RMSE_SE = cv_rmse_se,
      RMSE_Lower = pmax(0, cv_rmse - 1.96 * cv_rmse_se),
      RMSE_Upper = cv_rmse + 1.96 * cv_rmse_se,
      Bias = mean(biases, na.rm = TRUE),
      Estimate_SD = stats::sd(estimates, na.rm = TRUE),
      Eval_Time_s = mean(eval_times, na.rm = TRUE),
      Tuning_Time_s = mean(tuning_times, na.rm = TRUE),
      Occupied_Cell_Fraction = mean(cv_diag$Occupied_Cell_Fraction, na.rm = TRUE),
      Usable_Cell_Fraction = mean(cv_diag$Usable_Cell_Fraction, na.rm = TRUE),
      Skipped_Occupied_Cell_Fraction = mean(cv_diag$Skipped_Occupied_Cell_Fraction, na.rm = TRUE),
      Skipped_Point_Fraction = mean(cv_diag$Skipped_Point_Fraction, na.rm = TRUE),
      Coverage_Fraction = mean(cv_diag$Coverage_Fraction, na.rm = TRUE),
      Mean_Occupied_Cell_Size = mean(cv_diag$Mean_Occupied_Cell_Size, na.rm = TRUE),
      Median_Occupied_Cell_Size = mean(cv_diag$Median_Occupied_Cell_Size, na.rm = TRUE),
      Max_Occupied_Cell_Size = mean(cv_diag$Max_Occupied_Cell_Size, na.rm = TRUE),
      Sorting_Cost_Proxy = mean(cv_diag$Sorting_Cost_Proxy, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  )

  list(
    summary = summary,
    oracle_grid = add_metadata(oracle_grid, "OracleGrid"),
    cv_selected = add_metadata(cv_selected, "CV"),
    cv_table = add_metadata(cv_table, "CV")
  )
}

if (mode == "quick") {
  n_reps <- 2
  n_folds <- 2
} else {
  n_reps <- 10
  n_folds <- 3
}
coverage_min <- 0.95
coverage_min <- as.numeric(get_arg_value("coverage-min", coverage_min))
occupancy_min <- as.integer(get_arg_value("occupancy-min", 10))
occupancy_fraction_min <- as.numeric(get_arg_value("occupancy-fraction-min", 0.95))

set.seed(42)
config_grid <- make_config_grid(mode)

cat("============================================================\n")
cat("PSS regime diagnostics mode:", mode, "\n")
cat("Settings:", nrow(config_grid), " | reps:", n_reps, " | folds:", n_folds, "\n")
cat("CV coverage constraint:", coverage_min, "\n")
cat("CV occupancy constraint: fraction(n_k >=", occupancy_min, ") >=", occupancy_fraction_min, "\n")
cat("============================================================\n")

all_results <- vector("list", nrow(config_grid))

for (i in seq_len(nrow(config_grid))) {
  cfg <- config_grid[i, ]
  cat(sprintf(
    "[%d/%d] family=%s, n=%d, d=%d, rho=%.2f\n",
    i, nrow(config_grid), cfg$family, cfg$n, cfg$d, cfg$rho
  ))
  all_results[[i]] <- run_single_setting(
    family = cfg$family,
    n = cfg$n,
    d = cfg$d,
    rho = cfg$rho,
    n_reps = n_reps,
    n_folds = n_folds,
    seed = 42 + 10000L * i,
    mode = mode,
    coverage_min = coverage_min,
    occupancy_min = occupancy_min,
    occupancy_fraction_min = occupancy_fraction_min
  )
}

summary_df <- dplyr::bind_rows(lapply(all_results, `[[`, "summary"))
oracle_grid_df <- dplyr::bind_rows(lapply(all_results, `[[`, "oracle_grid"))
cv_selected_df <- dplyr::bind_rows(lapply(all_results, `[[`, "cv_selected"))
cv_table_df <- dplyr::bind_rows(lapply(all_results, `[[`, "cv_table"))

summary_path <- file.path(out_dir, paste0("pss_regime_summary_", mode, "_", timestamp, ".csv"))
oracle_path <- file.path(out_dir, paste0("pss_oracle_grid_", mode, "_", timestamp, ".csv"))
selected_path <- file.path(out_dir, paste0("pss_cv_selected_ell_", mode, "_", timestamp, ".csv"))
cv_table_path <- file.path(out_dir, paste0("pss_cv_tables_", mode, "_", timestamp, ".csv"))

write.csv(summary_df, summary_path, row.names = FALSE)
write.csv(oracle_grid_df, oracle_path, row.names = FALSE)
write.csv(cv_selected_df, selected_path, row.names = FALSE)
write.csv(cv_table_df, cv_table_path, row.names = FALSE)

cat("\nSaved:\n")
cat("  ", summary_path, "\n")
cat("  ", oracle_path, "\n")
cat("  ", selected_path, "\n")
cat("  ", cv_table_path, "\n\n")

print(knitr::kable(summary_df, digits = 5))
