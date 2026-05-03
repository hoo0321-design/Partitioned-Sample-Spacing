#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
})

args <- commandArgs(trailingOnly = TRUE)
get_arg_value <- function(name, default = NA_character_) {
  prefix <- paste0("--", name, "=")
  hit <- args[startsWith(args, prefix)]
  if (!length(hit)) return(default)
  sub(prefix, "", hit[[1]], fixed = TRUE)
}

base_dir <- get_arg_value("base-dir")
if (is.na(base_dir) || !nzchar(base_dir)) {
  stop("--base-dir is required")
}

repo_root <- normalizePath(getwd(), mustWork = TRUE)
source(file.path(repo_root, "PSS", "PSS_entropy.R"))
source(file.path(repo_root, "CADEE", "CADEE.R"))

settings <- read.csv(file.path(base_dir, "settings.csv"), stringsAsFactors = FALSE)

ell_grid_for_d <- function(d) {
  if (d <= 5) return(1:20)
  if (d <= 10) return(1:10)
  if (d <= 20) return(1:5)
  1:3
}

summarize_estimates <- function(estimates, true_entropy, times, method, optimal_param,
                                tuning, diagnostics = NULL) {
  errors <- estimates - true_entropy
  squared_errors <- errors^2
  rmse <- sqrt(mean(squared_errors, na.rm = TRUE))
  mse_se <- stats::sd(squared_errors, na.rm = TRUE) / sqrt(length(squared_errors))
  rmse_se <- if (is.finite(rmse) && rmse > 0) mse_se / (2 * rmse) else NA_real_

  out <- data.frame(
    Method = method,
    Tuning = tuning,
    Optimal_Param = optimal_param,
    N_Reps = length(estimates),
    RMSE = rmse,
    RMSE_SE = rmse_se,
    Bias = mean(errors, na.rm = TRUE),
    Abs_Error = mean(abs(errors), na.rm = TRUE),
    Estimate_SD = stats::sd(estimates, na.rm = TRUE),
    Eval_Time_s = mean(times, na.rm = TRUE),
    Train_Time_s = 0,
    stringsAsFactors = FALSE
  )

  if (!is.null(diagnostics) && nrow(diagnostics)) {
    out$Coverage_Fraction <- mean(diagnostics$Coverage_Fraction, na.rm = TRUE)
    out$Skipped_Point_Fraction <- mean(diagnostics$Skipped_Point_Fraction, na.rm = TRUE)
    out$Mean_Occupied_Cell_Size <- mean(diagnostics$Mean_Occupied_Cell_Size, na.rm = TRUE)
    out$Usable_Cell_Fraction <- mean(diagnostics$Usable_Cell_Fraction, na.rm = TRUE)
  }
  out
}

summary_rows <- list()
estimate_rows <- list()
per_ell_rows <- list()
groups <- unique(settings[c("experiment", "distribution", "n", "d", "rho")])

for (group_id in seq_len(nrow(groups))) {
  group <- groups[group_id, ]
  rows <- settings[
    settings$experiment == group$experiment &
      settings$distribution == group$distribution &
      settings$n == group$n &
      settings$d == group$d &
      settings$rho == group$rho,
    ,
    drop = FALSE
  ]

  datasets <- lapply(
    rows$data_file,
    function(file_name) as.matrix(read.csv(file.path(base_dir, "datasets", file_name)))
  )
  true_entropy <- rows$true_entropy[[1]]
  cat(sprintf(
    "[%d/%d] R methods %s %s n=%d d=%d rho=%.1f reps=%d\n",
    group_id, nrow(groups), group$experiment, group$distribution,
    group$n, group$d, group$rho, nrow(rows)
  ))

  pss_by_ell <- lapply(ell_grid_for_d(group$d), function(ell) {
    estimates <- numeric(length(datasets))
    times <- numeric(length(datasets))
    diagnostics <- vector("list", length(datasets))

    for (rep_id in seq_along(datasets)) {
      timed <- system.time({
        out <- calculate_entropy_pss_d(datasets[[rep_id]], ell, return_diagnostics = TRUE)
      })
      estimates[[rep_id]] <- out$entropy
      times[[rep_id]] <- timed[["elapsed"]]
      diagnostics[[rep_id]] <- out$diagnostics
      estimate_rows[[length(estimate_rows) + 1L]] <<- data.frame(
        Experiment = group$experiment,
        Distribution = group$distribution,
        Dimensions = group$d,
        N_Samples = group$n,
        Correlation = group$rho,
        Replicate = rows$replicate[[rep_id]],
        Method = "PSS",
        Optimal_Param = ell,
        Estimate = estimates[[rep_id]],
        True_Entropy = true_entropy,
        Eval_Time_s = times[[rep_id]],
        Train_Time_s = 0,
        stringsAsFactors = FALSE
      )
    }

    diag_df <- bind_rows(diagnostics)
    errors <- estimates - true_entropy
    data.frame(
      ell = ell,
      RMSE = sqrt(mean(errors^2, na.rm = TRUE)),
      Bias = mean(errors, na.rm = TRUE),
      Eval_Time_s = mean(times, na.rm = TRUE),
      Coverage_Fraction = mean(diag_df$Coverage_Fraction, na.rm = TRUE),
      Skipped_Point_Fraction = mean(diag_df$Skipped_Point_Fraction, na.rm = TRUE),
      Mean_Occupied_Cell_Size = mean(diag_df$Mean_Occupied_Cell_Size, na.rm = TRUE),
      Usable_Cell_Fraction = mean(diag_df$Usable_Cell_Fraction, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  })

  pss_per_ell <- bind_rows(pss_by_ell)
  pss_per_ell <- transform(
    pss_per_ell,
    Experiment = group$experiment,
    Distribution = group$distribution,
    Dimensions = group$d,
    N_Samples = group$n,
    Correlation = group$rho
  )
  per_ell_rows[[length(per_ell_rows) + 1L]] <- pss_per_ell

  stable <- pss_per_ell[
    pss_per_ell$Coverage_Fraction >= 0.95 &
      pss_per_ell$Skipped_Point_Fraction <= 0.05 &
      pss_per_ell$Mean_Occupied_Cell_Size >= 2,
    ,
    drop = FALSE
  ]
  if (!nrow(stable)) {
    stable <- pss_per_ell[
      order(-pss_per_ell$Coverage_Fraction, pss_per_ell$RMSE),
      ,
      drop = FALSE
    ][1, ]
  }
  ell_star <- stable$ell[which.min(stable$RMSE)]

  best_estimates <- numeric(length(datasets))
  best_times <- numeric(length(datasets))
  best_diagnostics <- vector("list", length(datasets))
  for (rep_id in seq_along(datasets)) {
    timed <- system.time({
      out <- calculate_entropy_pss_d(datasets[[rep_id]], ell_star, return_diagnostics = TRUE)
    })
    best_estimates[[rep_id]] <- out$entropy
    best_times[[rep_id]] <- timed[["elapsed"]]
    best_diagnostics[[rep_id]] <- out$diagnostics
  }
  summary_rows[[length(summary_rows) + 1L]] <- cbind(
    data.frame(
      Experiment = group$experiment,
      Distribution = group$distribution,
      Dimensions = group$d,
      N_Samples = group$n,
      Correlation = group$rho,
      stringsAsFactors = FALSE
    ),
    summarize_estimates(
      best_estimates,
      true_entropy,
      best_times,
      "PSS",
      ell_star,
      "Oracle-ell-coverage",
      bind_rows(best_diagnostics)
    )
  )

  cadee_estimates <- numeric(length(datasets))
  cadee_times <- numeric(length(datasets))
  for (rep_id in seq_along(datasets)) {
    timed <- system.time({
      cadee_estimates[[rep_id]] <- copulasH_R(datasets[[rep_id]])
    })
    cadee_times[[rep_id]] <- timed[["elapsed"]]
    estimate_rows[[length(estimate_rows) + 1L]] <- data.frame(
      Experiment = group$experiment,
      Distribution = group$distribution,
      Dimensions = group$d,
      N_Samples = group$n,
      Correlation = group$rho,
      Replicate = rows$replicate[[rep_id]],
      Method = "CADEE",
      Optimal_Param = NA_real_,
      Estimate = cadee_estimates[[rep_id]],
      True_Entropy = true_entropy,
      Eval_Time_s = cadee_times[[rep_id]],
      Train_Time_s = 0,
      stringsAsFactors = FALSE
    )
  }
  summary_rows[[length(summary_rows) + 1L]] <- cbind(
    data.frame(
      Experiment = group$experiment,
      Distribution = group$distribution,
      Dimensions = group$d,
      N_Samples = group$n,
      Correlation = group$rho,
      stringsAsFactors = FALSE
    ),
    summarize_estimates(cadee_estimates, true_entropy, cadee_times, "CADEE", NA_real_, "None")
  )
}

summary <- bind_rows(summary_rows)
estimates <- bind_rows(estimate_rows)
per_ell <- bind_rows(per_ell_rows)

write.csv(summary, file.path(base_dir, "r_pss_cadee_summary.csv"), row.names = FALSE)
write.csv(estimates, file.path(base_dir, "r_pss_cadee_estimates.csv"), row.names = FALSE)
write.csv(per_ell, file.path(base_dir, "pss_per_ell.csv"), row.names = FALSE)

cat("\n=== R method summary ===\n")
print(summary)

