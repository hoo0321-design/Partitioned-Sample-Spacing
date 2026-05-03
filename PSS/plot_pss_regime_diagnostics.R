#==============================================================================
# Paper plots for PSS regime diagnostics
#
# Usage from repository root:
#   Rscript PSS/plot_pss_regime_diagnostics.R --mode=full --rho=0.5
#   Rscript PSS/plot_pss_regime_diagnostics.R --mode=quick
#   Rscript PSS/plot_pss_regime_diagnostics.R --summary=path/to/summary.csv
#
# Inputs are produced by:
#   Rscript PSS/run_pss_regime_diagnostics.R --full
#
# Outputs:
#   results/pss_diagnostics/figures/<run_id>/*.pdf
#   results/pss_diagnostics/figures/<run_id>/*.png
#==============================================================================

pkgs <- c("ggplot2", "dplyr", "tidyr", "readr", "scales", "stringr")
to_install <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]
if (length(to_install)) install.packages(to_install, repos = "https://cloud.r-project.org")
invisible(lapply(pkgs, require, character.only = TRUE))

args <- commandArgs(trailingOnly = TRUE)

get_arg_value <- function(name, default = NA_character_) {
  prefix <- paste0("--", name, "=")
  hit <- args[startsWith(args, prefix)]
  if (!length(hit)) return(default)
  sub(prefix, "", hit[[1]], fixed = TRUE)
}

input_dir <- get_arg_value("input-dir", file.path("results", "pss_diagnostics"))
mode <- get_arg_value("mode", "full")
rho_arg <- get_arg_value("rho", NA_character_)
summary_arg <- get_arg_value("summary", NA_character_)
oracle_arg <- get_arg_value("oracle", NA_character_)
cv_table_arg <- get_arg_value("cv-table", NA_character_)
output_arg <- get_arg_value("output-dir", NA_character_)
format_arg <- get_arg_value("format", "both")
include_titles <- "--with-titles" %in% args

latest_file <- function(pattern) {
  files <- list.files(input_dir, pattern = pattern, full.names = TRUE)
  if (!length(files)) {
    stop("No matching files for pattern: ", pattern, " in ", input_dir, call. = FALSE)
  }
  files[which.max(file.info(files)$mtime)]
}

extract_run_id <- function(path) {
  stem <- tools::file_path_sans_ext(basename(path))
  stringr::str_match(stem, paste0("_(quick|full)_([0-9]{8}_[0-9]{6})$"))[, 3]
}

summary_path <- if (is.na(summary_arg)) {
  latest_file(paste0("^pss_regime_summary_", mode, "_.*\\.csv$"))
} else {
  summary_arg
}

run_id <- extract_run_id(summary_path)
if (is.na(run_id)) {
  run_id <- format(Sys.time(), "%Y%m%d_%H%M%S")
}

oracle_path <- if (is.na(oracle_arg)) {
  candidate <- file.path(input_dir, paste0("pss_oracle_grid_", mode, "_", run_id, ".csv"))
  if (file.exists(candidate)) candidate else latest_file(paste0("^pss_oracle_grid_", mode, "_.*\\.csv$"))
} else {
  oracle_arg
}

cv_table_path <- if (is.na(cv_table_arg)) {
  candidate <- file.path(input_dir, paste0("pss_cv_tables_", mode, "_", run_id, ".csv"))
  if (file.exists(candidate)) candidate else latest_file(paste0("^pss_cv_tables_", mode, "_.*\\.csv$"))
} else {
  cv_table_arg
}

figure_dir <- if (is.na(output_arg)) {
  file.path(input_dir, "figures", paste0(mode, "_", run_id))
} else {
  output_arg
}
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

summary_df <- readr::read_csv(summary_path, show_col_types = FALSE)
oracle_df <- readr::read_csv(oracle_path, show_col_types = FALSE)
cv_table_df <- readr::read_csv(cv_table_path, show_col_types = FALSE)

if (is.na(rho_arg)) {
  rho_focus <- max(summary_df$Correlation, na.rm = TRUE)
} else {
  rho_focus <- as.numeric(rho_arg)
}

nearest_value <- function(values, target) {
  values <- sort(unique(values))
  values[which.min(abs(values - target))]
}

max_n <- max(summary_df$N_Samples, na.rm = TRUE)
conv_d <- nearest_value(summary_df$Dimensions, 5)

summary_focus <- summary_df %>%
  dplyr::filter(abs(Correlation - rho_focus) < 1e-12) %>%
  dplyr::mutate(
    Distribution = factor(Distribution, levels = c("Normal", "Gamma", "Beta", "Lognormal")),
    Tuning = factor(Tuning, levels = c("Oracle", "CV")),
    RMSE_Lower = pmax(RMSE_Lower, 1e-12),
    RMSE_Upper = pmax(RMSE_Upper, 1e-12),
    RMSE = pmax(RMSE, 1e-12)
  )

if (!nrow(summary_focus)) {
  stop("No rows found for rho=", rho_focus, ". Available rho values: ",
       paste(sort(unique(summary_df$Correlation)), collapse = ", "), call. = FALSE)
}

method_cols <- c(Oracle = "#0072B2", CV = "#D55E00")
dist_cols <- c(Normal = "#0072B2", Gamma = "#009E73", Beta = "#D55E00", Lognormal = "#CC79A7")

paper_theme <- ggplot2::theme_bw(base_size = 11) +
  ggplot2::theme(
    panel.grid.minor = ggplot2::element_blank(),
    panel.grid.major = ggplot2::element_line(linewidth = 0.25, color = "grey86"),
    strip.background = ggplot2::element_rect(fill = "grey94", color = "grey55"),
    strip.text = ggplot2::element_text(face = "bold"),
    legend.position = "top",
    legend.title = ggplot2::element_blank(),
    plot.title.position = "plot",
    plot.title = ggplot2::element_text(face = "bold", size = 12),
    plot.subtitle = ggplot2::element_text(size = 10),
    axis.title = ggplot2::element_text(face = "bold")
  )

save_plot <- function(plot, name, width = 7.2, height = 4.6) {
  if (!include_titles) {
    plot <- plot +
      ggplot2::labs(title = NULL, subtitle = NULL) +
      ggplot2::theme(
        plot.title = ggplot2::element_blank(),
        plot.subtitle = ggplot2::element_blank()
      )
  }
  if (format_arg %in% c("pdf", "both")) {
    ggplot2::ggsave(file.path(figure_dir, paste0(name, ".pdf")), plot,
                    width = width, height = height)
  }
  if (format_arg %in% c("png", "both")) {
    ggplot2::ggsave(file.path(figure_dir, paste0(name, ".png")), plot,
                    width = width, height = height, dpi = 450)
  }
}

# Figure 1: CV-to-oracle gap. This is the cleanest answer to the tuning question.
gap_df <- summary_focus %>%
  dplyr::select(Distribution, Dimensions, N_Samples, Tuning, RMSE) %>%
  tidyr::pivot_wider(names_from = Tuning, values_from = RMSE) %>%
  dplyr::filter(!is.na(Oracle), !is.na(CV)) %>%
  dplyr::mutate(
    CV_Oracle_Ratio = CV / Oracle,
    Ratio_Label = sprintf("%.2fx", CV_Oracle_Ratio),
    N_Samples = factor(N_Samples),
    Dimensions = factor(Dimensions)
  )

readr::write_csv(gap_df, file.path(figure_dir, "pss_cv_oracle_gap_table.csv"))

p_gap <- ggplot2::ggplot(gap_df, ggplot2::aes(x = Dimensions, y = N_Samples, fill = CV_Oracle_Ratio)) +
  ggplot2::geom_tile(color = "white", linewidth = 0.5) +
  ggplot2::geom_text(ggplot2::aes(label = Ratio_Label), size = 3.2) +
  ggplot2::facet_wrap(~ Distribution, nrow = 1) +
  ggplot2::scale_fill_gradient2(
    low = "#2166AC", mid = "white", high = "#B2182B",
    midpoint = 1.25, trans = "log10",
    labels = function(x) paste0(scales::number(x, accuracy = 0.01), "x")
  ) +
  ggplot2::labs(
    title = "CV tracks the oracle partition except in sparse regimes",
    subtitle = paste0("Cell values are RMSE(CV) / RMSE(oracle), rho = ", rho_focus),
    x = "Dimension d",
    y = "Sample size n",
    fill = "RMSE ratio"
  ) +
  ggplot2::guides(fill = ggplot2::guide_colorbar(
    title.position = "top", barwidth = 0.35, barheight = 3.2
  )) +
  paper_theme +
  ggplot2::theme(legend.position = "right")
save_plot(p_gap, "figure_pss_cv_oracle_gap_heatmap", width = 9.2, height = 3.8)

# Figure 2: Empirical convergence with error bars.
conv_df <- summary_focus %>%
  dplyr::filter(Dimensions == conv_d) %>%
  dplyr::mutate(N_Samples = as.numeric(N_Samples))

p_conv <- ggplot2::ggplot(
  conv_df,
  ggplot2::aes(x = N_Samples, y = RMSE, color = Tuning, shape = Tuning, group = Tuning)
) +
  ggplot2::geom_ribbon(
    ggplot2::aes(ymin = RMSE_Lower, ymax = RMSE_Upper, fill = Tuning),
    alpha = 0.12, color = NA, show.legend = FALSE
  ) +
  ggplot2::geom_line(linewidth = 0.75) +
  ggplot2::geom_point(size = 2.2) +
  ggplot2::facet_wrap(~ Distribution, nrow = 1) +
  ggplot2::scale_x_log10(breaks = sort(unique(conv_df$N_Samples)), labels = scales::comma) +
  ggplot2::scale_y_log10(labels = scales::label_number(accuracy = 0.01)) +
  ggplot2::scale_color_manual(values = method_cols) +
  ggplot2::scale_fill_manual(values = method_cols) +
  ggplot2::labs(
    title = "Empirical convergence across expanded distributions",
    subtitle = paste0("Dimension d = ", conv_d, ", rho = ", rho_focus, "; bands are +/- 1.96 SE"),
    x = "Sample size n",
    y = "RMSE"
  ) +
  paper_theme
save_plot(p_conv, "figure_pss_empirical_convergence", width = 9.2, height = 3.8)

# Figure 3: Occupancy diagnostics under the CV-selected partition.
diag_df <- summary_focus %>%
  dplyr::filter(Tuning == "CV", N_Samples == max_n) %>%
  dplyr::select(
    Distribution, Dimensions, Coverage_Fraction,
    Skipped_Point_Fraction, Skipped_Occupied_Cell_Fraction
  ) %>%
  tidyr::pivot_longer(
    cols = c(Coverage_Fraction, Skipped_Point_Fraction, Skipped_Occupied_Cell_Fraction),
    names_to = "Metric", values_to = "Value"
  ) %>%
  dplyr::mutate(
    Metric = dplyr::recode(
      Metric,
      Coverage_Fraction = "Covered validation mass",
      Skipped_Point_Fraction = "Skipped point fraction",
      Skipped_Occupied_Cell_Fraction = "Skipped occupied-cell fraction"
    )
  )

p_diag <- ggplot2::ggplot(
  diag_df,
  ggplot2::aes(x = Dimensions, y = Value, color = Distribution, group = Distribution)
) +
  ggplot2::geom_hline(
    data = data.frame(Metric = "Covered validation mass"),
    ggplot2::aes(yintercept = 0.95),
    inherit.aes = FALSE, linetype = "dashed", color = "grey35", linewidth = 0.45
  ) +
  ggplot2::geom_line(linewidth = 0.75) +
  ggplot2::geom_point(size = 2.1) +
  ggplot2::facet_wrap(~ Metric, ncol = 1, scales = "free_y") +
  ggplot2::scale_x_continuous(breaks = sort(unique(diag_df$Dimensions))) +
  ggplot2::scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  ggplot2::scale_color_manual(values = dist_cols) +
  ggplot2::labs(
    title = "Coverage and occupancy constraints expose sparse-cell failure modes",
    subtitle = paste0("CV-selected partitions at n = ", scales::comma(max_n), ", rho = ", rho_focus),
    x = "Dimension d",
    y = "Fraction"
  ) +
  paper_theme
save_plot(p_diag, "figure_pss_occupancy_diagnostics", width = 7.2, height = 6.2)

# Appendix Figure A: RMSE by dimension at the largest n.
rmse_d_df <- summary_focus %>%
  dplyr::filter(N_Samples == max_n)

p_rmse_d <- ggplot2::ggplot(
  rmse_d_df,
  ggplot2::aes(x = Dimensions, y = RMSE, color = Tuning, shape = Tuning, group = Tuning)
) +
  ggplot2::geom_errorbar(ggplot2::aes(ymin = RMSE_Lower, ymax = RMSE_Upper), width = 0.16, alpha = 0.65) +
  ggplot2::geom_line(linewidth = 0.75) +
  ggplot2::geom_point(size = 2.2) +
  ggplot2::facet_wrap(~ Distribution, nrow = 1) +
  ggplot2::scale_x_continuous(breaks = sort(unique(rmse_d_df$Dimensions))) +
  ggplot2::scale_y_log10(labels = scales::label_number(accuracy = 0.01)) +
  ggplot2::scale_color_manual(values = method_cols) +
  ggplot2::labs(
    title = "Finite-sample behavior as dimension increases",
    subtitle = paste0("n = ", scales::comma(max_n), ", rho = ", rho_focus, "; intervals are +/- 1.96 SE"),
    x = "Dimension d",
    y = "RMSE"
  ) +
  paper_theme
save_plot(p_rmse_d, "appendix_pss_rmse_by_dimension", width = 9.2, height = 3.8)

# Appendix Figure B: selected ell under oracle and CV.
ell_df <- summary_focus %>%
  dplyr::filter(N_Samples == max_n)

p_ell <- ggplot2::ggplot(
  ell_df,
  ggplot2::aes(x = Dimensions, y = Selected_Ell_Mean, color = Tuning, shape = Tuning, group = Tuning)
) +
  ggplot2::geom_line(linewidth = 0.75) +
  ggplot2::geom_point(size = 2.2) +
  ggplot2::facet_wrap(~ Distribution, nrow = 1) +
  ggplot2::scale_x_continuous(breaks = sort(unique(ell_df$Dimensions))) +
  ggplot2::scale_y_continuous(breaks = scales::pretty_breaks()) +
  ggplot2::scale_color_manual(values = method_cols) +
  ggplot2::labs(
    title = "Selected partition count",
    subtitle = paste0("n = ", scales::comma(max_n), ", rho = ", rho_focus),
    x = "Dimension d",
    y = "Selected ell"
  ) +
  paper_theme
save_plot(p_ell, "appendix_pss_selected_ell", width = 9.2, height = 3.8)

# Appendix Figure C: CV objective and feasibility for a representative dimension.
cv_curve_df <- cv_table_df %>%
  dplyr::filter(abs(Correlation - rho_focus) < 1e-12, Dimensions == conv_d, N_Samples == max_n) %>%
  dplyr::group_by(Distribution, ell) %>%
  dplyr::summarise(
    cv_score = mean(cv_score, na.rm = TRUE),
    cv_score_se = stats::sd(cv_score, na.rm = TRUE) / sqrt(dplyr::n()),
    cv_coverage = mean(cv_coverage, na.rm = TRUE),
    stable_cell_fraction = mean(stable_cell_fraction, na.rm = TRUE),
    feasible = all(feasible),
    .groups = "drop"
  ) %>%
  dplyr::mutate(
    ell = as.numeric(ell),
    Feasible = ifelse(feasible, "Feasible", "Filtered")
  )

if (nrow(cv_curve_df)) {
  p_cv_curve <- ggplot2::ggplot(
    cv_curve_df,
    ggplot2::aes(x = ell, y = cv_score, color = Feasible, group = 1)
  ) +
    ggplot2::geom_line(color = "grey45", linewidth = 0.65) +
    ggplot2::geom_errorbar(ggplot2::aes(ymin = cv_score - cv_score_se, ymax = cv_score + cv_score_se),
                           width = 0.12, color = "grey50", alpha = 0.7) +
    ggplot2::geom_point(size = 2.3) +
    ggplot2::facet_wrap(~ Distribution, nrow = 1, scales = "free_y") +
    ggplot2::scale_x_continuous(breaks = sort(unique(cv_curve_df$ell))) +
    ggplot2::scale_color_manual(values = c(Feasible = "#009E73", Filtered = "#999999")) +
    ggplot2::labs(
      title = "Coverage- and occupancy-constrained CV objective",
      subtitle = paste0("Representative setting: n = ", scales::comma(max_n),
                        ", d = ", conv_d, ", rho = ", rho_focus),
      x = "Partitions per axis ell",
      y = "Validation negative log-likelihood"
    ) +
    paper_theme
  save_plot(p_cv_curve, "appendix_pss_cv_curve", width = 9.2, height = 3.8)
}

cat("Loaded:\n")
cat("  summary: ", summary_path, "\n", sep = "")
cat("  oracle:  ", oracle_path, "\n", sep = "")
cat("  cv:      ", cv_table_path, "\n", sep = "")
cat("Saved paper plots to:\n")
cat("  ", figure_dir, "\n", sep = "")
cat("\nRecommended main-paper figures:\n")
cat("  1. figure_pss_cv_oracle_gap_heatmap\n")
cat("  2. figure_pss_empirical_convergence\n")
cat("  3. figure_pss_occupancy_diagnostics\n")
cat("Appendix candidates:\n")
cat("  appendix_pss_rmse_by_dimension\n")
cat("  appendix_pss_selected_ell\n")
cat("  appendix_pss_cv_curve\n")
