#==============================================================================
# d-DIMENSIONAL PARTITIONED SAMPLE-SPACING (PSS) JOINT ENTROPY ESTIMATOR
#
# calculate_entropy_pss_d(data, n_partitions, return_diagnostics = FALSE)
#
# INPUT
#   data          : numeric matrix/data.frame with n rows (samples) and d cols (dimensions)
#   n_partitions  : ℓ, number of equal-width partitions per axis (total cells = ℓ^d)
#
# OUTPUT
#   A single numeric value:  \hat H_{n,ℓ} = -(1/n) * Σ_{v=1}^n log \hat f_{n,ℓ}(X_v)
#
# METHOD :
#   1) Partition each coordinate range into ℓ equal-width intervals → ℓ^d hyperrectangles.
#   2) For each occupied cell P_k with n_k points, sort the points marginally and
#      form m-spacings with m_k = ⌊√n_k + 1/2⌋.
#   3) Define the local (piecewise) density at X_v in P_k by
#         \hat f_{n,ℓ}(X_v) = (n_k / n) * ∏_{j=1}^d [ 2 m_k / (n_k Δx_{k j, a_j}) ],
#      where Δx_{k j, a_j} = x_{k j,(a_j+m_k)} - x_{k j,(a_j-m_k)} are marginal m-spacings.
#   4) Plug-in entropy:  \hat H_{n,ℓ} = -(1/n) * Σ_v log \hat f_{n,ℓ}(X_v).
#
# THEORY NOTES (summarized for context):
#   • Choose m_k = ⌊√n_k + 1/2⌋ so m_k/n_k → 0 while m_k → ∞ (standard spacing regime).
#   • Growth condition like ℓ(n)^d = o(n) ensures enough points per cell for stable spacings.
#   • Points yielding zero spacings (ties) are skipped in that cell; their aggregate effect
#     vanishes asymptotically as the estimator’s total mass approaches 1.
#==============================================================================
library(MASS)
library(parallel)
library(FNN)     # using FNN::entropy for kNN entropy
library(dplyr)
library(knitr)

# --- Core estimator (PSS naming, line-by-line commented) ---------------------
calculate_entropy_pss_d <- function(data, n_partitions, return_diagnostics = FALSE) {
  # Notation
  data <- as.matrix(data)
  d <- ncol(data)           # dimension
  n_samples <- nrow(data)   # sample size n
  l <- n_partitions         # ℓ (partitions per axis)

  # 1) Equal-width partition boundaries along each observed axis range
  all_coords <- as.data.frame(data)
  all_boundaries <- lapply(
    all_coords,
    function(col) seq(min(col), max(col), length.out = l + 1)
  )

  # 2) Assign each sample to its axiswise bin (1..ℓ); rightmost interval closed
  all_indices_mat <- do.call(
    cbind,
    mapply(
      findInterval, all_coords, all_boundaries,
      MoreArgs = list(rightmost.closed = TRUE),
      SIMPLIFY = FALSE
    )
  )
  # guard: if a point falls exactly on the max, clamp to ℓ
  all_indices_mat <- pmin(all_indices_mat, l)

  # 3) Encode (a_1,...,a_d) as a stable cell key.
  # A numeric base-ell encoding can lose precision when ell^d exceeds 2^53.
  cell_keys <- do.call(paste, c(as.data.frame(all_indices_mat), sep = ":"))

  # Group row ids by occupied cell P_k
  point_indices_by_bin <- split(seq_len(n_samples), cell_keys)
  bin_sizes <- lengths(point_indices_by_bin)
  total_cells <- l^d
  valid_point_count <- 0L
  skipped_point_count_lt2 <- sum(bin_sizes[bin_sizes < 2])
  zero_spacing_point_count <- 0L
  neg_log_f_sum <- 0

  # ---- Work within one P_k: accumulate Σ(-log \hat f) over its points ----------
  process_single_bin_d <- function(bin_linear_index) {
    point_indices <- point_indices_by_bin[[as.character(bin_linear_index)]]

    # Cells with <2 points produce no valid m-spacings → contribute 0 to the sum
    if (is.null(point_indices) || length(point_indices) < 2) {
      return(list(sum_neg_log_f = 0, n_valid = 0L, n_zero_spacing = 0L))
    }

    n_k <- length(point_indices)
    # Spacing size m_k (PSS choice): grows like √n_k, yet m_k/n_k → 0
    m_k <- floor(sqrt(n_k) + 0.5)

    # (A) Sort marginally within the cell to access order statistics x_{k j,(r)}
    bin_data <- data[point_indices, , drop = FALSE]
    sorted_data_list <- as.list(as.data.frame(apply(bin_data, 2, sort)))

    # (B) Compute marginal m-spacings for each “row index” i and each margin j:
    #     Δx_{k j, i} = x_{(i+m_k)} - x_{(i-m_k)}, with indices clamped to [1, n_k]
    m_spacings_list <- lapply(sorted_data_list, function(col) {
      sapply(seq_len(n_k), function(i) col[min(n_k, i + m_k)] - col[max(1, i - m_k)])
    })
    # n_k × d matrix of spacings (per-point, per-margin)
    m_spacings_mat <- do.call(cbind, m_spacings_list)

    # (C) Exclude rows that have zero spacing in any margin (ties/duplicates)
    valid_indices <- rowSums(m_spacings_mat > 0) == d
    n_valid <- sum(valid_indices)
    n_zero_spacing <- n_k - n_valid
    if (!any(valid_indices)) {
      return(list(sum_neg_log_f = 0, n_valid = 0L, n_zero_spacing = n_zero_spacing))
    }

    # (D) Evaluate log \hat f_{n,ℓ}(X) for each valid row (PSS formula):
    #     log \hat f = log(n_k/n) + Σ_{j=1}^d log( 2 m_k / (n_k Δx_{k j, i}) )
    log_nk_over_n <- log(n_k / n_samples)
    log_spacing_term <- rowSums(
      log(2 * m_k / (n_k * m_spacings_mat[valid_indices, , drop = FALSE]))
    )
    log_f_hat_values <- log_nk_over_n + log_spacing_term

    # Return Σ(-log \hat f) over this cell
    list(
      sum_neg_log_f = sum(-log_f_hat_values),
      n_valid = as.integer(n_valid),
      n_zero_spacing = as.integer(n_zero_spacing)
    )
  }

  # 4) Sum cell-wise contributions
  bin_indices_to_process <- names(point_indices_by_bin)
  bin_results <- lapply(bin_indices_to_process, process_single_bin_d)
  neg_log_f_sum <- sum(vapply(bin_results, `[[`, numeric(1L), "sum_neg_log_f"))
  valid_point_count <- sum(vapply(bin_results, `[[`, integer(1L), "n_valid"))
  zero_spacing_point_count <- sum(vapply(bin_results, `[[`, integer(1L), "n_zero_spacing"))

  # 5) Plug-in joint entropy estimator: average over all n points
  entropy <- neg_log_f_sum / n_samples
  if (!return_diagnostics) return(entropy)

  occupied_cells <- length(bin_sizes)
  usable_cells <- sum(bin_sizes >= 2)
  skipped_occupied_cells_lt2 <- sum(bin_sizes < 2)
  skipped_point_count <- skipped_point_count_lt2 + zero_spacing_point_count

  list(
    entropy = entropy,
    diagnostics = data.frame(
      Dimensions = d,
      N_Samples = n_samples,
      Partitions_Per_Axis = l,
      Total_Cells = total_cells,
      Occupied_Cells = occupied_cells,
      Usable_Cells = usable_cells,
      Empty_Cells = total_cells - occupied_cells,
      Skipped_Occupied_Cells_lt2 = skipped_occupied_cells_lt2,
      Skipped_Points_lt2 = skipped_point_count_lt2,
      Zero_Spacing_Points = zero_spacing_point_count,
      Valid_Points = valid_point_count,
      Skipped_Points = skipped_point_count,
      Occupied_Cell_Fraction = occupied_cells / total_cells,
      Usable_Cell_Fraction = usable_cells / total_cells,
      Skipped_Occupied_Cell_Fraction = skipped_occupied_cells_lt2 / total_cells,
      Skipped_Point_Fraction = skipped_point_count / n_samples,
      Coverage_Fraction = valid_point_count / n_samples,
      Mean_Occupied_Cell_Size = mean(bin_sizes),
      Median_Occupied_Cell_Size = stats::median(bin_sizes),
      Max_Occupied_Cell_Size = max(bin_sizes),
      Sorting_Cost_Proxy = sum(bin_sizes * log(pmax(bin_sizes, 1))),
      stringsAsFactors = FALSE
    )
  )
}

# --- Gaussian-copula simulation utilities ------------------------------------

equicorr_matrix <- function(d, rho) {
  R <- matrix(rho, d, d)
  diag(R) <- 1
  R
}

is_posdef <- function(M) {
  all(eigen(M, symmetric = TRUE, only.values = TRUE)$values > 0)
}

entropy_gamma_margin <- function(shape, scale) {
  shape + log(scale) + lgamma(shape) + (1 - shape) * digamma(shape)
}

entropy_beta_margin <- function(shape1, shape2) {
  lbeta(shape1, shape2) -
    (shape1 - 1) * digamma(shape1) -
    (shape2 - 1) * digamma(shape2) +
    (shape1 + shape2 - 2) * digamma(shape1 + shape2)
}

entropy_lognormal_margin <- function(meanlog, sdlog) {
  meanlog + log(sdlog) + 0.5 * log(2 * pi * exp(1))
}

entropy_laplace_margin <- function(scale) {
  1 + log(2 * scale)
}

qlaplace <- function(u, scale) {
  ifelse(u < 0.5, scale * log(2 * u), -scale * log(2 * (1 - u)))
}

true_entropy_gaussian_copula <- function(d, rho, family,
                                         shape = 0.4, scale = 0.3,
                                         shape1 = 0.5, shape2 = 2,
                                         meanlog = 0, sdlog = 1,
                                         laplace_scale = 1 / sqrt(2)) {
  family <- tolower(family)
  R <- equicorr_matrix(d, rho)
  if (!is_posdef(R)) stop("Equicorrelation matrix is not positive definite.")
  copula_entropy <- 0.5 * as.numeric(determinant(R, logarithm = TRUE)$modulus)

  marginal_entropy <- switch(
    family,
    normal = 0.5 * log(2 * pi * exp(1)),
    gamma = entropy_gamma_margin(shape, scale),
    beta = entropy_beta_margin(shape1, shape2),
    lognormal = entropy_lognormal_margin(meanlog, sdlog),
    laplace = entropy_laplace_margin(laplace_scale),
    stop("Unsupported family: ", family)
  )

  d * marginal_entropy + copula_entropy
}

simulate_gaussian_copula <- function(n, d, rho, family,
                                     shape = 0.4, scale = 0.3,
                                     shape1 = 0.5, shape2 = 2,
                                     meanlog = 0, sdlog = 1,
                                     laplace_scale = 1 / sqrt(2),
                                     u_clip = 1e-12) {
  family <- tolower(family)
  R <- equicorr_matrix(d, rho)
  if (!is_posdef(R)) stop("Equicorrelation matrix is not positive definite.")
  Z <- MASS::mvrnorm(n, mu = rep(0, d), Sigma = R)

  if (family == "normal") return(Z)

  U <- pnorm(Z)
  U <- pmin(pmax(U, u_clip), 1 - u_clip)
  X <- switch(
    family,
    gamma = qgamma(U, shape = shape, scale = scale),
    beta = qbeta(U, shape1 = shape1, shape2 = shape2),
    lognormal = qlnorm(U, meanlog = meanlog, sdlog = sdlog),
    laplace = qlaplace(U, scale = laplace_scale),
    stop("Unsupported family: ", family)
  )
  matrix(X, nrow = n, ncol = d)
}

# --- PSS cross-validation utilities ------------------------------------------

.pss_bin_assign <- function(X, bounds) {
  X <- as.matrix(X)
  d <- ncol(X)
  ell <- length(bounds[[1]]) - 1L
  idx <- do.call(cbind, lapply(seq_len(d), function(j) {
    out <- findInterval(X[, j], bounds[[j]], rightmost.closed = TRUE)
    pmax(1L, pmin(ell, out))
  }))
  matrix(as.integer(idx), nrow = nrow(X), ncol = d)
}

.pss_cell_keys <- function(idx_mat) {
  do.call(paste, c(as.data.frame(idx_mat), sep = ":"))
}

pss_build_density_model <- function(X_train, ell) {
  X_train <- as.matrix(X_train)
  n_train <- nrow(X_train)
  d <- ncol(X_train)
  bounds <- lapply(seq_len(d), function(j) {
    rng <- range(X_train[, j])
    if (rng[1] == rng[2]) rng[2] <- rng[2] + .Machine$double.eps
    seq(rng[1], rng[2], length.out = ell + 1L)
  })

  idx_mat <- .pss_bin_assign(X_train, bounds)
  keys <- .pss_cell_keys(idx_mat)
  groups <- split(seq_len(n_train), keys)
  cells <- new.env(parent = emptyenv())

  for (key in names(groups)) {
    rows <- groups[[key]]
    n_k <- length(rows)
    if (n_k < 2L) next
    m_k <- floor(sqrt(n_k) + 0.5)
    cell_data <- X_train[rows, , drop = FALSE]
    cells[[key]] <- list(
      n_k = n_k,
      m_k = m_k,
      sorted = lapply(seq_len(d), function(j) sort(cell_data[, j]))
    )
  }

  list(bounds = bounds, ell = ell, d = d, n_train = n_train, cells = cells)
}

pss_log_density <- function(model, X_test) {
  X_test <- as.matrix(X_test)
  n_test <- nrow(X_test)
  idx_mat <- .pss_bin_assign(X_test, model$bounds)
  keys <- .pss_cell_keys(idx_mat)
  out <- rep(NA_real_, n_test)

  for (i in seq_len(n_test)) {
    cell <- model$cells[[keys[i]]]
    if (is.null(cell)) next
    log_spacing <- 0
    ok <- TRUE
    for (j in seq_len(model$d)) {
      s <- cell$sorted[[j]]
      rank_i <- findInterval(X_test[i, j], s, rightmost.closed = TRUE)
      rank_i <- pmax(1L, pmin(cell$n_k, rank_i))
      dx <- s[pmin(cell$n_k, rank_i + cell$m_k)] -
        s[pmax(1L, rank_i - cell$m_k)]
      if (!is.finite(dx) || dx <= 0) {
        ok <- FALSE
        break
      }
      log_spacing <- log_spacing + log(2 * cell$m_k / (cell$n_k * dx))
    }
    if (ok) out[i] <- log(cell$n_k / model$n_train) + log_spacing
  }

  out
}

pss_validation_cell_sizes <- function(model, X_test) {
  X_test <- as.matrix(X_test)
  n_test <- nrow(X_test)
  idx_mat <- .pss_bin_assign(X_test, model$bounds)
  keys <- .pss_cell_keys(idx_mat)
  out <- integer(n_test)

  for (i in seq_len(n_test)) {
    cell <- model$cells[[keys[i]]]
    out[i] <- if (is.null(cell)) 0L else as.integer(cell$n_k)
  }

  out
}

select_l_via_cv_pss <- function(data, l_range, n_folds = 3, coverage_penalty = 3,
                                coverage_min = 0.95, one_se_rule = TRUE,
                                occupancy_min = 10,
                                occupancy_fraction_min = coverage_min,
                                occupancy_penalty = coverage_penalty,
                                seed = 42) {
  data <- as.matrix(data)
  set.seed(seed)
  n <- nrow(data)
  fold_id <- sample(rep(seq_len(n_folds), length.out = n))

  cv_table <- do.call(rbind, lapply(l_range, function(ell) {
    nll_sum <- 0
    n_covered <- 0L
    fold_scores <- numeric(n_folds)
    fold_coverages <- numeric(n_folds)
    fold_stable_fractions <- numeric(n_folds)
    all_cell_sizes <- integer(0)
    for (fold in seq_len(n_folds)) {
      validation_rows <- which(fold_id == fold)
      train_rows <- which(fold_id != fold)
      model <- pss_build_density_model(data[train_rows, , drop = FALSE], ell)
      validation_data <- data[validation_rows, , drop = FALSE]
      logf <- pss_log_density(model, validation_data)
      cell_sizes <- pss_validation_cell_sizes(model, validation_data)
      ok <- is.finite(logf)
      n_covered <- n_covered + sum(ok)
      all_cell_sizes <- c(all_cell_sizes, cell_sizes)
      fold_coverages[fold] <- sum(ok) / length(validation_rows)
      fold_stable_fractions[fold] <- mean(cell_sizes >= occupancy_min)
      if (any(ok)) {
        fold_nll <- mean(-logf[ok])
        fold_scores[fold] <- fold_nll +
          coverage_penalty * (1 - fold_coverages[fold]) +
          occupancy_penalty * (1 - fold_stable_fractions[fold])
        nll_sum <- nll_sum + sum(-logf[ok])
      } else {
        fold_scores[fold] <- Inf
      }
    }
    coverage <- n_covered / n
    stable_fraction <- mean(all_cell_sizes >= occupancy_min)
    score <- if (n_covered == 0L) Inf else (nll_sum / n_covered) +
      coverage_penalty * (1 - coverage) +
      occupancy_penalty * (1 - stable_fraction)
    data.frame(
      ell = ell,
      cv_score = score,
      cv_score_se = stats::sd(fold_scores) / sqrt(n_folds),
      cv_coverage = coverage,
      min_fold_coverage = min(fold_coverages),
      stable_cell_fraction = stable_fraction,
      min_fold_stable_cell_fraction = min(fold_stable_fractions),
      validation_cell_size_q10 = as.numeric(stats::quantile(all_cell_sizes, 0.1, names = FALSE)),
      validation_cell_size_median = stats::median(all_cell_sizes),
      occupancy_min = occupancy_min,
      occupancy_fraction_min = occupancy_fraction_min,
      feasible = is.finite(score) &&
        coverage >= coverage_min &&
        stable_fraction >= occupancy_fraction_min,
      stringsAsFactors = FALSE
    )
  }))

  feasible_table <- cv_table[cv_table$feasible, , drop = FALSE]
  selection_rule <- "coverage_occupancy_constrained"
  if (nrow(feasible_table) == 0L) {
    feasible_table <- cv_table[
      order(-cv_table$cv_coverage, -cv_table$stable_cell_fraction, cv_table$cv_score, cv_table$ell),
      ,
      drop = FALSE
    ]
    selection_rule <- "fallback_max_coverage_occupancy"
  }

  best <- feasible_table[order(feasible_table$cv_score, feasible_table$ell), ][1, ]
  if (one_se_rule && nrow(feasible_table) > 1L && is.finite(best$cv_score_se)) {
    threshold <- best$cv_score + best$cv_score_se
    one_se_candidates <- feasible_table[
      feasible_table$cv_score <= threshold,
      ,
      drop = FALSE
    ]
    if (nrow(one_se_candidates) > 0L) {
      best <- one_se_candidates[order(one_se_candidates$ell), ][1, ]
      selection_rule <- paste0(selection_rule, "_one_se")
    }
  }

  list(
    ell_star = as.integer(best$ell),
    cv_table = cv_table,
    selection_rule = selection_rule,
    coverage_min = coverage_min,
    occupancy_min = occupancy_min,
    occupancy_fraction_min = occupancy_fraction_min
  )
}

summarize_pss_for_l <- function(datasets, H_true, ell) {
  estimates <- numeric(length(datasets))
  times <- numeric(length(datasets))
  diag_rows <- vector("list", length(datasets))

  for (i in seq_along(datasets)) {
    timed <- system.time({
      out <- calculate_entropy_pss_d(datasets[[i]], ell, return_diagnostics = TRUE)
    })
    estimates[i] <- out$entropy
    times[i] <- timed[["elapsed"]]
    diag_rows[[i]] <- out$diagnostics
  }

  diagnostics <- do.call(rbind, diag_rows)
  errors <- estimates - H_true
  squared_errors <- errors^2
  rmse <- sqrt(mean(squared_errors, na.rm = TRUE))
  mse_se <- stats::sd(squared_errors, na.rm = TRUE) / sqrt(length(squared_errors))
  rmse_se <- if (is.finite(rmse) && rmse > 0) mse_se / (2 * rmse) else NA_real_

  data.frame(
    ell = ell,
    N_Reps = length(datasets),
    RMSE = rmse,
    RMSE_SE = rmse_se,
    RMSE_Lower = pmax(0, rmse - 1.96 * rmse_se),
    RMSE_Upper = rmse + 1.96 * rmse_se,
    Bias = mean(errors, na.rm = TRUE),
    Estimate_SD = stats::sd(estimates, na.rm = TRUE),
    Eval_Time_s = mean(times, na.rm = TRUE),
    Occupied_Cell_Fraction = mean(diagnostics$Occupied_Cell_Fraction, na.rm = TRUE),
    Usable_Cell_Fraction = mean(diagnostics$Usable_Cell_Fraction, na.rm = TRUE),
    Skipped_Occupied_Cell_Fraction = mean(diagnostics$Skipped_Occupied_Cell_Fraction, na.rm = TRUE),
    Skipped_Point_Fraction = mean(diagnostics$Skipped_Point_Fraction, na.rm = TRUE),
    Coverage_Fraction = mean(diagnostics$Coverage_Fraction, na.rm = TRUE),
    Mean_Occupied_Cell_Size = mean(diagnostics$Mean_Occupied_Cell_Size, na.rm = TRUE),
    Median_Occupied_Cell_Size = mean(diagnostics$Median_Occupied_Cell_Size, na.rm = TRUE),
    Max_Occupied_Cell_Size = mean(diagnostics$Max_Occupied_Cell_Size, na.rm = TRUE),
    Sorting_Cost_Proxy = mean(diagnostics$Sorting_Cost_Proxy, na.rm = TRUE),
    stringsAsFactors = FALSE
  )
}
