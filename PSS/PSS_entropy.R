#==============================================================================
# d-DIMENSIONAL PARTITIONED SAMPLE-SPACING (PSS) JOINT ENTROPY ESTIMATOR
#
# calculate_entropy_pss_d(data, n_partitions)
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
calculate_entropy_pss_d <- function(data, n_partitions) {
  # Notation
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
  
  # 3) Linearize (a_1,...,a_d) → single id in {1,...,ℓ^d} (base-ℓ encoding)
  powers_of_l <- l^((1:d) - 1)
  linear_indices <- as.vector(1 + (all_indices_mat - 1) %*% powers_of_l)
  
  # Group row ids by occupied cell P_k
  point_indices_by_bin <- split(seq_len(n_samples), linear_indices)
  
  # ---- Work within one P_k: accumulate Σ(-log \hat f) over its points ----------
  process_single_bin_d <- function(bin_linear_index) {
    point_indices <- point_indices_by_bin[[as.character(bin_linear_index)]]
    
    # Cells with <2 points produce no valid m-spacings → contribute 0 to the sum
    if (is.null(point_indices) || length(point_indices) < 2) return(0)
    
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
    if (!any(valid_indices)) return(0)
    
    # (D) Evaluate log \hat f_{n,ℓ}(X) for each valid row (PSS formula):
    #     log \hat f = log(n_k/n) + Σ_{j=1}^d log( 2 m_k / (n_k Δx_{k j, i}) )
    log_nk_over_n <- log(n_k / n_samples)
    log_spacing_term <- rowSums(
      log(2 * m_k / (n_k * m_spacings_mat[valid_indices, , drop = FALSE]))
    )
    log_f_hat_values <- log_nk_over_n + log_spacing_term
    
    # Return Σ(-log \hat f) over this cell
    sum(-log_f_hat_values)
  }
  
  # 4) Sum cell-wise contributions
  bin_indices_to_process <- as.numeric(names(point_indices_by_bin))
  bin_neg_log_f_sums <- lapply(bin_indices_to_process, process_single_bin_d)
  
  # 5) Plug-in joint entropy estimator: average over all n points
  sum(unlist(bin_neg_log_f_sums)) / n_samples
}
