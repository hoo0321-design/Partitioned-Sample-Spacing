#==============================================================================
#  EEG Eye State — PSS (CV-ℓ*) → ICA → TC(before/after)
#  This script performs an ICA analysis on the EEG Eye State dataset,
#  using the PSS estimator for Total Correlation (TC) calculation.
#  The optimal hyperparameter ℓ* for PSS is selected via cross-validation.
#==============================================================================

# --- 1) Dependencies ---
pkgs <- c("foreign", "fastICA")
to_install <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]
if (length(to_install)) install.packages(to_install, repos = "https://cloud.r-project.org")
invisible(lapply(pkgs, require, character.only = TRUE))

# --- 2) Guard: Check for existence of the PSS estimator function ---
if (!exists("calculate_entropy_pss_d")) {
  stop("Missing function: calculate_entropy_pss_d(data, n_partitions). Please source it first.")
}

# --- 3) Whitening function ---
whiten_data <- function(X) {
  X <- as.matrix(X)
  Xc <- scale(X, center = TRUE, scale = FALSE)
  S  <- cov(Xc)
  eg <- eigen(S, symmetric = TRUE)
  lam <- pmax(eg$values, .Machine$double.eps)
  Winvhalf <- eg$vectors %*% diag(1/sqrt(lam)) %*% t(eg$vectors)
  Xw <- Xc %*% Winvhalf
  list(Xw = Xw, Winvhalf = Winvhalf)
}

# --- 4) PSS entropy and Total Correlation (TC) functions ---
# Wrapper for the core PSS entropy estimator
H_pss <- function(data, ell) {
  data <- as.matrix(data)
  if (ncol(data) == 0L) return(0)
  calculate_entropy_pss_d(data, n_partitions = ell)  # Assumes the function returns entropy in nats
}

# Calculates Total Correlation using the PSS estimator
tc_pss <- function(X, ell) {
  X <- as.matrix(X); d <- ncol(X)
  H_joint <- H_pss(X, ell)
  H_marg  <- sum(sapply(seq_len(d), function(j) H_pss(X[, j, drop = FALSE], ell)))
  H_marg - H_joint
}

# --- 5) PSS Cross-Validation (CV) helper functions ---
# These functions implement a cross-validated negative log-likelihood criterion
# to select the optimal partition parameter, ℓ.

.bin_assign <- function(X, bounds) {
  X <- as.matrix(X); d <- ncol(X)
  if (d < 1L) stop(".bin_assign: X has 0 columns.")
  ell <- length(bounds[[1]]) - 1L
  mats <- lapply(seq_len(d), function(j)
    findInterval(X[, j], bounds[[j]], rightmost.closed = TRUE))
  mat <- do.call(cbind, mats)
  mat <- pmax(1L, pmin(ell, mat))
  matrix(as.integer(mat), nrow = nrow(X), ncol = d)
}

.cell_keys <- function(idx_mat) {
  idx_mat <- as.matrix(idx_mat)
  if (nrow(idx_mat) == 0L) return(character(0))
  do.call(paste, c(as.data.frame(idx_mat), sep = ":"))
}

.pss_build <- function(X_train, ell) {
  X_train <- as.matrix(X_train)
  n_tr <- nrow(X_train); d <- ncol(X_train)
  if (d < 1L) stop(".pss_build: X_train has 0 columns.")
  if (n_tr < 1L) stop(".pss_build: X_train has 0 rows.")
  if (ell < 2L) stop(".pss_build: ell must be >= 2.")
  
  bounds <- lapply(seq_len(d), function(j) {
    rng <- range(X_train[, j])
    if (rng[1] == rng[2]) rng[2] <- rng[2] + .Machine$double.eps
    seq(rng[1], rng[2], length.out = ell + 1L)
  })
  idx_mat <- .bin_assign(X_train, bounds)
  keys    <- .cell_keys(idx_mat)
  groups  <- split(seq_len(n_tr), keys)
  
  cells <- new.env(parent = emptyenv())
  for (key in names(groups)) {
    idx <- groups[[key]]
    nk  <- length(idx); if (nk < 2L) next
    mk  <- floor(sqrt(nk) + 0.5)
    dat <- X_train[idx, , drop = FALSE]
    sorted_cols <- lapply(seq_len(d), function(j) sort(dat[, j]))
    cells[[key]] <- list(nk = nk, mk = mk, sorted = sorted_cols)
  }
  list(bounds = bounds, ell = ell, d = d, n_tr = n_tr, cells = cells)
}

.pss_logf <- function(model, X_test) {
  X_test <- as.matrix(X_test)
  n_te <- nrow(X_test); d <- model$d
  if (n_te == 0L) return(numeric(0))
  idx_mat <- .bin_assign(X_test, model$bounds)
  keys    <- .cell_keys(idx_mat)
  out <- rep(NA_real_, n_te)
  for (i in seq_len(n_te)) {
    cell <- model$cells[[ keys[i] ]]
    if (is.null(cell)) next
    nk <- cell$nk; mk <- cell$mk
    ok <- TRUE; dlogs <- 0.0
    for (j in seq_len(d)) {
      s <- cell$sorted[[j]]
      r <- findInterval(X_test[i, j], s, rightmost.closed = TRUE)
      r <- pmax(1L, pmin(nk, r))
      dx <- s[pmin(nk, r + mk)] - s[pmax(1L, r - mk)]
      if (!(is.finite(dx) && dx > 0)) { ok <- FALSE; break }
      dlogs <- dlogs + log(2 * mk / (nk * dx))
    }
    if (!ok) next
    out[i] <- log(nk / model$n_tr) + dlogs
  }
  out
}

select_ell_via_cv <- function(Xw, L_grid = 2:15, K = 3, lambda = 3, seed = 42) {
  set.seed(seed)
  n <- nrow(Xw)
  if (n < K) K <- max(2L, min(n, K))
  fold_id <- sample(rep(seq_len(K), length.out = n))
  scores <- lapply(L_grid, function(ell) {
    nll_sum <- 0.0; n_cov <- 0L
    for (f in seq_len(K)) {
      te <- which(fold_id == f)
      tr <- which(fold_id != f)
      mdl <- .pss_build(Xw[tr, , drop = FALSE], ell)
      logf <- .pss_logf(mdl, Xw[te, , drop = FALSE])
      ok <- is.finite(logf)
      n_cov  <- n_cov + sum(ok)
      if (any(ok)) nll_sum <- nll_sum + sum(-logf[ok])
    }
    coverage <- n_cov / n
    score <- if (n_cov == 0) Inf else (nll_sum / n_cov) + lambda * (1 - coverage)
    data.frame(ell = ell, score = score, coverage = coverage)
  })
  tab <- do.call(rbind, scores)
  tab <- tab[order(tab$ell), ]
  best <- tab[order(tab$score, tab$ell), ][1, , drop = FALSE]
  list(ell_star = as.integer(best$ell), cv_table = tab)
}

# --- 6) Data loading ---
load_eeg_eye_state <- function(path_arff) {
  df <- foreign::read.arff(path_arff)
  df <- as.data.frame(df)
  df <- df[complete.cases(df), ]
  X <- as.matrix(df[, 1:(ncol(df) - 1), drop = FALSE]) # last column is the class label
  storage.mode(X) <- "double"
  list(X = X)
}

# --- 7) Main pipeline: PSS-only ---
compare_pss_tc <- function(path_arff,
                           L_grid = 2:15,
                           Kfolds = 3,
                           lambda = 3,
                           seed = 2025) {
  set.seed(seed)
  dat <- load_eeg_eye_state(path_arff)
  X   <- dat$X
  n   <- nrow(X); d <- ncol(X)
  
  # Step 1: Whiten the data
  W  <- whiten_data(X)
  Xw <- W$Xw
  
  # Step 2: Select optimal ell* for PSS via Cross-Validation
  sel_ell <- select_ell_via_cv(Xw, L_grid = L_grid, K = Kfolds, lambda = lambda, seed = seed)
  ell_star <- sel_ell$ell_star
  
  # Step 3: Calculate Total Correlation (TC) BEFORE ICA
  t_pss_before <- system.time({ TC_PSS_before <- tc_pss(Xw, ell_star) })[3]
  
  # Step 4: Perform Independent Component Analysis (ICA)
  set.seed(seed) # for reproducible ICA
  t_ica <- system.time({
    ica <- fastICA::fastICA(Xw, n.comp = d, fun = "logcosh",
                            alg.typ = "parallel", maxit = 1000, tol = 1e-6)
  })[3]
  Y <- ica$S
  
  # Step 5: Calculate Total Correlation (TC) AFTER ICA
  t_pss_after <- system.time({ TC_PSS_after <- tc_pss(Y, ell_star) })[3]
  
  list(
    dims = c(n = n, d = d),
    ell_star = ell_star,
    ell_cv = sel_ell$cv_table,
    PSS_before = TC_PSS_before,
    PSS_after  = TC_PSS_after,
    PSS_delta  = TC_PSS_before - TC_PSS_after,
    time_PSS_before = t_pss_before,
    time_PSS_after  = t_pss_after,
    time_ICA = t_ica
  )
}

# ============================
#  EXAMPLE USAGE
# ============================
path_arff <- "./EEG Eye State.arff" # Assumes the ARFF file is in the same directory
res <- compare_pss_tc(
  path_arff = path_arff,
  L_grid    = 2:15,
  Kfolds    = 3,
  lambda    = 0, # Note: lambda=0 means no penalty for non-coverage
  seed      = 2025
)

# Print results
cat("\n--- ICA Experiment Results ---\n")
print(res$dims)
cat("\n--- Cross-Validation Table for ℓ ---\n")
print(res$ell_cv)
cat(sprintf("\nOptimal ℓ* selected by CV: %d\n", res$ell_star))

cat(sprintf("[PSS ] TC_before = %.6f, TC_after = %.6f, ΔTC = %.6f  (time: %.2fs → %.2fs)\n",
            res$PSS_before, res$PSS_after, res$PSS_delta,
            res$time_PSS_before, res$time_PSS_after))

cat(sprintf("[ICA ] Computation time = %.2fs\n", res$time_ICA))