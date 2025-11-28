# ============================================================
# CADEE (Recursive Copula Splitting) 
#   copulasH_R(X, range = NULL, level = 0, params = list())
# ============================================================

copulasH_R <- function(xs, range = NULL, level = 0L, params = list()) {
  xs <- as.matrix(xs)
  storage.mode(xs) <- "double"
  n <- nrow(xs); D <- ncol(xs)
  if (n == 0L) return(0)
  
  p <- .cadee_defaults(D)
  if (length(params)) p[names(params)] <- params
  
  # D==1 or too small: sum of 1D marginals
  if (D == 1L) return(.H1_marginal(xs[,1], range = if (is.null(range)) NULL else range[,1], p))
  if (n < p$minSamplesToProcessCopula) return(.sum_marginals(xs, range, p))
  
  # rank-transform to U(0,1) + sum of marginal entropies
  tr <- .rank_transform_and_H1(xs, range, p)
  H1s <- tr$H1s
  ys  <- tr$ys   # n x D in (0,1)
  
  # build dependence adjacency via p-values (+ 2D refinement)
  dep <- .dependence_mask(ys, p)   # logical D x D
  
  # if graph splits into ≥2 connected components → recurse per block
  comps <- .connected_components(dep)
  if (length(comps) >= 2L) {
    H <- H1s
    for (dims in comps) {
      if (length(dims) > 1L) {
        ysmall <- ys[, dims, drop = FALSE]
        H <- H + copulasH_R(ysmall, range = rbind(rep(0, length(dims)), rep(1, length(dims))),
                            level = level + 1L, params = p)
      }
    }
    return(H)
  }
  
  # small again? stop at marginals
  if (n < p$minSamplesToProcessCopula) return(H1s)
  
  # choose a split dimension: most correlated overall (R^2 sum)
  R <- suppressWarnings(stats::cor(ys, method = "pearson")) # Pearson on ranks == Spearman on xs
  R2 <- R^2; diag(R2) <- 0
  Rsum <- colSums(R2)
  largeDims <- which((max(Rsum) - Rsum) < 0.1 & Rsum > 0)
  if (!length(largeDims)) largeDims <- seq_len(D)
  maxCorrsDim <- if (isTRUE(p$randomLevels)) {
    sample(largeDims, 1L)
  } else {
    largeDims[(level %% length(largeDims)) + 1L]
  }
  
  # recursive equal partitions along chosen dim
  nparts <- p$numberOfPartitionsPerDim
  Hparts <- numeric(nparts)
  for (prt in seq_len(nparts)) {
    f <- (prt - 1) / nparts
    l <- if (prt == nparts) 1 else prt / nparts
    mask <- (ys[, maxCorrsDim] >= f) & (ys[, maxCorrsDim] < l)
    nIn <- sum(mask)
    if (!nIn) next
    ysub <- ys[mask, , drop = FALSE]
    ysub[, maxCorrsDim] <- (ysub[, maxCorrsDim] - f) * nparts
    Hparts[prt] <- copulasH_R(ysub, range = rbind(rep(0, D), rep(1, D)),
                              level = level + 1L, params = p)
  }
  H1s + mean(Hparts)
}

# -------------------- helpers --------------------

.cadee_defaults <- function(D) {
  list(
    randomLevels                       = FALSE,
    nbins1                             = 1000L,
    pValuesForIndependence             = 0.05,
    exponentForH2D                     = 0.62,
    acceptenceThresholdForIndependence = -0.7,   # same as original
    minSamplesToProcessCopula          = 5L * D,
    numberOfPartitionsPerDim           = 2L
  )
}

.sum_marginals <- function(xs, range, p) {
  D <- ncol(xs); H <- 0
  for (j in seq_len(D)) {
    rj <- if (is.null(range)) NULL else range[, j]
    H <- H + .H1_marginal(xs[, j], rj, p)
  }
  H
}

.rank_transform_and_H1 <- function(xs, range, p) {
  n <- nrow(xs); D <- ncol(xs)
  ys <- matrix(0, n, D)
  H1s <- 0
  for (j in seq_len(D)) {
    xj <- xs[, j]
    ord <- order(xj)
    # marginal entropy
    if (is.null(range)) H1 <- .H1_spacings(sort(xj), p) else H1 <- .H1_bins(sort(xj), p, range[, j])
    H1s <- H1s + H1
    # empirical CDF ranks: (0.5/n, 1.5/n, ..., (n-0.5)/n)
    r <- numeric(n); r[ord] <- ((0:(n-1)) + 0.5) / n
    ys[, j] <- r
  }
  list(H1s = H1s, ys = ys)
}

# build boolean adjacency for dependence
.dependence_mask <- function(ys, p) {
  n <- nrow(ys); D <- ncol(ys)
  R <- suppressWarnings(stats::cor(ys, method = "pearson"))
  # fast p-value matrix for Pearson r
  # t = r * sqrt((n-2)/(1-r^2)), df=n-2
  P <- matrix(0, D, D)
  for (i in seq_len(D)) for (j in seq_len(D)) if (i != j) {
    r <- R[i, j]
    if (is.finite(r) && abs(r) < 1) {
      tstat <- r * sqrt(max(1, n - 2) / max(1e-12, 1 - r^2))
      P[i, j] <- 2 * stats::pt(-abs(tstat), df = max(1, n - 2))
    } else {
      P[i, j] <- 0
    }
  }
  isCorr <- (P < p$pValuesForIndependence)
  diag(isCorr) <- FALSE
  
  # refine undecided pairs via coarse 2D entropy proxy on [0,1]^2
  undec <- which(!isCorr, arr.ind = TRUE)
  if (nrow(undec)) {
    nFactor <- n ^ p$exponentForH2D
    for (k in seq_len(nrow(undec))) {
      i <- undec[k, 1]; j <- undec[k, 2]
      if (i <= j) next
      H2 <- .H2_unit_square(cbind(ys[, i], ys[, j]), p)
      # keep original odd rule: set TRUE when H2*nFactor < negative threshold
      isCorr[i, j] <- isCorr[j, i] <- (H2 * nFactor < p$acceptenceThresholdForIndependence)
    }
  }
  isCorr
}

# connected components of an undirected graph (adjacency = logical matrix)
.connected_components <- function(adj) {
  D <- nrow(adj)
  if (D <= 1) return(list(seq_len(D)))
  # make symmetric, zero diag
  A <- adj | t(adj); diag(A) <- FALSE
  seen <- rep(FALSE, D)
  comps <- list()
  for (s in seq_len(D)) if (!seen[s]) {
    # BFS
    q <- s; seen[s] <- TRUE; comp <- c()
    while (length(q)) {
      v <- q[1]; q <- q[-1]; comp <- c(comp, v)
      nbrs <- which(A[v, ] & !seen)
      if (length(nbrs)) { seen[nbrs] <- TRUE; q <- c(q, nbrs) }
    }
    comps[[length(comps) + 1L]] <- comp
  }
  comps
}

# 1D: fixed-range bin estimator (natural log)
.H1_bins <- function(xs_sorted, p, range_j) {
  n <- length(xs_sorted)
  nbins <- max(min(p$nbins1, floor(n^0.4), floor(n/10)), 1L)
  br <- seq(range_j[1], range_j[2], length.out = nbins + 1L)
  # density histogram
  h <- hist(xs_sorted, breaks = br, plot = FALSE, include.lowest = TRUE, right = TRUE)
  dens <- h$density
  mask <- dens > 0
  if (!any(mask)) return(0)
  dx <- diff(br)[1]
  -dx * sum(dens[mask] * log(dens[mask]))
}

# 1D: m_n spacings estimator (natural log)
.H1_spacings <- function(xs_sorted, p) {
  n <- length(xs_sorted)
  if (n < 3L) return(0)
  mn <- floor(n^(1/3 - 0.01)); if (mn < 1L) mn <- 1L
  if (n - mn < 1L) return(0)
  sp <- xs_sorted[(mn + 1L):n] - xs_sorted[1L:(n - mn)]
  sp <- pmax(sp, .Machine$double.eps)
  (sum(log(sp)) + log(n / mn) * (n - mn)) / n
}

# 2D: coarse entropy on [0,1]^2 (proxy used in original for MI/independence)
.H2_unit_square <- function(X2, p) {
  n <- nrow(X2)
  if (n < 4L) return(0)
  nbins <- max(min(p$nbins1, floor(n^0.2), floor(n/10)), 2L)
  # clamp to [0,1]
  Y <- pmin(pmax(X2, 0), 1 - .Machine$double.eps)
  bins <- floor(Y * nbins)
  bins[bins < 0] <- 0; bins[bins > nbins - 1L] <- nbins - 1L
  idx <- bins[, 1] + bins[, 2] * nbins
  counts <- tabulate(idx + 1L, nbins ^ 2L)
  mask <- counts > 0
  if (!any(mask)) return(0)
  -sum(counts[mask] * log(counts[mask])) / n + log(n / (nbins * nbins))
}

# marginal dispatcher
.H1_marginal <- function(x, range, p) {
  x <- as.double(x)
  ord <- order(x)
  if (is.null(range)) .H1_spacings(x[ord], p) else .H1_bins(x[ord], p, range)
}

# ========================== USAGE EXAMPLE ==========================
# set.seed(42)
 X <- matrix(rnorm(2000), ncol = 5)
 H <- copulasH_R(X)                       # range unknown (m-spacings for marginals)
 H_known <- copulasH_R(X, range = rbind(apply(X, 2, min), apply(X, 2, max)))
 params <- list(numberOfPartitionsPerDim = 3L)
 H_tuned <- copulasH_R(X, params = params)
 H
 
 
 # True entropy of MVN
 true_entropy_mvn <- function(Sigma) {
   d <- ncol(Sigma)
   0.5 * log((2 * pi * exp(1))^d * det(Sigma))
 }
 
 # Wrapper for CADEE
 cadee_entropy <- function(X) {
   copulasH_R(X)   
 }
 
 # Empirical MSE experiment with correlation
 empirical_mse_cadee <- function(nrep = 100L, n = 500L, d = 5L, rho = 0.5) {
   #Covariance matrix
   Sigma <- matrix(rho, d, d); diag(Sigma) <- 1
   mu <- rep(0, d)
   
   trueH <- true_entropy_mvn(Sigma)
   est <- numeric(nrep); time <- numeric(nrep)
   
   for (r in seq_len(nrep)) {
     X <- MASS::mvrnorm(n, mu = mu, Sigma = Sigma)
     t0 <- proc.time()[3]
     est[r] <- cadee_entropy(X)
     
     time[r] <- proc.time()[3] - t0
   }
   
   bias <- mean(est) - trueH
   var  <- mean((est - mean(est))^2)
   rmse  <- sqrt(mean((est - trueH)^2))
   
   list(true = trueH, mean_est = mean(est),
        bias = bias, var = var, rmse = rmse,time_mean=mean(time))
 }
 
 # ================= USAGE ==================
 set.seed(123)
 res <- empirical_mse_cadee(nrep = 100, n = 30000, d = 10 , rho = 0)
 print(res)
 
 
 
 #==============================================================================
 #  CADEE on Gaussian-copula Multivariate Gamma
 #    H_true = sum_j H(Gamma(shape_j, scale_j)) + 0.5 * log det(R)
 #==============================================================================
 
 # --- 0) Libraries -------------------------------------------------------------
 pkgs <- c("MASS","dplyr","knitr")
 to_install <- pkgs[!sapply(pkgs, requireNamespace, quietly=TRUE)]
 if (length(to_install)) install.packages(to_install, repos="https://cloud.r-project.org")
 library(MASS); library(dplyr); library(knitr)
 
 # --- 1) Helpers ---------------------------------------------------------------
 equicorr <- function(d, rho) { R <- matrix(rho, d, d); diag(R) <- 1; R }
 is_posdef <- function(M) { ei <- eigen(M, symmetric=TRUE, only.values=TRUE)$values; all(ei>0) }
 
 # Gamma entropy (shape=k, scale=theta) in nats
 gamma_entropy <- function(shape, scale) {
   shape + log(scale) + lgamma(shape) + (1 - shape) * digamma(shape)
 }
 
 # True H: sum of Gamma entropies + 0.5 * log det(R)
 true_entropy_gamma_gausscop <- function(d, rho, shape=2, scale=1) {
   if (length(shape)==1) shape <- rep(shape, d)
   if (length(scale)==1) scale <- rep(scale, d)
   stopifnot(length(shape)==d, length(scale)==d)
   R <- equicorr(d, rho); if (!is_posdef(R)) stop("R not PD")
   0.5 * as.numeric(determinant(R, logarithm=TRUE)$modulus) +
     sum(mapply(gamma_entropy, shape, scale))
 }
 
 # Sampler: Z~N(0,R), U=Phi(Z), X_j=QGamma(U_j; shape_j, scale_j)
 rmgamma_gausscop <- function(n, d, rho, shape=2, scale=1) {
   if (length(shape)==1) shape <- rep(shape, d)
   if (length(scale)==1) scale <- rep(scale, d)
   R <- equicorr(d, rho); if (!is_posdef(R)) stop("R not PD")
   Z <- MASS::mvrnorm(n, mu=rep(0,d), Sigma=R)
   U <- pnorm(Z)
   X <- matrix(NA_real_, n, d)
   for (j in 1:d) X[, j] <- qgamma(U[, j], shape=shape[j], scale=scale[j])
   X
 }
 
 # CADEE wrapper (논문 기본값 반영)
 cadee_entropy <- function(X, cadee_params=list()) {
   defaults <- list(
     randomLevels = FALSE,
     nbins1 = 1000L,
     pValuesForIndependence = 0.05,
     exponentForH2D = 0.62,
     acceptenceThresholdForIndependence = -0.75, # 권장 컷
     minSamplesToProcessCopula = 5L * ncol(X),
     numberOfPartitionsPerDim = 2L
   )
   if (length(cadee_params)) defaults[names(cadee_params)] <- cadee_params
   copulasH_R(X, params = defaults)
 }
 
 # Empirical MSE
 empirical_mse_cadee_gamma <- function(nrep=50, n=5000, d=5, rho=0.5,
                                       shape=2, scale=1, cadee_params=list()) {
   Htrue <- true_entropy_gamma_gausscop(d, rho, shape, scale)
   est <- numeric(nrep); time <- numeric(nrep)
   for (r in seq_len(nrep)) {
     X <- rmgamma_gausscop(n, d, rho, shape, scale)
     t0 <- proc.time()[3]
     est[r] <- cadee_entropy(X, cadee_params)
     
     time[r] <- proc.time()[3] - t0
   }
   list(true=Htrue, mean_est=mean(est),
        bias=mean(est)-Htrue, mse=mean((est-Htrue)^2),
        rmse=sqrt(mean((est-Htrue)^2)),
        time_mean=mean(time), time_med=median(time))
 }
 
 # --- 2) Run (example) --------------------------------------------------------
 set.seed(123)
 res_gamma <- empirical_mse_cadee_gamma(
   nrep=100, n=30000, d=20, rho=0, shape=0.4, scale=0.3
 )
 print(res_gamma)
 
 
 
