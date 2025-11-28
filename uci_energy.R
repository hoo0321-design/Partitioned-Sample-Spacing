# ==============================================================================
# 0. library loading
# ==============================================================================
if(!require(FNN)) install.packages("FNN")
if(!require(caret)) install.packages("caret")
if(!require(tidyr)) install.packages("tidyr")
if(!require(dplyr)) install.packages("dplyr")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(e1071)) install.packages("e1071")
if(!require(class)) install.packages("class")

library(dplyr); library(ggplot2); library(FNN); library(caret); 
library(class); library(tidyr); library(e1071)

set.seed(42)

# ==============================================================================
# 1. UCI Appliances Energy data load (N=19735, d=26)
# ==============================================================================
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv"

cat("Downloading Appliances Energy dataset (approx. 2.5MB)...\n")
data <- read.csv(url)

# [Preprocessing]
# 1. Remove unnecessary columns: 'date' (string), 'rv1', 'rv2' (random noise variables)
data$date <- NULL
data$rv1 <- NULL
data$rv2 <- NULL

# 2. Define target ('Appliances' energy consumption)
# Convert regression target to binary classification via median threshold
target_val <- data$Appliances
threshold <- median(target_val)
y <- as.factor(ifelse(target_val > threshold, 1, 0))

# 3. Define features (exclude 'Appliances')
# T1~T9, RH_1~RH_9, Lights, T_out, Press_mm_hg, ... total 26 features
X <- as.matrix(data[, -1]) # exclude Appliances (1st column)

# Normalize continuous sensor data
preproc <- preProcess(X, method = c("range"))
X_scaled <- as.matrix(predict(preproc, X))

# [Stability trick] Add small random noise (sensor data may contain identical values)
noise <- matrix(rnorm(prod(dim(X_scaled)), mean = 0, sd = 1e-5), nrow = nrow(X_scaled))
X_scaled <- X_scaled + noise

# Train/Test Split (70:30)
train_idx <- createDataPartition(y, p = 0.7, list = FALSE)
X_train <- X_scaled[train_idx, ]
y_train <- y[train_idx]
X_test  <- X_scaled[-train_idx, ]
y_test  <- y[-train_idx]

cat("Data Ready: N_train =", nrow(X_train), " Dimension =", ncol(X_train), "\n")
cat("Note: Native d=26. Perfect for showing PSS robustness.\n")

# ==============================================================================
# 2. CV Functions (Fast Sparse PSS)
# ==============================================================================
# (Memory-efficient sparse indexing version)
cv_tune_pss <- function(X, ell_candidates, n_folds = 3) {
  folds <- createFolds(1:nrow(X), k = n_folds, list = TRUE)
  scores <- c(); d <- ncol(X)
  cat("Tuning PSS ell: ")
  for (l in ell_candidates) {
    if (d * log2(l) > 31) { cat(sprintf("[Skip l=%d] ", l)); scores <- c(scores, -Inf); next }
    fold_scores <- c(); powers <- l^((1:d)-1)
    for (i in 1:n_folds) {
      idx_val <- folds[[i]]; X_tr <- X[-idx_val, , drop=FALSE]; X_val <- X[idx_val, , drop=FALSE]
      n_tr <- nrow(X_tr)
      boundaries <- vector("list", d); bin_log_vol <- 0
      for(j in 1:d) {
        rng <- range(X_tr[,j]); boundaries[[j]] <- seq(rng[1]-1e-8, rng[2]+1e-8, length.out = l + 1)
        bin_log_vol <- bin_log_vol + log((rng[2]-rng[1])/l)
      }
      get_lin_idx <- function(mat) {
        idx_mat <- matrix(0, nrow=nrow(mat), ncol=d)
        for(j in 1:d) {
          v <- findInterval(mat[,j], boundaries[[j]])
          v[v < 1] <- 1; v[v > l] <- l
          idx_mat[,j] <- v
        }
        as.vector(1 + (idx_mat - 1) %*% powers)
      }
      lin_tr <- get_lin_idx(X_tr); lin_val <- get_lin_idx(X_val)
      unique_tr <- unique(lin_tr); mapped_tr <- match(lin_tr, unique_tr); tr_counts <- tabulate(mapped_tr)
      mapped_val <- match(lin_val, unique_tr); val_counts <- tr_counts[mapped_val]; val_counts[is.na(val_counts)] <- 0
      pos <- val_counts > 0; log_liks <- numeric(length(lin_val))
      if(any(pos)) log_liks[pos] <- log(val_counts[pos]) - log(n_tr) - bin_log_vol
      if(any(!pos)) log_liks[!pos] <- log(1e-6)
      fold_scores <- c(fold_scores, mean(log_liks))
    }
    avg <- mean(fold_scores); scores <- c(scores, avg); cat(sprintf("[%d: %.2f] ", l, avg))
  }
  cat("\n"); return(ell_candidates[which.max(scores)])
}

cv_tune_knn <- function(X, k_candidates, n_folds = 3) {
  folds <- createFolds(1:nrow(X), k = n_folds, list = TRUE)
  scores <- c(); d <- ncol(X); cd <- pi^(d/2)/gamma(d/2+1)
  cat("Tuning KNN k: ")
  for (k in k_candidates) {
    fold_scores <- c()
    for (i in 1:n_folds) {
      idx_val <- folds[[i]]; X_tr <- X[-idx_val, , drop=FALSE]; X_val <- X[idx_val, , drop=FALSE]
      knn_res <- get.knnx(X_tr, X_val, k=k); dist_k <- pmax(knn_res$nn.dist[,k], 1e-6)
      log_dens <- log(k) - log(nrow(X_tr)) - log(cd) - d*log(dist_k)
      fold_scores <- c(fold_scores, mean(log_dens))
    }
    avg <- mean(fold_scores); scores <- c(scores, avg); cat(sprintf("[%d: %.2f] ", k, avg))
  }
  cat("\n"); return(k_candidates[which.max(scores)])
}

# ==============================================================================
# 3. Hyperparameter Tuning
# ==============================================================================
# PSS: use entire training data (~13k)
# KNN: sample 3000 for speed difference
tune_idx <- createDataPartition(y_train, p = min(1, 3000/nrow(X_train)), list=FALSE)
X_tune <- X_train[tune_idx, ]

ell_cands <- c(2, 3, 4, 5) # d=26 → expected ell around 2 or 3
k_cands <- c(1,2,3,4,5,7,10)

cat("\n>>> Tuning PSS (Full Data) <<<\n")
time_tune_pss <- system.time({ best_ell <- cv_tune_pss(X_train, ell_cands, n_folds = 3) })
cat("PSS Tuning Time:", time_tune_pss[3], "s\n")

cat("\n>>> Tuning KNN (Subset) <<<\n")
time_tune_knn <- system.time({ best_k <- cv_tune_knn(X_tune, k_cands, n_folds = 3) })
cat("KNN Tuning Time:", time_tune_knn[3], "s\n")

cat("Selected Ell:", best_ell, "| Selected K:", best_k, "\n")

# ==============================================================================
# 4. Core Estimators
# ==============================================================================
calculate_entropy_pss_d <- function(data, n_partitions) {
  d <- ncol(data); n_samples <- nrow(data); l <- n_partitions 
  all_coords <- as.data.frame(data)
  all_boundaries <- lapply(all_coords, function(col) seq(min(col), max(col), length.out = l + 1))
  all_indices_mat <- do.call(cbind, mapply(findInterval, all_coords, all_boundaries, MoreArgs = list(rightmost.closed = TRUE), SIMPLIFY = FALSE))
  all_indices_mat <- pmin(all_indices_mat, l)
  powers_of_l <- l^((1:d) - 1)
  linear_indices <- as.vector(1 + (all_indices_mat - 1) %*% powers_of_l)
  point_indices_by_bin <- split(seq_len(n_samples), linear_indices)
  
  process_single_bin <- function(indices) {
    n_k <- length(indices); if (n_k < 2) return(0)
    m_k <- floor(sqrt(n_k) + 0.5); bin_data <- data[indices, , drop = FALSE]
    sorted_data_list <- apply(bin_data, 2, sort)
    m_spacings_list <- lapply(1:d, function(j) {
      col <- sorted_data_list[, j]
      sapply(1:n_k, function(i) col[min(n_k, i + m_k)] - col[max(1, i - m_k)])
    })
    m_spacings_mat <- do.call(cbind, m_spacings_list)
    valid <- rowSums(m_spacings_mat > 0) == d; if (!any(valid)) return(0)
    log_term <- rowSums(log(2 * m_k / (n_k * m_spacings_mat[valid, , drop=FALSE])))
    sum(-(log(n_k / n_samples) + log_term))
  }
  bin_sums <- sapply(point_indices_by_bin, process_single_bin)
  return(sum(bin_sums) / n_samples)
}

get_mi_pss <- function(X_sub, y, ell) {
  H_S <- calculate_entropy_pss_d(X_sub, ell)
  classes <- levels(y); H_cond <- 0
  for (c in classes) {
    idx <- which(y == c); if(length(idx)==0) next
    p_c <- length(idx) / length(y)
    H_cond <- H_cond + p_c * calculate_entropy_pss_d(X_sub[idx, , drop=FALSE], ell)
  }
  return(H_S - H_cond)
}

get_mi_knn <- function(X_sub, y, k=5) {
  X_sub <- as.matrix(X_sub); H_S <- FNN::entropy(X_sub, k=k)[k]
  classes <- levels(y); H_cond <- 0
  for (c in classes) {
    idx <- which(y == c); if(length(idx) <= k) next
    p_c <- length(idx) / length(y)
    H_cond <- H_cond + p_c * FNN::entropy(X_sub[idx, , drop=FALSE], k=k)[k]
  }
  return(H_S - H_cond)
}

# ==============================================================================
# 5. Forward Selection (Max 20 Steps)
# ==============================================================================
# Try selecting up to 20 features out of 26
run_forward_selection <- function(X, y, method=c("PSS", "KNN"), max_k=20, param=NULL) {
  n_features <- ncol(X); selected <- c(); candidates <- 1:n_features; history <- list()
  cat(sprintf("\n--- Starting %s Selection ---\n", method))
  for (step in 1:max_k) {
    best_mi <- -Inf; best_feat <- NULL
    for (feat in candidates) {
      current_subset <- c(selected, feat); X_sub <- X[, current_subset, drop=FALSE]
      if (method == "PSS") mi_val <- get_mi_pss(X_sub, y, ell = param)
      else mi_val <- get_mi_knn(X_sub, y, k = param)
      if (!is.na(mi_val) && mi_val > best_mi) { best_mi <- mi_val; best_feat <- feat }
    }
    if (is.null(best_feat)) break
    selected <- c(selected, best_feat); candidates <- setdiff(candidates, best_feat)
    cat(sprintf("Step %d: Added V%d (MI: %.4f)\n", step, best_feat, best_mi))
    history[[step]] <- list(features = selected, mi = best_mi)
  }
  return(history)
}

cat("\n>>> Running PSS Selection (Check Speed!)...\n")
time_pss <- system.time({ res_pss <- run_forward_selection(X_train, y_train, "PSS", max_k=20, param=best_ell) })
print(time_pss)

cat("\n>>> Running KNN Selection...\n")
time_knn <- system.time({ res_knn <- run_forward_selection(X_train, y_train, "KNN", max_k=20, param=best_k) })
print(time_knn)

# ==============================================================================
# 6. Evaluation (SVM & Naive Bayes)
# ==============================================================================
evaluate_svm <- function(history, X_tr, y_tr, X_te, y_te) {
  accs <- c()
  steps <- 1:length(history)
  for (i in steps) {
    feats <- history[[i]]$features
    model <- svm(x = X_tr[, feats, drop=FALSE], y = y_tr, kernel = "radial")
    pred <- predict(model, X_te[, feats, drop=FALSE])
    accs <- c(accs, mean(pred == y_te))
  }
  return(accs)
}

evaluate_nb <- function(history, X_tr, y_tr, X_te, y_te) {
  accs <- c()
  for (i in 1:length(history)) {
    feats <- history[[i]]$features
    model <- naiveBayes(x = X_tr[, feats, drop=FALSE], y = y_tr)
    pred <- predict(model, X_te[, feats, drop=FALSE])
    accs <- c(accs, mean(pred == y_te))
  }
  return(accs)
}

cat("\nEvaluating PSS subsets...\n")
acc_pss_svm <- evaluate_svm(res_pss, X_train, y_train, X_test, y_test)
acc_pss_nb  <- evaluate_nb(res_pss, X_train, y_train, X_test, y_test)

cat("Evaluating KNN subsets...\n")
acc_knn_svm <- evaluate_svm(res_knn, X_train, y_train, X_test, y_test)
acc_knn_nb  <- evaluate_nb(res_knn, X_train, y_train, X_test, y_test)

# ==============================================================================
# 7. Visualization
# ==============================================================================

paper_theme <- theme_bw(base_size = 16) + 
  theme(
    panel.border = element_rect(colour = "black", fill = NA, linewidth = 1.5),
    panel.grid.minor = element_blank(),
    axis.title = element_text(face = "bold", size = 18, color = "black"),
    axis.text = element_text(face = "bold", size = 14, color = "black"),
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(face = "bold", size = 16),
    legend.key.width = unit(1.5, "cm")
  )

df_svm <- data.frame(k = 1:length(acc_pss_svm), PSS = acc_pss_svm, KNN = acc_knn_svm) %>% 
  pivot_longer(cols = c("PSS", "KNN"), names_to = "Method", values_to = "Accuracy")

p_svm <- ggplot(df_svm, aes(x = k, y = Accuracy, color = Method)) +
  geom_line(linewidth = 1.5) + 
  geom_point(size = 4) +
  labs(y = "Accuracy", x = "Selected Features") + 
  scale_color_manual(values = c("PSS" = "blue", "KNN" = "red")) +
  scale_x_continuous(breaks = seq(0, max(df_svm$k), by = 5)) +
  paper_theme

print(p_svm)

df_mi <- data.frame(
  Step = 1:length(res_pss),
  PSS_MI = unlist(lapply(res_pss, function(x) x$mi)),
  KNN_MI = unlist(lapply(res_knn, function(x) x$mi))
) %>% pivot_longer(cols = c("PSS_MI", "KNN_MI"), names_to = "Method", values_to = "MI_Value")

p_mi <- ggplot(df_mi, aes(x = Step, y = MI_Value, color = Method)) +
  geom_line(linewidth = 1.5) + 
  geom_point(size = 4) +
  scale_color_manual(
    values = c("PSS_MI" = "blue", "KNN_MI" = "red"),
    labels = c("KNN", "PSS") 
  ) +
  scale_x_continuous(breaks = seq(0, max(df_mi$Step), by = 5)) +
  labs(y = "Estimated MI", x = "Selected Features") + 
  paper_theme

print(p_mi)
