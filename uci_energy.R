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

script_file <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_file)) {
  dirname(normalizePath(sub("^--file=", "", script_file[[1]]), mustWork = FALSE))
} else {
  getwd()
}
pss_source <- file.path(script_dir, "PSS", "PSS_entropy.R")
if (!file.exists(pss_source)) {
  pss_source <- file.path("PSS", "PSS_entropy.R")
}
source(pss_source)

set.seed(42)

# ==============================================================================
# 1. UCI Appliances Energy data load (N=19735, d=26)
# ==============================================================================
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv"

out_dir <- file.path("results", "uci_energy_feature_selection_latest")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
data_cache <- file.path(out_dir, "energydata_complete.csv")

if (file.exists(data_cache)) {
  cat("Loading cached Appliances Energy dataset...\n")
  data <- read.csv(data_cache)
} else {
  cat("Downloading Appliances Energy dataset (approx. 2.5MB)...\n")
  data <- read.csv(url)
  write.csv(data, data_cache, row.names = FALSE)
}

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
# 2. CV Functions
# ==============================================================================
cv_tune_pss <- function(X, ell_candidates, n_folds = 3,
                        occupancy_min = 10,
                        stable_coverage_tau = 0.99,
                        seed = 42) {
  cat("Tuning PSS ell with stable-coverage constrained CV:\n")
  sel <- select_l_via_sc_cv_pss(
    X,
    l_range = ell_candidates,
    n_folds = n_folds,
    occupancy_min = occupancy_min,
    tau = stable_coverage_tau,
    one_se_rule = FALSE,
    seed = seed
  )
  print(sel$cv_table[, c(
    "ell",
    "cv_score",
    "validation_nll",
    "stable_validation_coverage",
    "validation_cell_size_median"
  )])
  sel
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
time_tune_pss <- system.time({
  pss_cv <- cv_tune_pss(
    X_train,
    ell_cands,
    n_folds = 3,
    occupancy_min = 10,
    stable_coverage_tau = 0.99,
    seed = 42
  )
  best_ell <- pss_cv$ell_star
})
cat("PSS Tuning Time:", time_tune_pss[3], "s\n")

cat("\n>>> Tuning KNN (Subset) <<<\n")
time_tune_knn <- system.time({ best_k <- cv_tune_knn(X_tune, k_cands, n_folds = 3) })
cat("KNN Tuning Time:", time_tune_knn[3], "s\n")

cat("Selected Ell:", best_ell, "| Selected K:", best_k, "\n")

# ==============================================================================
# 4. Core Estimators
# ==============================================================================
# calculate_entropy_pss_d() is sourced from PSS/PSS_entropy.R so this downstream
# experiment uses the same shared PSS estimator as the main benchmarks.

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

# ==============================================================================
# 8. Save reproducible outputs
# ==============================================================================

history_to_df <- function(history, method) {
  do.call(rbind, lapply(seq_along(history), function(i) {
    data.frame(
      Method = method,
      Step = i,
      Feature_Index = history[[i]]$features[[length(history[[i]]$features)]],
      Selected_Features = paste(history[[i]]$features, collapse = ","),
      MI = history[[i]]$mi,
      stringsAsFactors = FALSE
    )
  }))
}

selection_df <- bind_rows(
  history_to_df(res_pss, "PSS"),
  history_to_df(res_knn, "KNN")
)

accuracy_df <- bind_rows(
  data.frame(Method = "PSS", Classifier = "SVM", Step = seq_along(acc_pss_svm), Accuracy = acc_pss_svm),
  data.frame(Method = "PSS", Classifier = "NaiveBayes", Step = seq_along(acc_pss_nb), Accuracy = acc_pss_nb),
  data.frame(Method = "KNN", Classifier = "SVM", Step = seq_along(acc_knn_svm), Accuracy = acc_knn_svm),
  data.frame(Method = "KNN", Classifier = "NaiveBayes", Step = seq_along(acc_knn_nb), Accuracy = acc_knn_nb)
)

params_df <- data.frame(
  N_Train = nrow(X_train),
  N_Test = nrow(X_test),
  Dimensions = ncol(X_train),
  Best_Ell = best_ell,
  Best_K = best_k,
  PSS_CV_Method = pss_cv$selection_rule,
  PSS_CV_Stable_Coverage_Tau = pss_cv$stable_coverage_min,
  PSS_CV_Occupancy_Min = pss_cv$occupancy_min,
  PSS_Tuning_Time_s = unname(time_tune_pss[3]),
  KNN_Tuning_Time_s = unname(time_tune_knn[3]),
  PSS_Selection_Time_s = unname(time_pss[3]),
  KNN_Selection_Time_s = unname(time_knn[3]),
  stringsAsFactors = FALSE
)

write.csv(selection_df, file.path(out_dir, "feature_selection_history.csv"), row.names = FALSE)
write.csv(accuracy_df, file.path(out_dir, "feature_selection_accuracy.csv"), row.names = FALSE)
write.csv(df_mi, file.path(out_dir, "feature_selection_mi_long.csv"), row.names = FALSE)
write.csv(params_df, file.path(out_dir, "feature_selection_params.csv"), row.names = FALSE)
write.csv(pss_cv$cv_table, file.path(out_dir, "pss_stable_coverage_cv_table.csv"), row.names = FALSE)

ggsave(file.path(out_dir, "figure6a_svm_accuracy.png"), p_svm, width = 7.2, height = 5.0, dpi = 300)
ggsave(file.path(out_dir, "figure6a_svm_accuracy.pdf"), p_svm, width = 7.2, height = 5.0)
ggsave(file.path(out_dir, "figure6b_estimated_mi.png"), p_mi, width = 7.2, height = 5.0, dpi = 300)
ggsave(file.path(out_dir, "figure6b_estimated_mi.pdf"), p_mi, width = 7.2, height = 5.0)

cat("\nSaved UCI Energy feature-selection outputs to:", out_dir, "\n")
cat("\nBest SVM accuracy by method:\n")
print(accuracy_df %>%
  filter(Classifier == "SVM") %>%
  group_by(Method) %>%
  summarise(Best_Accuracy = max(Accuracy), Best_Step = Step[which.max(Accuracy)], .groups = "drop"))
