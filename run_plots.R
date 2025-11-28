#==============================================================================
# MASTER PLOTTING SCRIPT (FINAL) — PSS 파랑 고정 + 로그축 NA/0 정리 + 파일명 기반 보정
#==============================================================================

# --- 1) Libraries ---
library(tidyverse)
library(ggrepel)
library(scales)
library(stringr)

# --- 2) Load All Simulation Data (attach source filename) ---
csv_files <- list.files(pattern = "\\.csv$")
if (length(csv_files) == 0) {
  stop("No CSV files found in the directory. Please run the simulation scripts first.")
}
read_with_src <- function(f) read_csv(f, show_col_types = FALSE) %>% mutate(.src = f)
all_data_raw <- purrr::map_dfr(csv_files, read_with_src)
cat("Loaded CSVs:\n"); print(csv_files)

# --- 3) DATA CLEANING & CANONICALIZATION -------------------------------------

# 3-1) 안전한 컬럼명 정리
safe_rename <- function(df) {
  out <- df
  if ("Dimensions" %in% names(out) && !"dimension" %in% names(out)) out <- rename(out, dimension = Dimensions)
  if (!"dimension" %in% names(out) && "d" %in% names(out))           out <- rename(out, dimension = d)
  if ("N_Samples" %in% names(out) && !"n" %in% names(out))           out <- rename(out, n = N_Samples)
  if ("Samples" %in% names(out) && !"n" %in% names(out))             out <- rename(out, n = Samples)
  if ("N" %in% names(out) && !"n" %in% names(out))                   out <- rename(out, n = N)
  if (!"RMSE" %in% names(out) && "rmse" %in% names(out))             out <- rename(out, RMSE = rmse)
  out
}
all_data <- safe_rename(all_data_raw)

# 3-2) 파일명 기반 Method 추론 (Method 없거나 비어 있을 때)
infer_method_from_src <- function(src) {
  s <- tolower(src)
  case_when(
    str_detect(s, "pss")                         ~ "PSS",
    str_detect(s, "um_tksg|umtksg|uksg")         ~ "UM_tKSG",
    str_detect(s, "um_tkl|umtkl|ukl")            ~ "UM_tKL",
    str_detect(s, "cadee")                       ~ "CADEE",
    str_detect(s, "(^|[_\\-])ksg([^a-z]|$)")     ~ "KSG",
    str_detect(s, "(^|[_\\-])kl([^a-z]|$)")      ~ "KL",
    TRUE                                         ~ NA_character_
  )
}

# Method 컬럼 채우기
if (!"Method" %in% names(all_data)) {
  guess <- intersect(names(all_data), c("method","Estimator","Estimator_Name","Algo","Algorithm"))
  if (length(guess) >= 1) {
    all_data <- rename(all_data, Method = !!sym(guess[1]))
  } else {
    all_data <- all_data %>% mutate(Method = NA_character_)
  }
}
all_data <- all_data %>%
  mutate(Method = if_else(is.na(Method) | Method == "", infer_method_from_src(.src), Method))

# 3-3) Method 표준화 (UKL/UKSG, 하이픈/공백/대소문자 보정)
canon_method <- function(x) {
  x0 <- as.character(x) |> str_trim() |> str_replace_all("[-\\s]+", "_")
  xu <- toupper(x0)
  case_when(
    xu %in% c("PSS")                      ~ "PSS",
    xu %in% c("UM_TKL","UM_T_KL","UKL")   ~ "UM_tKL",
    xu %in% c("UM_TKSG","UM_T_KSG","UKSG")~ "UM_tKSG",
    xu %in% c("CADEE")                    ~ "CADEE",
    xu %in% c("KSG")                      ~ "KSG",
    xu %in% c("KL")                       ~ "KL",
    TRUE                                  ~ x0
  )
}
all_data <- all_data %>% mutate(Method = canon_method(Method))

# 3-4) Distribution 보정(파일명에서 유추; MVN → Normal)
infer_dist_from_src <- function(src) {
  s <- tolower(src)
  case_when(
    str_detect(s, "gamma")                        ~ "Gamma",
    str_detect(s, "mvn|normal|gauss|gaussian")    ~ "Normal",
    TRUE                                          ~ NA_character_
  )
}
if (!"Distribution" %in% names(all_data)) {
  all_data <- all_data %>% mutate(Distribution = infer_dist_from_src(.src))
} else {
  all_data <- all_data %>%
    mutate(Distribution = if_else(is.na(Distribution) | Distribution == "",
                                  infer_dist_from_src(.src), Distribution))
}
all_data <- all_data %>%
  mutate(Distribution = if_else(tolower(Distribution) %in% c("mvn","gaussian","normal"),
                                "Normal", Distribution))

# 3-5) 시간 컬럼 합치기 → Eval_Time_s
if (!"Eval_Time_s" %in% names(all_data)) all_data <- all_data %>% mutate(Eval_Time_s = NA_real_)
for (cand in c("PSS_OptEval_Time","PSS_TimeMean_sec","Time_s","EvalTime","time_sec","elapsed")) {
  if (cand %in% names(all_data)) all_data <- all_data %>% mutate(Eval_Time_s = coalesce(Eval_Time_s, .data[[cand]]))
}

# 3-6) l_opt 세팅 (PSS에서만)
if (!"l_opt" %in% names(all_data)) all_data <- all_data %>% mutate(l_opt = NA_integer_)
for (cand in c("Optimal_Param","ell","ell_star","best_l","lstar","opt_l","l")) {
  if (cand %in% names(all_data)) {
    all_data <- all_data %>%
      mutate(l_opt = if_else(Method == "PSS",
                             coalesce(l_opt, suppressWarnings(as.integer(.data[[cand]]))),
                             l_opt))
  }
}

# 3-7) 로그축 안전 보정 — Time: 0/NA → 작은 양수
min_pos_time <- all_data %>% filter(!is.na(Eval_Time_s) & Eval_Time_s > 0) %>%
  summarise(min_val = min(Eval_Time_s, na.rm = TRUE)) %>% pull(min_val)
time_eps <- if (length(min_pos_time) > 0 && is.finite(min_pos_time)) min(min_pos_time/10, 1e-12) else 1e-12
all_data <- all_data %>%
  mutate(Eval_Time_s = if_else(is.na(Eval_Time_s) | Eval_Time_s <= 0, time_eps, Eval_Time_s))

# 3-8) RMSE 생성/보정 — (MSE만 있으면 sqrt, 0/NA는 작은 양수)
if (!"RMSE" %in% names(all_data) && "MSE" %in% names(all_data)) {
  all_data <- all_data %>% mutate(RMSE = sqrt(pmax(MSE, 0)))
}
if (!"RMSE" %in% names(all_data)) {
  stop("No RMSE column found (and no MSE to derive it). Please include RMSE or MSE in CSVs.")
}
min_pos_rmse <- all_data %>% filter(!is.na(RMSE) & RMSE > 0) %>%
  summarise(min_val = min(RMSE, na.rm = TRUE)) %>% pull(min_val)
rmse_eps <- if (length(min_pos_rmse) > 0 && is.finite(min_pos_rmse)) min(min_pos_rmse/10, 1e-12) else 1e-12
all_data <- all_data %>%
  mutate(RMSE = if_else(is.na(RMSE) | RMSE <= 0, rmse_eps, RMSE))

# --- Debug: 반드시 PSS가 포함되어야 함 ----------------------------------------
cat("\n[DEBUG] Counts by Method after canon:\n")
print(all_data %>% count(Method, sort = TRUE))

# --- 4) COMMON AESTHETICS -----------------------------------------------------
# 두 번째(참고) 스크립트와 동일 미학 매핑
method_levels <- c("PSS", "UM_tKSG", "UM_tKL", "CADEE", "KSG", "KL")
all_data$Method <- factor(all_data$Method, levels = method_levels)

color_map <- c(
  PSS     = "blue",
  UM_tKSG = "green",
  UM_tKL  = "red",
  CADEE   = "#e7298a",
  KSG     = "#666666",
  KL      = "#a6cee3"
)
shape_map <- c(PSS = 16, UM_tKSG = 15, UM_tKL = 17, CADEE = 8, KSG = 1, KL = 2)

plot_theme <- theme_minimal(base_size = 12) +
  theme(
    axis.title.x = element_text(size = 20, face = "bold", margin = margin(t = 8)),
    axis.title.y = element_text(size = 20, face = "bold", margin = margin(r = 8)),
    axis.text    = element_text(size = 16),
    legend.title = element_text(size = 16, face = "bold"),
    legend.text  = element_text(size = 14),
    legend.position = "none",
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1.0)
  )

scale_color_fixed <- scale_color_manual(values = color_map, limits = method_levels)
scale_shape_fixed <- scale_shape_manual(values = shape_map, limits = method_levels)

#==============================================================================
#  GENERATE PLOTS (8개) — 각 플롯마다 drop_na로 NA 제거 + 로그축 안전
#==============================================================================

# helper: 안전 레이블 데이터 생성
mk_pss_labels <- function(df, mult = 1.7) {
  df %>%
    filter(Method == "PSS", !is.na(l_opt), is.finite(RMSE), RMSE > 0) %>%
    mutate(y_label = RMSE * mult)
}

# --- Plot 1: RMSE vs d (Normal, N=3000) ---
plot_data_1 <- all_data %>%
  filter(Distribution == "Normal", n == 3000) %>%
  drop_na(dimension, RMSE, Method)
pss_labels_1 <- mk_pss_labels(plot_data_1, 1.7)

p1 <- ggplot(plot_data_1, aes(x = dimension, y = RMSE, color = Method, group = Method)) +
  geom_line(linewidth = 1.5, alpha = 0.95, na.rm = TRUE) +
  geom_point(aes(shape = Method), size = 5, na.rm = TRUE) +
  scale_y_log10() + scale_color_fixed + scale_shape_fixed +
  labs(x = "d", y = "RMSE") + plot_theme +
  geom_text_repel(
    data = pss_labels_1,
    aes(x = dimension, y = y_label, label = paste0("ℓ*=", l_opt)),
    inherit.aes = FALSE, color = color_map[["PSS"]], size = 7,
    segment.color = NA, box.padding = 0.2, point.padding = 0.2, seed = 42
  )
print(p1); ggsave("figure2a_rmse_vs_d_normal.png", p1, width = 8, height = 6, dpi = 300)
cat("\nSaved: figure2a_rmse_vs_d_normal.png\n")

# --- Plot 2: Time vs d (Normal, N=3000) ---
plot_data_2 <- all_data %>%
  filter(Distribution == "Normal", n == 3000) %>%
  drop_na(dimension, Eval_Time_s, Method)

p2 <- ggplot(plot_data_2, aes(x = dimension, y = Eval_Time_s, color = Method, group = Method)) +
  geom_line(linewidth = 1.5, alpha = 0.95, na.rm = TRUE) +
  geom_point(aes(shape = Method), size = 5, na.rm = TRUE) +
  scale_y_log10() + scale_color_fixed + scale_shape_fixed +
  labs(x = "d", y = "Time (s)") + plot_theme
print(p2); ggsave("figure2b_time_vs_d_normal.png", p2, width = 8, height = 6, dpi = 300)
cat("Saved: figure2b_time_vs_d_normal.png\n")

# --- Plot 3: RMSE vs N (Normal, d=10) ---
plot_data_3 <- all_data %>%
  filter(Distribution == "Normal", dimension == 10) %>%
  drop_na(n, RMSE, Method)
pss_labels_3 <- mk_pss_labels(plot_data_3, 1.7)

p3 <- ggplot(plot_data_3, aes(x = n, y = RMSE, color = Method, group = Method)) +
  geom_line(linewidth = 1.5, alpha = 0.95, na.rm = TRUE) +
  geom_point(aes(shape = Method), size = 5, na.rm = TRUE) +
  scale_x_log10(labels = scales::comma) + scale_y_log10() +
  scale_color_fixed + scale_shape_fixed +
  labs(x = "N", y = "RMSE") + plot_theme +
  geom_text_repel(
    data = pss_labels_3,
    aes(x = n, y = y_label, label = paste0("ℓ*=", l_opt)),
    inherit.aes = FALSE, color = color_map[["PSS"]], size = 7,
    segment.color = NA, box.padding = 0.2, point.padding = 0.2, seed = 42
  )
print(p3); ggsave("figure2c_rmse_vs_n_normal.png", p3, width = 8, height = 6, dpi = 300)
cat("Saved: figure2c_rmse_vs_n_normal.png\n")

# --- Plot 4: Time vs N (Normal, d=10) ---
plot_data_4 <- all_data %>%
  filter(Distribution == "Normal", dimension == 10) %>%
  drop_na(n, Eval_Time_s, Method)

p4 <- ggplot(plot_data_4, aes(x = n, y = Eval_Time_s, color = Method, group = Method)) +
  geom_line(linewidth = 1.5, alpha = 0.95, na.rm = TRUE) +
  geom_point(aes(shape = Method), size = 5, na.rm = TRUE) +
  scale_x_log10(labels = scales::comma) + scale_y_log10() +
  scale_color_fixed + scale_shape_fixed +
  labs(x = "N", y = "Time (s)") + plot_theme
print(p4); ggsave("figure2d_time_vs_n_normal.png", p4, width = 8, height = 6, dpi = 300)
cat("Saved: figure2d_time_vs_n_normal.png\n")

# --- Plot 5: RMSE vs d (Gamma, N=30000) ---
plot_data_5 <- all_data %>%
  filter(Distribution == "Gamma", n == 30000) %>%
  drop_na(dimension, RMSE, Method)
pss_labels_5 <- mk_pss_labels(plot_data_5, 2.0)

p5 <- ggplot(plot_data_5, aes(x = dimension, y = RMSE, color = Method, group = Method)) +
  geom_line(linewidth = 1.5, alpha = 0.95, na.rm = TRUE) +
  geom_point(aes(shape = Method), size = 5, na.rm = TRUE) +
  scale_y_log10() + scale_color_fixed + scale_shape_fixed +
  labs(x = "d", y = "RMSE") + plot_theme +
  geom_text_repel(
    data = pss_labels_5,
    aes(x = dimension, y = y_label, label = paste0("ℓ*=", l_opt)),
    inherit.aes = FALSE, color = color_map[["PSS"]], size = 7,
    segment.color = NA, box.padding = 0.2, point.padding = 0.2, seed = 42
  )
print(p5); ggsave("figure3a_rmse_vs_d_gamma.png", p5, width = 8, height = 6, dpi = 300)
cat("Saved: figure3a_rmse_vs_d_gamma.png\n")

# --- Plot 6: Time vs d (Gamma, N=30000) ---
plot_data_6 <- all_data %>%
  filter(Distribution == "Gamma", n == 30000) %>%
  drop_na(dimension, Eval_Time_s, Method)

p6 <- ggplot(plot_data_6, aes(x = dimension, y = Eval_Time_s, color = Method, group = Method)) +
  geom_line(linewidth = 1.5, alpha = 0.95, na.rm = TRUE) +
  geom_point(aes(shape = Method), size = 5, na.rm = TRUE) +
  scale_y_log10() + scale_color_fixed + scale_shape_fixed +
  labs(x = "d", y = "Time (s)") + plot_theme
print(p6); ggsave("figure3b_time_vs_d_gamma.png", p6, width = 8, height = 6, dpi = 300)
cat("Saved: figure3b_time_vs_d_gamma.png\n")

# --- Plot 7: RMSE vs N (Gamma, d=5) ---
plot_data_7 <- all_data %>%
  filter(Distribution == "Gamma", dimension == 5) %>%
  drop_na(n, RMSE, Method)
pss_labels_7 <- mk_pss_labels(plot_data_7, 1.7)

p7 <- ggplot(plot_data_7, aes(x = n, y = RMSE, color = Method, group = Method)) +
  geom_line(linewidth = 1.5, alpha = 0.95, na.rm = TRUE) +
  geom_point(aes(shape = Method), size = 5, na.rm = TRUE) +
  scale_x_log10(breaks = plot_data_7$n %>% unique() %>% sort(), labels = scales::comma) +
  scale_y_log10() + scale_color_fixed + scale_shape_fixed +
  labs(x = "N", y = "RMSE") + plot_theme +
  geom_text_repel(
    data = pss_labels_7,
    aes(x = n, y = y_label, label = paste0("ℓ*=", l_opt)),
    inherit.aes = FALSE, color = color_map[["PSS"]], size = 7,
    segment.color = NA, box.padding = 0.2, point.padding = 0.2, seed = 42
  )
print(p7); ggsave("figure3c_rmse_vs_n_gamma.png", p7, width = 8, height = 6, dpi = 300)
cat("Saved: figure3c_rmse_vs_n_gamma.png\n")

# --- Plot 8: Time vs N (Gamma, d=5) ---
plot_data_8 <- all_data %>%
  filter(Distribution == "Gamma", dimension == 5) %>%
  drop_na(n, Eval_Time_s, Method)

p8 <- ggplot(plot_data_8, aes(x = n, y = Eval_Time_s, color = Method, group = Method)) +
  geom_line(linewidth = 1.5, alpha = 0.95, na.rm = TRUE) +
  geom_point(aes(shape = Method), size = 5, na.rm = TRUE) +
  scale_x_log10(breaks = plot_data_8$n %>% unique() %>% sort(), labels = scales::comma) +
  scale_y_log10() + scale_color_fixed + scale_shape_fixed +
  labs(x = "N", y = "Time (s)") + plot_theme
print(p8); ggsave("figure3d_time_vs_n_gamma.png", p8, width = 8, height = 6, dpi = 300)
cat("Saved: figure3d_time_vs_n_gamma.png\n")

cat("\nAll plots have been generated and saved successfully!\n")
