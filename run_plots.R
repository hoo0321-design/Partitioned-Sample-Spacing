#==============================================================================
# MASTER PLOTTING SCRIPT (Robust Column Fix)
# - Fixes "Names must be unique" error by coalescing columns instead of renaming.
# - Displays plots in the R window AND saves them to files.
#==============================================================================

# --- 1. Setup & Data Loading -------------------------------------------------
library(tidyverse)
library(ggrepel)
library(scales)
library(stringr)

# Load all CSVs in the current directory
csv_files <- list.files(pattern = "\\.csv$")
if (length(csv_files) == 0) stop("No CSV files found. Please move Python output files here.")

cat("Loading", length(csv_files), "files...\n")

# Helper to read and attach filename
read_src <- function(f) read_csv(f, show_col_types = FALSE) %>% mutate(.src = basename(f))
raw_data <- map_dfr(csv_files, read_src)

# --- 2. Data Processing Pipeline ---------------------------------------------

# Helper function to merge synonymous columns (e.g., 'd' & 'Dimension')
unify_columns <- function(df) {
  # Safe accessor: returns the column as numeric if it exists, else NA
  g <- function(v) if(v %in% names(df)) as.numeric(df[[v]]) else rep(NA_real_, nrow(df))
  
  # Coalesce variations into master columns
  df$dimension   <- coalesce(g("dimension"), g("Dimensions"), g("d"))
  df$n           <- coalesce(g("n"), g("N"), g("N_Samples"), g("Samples"))
  df$rho         <- coalesce(g("rho"), g("Correlation"))
  df$RMSE        <- coalesce(g("RMSE"), g("rmse"))
  df$Eval_Time_s <- coalesce(g("Eval_Time_s"), g("Time_s"), g("time_sec"), 
                             g("PSS_OptEval_Time"), g("EvalTime"))
  return(df)
}

df <- raw_data %>%
  # 2.1 Unify Columns (Solves the "Names must be unique" error)
  unify_columns() %>%
  
  # 2.2 Infer 'Method' if missing
  mutate(
    Method = case_when(
      !is.na(Method) & Method != "" ~ Method,
      str_detect(tolower(.src), "pss") ~ "PSS",
      str_detect(tolower(.src), "um_tksg|umtksg|uksg") ~ "UM_tKSG",
      str_detect(tolower(.src), "um_tkl|umtkl|ukl") ~ "UM_tKL",
      str_detect(tolower(.src), "cadee") ~ "CADEE",
      str_detect(tolower(.src), "ksg") ~ "KSG",
      str_detect(tolower(.src), "kl") ~ "KL",
      TRUE ~ NA_character_
    )
  ) %>%
  
  # 2.3 Standardize Method Names & Distribution
  mutate(
    Method = case_when(
      toupper(str_replace_all(Method, "[-\\s]+", "_")) == "PSS" ~ "PSS",
      toupper(str_replace_all(Method, "[-\\s]+", "_")) %in% c("UM_TKL", "UKL") ~ "UM_tKL",
      toupper(str_replace_all(Method, "[-\\s]+", "_")) %in% c("UM_TKSG", "UKSG") ~ "UM_tKSG",
      toupper(str_replace_all(Method, "[-\\s]+", "_")) == "CADEE" ~ "CADEE",
      toupper(str_replace_all(Method, "[-\\s]+", "_")) == "KSG" ~ "KSG",
      toupper(str_replace_all(Method, "[-\\s]+", "_")) == "KL" ~ "KL",
      TRUE ~ Method
    ),
    Distribution = if_else(
      str_detect(tolower(coalesce(Distribution, .src)), "gamma"), "Gamma", "Normal"
    )
  ) %>%
  
  # 2.4 Cleanup & Safety
  mutate(
    Eval_Time_s = pmax(Eval_Time_s, 1e-12), # Prevent log(0) for time
    RMSE = pmax(RMSE, 1e-12),               # Prevent log(0) for RMSE
    l_opt = if_else(Method == "PSS", coalesce(as.integer(get0("l_opt", ifnotfound=NA)), 
                                              as.integer(get0("Optimal_Param", ifnotfound=NA))), NA_integer_)
  )

# --- 3. Visualization Configuration ------------------------------------------
method_levels <- c("PSS", "UM_tKSG", "UM_tKL", "CADEE", "KSG", "KL")
df$Method <- factor(df$Method, levels = method_levels)

cols   <- c(PSS="blue", UM_tKSG="green", UM_tKL="red", CADEE="#e7298a", KSG="#666666", KL="#a6cee3")
shapes <- c(PSS=16, UM_tKSG=15, UM_tKL=17, CADEE=8, KSG=1, KL=2)

common_theme <- theme_minimal(base_size = 12) +
  theme(
    axis.title = element_text(size = 20, face = "bold"),
    axis.text = element_text(size = 16),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1.0),
    legend.position = "none"
  )

# --- 4. Plotting Function ----------------------------------------------------
draw_and_save <- function(sub_data, x_var, y_var, filename, 
                          log_x = FALSE, use_comma_x = FALSE, label_mult = 1.7) {
  
  # Base Plot
  p <- ggplot(sub_data, aes(x = .data[[x_var]], y = .data[[y_var]], color = Method, group = Method)) +
    geom_line(linewidth = 1.5, alpha = 0.95) +
    geom_point(aes(shape = Method), size = 5) +
    scale_y_log10() +
    scale_color_manual(values = cols, limits = method_levels) +
    scale_shape_manual(values = shapes, limits = method_levels) +
    labs(x = ifelse(x_var=="dimension", "d", ifelse(x_var=="n", "N", expression(rho))), 
         y = ifelse(y_var=="RMSE", "RMSE", "Time (s)")) +
    common_theme
  
  # X-Axis Scaling
  if (log_x) {
    if (use_comma_x) {
      p <- p + scale_x_log10(labels = comma, breaks = sort(unique(sub_data[[x_var]])))
    } else {
      p <- p + scale_x_log10()
    }
  } else {
    p <- p + scale_x_continuous(breaks = sort(unique(sub_data[[x_var]])))
  }
  
  # PSS Labels (Only for RMSE plots)
  if (y_var == "RMSE") {
    lbl_data <- sub_data %>% 
      filter(Method == "PSS", !is.na(l_opt)) %>% 
      mutate(yl = RMSE * label_mult)
    
    p <- p + geom_text_repel(
      data = lbl_data, aes(x = .data[[x_var]], y = yl, label = paste0("ℓ*=", l_opt)),
      inherit.aes = FALSE, color = "blue", size = 7, seed = 42, segment.color = NA
    )
  }
  
  # Display & Save
  print(p)
  ggsave(filename, p, width = 8, height = 6, dpi = 300)
  cat("Displayed & Saved:", filename, "\n")
}

# --- 5. Execution ------------------------------------------------------------

# Filter Data Subsets
d_norm_d  <- df %>% filter(Distribution == "Normal", n == 3000) %>% drop_na(dimension)
d_norm_n  <- df %>% filter(Distribution == "Normal", dimension == 10) %>% drop_na(n)
d_gamm_d  <- df %>% filter(Distribution == "Gamma", n == 30000) %>% drop_na(dimension)
d_gamm_n  <- df %>% filter(Distribution == "Gamma", dimension == 5) %>% drop_na(n)

# New Datasets for Correlation (Rho)
d_norm_rho <- df %>% filter(Distribution == "Normal", dimension == 5, n == 20000) %>% drop_na(rho)
d_gamm_rho <- df %>% filter(Distribution == "Gamma", dimension == 7, n == 50000) %>% drop_na(rho)

# 1-4: Normal Distribution
draw_and_save(d_norm_d, "dimension", "RMSE",        "figure2a_rmse_vs_d_normal.png", log_x=TRUE, label_mult=1.7)
draw_and_save(d_norm_d, "dimension", "Eval_Time_s", "figure2b_time_vs_d_normal.png", log_x=TRUE)
draw_and_save(d_norm_n, "n",         "RMSE",        "figure2c_rmse_vs_n_normal.png", log_x=TRUE, use_comma_x=TRUE, label_mult=1.7)
draw_and_save(d_norm_n, "n",         "Eval_Time_s", "figure2d_time_vs_n_normal.png", log_x=TRUE, use_comma_x=TRUE)

# 5-8: Gamma Distribution
draw_and_save(d_gamm_d, "dimension", "RMSE",        "figure3a_rmse_vs_d_gamma.png",  log_x=TRUE, label_mult=2.0)
draw_and_save(d_gamm_d, "dimension", "Eval_Time_s", "figure3b_time_vs_d_gamma.png",  log_x=TRUE)
draw_and_save(d_gamm_n, "n",         "RMSE",        "figure3c_rmse_vs_n_gamma.png",  log_x=TRUE, use_comma_x=TRUE, label_mult=1.7)
draw_and_save(d_gamm_n, "n",         "Eval_Time_s", "figure3d_time_vs_n_gamma.png",  log_x=TRUE, use_comma_x=TRUE)

# 9-12: Correlation Robustness (Varying Rho)
draw_and_save(d_norm_rho, "rho", "RMSE",        "figure4a_rmse_vs_rho_normal.png", log_x=FALSE, label_mult=1.7)
draw_and_save(d_norm_rho, "rho", "Eval_Time_s", "figure4b_time_vs_rho_normal.png", log_x=FALSE)
draw_and_save(d_gamm_rho, "rho", "RMSE",        "figure4c_rmse_vs_rho_gamma.png",  log_x=FALSE, label_mult=1.7)
draw_and_save(d_gamm_rho, "rho", "Eval_Time_s", "figure4d_time_vs_rho_gamma.png",  log_x=FALSE)

cat("\nDone! All 12 plots displayed and saved.\n")
