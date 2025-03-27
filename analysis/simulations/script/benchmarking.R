########################### Setup ###########################
library(ggplot2)
library(dplyr)
library(tidyr)
library(ggpubr)

color_palette <- c(
  "FiTree" = "#e8702a",
  "fitree" = "#e8702a",
  "freq" = "#7bc043",
  "SCIFIL" = "#0ea7b5",
  "scifil" = "#0ea7b5",
  "fitclone" = "#ffbe4f",
  "Diffusion" = "#0c457d",
  "Diffusion_subclone" = "#0c457d",
  "Diffusion_mutation" = "#0c457d"
)

# visualize the color palette
color_df <- data.frame(
  method = names(color_palette),
  color = color_palette
)
ggplot(color_df, aes(x = method, fill = method)) +
  geom_bar(stat = "identity", aes(y = 1), width = 0.5) +  # Use a dummy y value for equal bars
  scale_fill_manual(values = color_palette) +
  theme_minimal() +
  labs(title = "Color Palette Preview", x = "Method", y = "") +
  theme(axis.text.y = element_blank(),  # Hide y-axis text
        axis.ticks.y = element_blank(),
        axis.title.y = element_blank(),
        axis.text.x = element_blank(),
        panel.grid = element_blank())

########################### Evaluate absolute fitness ###########################

# Read data
fitness_df_all <- read.csv("analysis/simulations/data/fitness_masked_normal.csv") # Figure 3ab
# fitness_df_all <- read.csv("analysis/simulations/data/fitness_normal.csv") # Figure S12
# fitness_df_all <- read.csv("analysis/simulations/data/fitness_conditional_normal.csv") # Figure S13


(g_mae <- fitness_df_all %>%
  filter(metric == "mae") %>%
  mutate(method = factor(method, levels = c("FiTree", "Diffusion", "SCIFIL", "fitclone", "Frequency"))) %>%
  ggplot(aes(x = factor(N), y = value, fill = method)) +
  facet_grid(
    rows = vars(observed),
    cols = vars(n),
    labeller = labeller(
      n = function(x) paste0("n = ", x),
      observed = function(x) ifelse(x == "observed", "observed only", "observed + unobserved")
    )
  ) +
  geom_boxplot(alpha = 0.8) +
  scale_fill_manual(
    values = color_palette,
    labels = c("FiTree", "Diffusion", "SCIFIL", "fitclone", "Frequency")
  ) +
  theme(
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12),
    axis.title.y = element_text(size = 14),
    axis.title.x = element_text(size = 14),
    legend.text = element_text(size = 14),
    legend.title = element_blank(),
    legend.position = "top",
    strip.text.x = element_text(size = 14),
    strip.text.y = element_text(size = 12)
  ) +
  labs(
    x = "Number of Trees (N)",
    y = "mean absolute error"
  ) +
  scale_y_log10())

(g_sign <- fitness_df_all %>%
  filter(metric == "sign") %>%
  mutate(method = factor(method, levels = c("FiTree", "Diffusion", "SCIFIL", "fitclone", "Frequency"))) %>%
  ggplot(aes(x = factor(N), y = value, fill = method)) +
  facet_grid(
    rows = vars(observed),
    cols = vars(n),
    labeller = labeller(
      n = function(x) paste0("n = ", x),
      observed = function(x) ifelse(x == "observed", "observed only", "observed + unobserved")
    )
  ) +
  geom_boxplot(alpha = 0.8) +
  scale_fill_manual(
    values = color_palette,
    labels = c("FiTree", "Diffusion", "SCIFIL", "fitclone", "Frequency")
  ) +
  theme(
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12),
    axis.title.y = element_text(size = 14),
    axis.title.x = element_text(size = 14),
    legend.text = element_text(size = 14),
    legend.title = element_blank(),
    legend.position = "top",
    strip.text.x = element_text(size = 14),
    strip.text.y = element_text(size = 12)
  ) +
  labs(
    x = "Number of Trees (N)",
    y = "sign agreement"
  ) +
  ylim(0, 1))



########################### Evaluate spearman correlation ###########################

# read data
sc_df_all <- read.csv("analysis/simulations/data/sc_masked_normal.csv") # Figure 3c
# sc_df_all <- read.csv("analysis/simulations/data/sc_normal.csv") # Figure S12
# sc_df_all <- read.csv("analysis/simulations/data/sc_conditional_normal.csv") # Figure S13

(g_sc <- sc_df_all %>%
    filter(method != "Diffusion_subclone") %>%
    mutate(
      method = factor(method, 
      levels = c("fitree", "Diffusion_mutation", "scifil", "fitclone", "freq"))
    ) %>%
    ggplot(aes(x = factor(N), y = value, fill = method)) +
    facet_grid(
      rows = vars(observed),
      cols = vars(n),
      labeller = labeller(
        n = function(x) paste0("n = ", x),
        observed = function(x) ifelse(x == "observed", "observed only", "observed + unobserved")
      )
    ) +
    geom_boxplot(alpha = 0.8) +
    scale_fill_manual(
      values = color_palette,
      labels = c("FiTree", "Diffusion", "SCIFIL", "fitclone", "Frequency")
    ) +
    theme(
      axis.text.x = element_text(size = 12),
      axis.text.y = element_text(size = 12),
      axis.title.y = element_text(size = 14),
      axis.title.x = element_text(size = 14),
      legend.text = element_text(size = 14),
      legend.title = element_blank(),
      legend.position = "top",
      strip.text.x = element_text(size = 14),
      strip.text.y = element_text(size = 12)
    ) +
    labs(
      x = "Number of Trees (N)",
      y = "Spearman Correlation"
    ) +
    scale_y_continuous(
      breaks = seq(-1, 1, 0.25),
      limits = c(sc_df_all %>% pull(value) %>% range(na.rm=TRUE))
    ))


########################### Evaluate runtime ###########################

# read data
runtime_df_all <- read.csv("analysis/simulations/data/simulation_runtimes.csv")

runtime_df_all[runtime_df_all$rule == "rule_run_fitclone", "runtime"] <- runtime_df_all[runtime_df_all$rule == "rule_run_fitclone", "runtime"] * 100


# plot runtime (faceted by n and N and grouped by rule)
(g_runtime <- runtime_df_all %>%
  filter(rule %in% c(
    "rule_run_fitree_masked_normal",
    "rule_run_diffusion",
    "rule_run_SCIFIL",
    "rule_run_fitclone"
  )) %>%
  mutate(
    rule = factor(
      rule,
      levels = c(
        "rule_run_fitree_masked_normal",
        "rule_run_diffusion",
        "rule_run_SCIFIL",
        "rule_run_fitclone"
      )
    )
  ) %>%
  ggplot(aes(x = factor(N), y = runtime, fill = rule)) +
  facet_grid(
    cols = vars(n),
    labeller = labeller(
      n = function(x) paste0("n = ", x)
    )
  ) +
  geom_boxplot(alpha = 0.8) +
  scale_y_log10(
    breaks = c(1, 60, 1800, 3600, 14400, 86400, 172800),
    labels = c("1s", "1m", "30m", "1h", "4h", "1d", "2d")
  ) +
  scale_fill_manual(
    values = c(
      "rule_run_fitree_masked_normal" = "#e8702a",
      "rule_run_diffusion" = "#0c457d",
      "rule_run_SCIFIL" = "#0ea7b5",
      "rule_run_fitclone" = "#ffbe4f"
    ),
    labels = c("FiTree", "Diffusion", "SCIFIL", "fitclone")
  ) +
  theme(
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12),
    axis.title.y = element_text(size = 14),
    axis.title.x = element_text(size = 14),
    legend.text = element_text(size = 14),
    legend.title = element_blank(),
    legend.position = "top",
    strip.text.x = element_text(size = 14),
    strip.text.y = element_text(size = 12)
  ) +
  labs(
    x = "Number of Trees (N)",
    y = "Runtime"
  )) 
