# Load necessary libraries
library(ggplot2)
library(dplyr)
library(tidyr)

# ETL
data <- read.csv("proportions_data.csv")

lookup <- c(
  "facet-1" = "Definiendum",
  "facet-2" = "Role",
  "facet-3" = "Permissible Action",
  "facet-4" = "Prohibitory Action",
  "facet-5" = "Condition",
  "facet-6" = "Evidence Object",
  "thematic" = "Fact Elements"
)

data <- data %>%
  mutate(experiment_id = ifelse(experiment_id %in% names(lookup), lookup[experiment_id], experiment_id))
colnames(data) <- gsub("\\.", "_", colnames(data))

# Summarize cumulative sum for each unique experiment_id and layer_depth
summary_data <- data %>%
  group_by(experiment_id, layer_depth) %>%
  summarize(
    mean_saulbase_mistral = mean(saulbase_mistral),
    mean_saulift_mistral = mean(saulift_mistral)
  )

# Plot for each experiment_id
p <- ggplot(summary_data, aes(x = layer_depth)) +
  geom_line(aes(y = mean_saulbase_mistral, color = "SaulLM-7B-Base")) +
  geom_line(aes(y = mean_saulift_mistral, color = "SaulLM-7B-Instruct")) +
  facet_wrap(~experiment_id, scales = "free", nrow = 2, ncol = 4) +
  labs(
    x = "Layer Depth", 
    y = "Mean Additional Attention to Legal Concepts\n(vs Mistral 7B)", 
    color = "Model"
  ) +
  theme_minimal() +
  scale_color_manual(values = c("SaulLM-7B-Base" = "blue", "SaulLM-7B-Instruct" = "red")) +
  geom_hline(yintercept = 0, color = "black", linetype = "solid", size = .5) +
  geom_smooth(aes(y = mean_saulbase_mistral, color = "SaulLM-7B-Base"), se = FALSE, linetype = "dashed", size = .5) +
  geom_smooth(aes(y = mean_saulift_mistral, color = "SaulLM-7B-Instruct"), se = FALSE, linetype = "dashed", size = .5) +
  theme(
    legend.position = c(0.99, 0.2),
    legend.box.background = element_rect(color = "black"),
    legend.justification = "right",
    strip.text = element_text(size = 14),
    axis.title.y = element_text(size = 14),
    axis.title.x = element_text(size = 16.5),
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12),
    legend.text = element_text(size = 14),
    legend.title = element_text(size = 14),
    legend.title.align = 0.5
  )
p
ggsave("plot_high_quality.png", plot = p, width = 10.5, height = 4, units = "in", dpi = 600)
