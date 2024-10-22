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

# Summarize medians for each unique experiment_id and layer_depth
summary_data <- data %>%
  group_by(experiment_id, layer_depth) %>%
  summarize(
    median_saulbase_mistral = median(saulbase_mistral),
    median_saulift_mistral = median(saulift_mistral)
  )

# Plot for each experiment_id
ggplot(summary_data, aes(x = layer_depth)) +
  geom_line(aes(y = median_saulbase_mistral, color = "SaulLM-7B-Base")) +
  geom_line(aes(y = median_saulift_mistral, color = "SaulLM-7B-Instruct")) +
  facet_wrap(~experiment_id, scales = "free", nrow = 2, ncol = 4) +
  labs(
    x = "Layer Depth", 
    y = "Median Change in Attention", 
#    title = "Median Change in Proportion of Attention Directed to Legal Code and Facet Tokens\nAfter Continued Pretraining and Fine-Tuning of Mistral 7B on Legal Corpora \n(By Code Type and Layer Depth)",
    color = "Model"
    ) +
  theme_minimal() +
  scale_color_manual(values = c("SaulLM-7B-Base" = "blue", "SaulLM-7B-Instruct" = "red"))+
  geom_hline(yintercept = 0, color = "black", linetype = "solid", size = .5) +
  geom_smooth(aes(y = median_saulbase_mistral, color = "SaulLM-7B-Base"), se = FALSE, linetype = "dashed", size = .5) +
  geom_smooth(aes(y = median_saulift_mistral, color = "SaulLM-7B-Instruct"), se = FALSE, linetype = "dashed", size = .5) +
  theme(
    legend.position = c(0.98, 0.2),  # Adjust the position of the legend inside the plot area
    legend.box.background = element_rect(color = "black"),  # Optional: Add a border around the legend box
    legend.justification = "right"  # Align the legend to the right
  )
