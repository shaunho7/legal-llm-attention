# Load necessary libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(moments)
library(entropy)
library(knitr)

# Load the CSV file
data <- read.csv("proportions_data.csv")

lookup <- c(
  "facet-1" = "Definiendum",
  "facet-2" = "Role",
  "facet-3" = "Permissible Action",
  "facet-4" = "Prohibitory Action",
  "facet-5" = "Condition",
  "facet-6" = "Evidence_Object",
  "thematic" = "Theme Facts"
)

# Perform the replacement using dplyr's mutate function
data <- data %>%
  mutate(experiment_id = ifelse(experiment_id %in% names(lookup), lookup[experiment_id], experiment_id))

# Replace '.' with '_' in column names
colnames(data) <- gsub("\\.", "_", colnames(data))

# Set layout for two plots side by side and add outer margin for the supertitle
par(mfrow = c(1, 2), oma = c(0, 0, 3, 0))  # 3 lines of space at the top for the supertitle


# QQ-plot for saulbase_mistral
qqnorm(data$saulbase_mistral, main = "SaulLM-7B-Base over Mistral-7B", font.main = 1)
qqline(data$saulbase_mistral, col = "blue")

# QQ-plot for saulift_mistral
qqnorm(data$saulift_mistral, main = "SaulLM-7B-Instruct over Mistral-7B", font.main = 1)
qqline(data$saulift_mistral, col = "red")

# Add supertitle using mtext
mtext("Changes in Proportion of Total Attention to Legal Codes and Facets", outer = TRUE, side = 3, line = 0, cex = 1.4, font = 2)