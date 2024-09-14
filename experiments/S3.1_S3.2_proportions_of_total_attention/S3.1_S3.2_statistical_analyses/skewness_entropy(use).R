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

# Compute skewness table
skewness_table <- data %>%
  group_by(experiment_id) %>%
  summarise(
    saulbase_mistral_skew = skewness(saulbase_mistral),
    saulift_mistral_skew = skewness(saulift_mistral)
  )

# Print skewness table
kable(skewness_table, caption = "Skewness for Saulbase Mistral and Saulift Mistral by Experiment ID")


# Compute kurtosis table
kurtosis_table <- data %>%
  group_by(experiment_id) %>%
  summarise(
    saulbase_mistral_kurtosis = kurtosis(saulbase_mistral),
    saulift_mistral_kurtosis = kurtosis(saulift_mistral)
  )

# Print kurtosis table
kable(kurtosis_table, caption = "Kurtosis for Saulbase Mistral and Saulift Mistral by Experiment ID")


# Compute entropy table
numBins = 11

entropy_table <- data %>%
  group_by(experiment_id) %>%
  summarise(
    saulbase_mistral_entropy = entropy(discretize(saulbase_mistral, numBins = numBins)),
    saulift_mistral_entropy = entropy(discretize(saulift_mistral, numBins = numBins))
  )

# Print entropy table
kable(entropy_table, caption = "Entropy for Saulbase Mistral and Saulift Mistral by Experiment ID")
