# Essential libraries for data manipulation and modeling
library(tidyverse)  # for data wrangling and manipulation (dplyr, tidyr, etc.)
library(tidymodels) # for model workflows and splitting data
library(glmnet)     # for fitting elastic net models (used for the multi-class model)
library(Matrix)     # for handling sparse matrices (if necessary)
library(sparsesvd)  # for singular value decomposition (if used for dimensionality reduction)
multi_model <- readRDS('results/multi-class-model.rds')
multi_proj_out <- readRDS('results/multi-proj-out.rds')


# Load the pre-trained model
multi_model <- readRDS('results/multi-class-model.rds')

# Preprocess the test data (same as in the model, using the relevant NLP function)
Processed_test_data <- nlp_function(clean_df)  # clean_df should be your test data
id <- Processed_test_data %>% select(.id)  # Keep the .id for later reference
processed_data <- Processed_test_data %>% select(-.id) %>% as.matrix()

# Ensure the test data has the same features as the training data
# Create a matrix with zeroes for missing terms
missing_terms <- setdiff(training_vocabulary, colnames(processed_data))

if (length(missing_terms) > 0) {
  # Create a matrix with 0 for missing terms
  missing_matrix <- matrix(0, nrow = nrow(processed_data), ncol = length(missing_terms))
  colnames(missing_matrix) <- missing_terms
  
  # Add the missing terms to the processed data
  processed_data <- cbind(processed_data, missing_matrix)
}

# Reorder columns to match training_vocabulary
processed_data <- processed_data[, training_vocabulary, drop = FALSE]

# Project the test data onto the same principal components as the training data
test_dtm_projected <- reproject_fn(.dtm = processed_data, multi_proj_out)  # Use projection_fn from your model

# Convert the projected test data to a matrix for predictions
x_test <- as.matrix(test_dtm_projected)

# Make predictions using the pre-trained multi-class model
preds_multi <- predict(multi_model, 
                       newx = x_test,
                       type = 'response')

# Extract the predicted class labels based on the highest probability for each sample
pred_class <- as_tibble(preds_multi[, , 1]) %>% 
  mutate(row = row_number()) %>%
  pivot_longer(-row, 
               names_to = 'label',
               values_to = 'probability') %>%
  group_by(row) %>%
  slice_max(probability, n = 1) %>%
  pull(label)

# Prepare the prediction results
pred_df <- id %>%
  bind_cols(mclass.pred = pred_class) %>%
  select(.id, mclass.pred)

# Save the predictions (or use it for further analysis)
save(pred_df, file = 'results/multi-class-preds.RData')

# View the predictions (optional)
print(pred_df)

View(pred_df)

# Check the first few rows of processed data
head(Processed_test_data)
