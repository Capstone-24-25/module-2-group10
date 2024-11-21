#require(tidyverse)
#require(keras)
#require(tensorflow)
load('data/claims-test.RData')
load('data/claims-raw.RData')
source('scripts/preprocessing.R')
library(glmnet)
#tf_model <- load_model_tf('results/example-model')

training_vocabulary <- readRDS('results/Training-vocabulary.rds') %>% as.vector()
bin_model <- readRDS('results/binary-class-model.rds')
bin_proj_out <- readRDS('results/binary-proj-out.rds')
multi_model <- readRDS('results/multi-class-model.rds')
multi_proj_out <- readRDS('results/multi-proj-out.rds')

# apply preprocessing pipeline
clean_df <- claims_test %>%
  parse_data() %>%
  select(.id, text_clean)

# grab input
#x <- clean_df %>%
#  pull(text_clean)

# compute predictions
#preds <- predict(tf_model, x) %>%
#  as.numeric()

#class_labels <- claims_raw %>% pull(bclass) %>% levels()

#pred_classes <- factor(preds > 0.5, labels = class_labels)

# export (KEEP THIS FORMAT IDENTICAL)
#pred_df <- clean_df %>%
#  bind_cols(bclass.pred = pred_classes) %>%
#  select(.id, bclass.pred)

#save(pred_df, file = 'results/example-preds.RData')

#preds




#preprocess the data
Processed_test_data <- nlp_function(clean_df)
id <- Processed_test_data %>% select(.id)
processed_data <- Processed_test_data %>% select(-.id) %>% as.matrix()


processed_data <- processed_data[, colnames(processed_data) %in% training_vocabulary, drop = FALSE]


missing_terms <- setdiff(training_vocabulary, colnames(processed_data))


missing_matrix <- matrix(0, nrow = nrow(processed_data), ncol = length(missing_terms))
colnames(missing_matrix) <- missing_terms


processed_data <- cbind(processed_data, missing_matrix)

processed_data <- processed_data[, training_vocabulary, drop = FALSE]





#Binary Classification
test_dtm_projected <- reproject_fn(.dtm = processed_data, bin_proj_out)

x_test <- as.matrix(test_dtm_projected)

preds <- predict(bin_model, 
                 newx = x_test,
                 type = 'response')

class_labels <- claims_raw %>% pull(bclass) %>% levels()

pred_classes <- factor(preds > 0.5, labels = class_labels)


preds_group10 <- cbind(id, pred_classes)




# Multi-class prediction
test_dtm_proj <- reproject_fn(.dtm = processed_data, multi_proj_out)
x_test_multi <- as.matrix(test_dtm_proj)

# Get predictions from the model
# For multi-class classification using glmnet models
# Assuming `multi_model` is a multnet object
# Assuming multi_model is your glmnet-based multi-class model
preds <- predict(multi_model, newx = x_test_multi, type = "response", s = 0.1)  # Adjust s (lambda) if needed

# Check the structure of the predictions
str(preds)  # Ensure that it is a matrix of probabilities

# Get the class labels from the raw data (assuming mclass contains the target variable)
class_labels <- claims_raw %>% pull(mclass) %>% levels()

# If preds is a matrix of probabilities, predict the class by selecting the highest probability for each instance
pred_class <- apply(preds, 1, function(x) class_labels[which.max(x)])

# Convert the predicted class labels into a factor
pred_class <- factor(pred_class, levels = class_labels)

# Combine the predictions with the original ids for output
m_preds_group10 <- cbind(id, pred_class)


preds_group10

dim(preds_group10)
dim(m_preds_group10)

library(dplyr)

# Join preds_group10 and m_preds_group10 by the .id column
joined_data <- preds_group10 %>%
  left_join(m_preds_group10, by = ".id") %>%
  rename(
    binaryclassprediction = pred_classes,
    multiclassprediction = pred_class
  )

joined_data

# Save the joined data to an RData file
save(joined_data, file = "preds-group10.RData")
