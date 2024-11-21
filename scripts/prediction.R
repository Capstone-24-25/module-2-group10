#require(tidyverse)
#require(keras)
#require(tensorflow)
load('data/claims-test.RData')
load('data/claims-raw.RData')
source('scripts/preprocessing.R')

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





#Multi-class
test_dtm_proj <- reproject_fn(.dtm = processed_data, multi_proj_out)

x_test_multi <- as.matrix(test_dtm_proj)


#multi_model

