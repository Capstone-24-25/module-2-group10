library(tidyverse)
library(tidymodels)
library(modelr)
library(Matrix)
library(sparsesvd)
library(glmnet)
library(tidyverse)
library(tidymodels)
library(modelr)
library(Matrix)
library(sparsesvd)
library(glmnet)
library(pROC)

#### Loading Data and applying NLP Function

source('scripts/preprocessing.R')
  
load('data/claims-clean-example.RData')
  
claims <- m_nlp_fn(claims_clean)




#Binary Classification - From Task1-2.R.
set.seed(102722)

processed_data <- nlp_fn(claims_clean)

bin_training <- processed_data %>% select(-.id, -bclass)
bin_train_labels <- processed_data %>% select(.id, bclass)

training_vocab <- colnames(as.matrix(bin_training))

bin_proj_out <- projection_fn(.dtm = bin_training, .prop = 0.7)
bin_train_dtm_projected <- bin_proj_out$data

bin_proj_out$n_pc


bin_train <- bin_train_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(bin_train_dtm_projected)


bin_x_train <- bin_train %>% select(-bclass) %>% as.matrix()
bin_y_train <- bin_train_labels %>% pull(bclass)

alpha_enet <- 0.3
bin_fit_reg <- glmnet(x = bin_x_train, 
                  y = bin_y_train, 
                  family = 'binomial',
                  alpha = alpha_enet)

set.seed(102722)
bin_cvout <- cv.glmnet(x = bin_x_train, 
                   y = bin_y_train, 
                   family = 'binomial',
                   alpha = alpha_enet)

# store optimal strength
bin_lambda_opt <- bin_cvout$lambda.min

# view results
bin_cvout

final_binomial_model <- glmnet(x = bin_x_train, 
                               y = bin_y_train, 
                               family = 'binomial',
                               alpha = alpha_enet,
                               lambda = bin_lambda_opt)
final_binomial_model


#save the vocab
saveRDS(training_vocab, file='results/Training-vocabulary.rds')
saveRDS(bin_proj_out, file = 'results/binary-proj-out.rds')
saveRDS(final_binomial_model, file = 'results/binary-class-model.rds')




#Multi Class Model
#### Partitioning Data for Modeling

# partition data
set.seed(102722)
partitions <- claims %>% initial_split(prop = 0.8)

# separate DTM from labels
test_dtm <- testing(partitions) %>%
  select(-.id, -mclass)
test_labels <- testing(partitions) %>%
  select(.id, mclass)

# same, training set
train_dtm <- training(partitions) %>%
  select(-.id, -mclass)
train_labels <- training(partitions) %>%
  select(.id, mclass)


# find projections based on training data
proj_out <- projection_fn(.dtm = train_dtm, .prop = 0.7)
train_dtm_projected <- proj_out$data

proj_out$n_pc

train <- train_labels %>%
  transmute(mclass = factor(mclass)) %>%
  bind_cols(train_dtm_projected)

x_train <- train %>% select(-mclass) %>% as.matrix()
y_train_multi <- train_labels %>% pull(mclass)


#### Modeling

# fit enet model
alpha_enet <- 0.2
fit_reg_multi <- glmnet(x = x_train, 
                        y = y_train_multi, 
                        family = 'multinomial',
                        alpha = alpha_enet)

# choose a strength by cross-validation
set.seed(102722)
cvout_multi <- cv.glmnet(x = x_train, 
                         y = y_train_multi, 
                         family = 'multinomial',
                         alpha = alpha_enet)



# view results
cvout_multi$lambda.min

#0.01223359



# project test data onto PCs
test_dtm_projected <- reproject_fn(.dtm = test_dtm, proj_out)

# coerce to matrix
x_test <- as.matrix(test_dtm_projected)

preds_multi <- predict(fit_reg_multi, 
                       s = cvout_multi$lambda.min, 
                       newx = x_test,
                       type = 'response')

as_tibble(preds_multi[, , 1]) 


pred_class <- as_tibble(preds_multi[, , 1]) %>% 
  mutate(row = row_number()) %>%
  pivot_longer(-row, 
               names_to = 'label',
               values_to = 'probability') %>%
  group_by(row) %>%
  slice_max(probability, n = 1) %>%
  pull(label)

pred_tbl <- table(pull(test_labels, mclass), pred_class)

pred_tbl


#### Metrics

accuracy <- mean(pred_class == pull(test_labels, mclass))
print(paste("Accuracy: ", accuracy))

confusion_matrix <- table(pull(test_labels, mclass), pred_class)
print(confusion_matrix)

pred_probs <- as.data.frame(preds_multi[, , 1])  # Convert the matrix to a dataframe
true_labels <- as.factor(pull(test_labels, mclass))

# calculate AUC
auc_results <- multiclass.roc(true_labels, pred_probs)
print(auc_results)

# average AUC score (mean of AUC across all classes)
mean_auc <- mean(auc_results$auc)
print(paste("Average AUC: ", mean_auc))


#### Saving Model

saveRDS(proj_out, file = 'results/multi-proj-out.rds')
saveRDS(fit_reg_multi, file = "results/multi-class-model.rds")


  
