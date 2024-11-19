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

#### NLP Function to Handle Multi-Class

m_nlp_fn <- function(parse_data.out){
  out <- parse_data.out %>% 
    unnest_tokens(output = token, 
                  input = text_clean, 
                  token = 'words',
                  stopwords = str_remove_all(stop_words$word, 
                                             '[[:punct:]]')) %>%
    mutate(token.lem = lemmatize_words(token)) %>%
    filter(str_length(token.lem) > 2) %>%
    count(.id, mclass, token.lem, name = 'n') %>%  # Changed bclass to mclass
    bind_tf_idf(term = token.lem, 
                document = .id,
                n = n) %>%
    pivot_wider(id_cols = c('.id', 'mclass'),  # Changed bclass to mclass
                names_from = 'token.lem',
                values_from = 'tf_idf',
                values_fill = 0)
  return(out)
}


#### Loading Data and applying NLP Function

source('scripts/preprocessing.R')
  
load('data/claims-clean-example.RData')
  
url <- 'https://raw.githubusercontent.com/pstat197/pstat197a/main/materials/activities/data/'
  
# load a few functions for the activity
source(paste(url, 'projection-functions.R', sep = ''))
claims <- m_nlp_fn(claims_clean)


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
y_train <- train_labels %>% pull(mclass)
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
cvout_multi

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

save(fit_reg_multi, file = "results/multi-class-model.RData")



  
