library(tidyverse)
library(tidymodels)
library(modelr)
library(Matrix)
library(sparsesvd)
library(glmnet)

source('scripts/preprocessing.R')

load('data/claims-clean-example.RData')

url <- 'https://raw.githubusercontent.com/pstat197/pstat197a/main/materials/activities/data/'

# load a few functions for the activity
source(paste(url, 'projection-functions.R', sep = ''))


set.seed(102722)

processed_data <- nlp_fn(claims_clean)

partitions <- processed_data %>% initial_split(prop = 0.8)


# separate DTM from labels
test_dtm <- testing(partitions) %>%
  select(-.id, -bclass)
test_labels <- testing(partitions) %>%
  select(.id, bclass)

# same, training set
train_dtm <- training(partitions) %>%
  select(-.id, -bclass)
train_labels <- training(partitions) %>%
  select(.id, bclass)


proj_out <- projection_fn(.dtm = train_dtm, .prop = 0.7)
train_dtm_projected <- proj_out$data

proj_out$n_pc


train <- train_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(train_dtm_projected)

#this has an error
#fit <- glm(bclass~.,data=train, family='binomial')


x_train <- train %>% select(-bclass) %>% as.matrix()
y_train <- train_labels %>% pull(bclass)

alpha_enet <- 0.3
fit_reg <- glmnet(x = x_train, 
                  y = y_train, 
                  family = 'binomial',
                  alpha = alpha_enet)

set.seed(102722)
cvout <- cv.glmnet(x = x_train, 
                   y = y_train, 
                   family = 'binomial',
                   alpha = alpha_enet)

# store optimal strength
lambda_opt <- cvout$lambda.min

# view results
cvout

#to test

test_dtm_projected <- reproject_fn(.dtm = test_dtm, proj_out)

x_test <- as.matrix(test_dtm_projected)

preds <- predict(fit_reg, 
                 s = lambda_opt, 
                 newx = x_test,
                 type = 'response')


pred_df <- test_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(pred = as.numeric(preds)) %>%
  mutate(bclass.pred = factor(pred > 0.5, 
                              labels = levels(bclass)))

# define classification metric panel 
panel <- metric_set(sensitivity, 
                    specificity, 
                    accuracy, 
                    roc_auc)

# compute test set accuracy
pred_df %>% panel(truth = bclass, 
                  estimate = bclass.pred, 
                  pred, 
                  event_level = 'second')

#Including headers
#sensitivity binary         0.813
#specificity binary         0.769
#accuracy    binary         0.793
#roc_auc     binary         0.876


#Not including headers
#sensitivity binary         0.842
#specificity binary         0.740
#accuracy    binary         0.797
#roc_auc     binary         0.857

#These metrics are ran from identical code except the fact that the 
#data includes the headers. From this data, we can see that there isn't much of a 
# difference between including the headers and not including them. So to 
#answer the question, No, binary class predictions aren't improved when using 
#Logistic PCR.






#Task2

claims_bigrams <- claims_clean %>% 
  unnest_tokens(output = token, 
                input = text_clean, 
                token = 'ngrams',
                n = 2,
                stopwords = str_remove_all(stop_words$word, 
                                           '[[:punct:]]')) %>%
  mutate(token.lem = lemmatize_words(token)) %>%
  filter(str_length(token.lem) > 2) %>%
  count(.id, bclass, token.lem, name = 'n') %>%
  bind_tf_idf(term = token.lem, 
              document = .id,
              n = n) %>%
  pivot_wider(id_cols = c('.id', 'bclass'),
              names_from = 'token.lem',
              values_from = 'tf_idf',
              values_fill = 0)




set.seed(102722)
bi_partitions <- claims_bigrams %>% initial_split(prop = 0.8)

#training data and labels  
bi_train_dtm <- training(bi_partitions) %>%
  select(-.id, -bclass)
bi_train_labels <- training(bi_partitions) %>%
  select(.id, bclass)

#test data and labels

bi_test_dtm <- testing(bi_partitions) %>%
  select(-.id, -bclass)
bi_test_labels <- testing(bi_partitions) %>%
  select(.id, bclass)


bi_proj_out <- projection_fn(.dtm=bi_train_dtm, .prop = 0.7)
bi_train_dtm_proj <- bi_proj_out$data

bi_proj_out$n_pc

bi_train <- bi_train_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(bi_train_dtm_proj)


x_bi_train <- bi_train %>% select(-bclass) %>% as.matrix()
y_bi_train <- bi_train_labels %>% pull(bclass)

alpha_enet <- 0.3
fit_reg_bi <- glmnet(x = x_bi_train, 
                  y = y_bi_train, 
                  family = 'binomial',
                  alpha = alpha_enet)

set.seed(102722)
bi_cvout <- cv.glmnet(x = x_bi_train, 
                   y = y_bi_train, 
                   family = 'binomial',
                   alpha = alpha_enet)

bi_lambda_opt <- bi_cvout$lambda.min


bi_test_dtm_projected <- reproject_fn(.dtm = bi_test_dtm, bi_proj_out)

x_bi_test <- as.matrix(bi_test_dtm_projected)

bi_preds <- predict(fit_reg_bi, 
                 s = bi_lambda_opt, 
                 newx = x_bi_test,
                 type = 'response')



bi_pred_df <- bi_test_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(pred = as.numeric(bi_preds)) %>%
  mutate(bclass.pred = factor(pred > 0.5, 
                              labels = levels(bclass)))

# define classification metric panel 
bi_panel <- metric_set(sensitivity, 
                    specificity, 
                    accuracy, 
                    roc_auc)

# compute test set accuracy
bi_pred_df %>% panel(truth = bclass, 
                  estimate = bclass.pred, 
                  pred, 
                  event_level = 'second')

#1 sensitivity binary        0.987 
#2 specificity binary        0.0422
#3 accuracy    binary        0.589 
#4 roc_auc     binary        0.750 

# Based on these results, we can see that the accuracy is a lot lower than using 
# unigrams which makes us think that the bigrams do not capture additional information 
# about the claims status of a page. This could be due to many reasons such as sparsity 
# of bigram features, overfitting, loss of contextual information, and increased model 
#complexity,






