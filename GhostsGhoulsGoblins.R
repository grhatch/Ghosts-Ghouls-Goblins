library(tidyverse)
library(vroom)
library(tidymodels)
library(stacks)
library(ggmosaic)
library(embed)
library(ggplot2)
library(bonsai)
library(lightgbm)
library(parallel)
library(discrim)



# step_impute_mean()
# step_impute_median()
# step_impute_mode()

missing <- vroom("./STAT348/Ghosts-Ghouls-Goblins/ghouls-goblins-and-ghosts-boo/trainWithMissingValues.csv")
train <- vroom("./STAT348/Ghosts-Ghouls-Goblins/ghouls-goblins-and-ghosts-boo/train.csv/train.csv")
test <- vroom("./STAT348/Ghosts-Ghouls-Goblins/ghouls-goblins-and-ghosts-boo/test.csv/test.csv")


fill_recipe <- recipe(type~., data=missing) %>%
  #step_impute_mean() %>%
  #step_impute_mode()
  step_impute_bag(bone_length, impute_with = imp_vars(rotting_flesh, hair_length, has_soul, color, type), trees = 500) %>%
  step_impute_bag(rotting_flesh, impute_with = imp_vars(bone_length, hair_length, has_soul, color, type), trees = 500) %>%
  step_impute_bag(hair_length, impute_with = imp_vars(bone_length, rotting_flesh, has_soul, color, type), trees = 500) %>%
  step_impute_bag(has_soul, impute_with = imp_vars(rotting_flesh, hair_length, bone_length, color, type), trees = 500) %>%
  step_impute_bag(color, impute_with = imp_vars(rotting_flesh, hair_length, has_soul, bone_length, type), trees = 500)
  

prep <- prep(fill_recipe)
baked <- bake(prep, new_data = missing)

rmse_vec(train[is.na(missing)], baked[is.na(missing)]) #see how well imputed set does


###################
## Random Forest ##
###################

rf_model <- rand_forest(mtry = tune(),
                        min_n=tune(),
                        trees=1000) %>%
  set_engine("ranger") %>%
  set_mode("classification")


train_new <- train %>%
  mutate(color = as.factor(color))

rf_recipe <- recipe(type~., data=train_new) %>% 
  step_lencode_glm(color, outcome = vars(type))  #target encoding
  #step_normalize(all_numeric_predictors()) #make mean 0, sd=1


?step_lencode_mixed
#%>%
 # step_rm()
  #step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  #step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
  #step_dummy(all_nominal_predictors()) # dummy variable encoding
 # step_lencode_mixed(all_nominal_predictors(), outcome = vars(type)) %>% #target encoding
  #step_normalize(all_numeric_predictors())

#prep <- prep(rf_recipe)
#baked <- bake(prep, new_data = train)

# set up workflow
rf_wf <- workflow() %>%
  add_recipe(rf_recipe) %>%
  add_model(rf_model)

L <- 5
## Grid of values to tune over; these should be params in the model
rf_tuning_grid <- grid_regular(mtry(range = c(1,6)),
                               min_n(),
                               levels = L) ## L^2 total tuning possibilities

K <- 5
## Split data for CV
rf_folds <- vfold_cv(train_new, v = K, repeats=1)

## Run CV
rf_CV_results <- rf_wf %>%
  tune_grid(resamples=rf_folds,
            grid=rf_tuning_grid,
            metrics=metric_set(accuracy))

## Find Best Tuning Parameters
rf_bestTune <- rf_CV_results %>%
  select_best("accuracy")


## Finalize the Workflow & fit it
rf_final_wf <-
  rf_wf %>%
  finalize_workflow(rf_bestTune) %>%
  fit(data=train)

## Predict
rf_pred <- rf_final_wf %>%
  predict(new_data = test, type="class") %>%
  bind_cols(.,test) %>% # bind predictions with test data
  select(id, .pred_class) %>% # Just keep datetime and predictions
  rename(type = .pred_class) # rename pred to count (for submission to Kaggle)

vroom_write(rf_pred, "rf_ghost.csv", delim = ',')


####################
## Neural Network ##
####################

# Deep learning:
# good with image recognition
# Large Language models (Chat GPT)
# 
# All about transformations
# activation function = magic


nn_recipe <- recipe(type~., data=train) %>% 
  step_normalize(all_numeric_predictors()) %>%
  #step_lencode_mixed(all_nominal_predictors(), outcome = vars(type)) %>% #target encoding
  step_dummy(all_nominal_predictors()) %>%
  step_range(all_numeric_predictors(), min=0, max=1)

nn_model <- mlp(hidden_units = tune(),
                epochs = 50) %>%
  set_engine("nnet") %>% #verbose = 0 prints out less #or nnet
  set_mode("classification")

# set up workflow
nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model)

maxHiddenUnits <- 5
nn_tuneGrid <- grid_regular(hidden_units(range=c(1, maxHiddenUnits)),
                            levels=5)
K <- 5
## Split data for CV
nn_folds <- vfold_cv(train, v = K, repeats=1)

# Run CV
tuned_nn <- nn_wf %>%
  tune_grid(resamples=nn_folds,
            grid=nn_tuneGrid,
            metrics=metric_set(accuracy))

tuned_nn %>% collect_metrics() %>%
  filter(.metric=="accuracy") %>%
  ggplot(aes(x=hidden_units, y=mean)) + geom_line()


## Find Best Tuning Parameters
nn_bestTune <- tuned_nn %>%
  select_best("accuracy")


## Finalize the Workflow & fit it
nn_final_wf <-
  nn_wf %>%
  finalize_workflow(nn_bestTune) %>%
  fit(data=train)

## Predict
nn_pred <- nn_final_wf %>%
  predict(new_data = test, type="class") %>%
  bind_cols(.,test) %>% # bind predictions with test data
  select(id, .pred_class) %>% # Just keep datetime and predictions
  rename(type = .pred_class) # rename pred to count (for submission to Kaggle)

vroom_write(nn_pred, "nn_ghost.csv", delim = ',')




##############
## Boosting ##
##############

cluster <- makePSOCKcluster(4)
doParallel::registerDoParallel(cluster)


# target encode on color! make color a factor
# use bayes

boost_recipe <- recipe(type~., data=train) %>% 
  step_normalize(all_numeric_predictors()) %>%
  #step_lencode_mixed(all_nominal_predictors(), outcome = vars(type)) %>% #target encoding
  step_dummy(all_nominal_predictors()) %>%
  step_range(all_numeric_predictors(), min=0, max=1)
  



boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
  set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("classification")

bart_model <- parsnip::bart(trees=tune()) %>% # BART figures out depth and learn_rate
  set_engine("dbarts") %>% # might need to install
  set_mode("classification")

# set up workflow
boost_wf <- workflow() %>%
  add_recipe(boost_recipe) %>%
  add_model(boost_model)


boost_tuneGrid <- grid_regular(tree_depth(), trees(), learn_rate(),
                            levels=5)
K <- 5
## Split data for CV
boost_folds <- vfold_cv(train, v = K, repeats=1)

# Run CV
tuned_boost <- boost_wf %>%
  tune_grid(resamples=boost_folds,
            grid=boost_tuneGrid,
            metrics=metric_set(accuracy))


## Find Best Tuning Parameters
boost_bestTune <- tuned_boost %>%
  select_best("accuracy")


## Finalize the Workflow & fit it
boost_final_wf <-
  boost_wf %>%
  finalize_workflow(boost_bestTune) %>%
  fit(data=train)

## Predict
boost_pred <- boost_final_wf %>%
  predict(new_data = test, type="class") %>%
  bind_cols(.,test) %>% # bind predictions with test data
  select(id, .pred_class) %>% # Just keep datetime and predictions
  rename(type = .pred_class) # rename pred to count (for submission to Kaggle)

vroom_write(boost_pred, "boost_ghost.csv", delim = ',')

stopCluster(cluster)








#####################
#### NAIVE BAYES ####
#####################
nb_recipe <- recipe(type~., data=train_new) %>% 
  step_lencode_glm(color, outcome = vars(type)) %>%  #target encoding
  step_normalize(all_numeric_predictors()) #make mean 0, sd=1

nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_wf <- workflow() %>%
  add_recipe(nb_recipe) %>%
  add_model(nb_model)

tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels=3)

folds <- vfold_cv(train_new, v=15, repeats=15)

CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(accuracy))

bestTune <- CV_results %>%
  select_best("accuracy")

final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train_new)

ghost_predictions <- predict(final_wf,
                             new_data=test,
                             type="class") %>%
  bind_cols(., test) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)

vroom_write(x=ghost_predictions, file="./GGGBayesPreds2.csv", delim=",")
