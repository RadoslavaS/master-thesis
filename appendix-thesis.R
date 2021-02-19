##############################################################
############## Loading of the packages used ##################
##############################################################

library(readr)
library(rpart)
library(e1071) 
library(rBayesianOptimization) 
library(MlBayesOpt) 
library(randomForest)
library(glmnet)
library(keras)
library(tensorflow)
library(caret)
library(mixtools)
import::from(zeallot, "%<-%")

##############################################################
################# Loading of the data ########################
##############################################################
kddcup_names = read_delim("kddcup.names.txt", 
              ":", escape_double = FALSE, col_names = FALSE, 
              trim_ws = TRUE, skip = 1)
variable_type = kddcup_names[,2]
kddcup_names = rbind(kddcup_names[,1], "attack_type", 
                     "difficulty_level")
train_kdd <- read_csv("KDDTrain+.txt",
                      col_names = t(kddcup_names)) 
test_kdd <- read_csv("KDDTest+.txt",
                     col_names = t(kddcup_names))

############# binary and attack labels #######################
y_train = ifelse(train_kdd$attack_type=="normal", 0, 1)
y_test = ifelse(test_kdd$attack_type=="normal", 0, 1)
label_train = train_kdd$attack_type
label_test = test_kdd$attack_type

############### original feature space #######################
x_train_orig = data.frame(train_kdd[,-c(42,43)])
x_test_orig = data.frame(test_kdd[,-c(42,43)])
for (i in 1:41){
  vartype = variable_type[i,]
  if(vartype=="symbolic."){
    x_train_orig[,i] = as.factor(x_train_orig[,i])
    x_test_orig[,i] = as.factor(x_test_orig[,i])
    }
}

################# dummy variables ############################ 
dmy <- dummyVars(" ~ .", data = rbind(train_kdd, 
            test_kdd)[,which(variable_type=="symbolic.")], 
            fullRank = T)
dmy_pred <- data.frame(predict(dmy, 
              newdata = rbind(train_kdd, test_kdd)))

x_train_119 = data.frame(cbind(dmy_pred[1:nrow(train_kdd),], 
  train_kdd[,-c(which(variable_type=="symbolic."), 42,43)]))
x_test_119 = data.frame(cbind(
  dmy_pred[(nrow(train_kdd)+1):nrow(dmy_pred),], 
  test_kdd[,-c(which(variable_type=="symbolic."), 42,43)]))

################## minmax transformation #####################
pp_minmax = preProcess(x_train_119, 
                method = c("range"))
x_train_minmax = predict(pp_minmax, x_train_119)
x_test_minmax = predict(pp_minmax, x_test_119)

########### for feed forward neural network ##################
pp_zscore = preProcess(x_train_119, 
                method = c("center", "scale", "zv"))
#removed: num_outbound_cmds
x_train_zscore = predict(pp_zscore, x_train_119)
x_test_zscore = predict(pp_zscore, x_test_119)

################# for Autoencoder ############################
dmy_protocol <- dummyVars(" ~ .", 
                          data = rbind(train_kdd, 
                          test_kdd)[,"protocol_type"], 
                          fullRank = F)
protocol = data.frame(predict(dmy_protocol, 
                              newdata = rbind(train_kdd, 
                                              test_kdd)))

dmy_service <- dummyVars(" ~ .", 
                         data = rbind(train_kdd, 
                         test_kdd)[,"service"], 
                         fullRank = F)
service = data.frame(predict(dmy_service, 
                             newdata = rbind(train_kdd, 
                                             test_kdd)))

dmy_flag <- caret::dummyVars(" ~ .", 
                             data = rbind(train_kdd, 
                             test_kdd)[,"flag"], 
                             fullRank = F)
flag = data.frame(predict(dmy_flag, 
                          newdata = rbind(train_kdd, 
                                          test_kdd)))

binary_variables = which(sapply(1:41, function(i) 
  length(table(train_kdd[,i])))==2)
binary_all = rbind(train_kdd, test_kdd)[,binary_variables]

continuous_all = rbind(train_kdd, test_kdd)[, -c(which(
  colnames(train_kdd)%in%c("protocol_type", 
                           "service", "flag")), 
  binary_variables, 42,43)]

contin_train0 =  train_kdd[which(y_train==0), -c(which(
  colnames(train_kdd)%in%c("protocol_type", 
                           "service", "flag")), 
  binary_variables, 42,43)]
contin_train0 = data.frame(contin_train0)

## minmax transformation of contin. var-s ##
## done by only benign samples
pp = caret::preProcess(contin_train0, 
                       method = c("range", "zv"))
#removed: wrong_fragment, num_outbound_cmds
minmax_all = predict(pp, continuous_all)

x_train_forAE = data.frame(cbind(protocol, service, 
 flag, binary_all,minmax_all)[1:nrow(train_kdd),])
x_test_forAE = data.frame(cbind(protocol, service, 
 flag, binary_all, minmax_all)[(nrow(
   train_kdd)+1):nrow(flag),])

##############################################################
############### Evaluation measures ##########################
##############################################################

eval.metrics = function(predicted_labels, true_labels){
  
  conf_matrix_freq = table(true_labels, predicted_labels)
  conf_matrix_perc = conf_matrix_freq/sum(conf_matrix_freq)
  
  # positive = attack 
  # negative = normal behavior
  
  TN = conf_matrix_freq[1,1] 
  FP = conf_matrix_freq[1,2] 
  FN = conf_matrix_freq[2,1] 
  TP = conf_matrix_freq[2,2] 
  
  ACC = (TP + TN) / (TP+TN+FP+FN)
  TPR = TP / (TP + FN)  
  TNR = TN / (TN + FP)  
  BACC = (TPR + TNR)/2
  FAR = FP / (TN + FP)  # 1 - TNR
  PPV = TP / (TP + FP)
  NPV = TN / (TN + FN)
  kappa = fmsb::Kappa.test(x=c(predicted_labels), 
                           y=true_labels)
  
  output = list("conf_matrix_freq" = conf_matrix_freq, 
                "conf_matrix_perc" = conf_matrix_perc,
                "ACC" = ACC, "BACC" = BACC,
                "TPR" = TPR, "TNR" = TNR, 
                "FAR" = FAR, "NPV" = NPV, 
                "PPV" = PPV, "K" = kappa)
  return(output)
}

##############################################################
################## Decision Tree #############################
##############################################################

c(x_train, x_test) %<-% list(x_train_orig, x_test_orig)

###################### CV function ###########################
cv.dt <- function(cp.par, minsplit.par) {
  cv <- tune.rpart(as.factor(y_train)~., 
                   data = cbind(x_train, y_train), 
                   cp=cp.par, 
                   minsplit = minsplit.par)
  measure = 1 - cv$best.performance #performance=misclass.err 
  list(Score = measure, 
       Pred = 0)}

################## Bayesian Optimization #####################
set.seed(21)
dt_BO <- BayesianOptimization(cv.dt,
            bounds = list(cp.par = c(0,0.05), 
                          minsplit.par = c(1L, 200L)), 
            init_grid_dt = NULL, 
            init_points = 5, 
            n_iter = 4,
            acq = "ei", 
            eps = 0.5,
            verbose = TRUE)

################## Final Decision Tree #######################
dt_model <- rpart(as.factor(y_train)~., 
             data = cbind(x_train, y_train),
             cp = dt_BO$Best_Par["cp.par"], 
             minsplit = dt_BO$Best_Par["minsplit.par"]) 
dt_predict = predict(dt_model, x_test, type="class") 
dt_eval = eval.metrics(as.numeric(dt_predict)-1, y_test)

dt_predict_train = predict(dt_model, x_train, type="class") 
dt_eval_train = eval.metrics(as.numeric(dt_predict_train)-1,
                             y_train)

##############################################################
################## Random Forest #############################
##############################################################

c(x_train, x_test) %<-% list(x_train_orig, x_test_orig)

#### variable service has 70 levels
##split 3rd variable to 2, bcs 53 levels is max for RF fun.
train_x3_levels = levels(x_train[,3])
x_train[,"x3_1"] = sapply(1:nrow(x_train), function(iter){
  ifelse(x_train[iter,3]%in%train_x3_levels[1:35], 
         as.character(x_train[iter,3]), "not")})
x_train[,"x3_2"] = sapply(1:nrow(x_train), function(iter){
  ifelse(x_train[iter,3]%in%train_x3_levels[36:70], 
         as.character(x_train[iter,3]), "not")})
x_train = x_train[,-3]
x_train[,"x3_1"] = as.factor(x_train[,"x3_1"])
x_train[,"x3_2"] = as.factor(x_train[,"x3_2"])
## the same for test set
x_test[,"x3_1"] = sapply(1:nrow(x_test), function(iter){
  ifelse(x_test[iter,3]%in%train_x3_levels[1:35], 
         as.character(x_test[iter,3]), "not")})
x_test[,"x3_2"] = sapply(1:nrow(x_test), function(iter){
  ifelse(x_test[iter,3]%in%train_x3_levels[36:70], 
         as.character(x_test[iter,3]), "not")})
x_test = x_test[,-3]
x_test[,"x3_1"] = as.factor(x_test[,"x3_1"])
x_test[,"x3_2"] = as.factor(x_test[,"x3_2"])

# to have the same levels of the x3_1 and x3_2:  
x_test <- rbind(x_train[1, ] , x_test)
x_test <- x_test[-1,]

###################### CV function ###########################
cv.rf <- function(mtry.par, ntree.par, nodesize.par) {
  #if y is a factor then method = "class" is assumed
  cv <- tune.randomForest(as.factor(y_train)~., 
                          data = cbind(x_train, y_train), 
                          mtry=mtry.par, 
                          ntree=ntree.par, 
                          nodesize=nodesize.par)
  measure = 1- cv$best.performance #performance=misclass.err 
  list(Score = measure, 
       Pred = 0)}

################## Bayesian Optimization #####################
set.seed(20)
rf_BO <- BayesianOptimization(cv.rf,
             bounds = list(mtry.par = c(2L, 20L), 
                           ntree.par=c(100L, 3000L),
                           nodesize.par=c(1L, 10L)), 
             init_grid_dt = NULL, 
             init_points = 5, 
             n_iter = 4,
             acq = "ei", 
             verbose = TRUE)

################## Final Random Forest #######################
set.seed(101)
rf_model = randomForest(x=x_train, y=as.factor(y_train), 
             ntree = rf_BO$Best_Par["ntree.par"] , 
             mtry = rf_BO$Best_Par["mtry.par"], 
             nodesize = rf_BO$Best_Par["nodesize.par"])
rf_predict = predict(rf_model, x_test)  
rf_eval = eval.metrics(as.numeric(rf_predict)-1, y_test)

rf_predict_train = predict(rf_model, x_train)  
rf_eval_train = eval.metrics(as.numeric(rf_predict_train)-1,
                             y_train)

##############################################################
################# Support Vector Machines ####################
##############################################################
 
c(x_train, x_test) %<-% list(x_train_minmax, x_test_minmax)

################## Bayesian Optimization #####################
svm_BO = svm_cv_opt(data = cbind(x_train, y = y_train), 
                               label = y,
                               svm_kernel = "radial",
                               n_folds = 3, 
                               init_points = 4 ,
                               acq = "ei",  
                               n_iter = 5)

################## Final SVM model ###########################
svm_model = svm(as.factor(y_train)~., 
                  data=data.frame(x_train, y_train),
                  kernel = "radial", 
                  gamma = svm_BO$Best_Par["gamma_opt"], 
                  cost = svm_BO$Best_Par["cost_opt"])

svm_predict = predict(svm_model, x_test)
svm_eval = eval.metrics(as.numeric(svm_predict)-1, y_test)

svm_predict_train = predict(svm_model, x_train)
svm_eval_train = eval.metrics(as.numeric(svm_predict_train)-1,
                              y_train)

##############################################################
################### Logistic Regression ######################
##############################################################

c(x_train, x_test) %<-% list(x_train_minmax, x_test_minmax)

x_train = data.matrix(x_train) 
x_test = data.matrix(x_test) 
y_train = as.factor(y_train)

###################### CV function ###########################
cv.lr <- function(alphapar) {
  set.seed(66)
  cv <- cv.glmnet(x=x_train, y=y_train, alpha=alphapar, 
                  family="binomial", type.measure="class")
  measure = 1 - cv$cvm[which(cv$lambda==cv$lambda.1se)]  
  list(Score = measure,
       Pred = 0)}

################## Bayesian Optimization #####################
set.seed(10)
lr_BO <- BayesianOptimization(cv.lr,
                              bounds=list(alphapar=c(0,1)),
                              init_grid_dt = NULL, 
                              init_points = 5, 
                              n_iter = 4,
                              acq = "ei",
                              verbose = TRUE)

############## Final Logistic Regression #####################
set.seed(66)
lr_model <- glmnet(x=x_train, y=y_train, 
                   family="binomial", 
                   alpha=lr_BO$Best_Par, 
                   lambda = cv.glmnet(x=x_train, y=y_train, 
                            family="binomial", 
                            alpha=lr_BO$Best_Par)$lambda.1se)
lr_predict = predict(lr_model, newx=x_test, 
                        type="response") 
lr_eval = eval.metrics(ifelse(lr_predict<0.5, 0, 1), y_test)

lr_predict_train = predict(lr_model, newx=x_train, 
                     type="response") 
lr_eval_train=eval.metrics(ifelse(lr_predict_train<0.5, 0, 1), 
                           y_train)


##############################################################
############# Feed Forward Neural Network ####################
##############################################################

c(x_train, x_test) %<-% list(x_train_zscore, x_test_zscore)

############# chage format of the data #######################
features = names(x_train)
response = c(y_train, y_test)
fmla_nn = as.formula(paste("as.factor(response) ~ ", 
                           paste(features, collapse= "+")))
x <- model.matrix(fmla_nn, cbind(rbind(x_train, x_test), 
                                 response))
X_train  = x[1:nrow(x_train),]
X_test = x[(nrow(x_train)+1):nrow(x),]
shape = dim(x)[2]

########### function for setting up NN model #################
setup.nn <- function(nHidLay, nHidNeur){
  ## nHiddenLayers: assume at least 1 
  ## nHiddenNeurons: number of hidden neurons in first HL, 
  #######  ...then it decreases in next HLs. 
  
  model <- keras_model_sequential()
  model %>% layer_dense(units=nHidNeur, 
                        activation="selu", 
                        input_shape=shape, 
            kernel_initializer = initializer_lecun_normal(2)) 
  
  if (nHiddenLayers>1){
    nlayers=1
    while (nlayers < nHidLay)
    {
      model %>% layer_dense(
      units=nHidNeur-floor(((nlayers)*10^(-1))*nHidNeur),
      activation="selu", 
      kernel_initializer = initializer_lecun_normal(2)) 
      nlayers=nlayers+1
    }
  } 
  
  # output layer
  model %>% layer_dense(units=1, activation="sigmoid") 
  model 
}

############# setting of cross validation ####################
flds <- createFolds(y_train , k = 5, list = TRUE, 
                    returnTrain = FALSE)

compile.nn = function(learnRate, momentum, 
                      nHidLay, nHidNeur, 
                      nEpochs=150,
                      X_matrix, y_vector){  
  
  model = setup.nn(nHidLay, nHidNeur)
  model %>% compile(loss="binary_crossentropy",
                    optimizer=optimizer_sgd(lr=learnRate,
                                         momentum=momentum),
                    metrics="accuracy")
  history <-  model %>% fit(X_matrix, y_vector, 
                            batch_size=128, 
                            epochs=nEpochs,
                            verbose=1,
                            callback_early_stopping(
                              monitor = "loss", 
                              min_delta = 0.000000001, 
                              patience = 10))
  return(model)
}

cv.nn = function(learnRate, momentum, 
                 nHidLay, nHidNeur, 
                 nEpochs=150){
  cv = sapply(1:length(flds), function(i){
    ind = flds[[i]]
    mdl = compile.nn(learnRate, momentum, 
                     nHidLay, nHidNeur, 
                     nEpochs, 
                     X_train[-c(ind),], 
                     y_train[-c(ind)])
    pred = mdl %>% predict(X_train[ind,])
    return(
      sum(ifelse(pred<0.5, 0,1)==y_train[ind])/length(ind))
  })
  return(list(Score = mean(cv), Pred = 0))
}

################## Bayesian Optimization #####################
nn_BO <- BayesianOptimization(cv.nn,
            bounds = list(learnRate = c(0.0001,0.05), 
                          momentum=c(0,0.9),
                          nHidLay=c(1L, 6L), 
                          nHidNeur=c(10L, 100L)), 
            init_grid_dt = NULL, 
            init_points = 5, 
            n_iter = 4,
            acq = "ei", 
            verbose = TRUE)

################## Final Neural Network ######################
nn_model = setup.nn(nHidLay=nn_BO$Best_Par["nHidLay"], 
                    nHidNeur=nn_BO$Best_Par["nHidNeur"])

nn_model %>% compile(loss="binary_crossentropy",
                optimizer=optimizer_sgd(
                  lr=nn_BO$Best_Par["learnRate"],
                  momentum=nn_BO$Best_Par["momentum"]),
                metrics="accuracy")

history <-  nn_model %>% fit(X_train, y_train, 
                                batch_size=128, 
                                epochs=500,
                                verbose=1,
                                callback_early_stopping(
                                  monitor = "loss", 
                                  min_delta = 0.0001, 
                                  patience = 10))

nn_predict <- nn_model %>% predict(X_test)
nn_eval = eval.metrics(ifelse(nn_predict<0.5, 0, 1), 
                       y_test) 

nn_predict_train = nn_model %>% predict(X_train)
nn_eval_train=eval.metrics(ifelse(nn_predict_train<0.5, 0, 1), 
                           y_train)

##############################################################
############### Semi-supervised approach #####################
##############################################################

c(x_train, x_test) %<-% list(x_train_forAE, x_test_forAE)
c(x_train_all, y_train_all) %<-% list(x_train_forAE, y_train)

## use only benign samples
x_train = x_train[which(y_train==0),] 
y_train = y_train[which(y_train==0)]

features = names(x_train)
response = c(y_train, y_test)
fmla_ae = as.formula(paste("as.factor(response) ~ ", 
                           paste(features, collapse= "+")))
x <- model.matrix(fmla_ae, cbind(rbind(x_train, x_test), 
                                 response))[,-1]
X_train  = x[1:nrow(x_train),]
X_test = x[(nrow(x_train)+1):nrow(x),]
shape = dim(x)[2]

# train samples incl. attacks:
X_train_all = model.matrix(fmla_ae, cbind(x_train_all,
                                response = y_train_all))[,-1]

##############################################################
##################### Autoencoder ############################
##############################################################

setup.ae <- function(nHidLay, nHidNeur, botNeur){ 
  
  activF = "selu"  # actif. f. in hidden layers
  activBTN = "tanh"  # activ. f. in bottleneck
  inputs <- layer_input(shape = c(shape)) 
  
  if (nHidLay==1){
    hidden_layers <- inputs %>%
      layer_dense(units=nHidNeur, activation=activF)  %>%
      layer_dense(units=botNeur, activation=activBTN) %>% 
      layer_dense(units=nHidNeur, activation=activF) 
    
  } else if (nHidLay==0){
    hidden_layers <- inputs %>%
      layer_dense(units=botNeur, activation=activBTN)
    
  } else if (nHidLay==2){
    hidden_layers <- inputs %>%
      layer_dense(units=nHidNeur, activation=activF)  %>% 
      layer_dense(units=nHidNeur-floor((nHidNeur-botNeur)/2), 
                  activation=activF)  %>% 
      layer_dense(units=botNeur, activation=activBTN) %>% 
      layer_dense(units=nHidNeur-floor((nHidNeur-botNeur)/2), 
                  activation=activF)  %>% 
      layer_dense(units=nHidNeur, activation=activF) 
  }
 
  out_protocol <- hidden_layers %>%
    layer_dense(units=3, activation="softmax", 
                name = 'out_protocol') 
  
  out_service <- hidden_layers %>%
    layer_dense(units=70, activation="softmax", 
                name = 'out_service') 
  
  out_flag <- hidden_layers %>%
    layer_dense(units=11, activation="softmax", 
                name = 'out_flag') 
  
  binary_outputs <- hidden_layers %>%
    layer_dense(units=5, activation="sigmoid", 
                name = 'binary_out') 
  
  contin_outputs <- hidden_layers %>%
    layer_dense(units=31, activation="linear", 
                name = 'contin_out') 
  
  model <- keras_model(inputs = inputs, 
                       outputs = c(out_protocol,
                                   out_service, 
                                   out_flag, 
                                   binary_outputs, 
                                   contin_outputs))
  model
}

compile.ae <- function(learRate,  botNeur,
                  nHidLay, nHidNeur, 
                  nEpochs=500, 
                  w_protocol, w_service, 
                  w_flag, w_binary, w_contin, 
                  beta1.par, beta2.par){  
  
  sum_of_weights = w_protocol + w_service + 
    w_flag + w_binary + w_contin
  
  model = setup.ae(nHidLay, nHidNeur, botNeur)
  
  model %>% compile(
    optimizer = optimizer_adam(lr=learnRate, 
                               beta_1 = beta1.par, 
                               beta_2 = beta2.par),
    loss = list(out_protocol = "categorical_crossentropy", 
                out_service= "categorical_crossentropy",
                out_flag = "categorical_crossentropy",  
                binary_out = "binary_crossentropy", 
                contin_out = 'mean_absolute_error'),
    loss_weights = list(
      out_protocol = w_protocol/sum_of_weights, 
      out_service= w_service/sum_of_weights,
      out_flag = w_flag/sum_of_weights,  
      binary_out = w_binary/sum_of_weights, 
      contin_out = w_contin/sum_of_weights)
  )
  
  history <- model %>% fit(X_train, 
                          list(X_train[, 1:3], 
                               X_train[, 4:73], 
                               X_train[, 74:84], 
                               X_train[, 85:89], 
                               X_train[, 90:120]), 
                          batch_size=120, 
                          epochs=nEpochs,
                          verbose=1, validation_split=0.2,
                          callback_early_stopping(
                            monitor = "val_loss", 
                            min_delta = 0.000000001, 
                            patience = 20, 
                            restore_best_weights = T))
  measure = -history$metrics$val_loss[length(
                            history$metrics$val_loss)] 
  n_ep_used = length(history$metrics$val_loss)
  return(list(Score = measure, Pred = n_ep_used)) 
}

################## Bayesian Optimization #####################
ae_BO <- BayesianOptimization(compile.ae,
              bounds = list(learnRate = c(0.0001, 0.05), 
                            botNeur = c(2L, 40L),
                            nHidLay=c(1L,2L), 
                            nHidNeur=c(40L, 50L), 
                            w_protocol = c(1, 10), 
                            w_service = c(1, 10),
                            w_flag = c(1, 10), 
                            w_binary  = c(1, 10), 
                            w_contin= c(1, 10), 
                            beta1.par = c(0.8,0.99), 
                            beta2.par = c(0.8,0.99)),
                                init_grid_dt = NULL, 
                                init_points = 3, 
                                n_iter = 7,
                                acq = "ei", 
                                eps = 0.5,
                                verbose = TRUE)

c(lr, btlnck,  nhl, 
  nhn, wp, ws, wf, 
  wb, wc, b1, b2, ne) %<-% list(
    ae_BO$Best_Par["learnRate"],
    ae_BO$Best_Par["botNeur"], 
    ae_BO$Best_Par["nHidLay"],
    ae_BO$Best_Par["nHidNeur"],
    ae_BO$Best_Par["w_protocol"], 
    ae_BO$Best_Par["w_service"], 
    ae_BO$Best_Par["w_flag"], 
    ae_BO$Best_Par["w_binary"], 
    ae_BO$Best_Par["w_contin"],
    ae_BO$Best_Par["beta1.par"], 
    ae_BO$Best_Par["beta2.par"],
    500)

sum_of_final_weights = wp + ws + wf + wb + wc

ae_model = setup.ae(nHidLay=nhl, nHidNeur=nhn, 
                         botNeur = btlnck)

ae_model %>% compile(
  optimizer = optimizer_adam(lr=lr, 
                             beta_1=b1, beta_2=b2),
  loss = list(out_protocol = "categorical_crossentropy", 
              out_service= "categorical_crossentropy",
              out_flag = "categorical_crossentropy",  
              binary_out = "binary_crossentropy", 
              contin_out = 'mean_absolute_error'),
  loss_weights = list(
    out_protocol = wp/sum_of_final_weights, 
    out_service= ws/sum_of_final_weights,
    out_flag = wf/sum_of_final_weights,  
    binary_out = wb/sum_of_final_weights, 
    contin_out = wc/sum_of_final_weights)
)

history_ae <- ae_model %>% fit(X_train, 
                               list(X_train[, 1:3], 
                                    X_train[, 4:73], 
                                    X_train[, 74:84], 
                                    X_train[, 85:89], 
                                    X_train[, 90:120]), 
                            batch_size=120, 
                            epochs=ne,
                            verbose=1,
                            callback_early_stopping(
                              monitor = "loss", 
                              min_delta = 0.000000001, 
                              patience = 20))


##############################################################
################## Gaussian Mixture ##########################
##############################################################
GMM.prediction = function(train.data, test.data, AE.model, 
                          plot=NULL){
  if(is.null(plot)){plot=FALSE}
  predicted.train <- AE.model %>% predict(train.data)
  predicted.test <- AE.model %>% predict(test.data)
  
  predicted_difference_train <- cbind(predicted.train[[1]], 
                                      predicted.train[[2]],
                                      predicted.train[[3]], 
                                      predicted.train[[4]],
                                      predicted.train[[5]]) -  
                                 train.data
  predicted_difference_test <- cbind(predicted.test[[1]], 
                                     predicted.test[[2]],
                                     predicted.test[[3]], 
                                     predicted.test[[4]], 
                                     predicted.test[[5]]) - 
                                 test.data
  
  log_train = log(rowMeans(abs(predicted_difference_train)))
  log_test = log(rowMeans(abs(predicted_difference_test)))
  
  threshold = quantile(log_train, 0.9) 
  
  set.seed(66)
  mean1 = mean(log_test[which(log_test<threshold)])
  mean2 = mean(log_test[which(log_test>threshold)])
  mixmdl = mixtools::normalmixEM(log_test, k=2, 
                                 maxrestarts = 20, 
                                 mean.constr = c(mean1,mean2)
  )
  
  ## to overcome possible problem on tails
  sure_normal = which(log_test<mean1)  
  sure_attack = which(log_test>mean2)
  clusters = sapply(1:nrow(test.data), 
                    function(i) if(i%in%sure_normal){0}
                    else if(i%in%sure_attack){1}
                    else{as.numeric(which.max(
                      mixmdl$posterior[i,]))-1})
  
  if(plot==TRUE){
    plot(mixmdl, which = 2,xlab2=" ",
         col2=c("blue", "orange"), main2="")
    abline(v=threshold, lwd=2) 
    legend("topright", 
           legend = c("normal", "attack", 
                      "0.9 quantile of log(train)"),  
           pch=15,col=c("blue", "orange", "black"))
    
  }
  return("prediction" = clusters)
}

####### Final prediction of Semi-supervised model ############
gmm_predict = GMM.prediction(X_train, X_test, ae_model,T) 
gmm_eval = eval.metrics(gmm_predict, y_test)

gmm_predict_train = GMM.prediction(X_train, X_train_all,
                                   ae_model, T)
gmm_eval_train = eval.metrics(gmm_pred_train, y_train_all)









