#set working directory
#path <- "uncomment this and paste your datafile path here"
setwd(path)

getwd()

#load libraries
library(data.table)
library(mlr)

#set variable names
setcol = c("time", "signal", "open_channels")
setcoltest = c("time", "signal")

#load data
train <- read.table("train.csv", header = F, sep = ",", col.names = setcol, na.strings = c(" ?"), stringsAsFactors = F)
test <- read.table("test.csv",header = F,sep = ",",col.names = setcoltest,skip = 1, na.strings = c(" ?"),stringsAsFactors = F)

#convert data frame to data table
setDT(train) 
setDT(test)

#check missing values 
table(is.na(train))
sapply(train, function(x) sum(is.na(x))/length(x))*100

table(is.na(test))
sapply(test, function(x) sum(is.na(x))/length(x))*100

#quick data cleaning
library(stringr)

#remove leading whitespaces
char_col <- colnames(train)[ sapply (test,is.character)]
for(i in char_col) set(train,j=i,value = str_trim(train[[i]],side = "left"))

for(i in char_col) set(test,j=i,value = str_trim(test[[i]],side = "left"))

#set all missing value as "Missing" 
train[is.na(train)] <- "Missing" 
test[is.na(test)] <- "Missing"


#using one hot encoding
labels <- train$signal
ts_label <- test$signal

new_tr1 = data.matrix(train, rownames.force = NA)
new_ts1 = data.matrix(test, rownames.force = NA)

#convert factor to numeric 
labels <- as.numeric(labels)-1
ts_label <- as.numeric(ts_label)-1

#preparing matrix 
dtrain <- xgb.DMatrix(data = new_tr1,label = labels) 
dtest <- xgb.DMatrix(data = new_ts1,label=ts_label)

#default parameters
params <- list(booster = "gblinear", objective = "multi:softmax",num_class =11, eta=0.3, gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)

xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 79, nfold = 5, showsd = T, stratified = T, print.every.n = 10, early.stop.round = 20, maximize = F)
##best iteration = 79

summary(xgbcv)
