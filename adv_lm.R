#library
library(caret)
library(mlbench)
library(psych)
library(glmnet)

#data 
data("BostonHousing")
data <- BostonHousing

str(data)

pairs.panels(data[,c(-4,-14)],cex=2)

#data partition 
set.seed(222)
ind <- sample(2,nrow(data),replace=T,prob=c(0.7,0.3))
train <- data[ind==1,]
test <- data[ind==2,]

#custom control parameters

custom <- trainControl(method="repeatedcv",
                       number=10,
                       repeats = 5,
                       verboseIter = T)
?trainControl

#liner model
set.seed(1234)
lm <- train(medv~.,
            train,
            method="lm",
            trControl=custom)
#results

lm$results
lm
summary(lm)
plot(lm$finalModel)

#ridge regression
set.seed(1234)
ridge <- train(medv~.,
               train,
               method="glmnet",
               tuneGrid=expand.grid(alpha=0,
                                    lambda=seq(0.0001,1,length=5)),
               trControl=custom)
ridge$results
ridge
summary(ridge)
plot(ridge)
plot(ridge$finalModel,xvar="lambda",label = T)
#upper axis is no of vars
#lower axis is log lambda
#left axis coeff of vars
plot(ridge$finalModel,xvar="dev",label = T)
#lower axis give the fraction of explaination 
#after 0.7 its takes a good jump so likely to be over fitted
plot(varImp(ridge,scale=T))

#Lasso
set.seed(1234)
lasso <- train(medv~.,
               train,
               method="glmnet",
               tuneGrid=expand.grid(alpha=1,
                                    lambda=seq(0.001,1,length=5)),
               trControl=custom)
lasso$results
summary(lasso)
plot(lasso)
plot(lasso$finalModel,xvar="lambda",label=T)
plot(lasso$finalModel,xvar="dev",label=T)
plot(varImp(lasso,scale = T))

#Elastic Net
set.seed(1234)
en <- train(medv~.,
            train,
            method="glmnet",
            tuneGrid=expand.grid(alpha=seq(0,1,length=5),
                                 lambda = seq(0.001,1,length=5)),
            trControl=custom)
plot(en)
#as lambda lowers rmse lowers so lets re run 

en <- train(medv~.,
            train,
            method="glmnet",
            tuneGrid=expand.grid(alpha=seq(0,1,length=5),
                                 lambda = seq(0.001,0.2,length=5)),
            trControl=custom)
plot(en)
plot(en$finalModel,xvar="lambda")
plot(en$finalModel,xvar="dev")
plot(varImp(en,scale=F))

#comparing models 
model_list <- list(linear_model=lm,Ridge=ridge,Lasso=lasso,Elasticnet=en)
res <- resamples(model_list)
summary(res)
bwplot(res)
xyplot(res,metric="RMSE")

#best model
en$bestTune
best <- en$bestTune

#prediction
p1 <- predict(en,train)
sqrt(mean((train$medv-p1)^2))
p2 <- predict(en,test)
sqrt(mean((test$medv-p2)^2))
