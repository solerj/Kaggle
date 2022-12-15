setwd("D:/1_DataScience/repos/Kaggle_HousePrices")
library(ggplot2)
#install.packages("tidymodels")

# might want to use tidymodels library for recipe and baking ETL
# cv.glmnet applies cv to get optimal lambda only - try glmnetUtils library (cva.glmnet)
# might want to optimise xgBoost parameters based on linear regression predictions - this could help with setting right parameters against overfitting
# difference between ensembling and stacking?
# lime package includes SHAP?
# apply training on training set only!
# randomForestExplainer
# random search instead of grid search


# library(cva.glmnet)


trainData <- read.csv("train.csv"
                      , stringsAsFactors = FALSE)
trainData$target <- log(trainData$SalePrice)
trainData$SalePrice <- c()
testData  <- read.csv("test.csv"
                      , stringsAsFactors = FALSE)
testData$target <- NA
myData <- rbind(trainData, testData)


# remove unwated features from EDA
toRemove <- c("Id"
              , "LowQualFinSF"
              , "BsmtFullBath"
              , "BsmtHalfBath"
              , "HalfBath"
              , "BedroomAbvGr"
              , "KitchenAbvGr"
              , "GarageYrBlt"
              , "X3SsnPorch"
              , "PoolArea"
              , "MiscVal"
              , "MoSold")
myData[,toRemove] <- c()


# ad-hoc feature updates
myData$YearBuilt <- round(myData$YearBuilt/25, 0)*25


# bulk feature updates

setToCharacter <- c("MSSubClass"
                    , "YrSold"
                    , "YearBuilt")

for (i in setToCharacter){
  myData[, i] <- as.character(myData[, i])
  colnames(myData)[colnames(myData) == i] <- paste0(i, "_")
}

 # ----------------

setToLn <- c("LotFrontage"
             , "LotArea"
             , "TotalBsmtSF"
             , "X1stFlrSF"
             , "X2ndFlrSF"
             , "GrLivArea"
             , "WoodDeckSF"
             , "OpenPorchSF"
             , "EnclosedPorch"
             , "ScreenPorch"
             )

for (i in setToLn){
  toBeUpdated <- myData[, i] != 0 & !is.na(myData[, i])
  myData[toBeUpdated, i] <- log(myData[toBeUpdated, i])
  colnames(myData)[colnames(myData) == i] <- paste0(i, "_ln")
}

str(myData)

# ----------------

classFromNumeric <- rbind(c("OverallCond",     5)
                          , c("YearRemodAdd",  1950)
                          , c("MasVnrArea",    0)
                          , c("BsmtFinSF1",    0)
                          , c("BsmtFinSF2",    0)
                          , c("TotalBsmtSF",   0)
                          , c("X2ndFlrSF",     0)
                          , c("WoodDeckSF",    0)
                          , c("OpenPorchSF",   0)
                          , c("EnclosedPorch", 0)
                          , c("ScreenPorch",   0)
                          )

# table(myData$OverallCond)
# plot(myData[,c("OverallCond", "target")])

for (i in 1:nrow(classFromNumeric)){
  colName <- colnames(myData)[grep(classFromNumeric[i,1], colnames(myData))]
  myData$orig <- myData[, colName]
  myData$new1 <- myData[, colName] == classFromNumeric[i,2]
  fixFit <- lm(orig ~ target, data = myData[!(myData$new1) & !is.na(myData$target), ])
  myData$orig[(myData$new1)] <- mean(predict.lm(fixFit
                                                , data.frame(target = myData$target[(myData$new1) & !is.na(myData$target)])))
  colnames(myData)[colnames(myData) == "orig"] <- paste0(colName
                                                         , "_")
  colnames(myData)[colnames(myData) == "new1"] <- paste0(colName
                                                         , "_is"
                                                         , classFromNumeric[i,2])
  myData[, colName] <- c()
}

# plot(myData[,c("OverallCond_", "target")])

# table(myData$OverallCond_)
# table(myData$OverallCond_is5)
# str(myData)
# dim(myData)

# testing methodology for linear + certain different value ----------------------------------------

#       plot(myData$OverallCond_, myData$target)
#       
#       myTibble <- myData %>%
#         group_by(OverallCond_is5) %>%
#         summarise(avgSalePrice = mean(target, na.rm = T))
#       
#       
#       tryFit <- glm(formula = target ~ OverallCond_ + OverallCond_is5
#                     , family = gaussian
#                     , data = myData[!is.na(myData$target),])
#       summary(tryFit)
#       
#       tryFitPlot <- c()
#       for (i in 1:10){
#         tryFitPlot <- c(tryFitPlot
#                         , tryFit$coefficients[1] + i*tryFit$coefficients[2] + (i==5)*tryFit$coefficients[3])
#       }
#       plot(tryFitPlot)

# -------------------------------------------------------------------------------------------------

# one-hot-encode character features

toOneHoteEncode <- colnames(myData)[lapply(myData, class) == "character"]

str(myData)

for (i in toOneHoteEncode){
  mapTable <- table(myData[!is.na(myData$target), i])
  mapTable <- mapTable[order(mapTable)]
  
  mapToOther <- names(mapTable)[mapTable < 5]
  myData[myData[, i] %in% mapToOther, i] <- "other"
  
  myData[is.na(myData[, i]), i] <- "other"
  
  noMapMax <- names(mapTable)[mapTable == max(mapTable)]
  mapFeatures <- if (length(mapToOther) + sum(is.na(myData[, i])) > 0){
    c("other", names(mapTable)[!(names(mapTable) %in% c(mapToOther, noMapMax))])
  } else {
    names(mapTable)[!(names(mapTable) %in% c(mapToOther, noMapMax))]
  }
  
  for (j in mapFeatures){
    myData$newOneHot <- myData[, i] == j
    colnames(myData)[colnames(myData) == "newOneHot"] <- paste0(i
                                                                , "_"
                                                                , j)
  }
  myData[, i] <- c()
}


# str(myData)
# dim(myData)
# summary(myData)

# standardise numerical features

toStandardise <- colnames(myData)[lapply(myData, class) %in% c("numeric", "integer") &
                                    colnames(myData) != "target"]

for (i in toStandardise){
  myMean <- mean(myData[, i], na.rm = T)
  mySd <- sd(myData[, i], na.rm = T)
  myData[, i] <- (myData[, i]-myMean)/mySd
  myData[is.na(myData[, i]), i] <- 0
}


# standardise numerical features

toInteger <- colnames(myData)[lapply(myData, class) == "logical"]

for (i in toInteger){
  myData[, i] <- as.integer(myData[, i])
  myData[is.na(myData[, i]), i] <- 0
}

str(myData)
# summary(myData$target)
# dim(myData)
# summary(myData[, 1:121])
# summary(myData[, 122:242])

#p <- ggplot(trainData, aes(x=as.character(YearBuilt), y=log(SalePrice), color=YearBuilt)) + geom_boxplot()
#p

myDataMean <- mean(myData$target, na.rm = T)
max_rmse <- sqrt(mean((myData$target - myDataMean)^2, na.rm = T))

featureCols <- colnames(myData)[colnames(myData) != "target"]
xDf <- myData[!is.na(myData$target), featureCols]
x   <- as.matrix(xDf)
yDf <- myData[!is.na(myData$target), c("target")]
y   <- as.matrix(yDf)




# linear regression -----------------------------

#install.packages("glmnetUtils")
require(glmnetUtils)

glmLmabdaA <- cva.glmnet(x = x
                         , y = y
                         , alpha = seq(0, 1, len = 11)^3
                         , nfolds = 7)

plot(glmLmabdaA)

numAlphas <- length(glmLmabdaA$alpha)
glmLmabdaAsummary <- c()

for (i in 1:numAlphas){
  glmnet.model <- glmLmabdaA$modlist[[i]]
  min.mse <-  min(glmnet.model$cvm)
  min.lambda <- glmnet.model$lambda.min
  alpha.value <- glmLmabdaA$alpha[i]
  glmLmabdaAsummary <- rbind(glmLmabdaAsummary
                             , c(alpha.value, min.lambda, min.mse))
}

plot(glmLmabdaAsummary[,3])
bestAlpha  <- glmLmabdaAsummary[which.min(glmLmabdaAsummary[,3]), 1]
bestLambda <- glmLmabdaAsummary[which.min(glmLmabdaAsummary[,3]), 2]

myGlmA <- glmnet(x = x
                 , y = y
                 , alpha = bestAlpha
                 , lambda = bestLambda
                 , family="gaussian")

myGlmACoef <- as.data.frame(as.matrix(coef(myGlmA)))
colnames(myGlmACoef) <- "coef"
myGlmACoef$feature <- rownames(myGlmACoef)
linRegfeatureSubset <- myGlmACoef[myGlmACoef$coef>0, "feature"]
featureColsS <- featureCols[featureCols %in% linRegfeatureSubset]

# xg-boost --------------------------------------

#install.packages("xgboost")
library(xgboost)
library(dplyr)

dim(myData)
set.seed(19301)
myDataSxgb <- myData[!is.na(myData$target),]
myDataSxgb$cvSplit <- round(runif(nrow(myDataSxgb), 0.5, 7.5))

rmse_xgb <- c()
for (k in seq(from = 1, to = 7, by = 1)){
  
  myDataMxgb <- xgb.DMatrix(data = as.matrix(myDataSxgb[myDataSxgb$cvSplit != k
                                                        , featureColsS])   # featureCols
                            , label = as.matrix(myDataSxgb[myDataSxgb$cvSplit != k
                                                           , c("target")]))
  
  myDataMxgbT <- as.matrix(myDataSxgb[myDataSxgb$cvSplit == k
                                      , featureColsS])  # featureCols 
  
  
  for (subsample in 0.65){
    for (colsample_bytree in 0.65){
      for (max_depth in 6){
        for (nrounds in seq(from = 400, to = 700, by = 100)){
          for (eta in seq(from = 0.01, to = 0.05, by = 0.01)){
            params <- list(booster = "gbtree"
                          , objective = "reg:squarederror"
                          , eta = eta
                          , gamma = 0
                          , max_depth = max_depth
                          , min_child_weight = 1
                          , subsample = subsample
                          , colsample_bytree = colsample_bytree
                          )
          
            xgbT <- xgb.train(params = params
                              , data = myDataMxgb
                              , nrounds = nrounds
                             , prediction = T
                             , metrics = "rmse")
           
           xgbPredT <- predict(xgbT, newdata = myDataMxgbT)
           
           #xgbPred   <- predict(xgb,   newdata = newx)
           
           rmse <- sqrt(mean((myDataSxgb$target[myDataSxgb$cvSplit == k] -
                                xgbPredT)^2))
            
           rmse_xgb <- rbind(rmse_xgb
                             , c(subsample
                                 , colsample_bytree
                                  , max_depth
                                  , nrounds
                                  , eta
                                  , rmse))
            
            print(c(k
                    , subsample
                    , colsample_bytree
                    , max_depth
                    , nrounds
                    , eta
                    , rmse))
          }
        }
      }
    }
  }
}
rmse_xgb <- as.data.frame(rmse_xgb)
colnames(rmse_xgb) <- c("subsample", "colsample_bytree", "max_depth", "nrounds", "eta", "rmse")
rmse_xgbS <- rmse_xgb %>%
  group_by(subsample, colsample_bytree, max_depth, nrounds, eta) %>%
  summarise(avgRMSE = mean(rmse))
#rmse_xgbS
plot(rmse_xgbS$avgRMSE)
head(rmse_xgbS[order(rmse_xgbS$avgRMSE),], 10)

xgb_p <- ggplot(rmse_xgb, aes(x=params, y=rmse)) + geom_boxplot()
xgb_p


myDataM <- xgb.DMatrix(data = as.matrix(myData[!is.na(myData$target)
                                               , featureColsS])   # replaced from featureCols
                          , label = as.matrix(myData$target[!is.na(myData$target)]))

params <- list(booster = "gbtree"
               , objective = "reg:squarederror"
               , eta = 0.03
               , gamma = 0
               , max_depth = 6
               , min_child_weight = 1
               , subsample = 0.65
               , colsample_bytree = 0.65
               )

xgb <- xgb.train(params = params
                , data = myDataM
                , nrounds = 700
                , prediction = T
                , metrics = "rmse")


# EVALUATION ----------------------------------------------

rmse <- function(predictionColName){
  rmse <- sqrt(mean((myData$target[!is.na(myData$target)] -
                       myData[!is.na(myData$target), c(predictionColName)])^2))
  rmse
}

newx  <- as.matrix(myData[, featureCols])
newxS <- as.matrix(myData[, featureColsS])

myData$xgbPred   <- predict(xgb,   newdata = newxS)
myData$glm0Pred  <- predict(myGlm, newx = newx, s=0)
myData$glm1Pred  <- predict(myGlm, newx = newx, s=glmLmabda$lambda.min)
myData$glm2Pred  <- predict(myGlm, newx = newx, s=mean(glmLmabda$lambda.min, glmLmabda$lambda.1se)) #was mean(lambda)
myData$glm3Pred  <- predict(myGlm, newx = newx, s=glmLmabda$lambda.1se)
myData$glm4Pred  <- predict(myGlmA, newx = newx)
myData$randFPred <- predict(randF, newdata = newx)
myData$dTreePred <- predict(dTree, newdata = myData[,featureCols])
myData$nnPred    <- model %>% predict(newx)


myModels <- colnames(myData)[grep("Pred", colnames(myData))]

myModelEval <- c()
for (i in myModels){
  myModelEval <- rbind(myModelEval, c(i, rmse(i)))
}
plot(c(myModelEval[,2], max_rmse))

myModelEval <- rbind(myModelEval
                     , c("max_rmse", max_rmse))

myModelEval

plot(myData$target, myData$glm4Pred)
plot(myData$target, myData$xgbPred)
plot(myData$target, myData$randFPred)
plot(myData$target, myData$nnPred)


predictions <- cbind(testData$Id, exp(myData$xgb[is.na(myData$target)]))
predictions <- as.data.frame(predictions)
colnames(predictions) <- c("Id", "SalePrice")
head(predictions)

write.csv(predictions, file = "johnSolerSubmission.csv", quote = F, row.names = F)
