setwd("D:/1_Kaggle/repos/Kaggle_HousePrices")
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


str(myData)

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


# EVALUATION ----------------------------------------------

rmse <- function(predictionColName){
  rmse <- sqrt(mean((myData$target[!is.na(myData$target)] -
                       myData[!is.na(myData$target), c(predictionColName)])^2))
  rmse
}

newx  <- as.matrix(myData[, featureCols])
myData$pred  <- predict(myGlmA, newx = newx)

myModels <- colnames(myData)[grep("pred", colnames(myData))]

myModelEval <- c()
for (i in myModels){
  myModelEval <- rbind(myModelEval, c(i, rmse(i)))
}
plot(c(myModelEval[,2], max_rmse))

myModelEval <- rbind(myModelEval
                     , c("max_rmse", max_rmse))

myModelEval

plot(myData$target, myData$pred)


predictions <- cbind(testData$Id, exp(myData$pred[is.na(myData$target)]))
predictions <- as.data.frame(predictions)
colnames(predictions) <- c("Id", "SalePrice")
head(predictions)

write.csv(predictions, file = "johnSolerSubmission.csv", quote = F, row.names = F)
