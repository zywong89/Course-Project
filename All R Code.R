setwd("~/Desktop/Data Science Specialization/P8 Practical Machine Learning/Week 4 Regularized Regression and Combining Predictors/Course Project")
training <- read.csv("pml-training.csv") # To be seperated into training & validation set
testing <- read.csv("pml-testing.csv") # Leave aside

library(caret)
# Data Cleaning
# Convert measurements that are classified as Factor to Numeric
# Non-numeric Values are treated as missing.
factors <- c(12:17, 20, 23, 26, 69:74, 87:92, 95, 98, 101, 125:130, 133, 136, 139)
for (j in factors) {
        training[, j] <- gsub("#DIV/0!", "", training[, j])
        testing[, j] <- gsub("#DIV/0!", "", testing[, j])
        training[, j] <- as.numeric(training[, j])
        testing[, j] <- as.numeric(testing[, j])
}
# Remove Columns 1-7 as they are subject or time related variables,
# which do not make sense for including these variables in prediction.
training <- training[, -c(1:7)]
testing <- testing[, -c(1:7)]
# Remove zero or near zero variance predictors, the frequency cutoff is 95/5.
# The decision is based on training set.
nzvp <- nearZeroVar(training, freqCut = 95/5)
training <- training[, -nzvp]
testing <- testing[, -nzvp]
# Remove predictors with more than 95% missing values
miss <- apply(training, 2, function(x) mean(is.na(x)) > 0.95)
training <- training[, miss == FALSE]
testing <- testing[, miss == FALSE]
# Number of predictors reduced from 153 to 53.
dim(training); dim(testing)
# Split the training set into smaller training (for model building) and validation sets
set.seed(12345)
inBuild <- createDataPartition(y = training$classe, p = 0.75, list = FALSE)
building <- training[inBuild, ]
valid <- training[-inBuild, ]
dim(building); dim(valid)

# Data Exploration
with(building, plot(classe, col = "orange",
                    main = "Distribution of Exercise Manners",
                    xlab = "Activity", ylab = "Frequency"))

# Fit decision tree, random forest, and stochastic gradient boosting models, with default settings.
fitDT <- train(classe ~ ., data = building, method = "rpart")
fitLDA <- train(classe ~ ., data = building, method = "rlda")
fitRF <- train(classe ~ ., data = building, method = "rf", ntree = 100, do.trace = TRUE)
fitGBM <- train(classe ~ ., data = building, method = "gbm", distribution = "multinomial")
# Make predictions on validation set.
predDT <- predict(fitDT, newdata = valid)
predLDA <- predict(fitLDA, newdata = valid)
predRF <- predict(fitRF, newdata = valid)
predGBM <- predict(fitGBM, newdata = valid)
# Evaluate model performances.
confMatDT<- confusionMatrix(predDT, valid$classe)
confMatLDA <- confusionMatrix(predLDA, valid$classe)
confMatRF <- confusionMatrix(predRF, valid$classe)
confMatGBM <- confusionMatrix(predGBM, valid$classe)
# Variable Importance
par(mar = c(6, 7, 3, 2))
varImpDT <- data.frame(var = rownames(varImp(fitDT)$importance),
                       imp = varImp(fitDT)$importance$Overall)
varImpDT <- varImpDT[order(varImpDT$imp, decreasing = TRUE), ]
varImpDT[1:5, ]
with(varImpDT[1:5, ], barplot(imp, col = c("red", "orange", "yellow", "green", "blue"),
                              main = "Decision Tree Model",
                              sub = paste("Accuracy =",
                                          round(100 * confMatDT$overall["Accuracy"], 2), "%"),
                              width = 10, space = 0.2, cex.names = 0.7, las = 2,
                              names.arg = var, horiz = TRUE,
                              xlab = "Importance", xlim = c(0, 100)))
varImpLDA <- data.frame(var = rownames(varImp(fitLDA)$importance),
                        varImp(fitLDA)$importance, row.names = 1:52)
varImpLDA$imp <- rowMeans(varImpLDA[, -1])
varImpLDA <- varImpLDA[order(varImpLDA$imp, decreasing = TRUE), ]
varImpLDA[1:5, c("var", "imp")]
with(varImpLDA[5:1, ], barplot(imp, col = c("red", "orange", "yellow", "green", "blue"),
                               main = "Regularized Linear Discriminant Analysis Model",
                               sub = paste("Accuracy =",
                                           round(100 * confMatLDA$overall["Accuracy"], 2), "%"),
                               width = 10, space = 0.2, cex.names = 0.7, las = 2,
                               names.arg = var, horiz = TRUE,
                               xlab = "Importance", xlim = c(0, 100)))

varImpRF <- data.frame(var = rownames(varImp(fitRF)$importance),
                       imp = varImp(fitRF)$importance$Overall)
varImpRF <- varImpRF[order(varImpRF$imp, decreasing = TRUE), ]
varImpRF[1:5, ]
with(varImpRF[5:1, ], barplot(imp, col = c("red", "orange", "yellow", "green", "blue"),
                              main = "Random Forest Model",
                              sub = paste("Accuracy =",
                                          round(100 * confMatRF$overall["Accuracy"], 2), "%"),
                              width = 10, space = 0.2, cex.names = 0.7, las = 2,
                              names.arg = var, horiz = TRUE,
                              xlab = "Importance", xlim = c(0, 100)))

varImpGBM <- data.frame(var = rownames(varImp(fitGBM)$importance),
                        imp = varImp(fitGBM)$importance$Overall)
varImpGBM <- varImpGBM[order(varImpGBM$imp, decreasing = TRUE), ]
varImpGBM[1:5, ]
with(varImpGBM[5:1, ], barplot(imp, col = c("red", "orange", "yellow", "green", "blue"),
                               main = "Generalized Boosted Regression Model",
                               sub = paste("Accuracy =",
                                           round(100 * confMatGBM$overall["Accuracy"], 2), "%"),
                               width = 10, space = 0.2, cex.names = 0.7, las = 2,
                               names.arg = var, horiz = TRUE,
                               xlab = "Importance", xlim = c(0, 100)))

# Use RF model to predict on test set
testRF <- predict(fitRF, newdata = testing)
for (i in 1:nrow(testing)) {
        print(paste("Prediction of ID No.", i, ":", testRF[i]))
}