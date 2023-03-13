setwd("~/Michel/Insper/AULAS 4 SEMESTRE/Modelagem Preditiva/APS")



# bibliotecas -------------------------------------------------------------
library(tree)
library(ranger)
library(fastAdaboost)
library(gbm)
library(pROC)

# Churn -------------------------------------------------------------------

churn <- read.csv("churn.csv", stringsAsFactors = TRUE)
str(churn)

idx <- sample(1:nrow(churn), size = round(0.7 * nrow(churn)), replace = FALSE)

training <- churn[idx, ]
test <- churn[-idx, ]




# Regressão logística -----------------------------------------------------

rlog <- glm (Exited ~ ., data = training, family = binomial())

prob_rlog <- predict(rlog, newdata = test, type = "response")

y_hat_rlog <- ifelse(prob_rlog >= 0.5, "Yes", "No")

(erro_rlog <- mean(y_hat_rlog != test$Exited))




# Árvore de classificação -------------------------------------------------

ctree <- tree(Exited ~ . , data = training)

plot(ctree)
text(ctree)

cv <- cv.tree(ctree, FUN = prune.misclass)

plot(cv$size, cv$dev, type = "b", lwd = 2, col = "dark green",
     xlab = "Número de folhas",
     ylab = "Número de erros de classificação",
     main = "Validação cruzada em 10 lotes")

pruned <- prune.misclass(ctree, best = 6)

plot(pruned, type = "uniform")
text(pruned, cex = 0.95)

y_hat_pruned <- predict(pruned, test, type = "class")
mean(y_hat_pruned != test$Exited)



# Random Forest -----------------------------------------------------------

rf <- ranger(Exited ~ ., data = training, probability = TRUE)

prob_rf <- predict(rf, data = test)$predictions[, 2] 
y_hat_rf <- ifelse(prob_rf >= 0.5, "Yes", "No")

mean(y_hat_rf != test$Exited)

table(Predicted = y_hat_rf, Observed = test$Exited)


# Adaboosting -------------------------------------------------------------

boost <- adaboost(Exited ~ ., data = training, nIter = 50)

pred_boost <- predict(boost, newdata = test)
y_hat_boost <- pred_boost$class

(error_boost <- mean(y_hat_boost != test$Exited))


# Curva ROC ---------------------------------------------------------------

roc_rlog <- roc (test$Exited, prob_rlog)

plot.roc(roc_rlog, col = "blue", grid = TRUE, 
         xlab = "FPR (1 - Specificity)",
         ylab = "TPR (Sensitivity)",
         main = "Curva ROC", legacy.axes = TRUE, asp = FALSE, las = 1)


prob_pruned <- predict(pruned, test, type = "vector")[,2]
roc_tree <- roc (test$Exited, prob_pruned)
plot(roc_tree, col = "dark green", add = TRUE)


roc_rf <- roc(test$Exited, prob_rf)
plot(roc_rf, col = "red", add = TRUE)


pr_boost <- pred_boost$prob[,2]
roc_boost <- roc(test$Exited, pr_boost)
plot(roc_boost, col = "black", add = TRUE)

legend("bottomright", legend = c("Random Forest","Adaboosting","Árvore de Classificação","Regressão logística"),
       col = c("red", "black","dark green","blue"), lwd = 2, cex = 1)

auc(roc_rlog)
auc(roc_tree)
auc(roc_rf)
auc(roc_boost)


# used cars ---------------------------------------------------------------

ucars <- read.csv("used_cars.csv", stringsAsFactors = TRUE)
str(ucars)


idx <- sample(1:nrow(ucars), size = round(0.7 * nrow(ucars)), replace = FALSE)

training <- ucars[idx, ]
test <- ucars[-idx, ]



# Regressão linear --------------------------------------------------------

rlin <- lm(price ~ ., data = training)
y_hat_lin <- predict (rlin, newdata = test)
(rmse_lin <- sqrt(mean((y_hat_lin - test$price)^2)))



# Árvore de regressão -----------------------------------------------------

rtree <- tree(price ~ ., data = training)
y_hat_rtree <- predict(rtree, newdata = test)
(rmse_tree <- sqrt(mean((y_hat_rtree - test$price)^2)))



# Random forest -----------------------------------------------------------

rrf <- ranger(price ~ ., data = training)
y_hat_rrf <- predict (rrf, data = test)$predictions
(rmse_rf <- sqrt(mean((y_hat_rrf - test$price)^2)))



# Boosting ----------------------------------------------------------------

l2 <- gbm(price ~ ., data = training, n.trees = 5000,
          distribution = "gaussian", interaction.depth = 6,
          shrinkage = 0.001)
y_hat_l2 <- predict(l2, newdata = test, n.trees = 5000)
(rmse_l2 <- sqrt(mean((y_hat_l2 - test$price)^2)))








