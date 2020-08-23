Prediction Assignment: Exercise
===============================

Overview
--------

The main goal of the project is to predict the manner in which 6
participants performed some exercise as described below. This is the
“classe” variable in the training set. The machine learning algorithm
described here is applied to the 20 test cases available in the test
data and the predictions are submitted in appropriate format to the
Course Project Prediction Quiz for automated grading.

Background
----------

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement - a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it. In this project, your goal
will be to use data from accelerometers on the belt, forearm, arm, and
dumbell of 6 participants. They were asked to perform barbell lifts
correctly and incorrectly in 5 different ways. More information is
available from the website here:
(<a href="http://groupware.les.inf.puc-rio.br/har" class="uri">http://groupware.les.inf.puc-rio.br/har</a>)
(see the section on the Weight Lifting Exercise Dataset).

Read more:
(<a href="http://groupware.les.inf.puc-rio.br/har#ixzz3xsbS5bVX" class="uri">http://groupware.les.inf.puc-rio.br/har#ixzz3xsbS5bVX</a>)

Data Loading and Cleaning
-------------------------

### 1 Dataset overview

The training data for this project are available here:

(<a href="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv" class="uri">https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv</a>)

The test data are available here:

(<a href="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv" class="uri">https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv</a>)

The data for this project come from
<a href="http://groupware.les.inf.puc-rio.br/har" class="uri">http://groupware.les.inf.puc-rio.br/har</a>.
Full source:

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H.
“Qualitative Activity Recognition of Weight Lifting Exercises.
Proceedings of 4th International Conference in Cooperation with SIGCHI
(Augmented Human ’13)”. Stuttgart, Germany: ACM SIGCHI, 2013.

My special thanks to the above mentioned authors for being so generous
in allowing their data to be used for this kind of assignment.

A short description of the datasets content from the authors’ website:

“Six young health participants were asked to perform one set of 10
repetitions of the Unilateral Dumbbell Biceps Curl in five different
fashions: exactly according to the specification (Class A), throwing the
elbows to the front (Class B), lifting the dumbbell only halfway (Class
C), lowering the dumbbell only halfway (Class D) and throwing the hips
to the front (Class E).

Class A corresponds to the specified execution of the exercise, while
the other 4 classes correspond to common mistakes. Participants were
supervised by an experienced weight lifter to make sure the execution
complied to the manner they were supposed to simulate. The exercises
were performed by six male participants aged between 20-28 years, with
little weight lifting experience. We made sure that all participants
could easily simulate the mistakes in a safe and controlled manner by
using a relatively light dumbbell (1.25kg)."

### Environment setup

Loading the reqired libraries

    library(knitr)
    library(caret)

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    library(rpart)
    library(rpart.plot)
    library(rattle)

    ## Loading required package: tibble

    ## Loading required package: bitops

    ## Rattle: A free graphical interface for data science with R.
    ## Version 5.4.0 Copyright (c) 2006-2020 Togaware Pty Ltd.
    ## Type 'rattle()' to shake, rattle, and roll your data.

    library(randomForest)

    ## randomForest 4.6-14

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:rattle':
    ## 
    ##     importance

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

    library(corrplot)

    ## corrplot 0.84 loaded

    set.seed(12345)

### Data loading

The next step is loading the dataset from the URL provided above. The
training dataset is then partinioned in 2 to create a Training set (70%
of the data) for the modeling process and a Test set (with the remaining
30%) for the validations. The testing dataset is not changed and will
only be used for the quiz results generation.

    UrlTrain <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    UrlTest  <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

    training <- read.csv(url(UrlTrain))
    testing  <- read.csv(url(UrlTest))

    inTrain  <- createDataPartition(training$classe, p=0.7, list=FALSE)
    TrainSet <- training[inTrain, ]
    TestSet  <- training[-inTrain, ]
    dim(TrainSet)

    ## [1] 13737   160

    dim(TestSet)

    ## [1] 5885  160

Both created datasets have 160 variables. Those variables have plenty of
NA, that can be removed with the cleaning procedures below. The Near
Zero variance (NZV) variables are also removed and the ID variables as
well.

    NZV <- nearZeroVar(TrainSet)
    TrainSet <- TrainSet[, -NZV]
    TestSet  <- TestSet[, -NZV]
    dim(TrainSet)

    ## [1] 13737   104

    dim(TestSet)

    ## [1] 5885  104

Remove NA variables:

    AllNA    <- sapply(TrainSet, function(x) mean(is.na(x))) > 0.95
    TrainSet <- TrainSet[, AllNA==FALSE]
    TestSet  <- TestSet[, AllNA==FALSE]
    dim(TrainSet)

    ## [1] 13737    59

    dim(TestSet)

    ## [1] 5885   59

    #remove ID variables ie coloumn 1-5
    TrainSet <- TrainSet[, -(1:5)]
    TestSet  <- TestSet[, -(1:5)]
    dim(TrainSet)

    ## [1] 13737    54

    dim(TestSet)

    ## [1] 5885   54

With above process we have reduced our variables to 54

Exploratory Analysis
--------------------

### Correlation analysis

A correlation among variables is analysed before proceeding to the
modeling procedures.

    corMatrix <- cor(TrainSet[, -54])
    corrplot(corMatrix, order = "FPC", method = "color", type = "lower", 
             tl.cex = 0.8, tl.col = rgb(0, 0, 0))

![](project_files/figure-markdown_strict/unnamed-chunk-6-1.png)

The highly correlated variables are shown in dark colors in the graph
above. To make an even more compact analysis, a PCA (Principal
Components Analysis) could be performed as pre-processing step to the
datasets.As the correlations are quite few, this step is Omitted for
this assignment.

Predictive Model Building
-------------------------

Three methods will be applied to model the regressions (in the Train
dataset) and the best one (with higher accuracy when applied to the Test
dataset) will be used for the quiz predictions. The methods are:  
Random Forests  
Decision Tree and  
Generalized Boosted Model, as described below.

A Confusion Matrix is plotted at the end of each analysis to better
visualize the accuracy of the models.

### a)Random Forest:

    set.seed(12345)
    controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
    modFitRandForest <- train(classe ~ ., data=TrainSet, method="rf",
                              trControl=controlRF)
    modFitRandForest$finalModel

    ## 
    ## Call:
    ##  randomForest(x = x, y = y, mtry = param$mtry) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 27
    ## 
    ##         OOB estimate of  error rate: 0.23%
    ## Confusion matrix:
    ##      A    B    C    D    E  class.error
    ## A 3904    2    0    0    0 0.0005120328
    ## B    6 2647    4    1    0 0.0041384500
    ## C    0    5 2391    0    0 0.0020868114
    ## D    0    0    9 2243    0 0.0039964476
    ## E    0    0    0    5 2520 0.0019801980

    predictRandForest <- predict(modFitRandForest, newdata=TestSet)
    conf_mat_rf<-confusionMatrix(predictRandForest, as.factor(TestSet$classe))
    conf_mat_rf

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1674    1    0    0    0
    ##          B    0 1138    2    0    0
    ##          C    0    0 1024    2    0
    ##          D    0    0    0  962    1
    ##          E    0    0    0    0 1081
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.999           
    ##                  95% CI : (0.9978, 0.9996)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9987          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9991   0.9981   0.9979   0.9991
    ## Specificity            0.9998   0.9996   0.9996   0.9998   1.0000
    ## Pos Pred Value         0.9994   0.9982   0.9981   0.9990   1.0000
    ## Neg Pred Value         1.0000   0.9998   0.9996   0.9996   0.9998
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2845   0.1934   0.1740   0.1635   0.1837
    ## Detection Prevalence   0.2846   0.1937   0.1743   0.1636   0.1837
    ## Balanced Accuracy      0.9999   0.9994   0.9988   0.9989   0.9995

    plot(conf_mat_rf$table, col = conf_mat_rf$byClass, 
         main = paste("Random Forest - Accuracy =",
                      round(conf_mat_rf$overall['Accuracy'], 4)))

![](project_files/figure-markdown_strict/unnamed-chunk-9-1.png)

### b) Decision Tree

    set.seed(12345)
    modFitDecTree <- rpart(classe ~ ., data=TrainSet, method="class")
    fancyRpartPlot(modFitDecTree)

![](project_files/figure-markdown_strict/unnamed-chunk-10-1.png)

    predictDecTree <- predict(modFitDecTree, newdata=TestSet, type="class")
    confMatDecTree <- confusionMatrix(predictDecTree, as.factor(TestSet$classe))
    confMatDecTree

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1502  201   59   66   74
    ##          B   58  660   37   64  114
    ##          C    4   66  815  129   72
    ##          D   90  148   54  648  126
    ##          E   20   64   61   57  696
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.7342          
    ##                  95% CI : (0.7228, 0.7455)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.6625          
    ##                                           
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.8973   0.5795   0.7943   0.6722   0.6433
    ## Specificity            0.9050   0.9425   0.9442   0.9151   0.9579
    ## Pos Pred Value         0.7897   0.7074   0.7505   0.6079   0.7751
    ## Neg Pred Value         0.9568   0.9033   0.9560   0.9344   0.9226
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2552   0.1121   0.1385   0.1101   0.1183
    ## Detection Prevalence   0.3232   0.1585   0.1845   0.1811   0.1526
    ## Balanced Accuracy      0.9011   0.7610   0.8693   0.7936   0.8006

    plot(confMatDecTree$table, col = confMatDecTree$byClass, 
         main = paste("Decision Tree - Accuracy =",
                      round(confMatDecTree$overall['Accuracy'], 4)))

![](project_files/figure-markdown_strict/unnamed-chunk-12-1.png)

### c) Generalised Boosted Method

    set.seed(12345)
    controlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
    modFitGBM  <- train(classe ~ ., data=TrainSet, method = "gbm",
                        trControl = controlGBM, verbose = FALSE)
    modFitGBM$finalModel

    ## A gradient boosted model with multinomial loss function.
    ## 150 iterations were performed.
    ## There were 53 predictors of which 53 had non-zero influence.

    predictGBM <- predict(modFitGBM, newdata=TestSet)
    confMatGBM <- confusionMatrix(predictGBM, as.factor(TestSet$classe))
    confMatGBM

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1668   12    0    1    0
    ##          B    6 1115   12    1    3
    ##          C    0   12 1012   21    0
    ##          D    0    0    2  941    6
    ##          E    0    0    0    0 1073
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9871          
    ##                  95% CI : (0.9839, 0.9898)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9837          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9964   0.9789   0.9864   0.9761   0.9917
    ## Specificity            0.9969   0.9954   0.9932   0.9984   1.0000
    ## Pos Pred Value         0.9923   0.9807   0.9684   0.9916   1.0000
    ## Neg Pred Value         0.9986   0.9949   0.9971   0.9953   0.9981
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2834   0.1895   0.1720   0.1599   0.1823
    ## Detection Prevalence   0.2856   0.1932   0.1776   0.1613   0.1823
    ## Balanced Accuracy      0.9967   0.9871   0.9898   0.9873   0.9958

    plot(confMatGBM$table, col = confMatGBM$byClass, 
         main = paste("GBM - Accuracy =", round(confMatGBM$overall['Accuracy'], 4)))

![](project_files/figure-markdown_strict/unnamed-chunk-15-1.png)

Applying selected model to test data
------------------------------------

The accuracy of the 3 regression modeling methods above are:  
1. Random Forest: 99.6%  
2. Decision Tree: 73.68%  
3. GBM: 98.57%
