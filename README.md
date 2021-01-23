# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains data about a marketing campaign based on calls of an specific bank, and we seek to predict if the client has subscribed to a bank product. 
Afer executing both methods, AUTO ML and HyperDrive Hyperparameter Tuning, the best performing model was a VotingEnsemble with AutoML, with an accuracy of 0.9154, vs the LogisticRegression with Hyperparameter Tuning (C: 4, max_iter: 30) obtaining 0.9122.
The HyperParameter chosen were:
* C: Inverse of regularization strength, in order handle a possible overfitting issue. That means smaller values specify stronger regularization. Link about regularization: https://en.wikipedia.org/wiki/Regularization_%28mathematics%29
* max_iter: Maximum number of iterations used.

The AutoML method tested several models including RandomForest, LightGBM, XGBoostClassifier between others:
![Alt text](Automl.JPG?raw=true)

And the most accurated model, VotingEnsemble, is in fact a machine learning model that combines the predictions from multiple other models. The following depicts all the models that AutoML method used.


## Scikit-learn Pipeline
The pipeline used with Scikit-learn was a LogisticRegression algorithm including the following steps, after defining the compute cluster to use:
1. Obtain the dataset from https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv
2. Clean the data using One Hot Encoding for categorical features
3. Configure the parameter sampler (Random) in order to tune the C and max-iter parameters. The RandomParameterSampling is faster than a complete GridSearch.
4. Specify the early stopping policy, based on slack criteria, and a frequency and delay interval for evaluation. Basically it will discard any run if the policy does not meet. 

The policy that I´ve used was BanditPolicy, due to it helps defining a minimum required improvement in order to continue with the parameter search. So, if the minimum requirement does not met, the process stop and we save valuable computational time.

5. Configuration of the HyperDriveConfig using the estimator
6. Execution of the Experiment

## AutoML
In order to complete the AutoML implementation, the following steps were necessary:
1. Obtain the dataset from https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv
2. Clean the data using One Hot Encoding for categorical features.
3. As the auto_ml config requires a Dataset, we need to convert the cleaned dataframes (x and y) into a Dataset.
4. Configure the AutoMl run  specifying the task (classification), the metric (accuracy) and the number of cross validation between others.
5. Execution of the experiment.  

## Pipeline comparison
The AutoML results in a slightly better accuracy as it tested the several algorithms with different parameters, but it takes much longer to complete.

## Future work
Obtaining an accuracy of 0.9122 is relatively good, but we could an exhaustive search of the algorithm in combination with other hyperparameters, for instance solver and penalty, in order to achieve better results.
