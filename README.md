# Project Summary: Pima Indians Diabetes - EDA & Prediction

## Executive Summary
This project focuses on predicting diabetes among the Pima Indian population using various machine learning techniques. The dataset includes medical predictors such as glucose levels, blood pressure, BMI, and more. By applying exploratory data analysis (EDA) and advanced feature engineering techniques, the project aims to improve the accuracy of diabetes predictions. The models employed include LightGBM and a hybrid LightGBM-KNN approach, achieving accuracies of 89.8% and 90.6% respectively. This report highlights the steps taken, from data preprocessing and feature creation to model evaluation and performance enhancement.

## Table of Contents
1. [Introduction](#1-introduction)
    - [1.1 Project Background](#11-project-background)
    - [1.2 Objectives](#12-objectives)
2. [Data Overview](#2-data-overview)
    - [2.1 Dataset Description](#21-dataset-description)
    - [2.2 Initial Data Exploration](#22-initial-data-exploration)
3. [Data Preprocessing and Feature Engineering](#3-data-preprocessing-and-feature-engineering)
    - [3.1 Handling Missing Values](#31-handling-missing-values)
    - [3.2 Exploratory Data Analysis (EDA)](#32-exploratory-data-analysis-eda)
    - [3.3 Feature Creation](#33-feature-creation)
        - [3.3.1 Polynomial Features](#331-polynomial-features)
        - [3.3.2 Interaction Features](#332-interaction-features)
        - [3.3.3 Binary Features](#333-binary-features)
    - [3.4 Feature Encoding](#34-feature-encoding)
        - [3.4.1 One-Hot Encoding](#341-one-hot-encoding)
        - [3.4.2 Label Encoding](#342-label-encoding)
        - [3.4.3 Frequency Encoding](#343-frequency-encoding)
    - [3.5 Feature Scaling](#35-feature-scaling)
        - [3.5.1 Normalization](#351-normalization)
        - [3.5.2 Standardization](#352-standardization)
4. [Model Building and Evaluation](#4-model-building-and-evaluation)
    - [4.1 LightGBM Model](#41-lightgbm-model)
    - [4.2 LightGBM & KNN Model](#42-lightgbm-knn-model)
    - [4.3 Model Performance Metrics](#43-model-performance-metrics)
5. [Conclusion](#5-conclusion)
6. [References](#6-references)

# 1. Introduction

## 1.1 Project Background
Diabetes is a chronic disease that poses significant health risks worldwide. Among the Pima Indian population, diabetes prevalence is particularly high. This project leverages data science and machine learning techniques to predict diabetes occurrence using a dataset of medical variables collected from the Pima Indians. By enhancing predictive accuracy, the project aims to contribute to early detection and better management of diabetes.

## 1.2 Objectives
The primary objective of this project is to build and evaluate machine learning models that accurately predict diabetes in the Pima Indian population. This involves:
- Conducting exploratory data analysis (EDA) to understand the dataset.
- Preprocessing the data by handling missing values and encoding categorical variables.
- Engineering new features to improve model performance.
- Comparing the performance of different models, including LightGBM and a hybrid LightGBM-KNN model.
- Assessing model performance using various metrics and selecting the best model for deployment.

# 2. Data Overview

## 2.1 Dataset Description
The dataset comprises medical predictor variables and a target variable indicating the presence of diabetes. Key features include the number of pregnancies, glucose levels, blood pressure, skin thickness, insulin levels, BMI, age, and diabetes pedigree function.

## 2.2 Initial Data Exploration
Initial exploration involves inspecting the dataset for missing values, understanding the distribution of features, and examining the correlation between variables. This step sets the foundation for subsequent data preprocessing and feature engineering tasks.

*Figure 1: Glucose Distribution*  
![Feature engineering example](img/glucose_distribution.png)
The above figure shows the distribution of glucose levels among the Pima Indian population. Higher glucose levels are typically associated with a higher likelihood of diabetes.

*Figure 2: Blood Pressure Distribution*  
![Feature engineering example](img/blood_pressure_distribution.png)
The above figure displays the distribution of blood pressure readings in the dataset. Blood pressure is a significant predictor of diabetes.

*Figure 3: Glucose vs. Blood Pressure Scatter Plot with Box*  
![Feature engineering example](img/FeatureEngineering.png)
This scatter plot highlights the relationship between glucose levels and blood pressure, with a boxed region indicating the concentration of healthy individuals.

# 3. Data Preprocessing and Feature Engineering

## 3.1 Handling Missing Values
Missing values in the dataset are addressed using various strategies, including imputation with median values specific to the target class. For example, missing insulin values are replaced by the median insulin levels of diabetic and non-diabetic patients separately.

## 3.2 Exploratory Data Analysis (EDA)
EDA helps uncover patterns, anomalies, and relationships within the data. Techniques such as correlation matrices and distribution plots are used to visualize the data and inform feature engineering decisions.

## 3.3 Feature Creation
Creating new features from existing data can significantly enhance model performance. Techniques used include:
- **Polynomial Features**: Generating higher-order terms to capture non-linear relationships.
- **Interaction Features**: Combining features to capture interactions between variables.
- **Binary Features**: Creating binary indicators based on threshold values, such as age groups or glucose levels.

*Figure 4: N4 Bar Chart*  
![Feature engineering example](img/glucose_bloodpressure_barplot.png)
The bar chart above represents the distribution of a binary feature (N4) which combines glucose levels and blood pressure into a categorical variable.

*Figure 5: N4 Pie Chart*  
![Feature engineering example](img/glucose_bloodpressure_piechart.png)
The pie chart provides a visual representation of the binary feature N4, highlighting the proportion of diabetic vs. non-diabetic individuals.

## 3.4 Feature Encoding
Encoding categorical variables is crucial for machine learning models. Techniques used include:
- **One-Hot Encoding**: Converting categorical variables into binary vectors.
- **Label Encoding**: Assigning numerical labels to categories.
- **Frequency Encoding**: Replacing categories with their frequency counts.

## 3.5 Feature Scaling
Scaling features ensures that all variables contribute equally to the model. Techniques used include:
- **Normalization**: Scaling features to a range of [0, 1].
- **Standardization**: Transforming features to have a mean of 0 and a standard deviation of 1.

# 4. Model Building and Evaluation

## 4.1 LightGBM Model
LightGBM, a gradient boosting framework, is used for its efficiency and accuracy. Hyperparameters are optimized using Random Search CV to enhance model performance.

## 4.2 LightGBM & KNN Model
A hybrid model combining LightGBM and K-Nearest Neighbors (KNN) is employed to further improve accuracy. The best combination of hyperparameters is identified using Grid Search CV.

## 4.3 Model Performance Metrics
Model performance is evaluated using metrics such as accuracy, precision, recall, F1 score, and ROC-AUC. Confusion matrices and precision-recall curves are used to visualize model effectiveness.

*Figure 6: Final Model Performance Report*  
![Feature engineering example](img/EDA_Model_Performance_Report.png)
The final model performance report shows the confusion matrix, various performance metrics, the ROC curve, and the precision-recall curve for the LightGBM & KNN model.

# 5. Conclusion
This project demonstrates the effectiveness of advanced feature engineering and machine learning techniques in predicting diabetes among the Pima Indian population. The use of LightGBM and hybrid models highlights the importance of model selection and hyperparameter tuning in achieving high predictive accuracy.

# 6. References
- Kaggle Notebook by [@vincentlugat](https://www.kaggle.com/code/vincentlugat/pima-indians-diabetes-eda-prediction-0-906)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Exploratory Data Analysis Techniques](https://en.wikipedia.org/wiki/Exploratory_data_analysis)
