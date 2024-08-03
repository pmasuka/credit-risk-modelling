# Credit Risk Prediction with Machine Learning

This repository contains code and analysis for predicting credit risk using various machine learning models. The project involves preprocessing data, handling class imbalance, and training multiple classifiers to evaluate their performance.

## Table of Contents

- [Overview](#overview)
- [Data](#data)
- [Preprocessing](#preprocessing)
- [Models](#models)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [License](#license)

## Overview

This project aims to predict credit risk using a dataset of credit records. The dataset is used to train and evaluate several machine learning models, including Random Forest, XGBoost, LightGBM, Support Vector Machines (SVM), and Logistic Regression. The goal is to identify the best model for predicting whether a credit applicant is at risk of default.

## Data

The dataset used in this project is `german_credit_data.csv`, which contains information about credit applicants. It includes the following features:

- `Age`
- `Sex`
- `Job`
- `Housing`
- `Saving accounts`
- `Checking account`
- `Credit amount`
- `Duration`
- `Purpose`
- `Risk` (Target variable: `good` or `bad`)

## Preprocessing

1. **Loading Data:** The dataset is read into a Pandas DataFrame.
2. **Handling Missing Values:** Missing values in 'Saving accounts' and 'Checking account' are imputed with the mode.
3. **Exploratory Data Analysis (EDA):** Various plots are used to visualize data distributions and identify potential issues.
4. **Encoding Categorical Variables:** Categorical features are encoded using `LabelEncoder` to convert them into numerical format.
5. **Feature Scaling:** Numerical features are standardized using `StandardScaler`.
6. **Splitting Data:** The dataset is split into training, validation, and testing sets. SMOTE is used to handle class imbalance.

## Models

The following machine learning models are trained and evaluated:

- **Random Forest**
- **XGBoost**
- **LightGBM**
- **Support Vector Machines (SVM)**
- **Logistic Regression**

Hyperparameter tuning is performed for each model using `GridSearchCV` to find the best parameters.

## Evaluation

Model performance is evaluated based on several metrics:

- **Accuracy**
- **F1 Score**
- **ROC AUC Score**
- **Classification Report**
- **Confusion Matrix**

The best model is selected based on the validation score and then evaluated on the test set.
