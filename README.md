# loan-default-prediction
Loan Default Prediction Modelling for an imbalanced dataset
# Loan Default Prediction using Machine Learning

## Project Overview

This project uses supervised machine learning techniques to predict the likelihood of a borrower defaulting on a loan. 
It addresses a critical problem in the financial sector; reducing credit risk by identifying high-risk borrowers before loan approval.

By applying machine learning models like Logistic Regression with resampling techniques like RandomOverSampler), the project aims to build a predictive system that improves recall score for the minority class (loan defaulters).


## Dataset

- **Source**: Coursera's Loan Default Prediction Challenge
- **Rows**: 255,347  
- **Columns**: 18  
- **Type**: Structured tabular data
- The data is imbalanced with only 11.6% defaulters

The dataset simulates real-world loan data with features that include applicant demographics, financial history, credit scores, and loan characteristics. The target variable indicates whether a customer **defaulted** (`1`) or **did not default** (`0`).


## Problem Statement

Loan defaulting is a major financial risk. Traditional accuracy-based models fail to detect defaulters due to class imbalance (most customers don’t default). Hence, this project focuses on **F1-score** and **recall** for the minority class using advanced resampling and model tuning techniques.


##  Methods & Techniques

- Data Cleaning and Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering and Selection
- Handling Imbalanced Data using Random Over Sampler

- Models Used:
  - Logistic Regression
  - XGboost
  - Gaussion Naive Bayes

- Evaluation Metrics:
  - Accuracy
  - Recall
  - Confusion Matrix


##  Key Results

- Applied **Random Over Sampler** to improve model performance on minority class
- Achieved significant improvement in **Recall-score** for class `1` (defaulters)
- Explained why **Recall-score** is a better metric than accuracy in Loan default Predicion
