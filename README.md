# Diabetes Prediction Classification

## Project Overview

This project aims to predict the onset of diabetes based on a set of medical diagnostic measurements. A binary classification approach is employed using various machine learning models to determine whether a patient has diabetes or not. The project covers the entire machine learning workflow, including data exploration, preprocessing, model training, hyperparameter tuning, and a comparative evaluation of different classification algorithms.

## Dataset
[Kaggle Link](https://www.kaggle.com/code/emirhanhasrc/classification-eda-different-classifiers?scriptVersionId=252639794)
The dataset used is the **PIMA Indians Diabetes Database**. It contains several medical predictor variables and one target variable, `Outcome`.

### Features

| Feature                    | Description                                                              |
| -------------------------- | ------------------------------------------------------------------------ |
| **Pregnancies**            | Number of times pregnant                                                 |
| **Glucose**                | Plasma glucose concentration a 2 hours in an oral glucose tolerance test |
| **BloodPressure**          | Diastolic blood pressure (mm Hg)                                         |
| **SkinThickness**          | Triceps skin fold thickness (mm)                                         |
| **Insulin**                | 2-Hour serum insulin (mu U/ml)                                           |
| **BMI**                    | Body mass index (weight in kg/(height in m)^2)                           |
| **DiabetesPedigreeFunction** | A function that scores the likelihood of diabetes based on family history. |
| **Age**                    | Age in years                                                             |
| **Outcome**                | The target variable (0 for non-diabetic, 1 for diabetic)                 |

## Project Workflow

The project is structured into the following key steps:

### 1. Exploratory Data Analysis (EDA)

The initial phase involved a thorough exploration of the dataset to uncover patterns, anomalies, and relationships between features.
*   **Missing Value Identification**: An initial `describe()` of the data revealed that several columns (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) had minimum values of 0. These are physiologically improbable and were treated as missing values.
*   **Data Visualization**:
    *   `sns.pairplot()` was used to visualize the relationships between all features, color-coded by the `Outcome` variable.
    *   `sns.histplot()` helped in understanding the distribution of key features like `Age` and `DiabetesPedigreeFunction`.
    *   `sns.boxenplot()` provided insights into the age distribution for both diabetic and non-diabetic patients.
    *   `sns.heatmap()` of the correlation matrix was generated to understand the linear relationships between variables.

 <!-- Not: Gerçek bir resim URL'si eklemeniz gerekir -->

### 2. Data Preprocessing

Before model training, the data was preprocessed to handle inconsistencies and prepare it for the models.
*   **Splitting Data**: The dataset was split into features (`X`) and the target variable (`y`), followed by a standard train-test split.
*   **Missing Value Imputation**: To handle the zero-values identified during EDA, a median imputation strategy was employed. For each column with missing values, the `0`s were replaced with the median of that column calculated *only from the training set* to prevent data leakage.
*   **Feature Scaling**: `StandardScaler` was applied to both the training and testing sets. This scales the features to have a mean of 0 and a standard deviation of 1, which is crucial for distance-based algorithms like SVM and KNN, and generally beneficial for many others.

### 3. Model Training and Hyperparameter Tuning

An initial model was built, tuned, and then compared against a suite of other common classification models.
*   **AdaBoost Classifier**:
    1.  An initial `AdaBoostClassifier` was trained, achieving an accuracy of **75.97%**.
    2.  `GridSearchCV` was used to find the optimal hyperparameters for `n_estimators` and `learning_rate` to potentially improve the model's performance.

*   **Comparative Model Analysis**: After preprocessing and scaling the data, the following models were trained and evaluated:
    *   Logistic Regression
    *   K-Nearest Neighbors (KNN)
    *   Support Vector Machine (SVM)
    *   Gaussian Naive Bayes
    *   Random Forest

For each model, a classification report and a confusion matrix were generated to assess its performance thoroughly.

 <!-- Not: Gerçek bir resim URL'si eklemeniz gerekir -->
 <!-- Not: Gerçek bir resim URL'si eklemeniz gerekir -->

## Results

The performance of each model was evaluated based on accuracy, precision, recall, and F1-score. Below is a summary of the results on the test set. The scores for the positive class (1 - Diabetes) are highlighted.

| Model                 | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
| --------------------- | :------: | :-----------------: | :--------------: | :----------------: |
| **AdaBoost Classifier** | **0.760**|        0.59         |     **0.65**     |      **0.62**      |
| **Logistic Regression** | **0.753**|        0.59         |       0.59       |        0.59        |
| SVM                   |  0.747   |        0.58         |       0.57       |        0.57        |
| Random Forest         |  0.727   |        0.54         |       0.59       |        0.56        |
| Naive Bayes           |  0.721   |        0.53         |       0.57       |        0.55        |
| KNN                   |  0.695   |        0.49         |       0.57       |        0.53        |

### Conclusion

Based on the evaluation, the **AdaBoost Classifier** and **Logistic Regression** models provided the best performance. The AdaBoost model slightly outperformed the others, particularly in its ability to correctly identify diabetic patients (Recall of 0.65 for Class 1).

The project successfully demonstrates a complete pipeline for a classification task, emphasizing the importance of careful EDA and preprocessing for building robust machine learning models.
