<!-- markdownlint-disable -->
<h1 align="center">
   Practical Application Assignment 17.1: Comparing Classifiers 
    <br>
</h1>

<p align="center">
    <strong>🏆&nbsp; A comparison of classifiers: LogisticRegression, KNN, DecisionTree and Support Vector Machine.</strong>
</p>

<p align="center">
    <a href="https://github.com/pnanyaduba/kraftwerk/tree/main/practical_application_II_starter" title="Best-of-badge"><img src="https://ci.appveyor.com/api/projects/status/32r7s2skrgm9ubva?svg=true&passingText=master%20-%20OK"></a>
    <a href="#Contents" title="Project Count"><img src="https://img.shields.io/badge/projects-2nd-blue.svg?color=5ac4bf"></a>
    <a href="#Contribution" title="Contributions are welcome"><img src="https://img.shields.io/badge/contributions-welcome-green.svg"></a>
    <a href="#" title="Best-of Updates"><img src="https://img.shields.io/github/release-date/ml-tooling/best-of-ml-python?color=green&label=updated"></a>
    <a href="https://twitter.com/peteberc" title="Follow on Twitter"><img src="https://img.shields.io/twitter/follow/mltooling.svg?style=social&label=Follow"></a>
</p>

In this application, I explored a Portuguese Bank Marketing Data Set from the following site: [UCI Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing). Our goal is to use about four classifiers in modelling the data and then compare the performance of each of the classifiers. 

#### Business Objective
The business objective is to predict if a customer will subscribe a term deposit.

<h2 align="center">
   **Selected Model: the K-Nearest Neighbors (KNN) model was selected**
    <br>
</h2>

---

<p align="center">
     🧙‍♂️&nbsp; Click here to access the Project Notebook <a href="https://github.com/scriptstorydude/PortugeseBank/blob/main/PortugueseBank.ipynb">Practical App 3</a> <br>
   
</p>

---

## Report
The Analysis of the Logistic Regress, Decision Tree Classifier, KNeighborsClassifier and the Support Vector Machines was performed according to the following criteria
1. Imbalance Class Handling
2. Model Training Speed
3. Interpretable Results

Other criteria observed include
1. Accuracy
2. Precision
3. Recall
4. Specificity
5. Mean Squared Error

- **Logistic Regression Classifier**
    - SMOTE was used to handle imbalanced classes
    - Speed of Training is moderately low at 0.082s
    - Train Score performs slightly better than the Test Score
    - Accuracy and Specificity are not too high at 59. and 34.
    - Precision is low at 56.%
    - Recall is high at 85.%
    - Train and Test MSEs are relatively equal at .40
  
- **Decision Tree Classifier**
    - SMOTE was used to handle imbalanced classes
    - Speed of Training is moderately high at 0.085s
    - Accuracy and Specificity are very high at 75.7% and 69.5% respectively
    - Train Score at .59 performs slightly better than the Test Score  at .58
    - Precision is high at 72.7%
    - Recall is low at 82%
    - However, the Decision Tree Classifier appears to overfit as Train MSE(.1869) is lower than Test MSE (.2426)

- **KNearest Neighbors Classifier**
    - SMOTE was used to handle imbalanced classes
    - Speed of Training is high at 0.036s
    - Accuracy and Specificity are very high at 72% and 66% respectively
    - Train Score (.24) is slightly lower than the Test Score (.28)
    - Precision is high at 69.7%
    - Recall is low at 77.9%
    - However, the KNNeighbors Classifier appears to slightly overfit since Test MSE is higher than Train MSE
      
- **Support Vector Machine**
    - SMOTE was used to handle imbalanced classes
    - Speed of Training is least at .58s
    - Accuracy and Specificity is high at 59.51% and 52.09% respectively
    - Test Score (.632) is slightly lower than Train Score (.633)
    - Precision is 56.50.%
    - Recall is high at 67.56.%
    - Train and Test MSE are relatively equal


### **Select Best Model: K-Nearest Neighbors**
Based on the provided metrics, the K-Nearest Neighbors (KNN) model appears to be the most suitable choice. Here's an analysis of the key metrics:

Test Score (Accuracy): KNN achieved approximately 72.12%, which is higher than the other models.

Precision: At 69.72%, KNN demonstrates a good balance between true positive predictions and false positives.

Recall: With a recall of 77.91%, KNN effectively identifies actual positive cases.

Specificity: KNN has a specificity of 66.37%, indicating its ability to correctly identify negative cases.

Train and Test Mean Squared Error (MSE): KNN shows moderate MSE values, suggesting reasonable prediction errors.

While the Logistic Regression model has a slightly lower test score and precision, it exhibits higher specificity. The Decision Tree model shows higher accuracy but may be prone to overfitting, as indicated by the significant difference between train and test scores. The Support Vector Machine (SVM) model has comparable metrics but requires longer training time.

Considering the balance between accuracy, precision, recall, and training time, K-Nearest Neighbors stands out as the best model among the options evaluated.

## Contents on Jupyter Notebook Steps

- [Jupyter Notebook Link](https://github.com/pnanyaduba/PortugueseBank/blob/main/PortugueseBank.ipynb)
- [Jupyter Notebook Link](https://github.com/scriptstorydude/PortugueseBank/blob/main/PortugueseBank.ipynb)
- (https://github.com/scriptstorydude/PortugeseBank/blob/main/PortugueseBank.ipynb)
- Information about the data
- Dropping Unwanted Columns
- Encode the whole categorical columns
- Get the Correlation Matrix
- Plot the Scatter Matrix
- Define the Modelling Data
- Perform Principal Component Analysis of the Scaled Data
- Create four models - Logistic Regression, DecisionTree, KNN and SVC
- Evaluate the four models
- Summarize the results of the Evaluations
- Select the Best Model out of the four based on the summary
- Create a Report of the Analysis

<br>

## Machine Learning Libraries Used on Jupyter Notebook

- statsmodels.tsa.filters.filtertools as ft
- sklearn.metrics import mean_squared_error
- statsmodels.tsa.filters.filtertools import convolution_filter
- sklearn.feature_selection import SequentialFeatureSelector
- statsmodels.tsa.seasonal import _extrapolate_trend
- pandas.testing as tm
- statsmodels.tsa.arima_process as arima_process
- statsmodels.graphics.tsaplots as tsaplots
- numpy as np
- pandas as pd
- matplotlib.pyplot as plt
- statsmodels.api as sm
- statsmodels.tsa.seasonal import seasonal_decompose
- sklearn.preprocessing import OneHotEncoder
- sklearn.pipeline import Pipeline
- sklearn.preprocessing import StandardScaler
- sklearn.impute import SimpleImputer
- sklearn.compose import ColumnTransformer
- sklearn.decomposition import PCA
- sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
- sklearn.pipeline import make_pipeline
- sklearn.model_selection import train_test_split, GridSearchCV
- sklearn.preprocessing import PolynomialFeatures, OrdinalEncoder
- sklearn.linear_model import Ridge
- scipy import stats
- scipy.linalg import svd
- warnings
- warnings.filterwarnings('ignore')
