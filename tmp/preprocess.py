'''This module is processing data to make sure models can train on it'''

# For data manipulation and analysis
import pandas as pd
import numpy as np

# For visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px  # For interactive plotting with Plotly

# Handling warnings
import warnings

# For splitting data into training and testing sets
from sklearn.model_selection import train_test_split

# For preprocessing the data: Label Encoding and Standard Scaling
from sklearn.preprocessing import LabelEncoder, StandardScaler

# For building models
from sklearn.linear_model import LogisticRegression  # Logistic Regression model
from sklearn.neighbors import KNeighborsClassifier   # K-Nearest Neighbors model

# For evaluating models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# For checking data skewness
from scipy.stats import skew

# Additional Plotly imports for advanced plotting and graph creation
import plotly.express as px  # Repeated import of Plotly for consistency
import plotly.graph_objects as go  # For creating detailed graphs
import math
from plotly.subplots import make_subplots  # For creating subplots in Plotly
from numpy import linalg as LA  # Linear algebra functions for advanced computations

# Feature selection
from sklearn.feature_selection import RFE  # Recursive Feature Elimination for feature selection

# Data splitting again (re-import, same as above)
from sklearn.model_selection import train_test_split

# Support Vector Machine for classification
from sklearn.svm import SVC

# For resampling and dealing with imbalanced datasets
from sklearn.utils import resample


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
import graphviz
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

import glob

file_path = glob.glob('**/IBM_data.csv', recursive=True)[0]
df = pd.read_csv(file_path)
print(df.head())

#Data Cleaning & Reduction
df.drop(columns = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], axis=1, inplace=True)

# Data Preprocessing¶
attrition_counts = df['Attrition'].value_counts()
num_cols = df.select_dtypes(include=np.number).columns
numerical_data = df[num_cols]

# Outliers
features_with_outliers = []
for feature in num_cols:
    percentile25 = df[feature].quantile(0.25)
    percentile75 = df[feature].quantile(0.75)
    iqr = percentile75 - percentile25
    upper_limit = percentile75 + 1.5 * iqr
    lower_limit = percentile25 - 1.5 * iqr
    outliers = df[(df[feature] > upper_limit) | (df[feature] < lower_limit)]
    proportion_of_outliers = len(outliers) / len(df) * 100
    if len(outliers) > 0:
        features_with_outliers.append(feature)
        print(f"Feature: {feature}")
        print(f"Number of outliers: {len(outliers)}")
        print(f"Proportion of outliers: {proportion_of_outliers:.2f}%\n")

#Skewness
skewed_features = {}
skewed_columns = []


for feature in num_cols:
    skewness = df[feature].skew()
    skewed_features[feature] = skewness
    if skewness > 0.5:
        print(f"{feature} is right skewed with skewness: {skewness}")

for feature in num_cols:
    if skewed_features[feature] > 0.5:
        skewed_columns.append(feature)
        df[feature] = np.log1p(df[feature])


print("Log transformation applied to right-skewed features.")

df['WorkExperience'] = df[['TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole',
                                'YearsSinceLastPromotion', 'YearsWithCurrManager']].mean(axis=1)

df['OverallSatisfaction'] = (
    df[['JobSatisfaction' , 'EnvironmentSatisfaction', 'RelationshipSatisfaction', 'WorkLifeBalance']].mean(axis=1)
)

df=df.drop(['TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole',
            'YearsSinceLastPromotion', 'JobSatisfaction' , 'EnvironmentSatisfaction',
            'RelationshipSatisfaction', 'WorkLifeBalance'],axis=1)


#Categorical Encoding
cat_cols = df.select_dtypes(include = ['object'])

# Create a LabelEncoder object
label_encoder = LabelEncoder()
categorical_cols = cat_cols
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

print(df.head())
print(df.columns)
