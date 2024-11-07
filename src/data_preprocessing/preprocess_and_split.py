"""
This module processes data for employee attrition prediction.
It includes functions for loading, cleaning, preprocessing, and splitting the data.
"""

import glob
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

def load_data():
    """
    Load the data from a CSV file.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    file_path = glob.glob('**/IBM_data.csv', recursive=True)[0]
    return pd.read_csv(file_path)

def clean_data(df):
    """
    Clean the data by dropping unnecessary columns.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    df.drop(columns=['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], axis=1, inplace=True)
    return df

def handle_outliers(df):
    """
    Detect and print information about outliers in numerical features.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        list: A list of features that contain outliers.
    """
    num_cols = df.select_dtypes(include=np.number).columns
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
    return features_with_outliers

def handle_skewness(df):
    """
    Handle skewness in numerical features by applying log transformation.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        tuple: A tuple containing the transformed DataFrame and a list of skewed columns.
    """
    num_cols = df.select_dtypes(include=np.number).columns
    skewed_features = {}
    skewed_columns = []
    for feature in num_cols:
        skewness = df[feature].skew()
        skewed_features[feature] = skewness
        if skewness > 0.5:
            print(f"{feature} is right skewed with skewness: {skewness}")
            skewed_columns.append(feature)
            df[feature] = np.log1p(df[feature])
    print("Log transformation applied to right-skewed features.")
    return df, skewed_columns

def feature_engineering(df):
    """
    Perform feature engineering by creating new features and dropping redundant ones.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with engineered features.
    """
    df['WorkExperience'] = df[['TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole',
                               'YearsSinceLastPromotion', 'YearsWithCurrManager']].mean(axis=1)
    df['OverallSatisfaction'] = df[['JobSatisfaction', 'EnvironmentSatisfaction', 
                                    'RelationshipSatisfaction', 'WorkLifeBalance']].mean(axis=1)
    df = df.drop(['TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole',
                  'YearsSinceLastPromotion', 'JobSatisfaction', 'EnvironmentSatisfaction',
                  'RelationshipSatisfaction', 'WorkLifeBalance'], axis=1)
    return df

def encode_categorical(df):
    """
    Encode categorical variables using LabelEncoder.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with encoded categorical variables.
    """
    label_encoder = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = label_encoder.fit_transform(df[col])
    return df

def split_data(df, target_column='Attrition'):
    """
    Split the data into features and target, then into train and test sets.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def scale_features(X_train, X_test):
    """
    Scale the features using StandardScaler.

    Args:
        X_train (np.array): Training features.
        X_test (np.array): Testing features.

    Returns:
        tuple: Scaled X_train and X_test.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def apply_smote(X_train, y_train):
    """
    Apply SMOTE to balance the classes in the training data.

    Args:
        X_train (np.array): Training features.
        y_train (np.array): Training labels.

    Returns:
        tuple: Resampled X_train and y_train.
    """
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled

def preprocess_data():
    """
    Main function to preprocess the data.

    This function orchestrates the entire preprocessing pipeline including
    loading, cleaning, handling outliers and skewness, feature engineering,
    encoding categorical variables, splitting the data, scaling features,
    and applying SMOTE.

    Returns:
        tuple: X_train_resampled, X_test_scaled, y_train_resampled, y_test
    """
    df = load_data()
    print("Data loaded. Shape:", df.shape)
    
    df = clean_data(df)
    print("Data cleaned. Shape:", df.shape)
    
    features_with_outliers = handle_outliers(df)
    print(f"Features with outliers: {features_with_outliers}")
    
    df, skewed_columns = handle_skewness(df)
    print(f"Skewed columns handled: {skewed_columns}")
    
    df = feature_engineering(df)
    print("Feature engineering completed. Shape:", df.shape)
    
    df = encode_categorical(df)
    print("Categorical variables encoded.")
    
    X_train, X_test, y_train, y_test = split_data(df)
    print("Data split into train and test sets.")
    
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    print("Features scaled.")
    
    X_train_resampled, y_train_resampled = apply_smote(X_train_scaled, y_train)
    print("SMOTE applied to balance classes.")
    
    return X_train_resampled, X_test_scaled, y_train_resampled, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data()
    print("Preprocessing completed. Data is ready for modeling.")