'''This module is processing data to make sure models can train on it'''
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing.preprocess import df

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

X = df.drop('Attrition', axis=1)  # Selecting all columns except the target
y = df['Attrition']                # Selecting the target column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Solve imbalance data problem, make sure minority class are well represented. 
#Pour chaque exemple de la classe minoritaire, il trouve ses k plus proches voisins (par défaut k=5).

smote = SMOTE() #(Synthetic Minority Over-sampling Technique)
print("Before Smoote" , y_train.value_counts())

X_train, y_train = smote.fit_resample(X_train, y_train)

print("\n After Smoote" , y_train.value_counts())



# splitting.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data():
    # Chargez vos données ici
    df = pd.read_csv('chemin/vers/votre/fichier.csv')
    
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    return X_train, X_test, y_train, y_test

# Ne pas exécuter le code ici, juste définir la fonction