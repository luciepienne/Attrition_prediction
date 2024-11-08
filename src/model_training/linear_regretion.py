import pickle
import numpy as np
import json

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing.preprocess_and_split import preprocess_data

# Charger et prétraiter les données
X_train, X_test, y_train, y_test, feature_names, feature_types, encoding_dict = preprocess_data()

# Create a Logistic Regression model
lr_model = LogisticRegression()

# Train the model
lr_model.fit(X_train, y_train)
y_train_predict = lr_model.predict(X_train)

# Make predictions on the test data

y_test_predict = lr_model.predict(X_test)

# Classification Report
print("=================================================================================================")
print("Classification Report for Logistic Regression Model (Train Set):")
print(classification_report(y_train, y_train_predict))

print("=================================================================================================")
print("Classification Report for Logistic Regression Model (Test Set):")
print(classification_report(y_test, y_test_predict))

with open('models/linear_regression_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
print("Models trained and saved successfully.")

# Sauvegarder les informations sur les features dans un fichier JSON
feature_info = {
    'feature_names': feature_names,
    'feature_types': {name: str(dtype) for name, dtype in feature_types.items()},
    'encoding_dict': {
        key: {str(k): int(v) for k, v in value.items()}
        for key, value in encoding_dict.items()
    }
}

def convert_to_json_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

with open('models/feature_info_lr.json', 'w') as f:
    json.dump(feature_info, f, indent=2, default=convert_to_json_serializable)

print("Noms des features:", feature_names)
print("Types des features:", feature_types)
print("Dictionnaire d'encodage:", encoding_dict)
print("Modèle entraîné et sauvegardé avec succès. Informations sur les features sauvegardées dans feature_info.json.")