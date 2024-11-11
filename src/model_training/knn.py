import json
import os
import pickle
import sys

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing.preprocess_and_split import preprocess_data

# Charger et prétraiter les données
X_train, X_test, y_train, y_test, feature_names, feature_types, encoding_dict = (
    preprocess_data()
)

# Create a K-Nearest Neighbors model (with 5 neighbors, you can adjust this value)
knn_model = KNeighborsClassifier(
    n_neighbors=10,
    weights="uniform",
    algorithm="auto",
    leaf_size=50,
    p=2,
    metric="minkowski",
    metric_params=None,
    n_jobs=None,
)

# Train the model
knn_model.fit(X_train, y_train)

# Make predictions on the training data
y_train_predict = knn_model.predict(X_train)

# Make predictions on the test data
y_test_predict = knn_model.predict(X_test)

# Classification Report for Train Set
print(
    "================================================================================================="
)
print("Classification Report for KNN Model (Train Set):")
print(classification_report(y_train, y_train_predict))

# Classification Report for Test Set
print(
    "================================================================================================="
)
print("Classification Report for KNN Model (Test Set):")
print(classification_report(y_test, y_test_predict))

# Sauvegarder le modèle
with open("models/knn_model.pkl", "wb") as f:
    pickle.dump(knn_model, f)

# Sauvegarder les informations sur les features dans un fichier JSON
feature_info = {
    "feature_names": feature_names,
    "feature_types": {name: str(dtype) for name, dtype in feature_types.items()},
    "encoding_dict": {
        key: {str(k): int(v) for k, v in value.items()}
        for key, value in encoding_dict.items()
    },
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


with open("models/feature_info.json", "w") as f:
    json.dump(feature_info, f, indent=2, default=convert_to_json_serializable)

print("Noms des features:", feature_names)
print("Types des features:", feature_types)
print("Dictionnaire d'encodage:", encoding_dict)
print(
    "Modèle entraîné et sauvegardé avec succès. Informations sur les features sauvegardées dans feature_info.json."
)
