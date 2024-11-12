import json
import os
import pickle
import sys

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models.signature import infer_signature
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing.preprocess_and_split import preprocess_data

# Configurez MLflow
mlflow.set_tracking_uri("http://localhost:5000")  # Ajustez l'URL si nécessaire
mlflow.set_experiment("KNN_Employee_Attrition")

# Charger et prétraiter les données
X_train, X_test, y_train, y_test, feature_names, feature_types, encoding_dict = (
    preprocess_data()
)

# Create a K-Nearest Neighbors model (with 5 neighbors, you can adjust this value)
knn_model = KNeighborsClassifier(
    n_neighbors=10,
    weights="uniform",
    algorithm="auto",
    leaf_size=150,
    p=2,
    metric="minkowski",
    metric_params=None,
    n_jobs=None,
)
with mlflow.start_run():
    # Entraînez le modèle
    knn_model.fit(X_train, y_train)

    # Faites des prédictions
    y_train_predict = knn_model.predict(X_train)
    y_test_predict = knn_model.predict(X_test)

    # Calculez les métriques
    train_accuracy = accuracy_score(y_train, y_train_predict)
    test_accuracy = accuracy_score(y_test, y_test_predict)

    # Enregistrez les paramètres du modèle
    mlflow.log_param("n_neighbors", knn_model.n_neighbors)
    mlflow.log_param("weights", knn_model.weights)
    mlflow.log_param("algorithm", knn_model.algorithm)
    mlflow.log_param("leaf_size", knn_model.leaf_size)

    # Enregistrez les métriques
    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("test_accuracy", test_accuracy)

    # Enregistrez le rapport de classification comme artefact
    train_report = classification_report(y_train, y_train_predict)
    test_report = classification_report(y_test, y_test_predict)
    with open("classification_report.txt", "w") as f:
        f.write("Train Set:\n")
        f.write(train_report)
        f.write("\nTest Set:\n")
        f.write(test_report)
    mlflow.log_artifact("classification_report.txt")
# # Train the model
# knn_model.fit(X_train, y_train)

# # Make predictions on the training data
# y_train_predict = knn_model.predict(X_train)

# # Make predictions on the test data
# y_test_predict = knn_model.predict(X_test)

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

# Enregistrez le modèle
signature = infer_signature(X_train, y_train_predict)
mlflow.sklearn.log_model(knn_model, "knn_model", signature=signature)

print("Model trained and logged to MLflow")


# Sauvegarder le modèle
with open("models/knn_modelmlflow.pkl", "wb") as f:
    pickle.dump(knn_model, f)

# # Sauvegarder les informations sur les features dans un fichier JSON
# feature_info = {
#     "feature_names": feature_names,
#     "feature_types": {name: str(dtype) for name, dtype in feature_types.items()},
#     "encoding_dict": {
#         key: {str(k): int(v) for k, v in value.items()}
#         for key, value in encoding_dict.items()
#     },
# }


# def convert_to_json_serializable(obj):
#     if isinstance(obj, np.integer):
#         return int(obj)
#     elif isinstance(obj, np.floating):
#         return float(obj)
#     elif isinstance(obj, np.ndarray):
#         return obj.tolist()
#     else:
#         return obj


# with open("models/feature_info.json", "w") as f:
#     json.dump(feature_info, f, indent=2, default=convert_to_json_serializable)

# print("Noms des features:", feature_names)
# print("Types des features:", feature_types)
# print("Dictionnaire d'encodage:", encoding_dict)
# print(
#     "Modèle entraîné et sauvegardé avec succès"
# )
