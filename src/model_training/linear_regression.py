import json
import os
import pickle
import sys

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing.preprocess_and_split import preprocess_data

# Configurez MLflow
mlflow.set_tracking_uri("http://localhost:5000")  # Ajustez l'URL si nécessaire
mlflow.set_experiment("Linear_Regression_Attrition")


## Charger et prétraiter les données
X_train, X_test, y_train, y_test, feature_names, feature_types, encoding_dict = (
    preprocess_data()
)

# Créer un modèle de régression logistique
lr_model = LogisticRegression()

with mlflow.start_run():
    # Entraîner le modèle
    lr_model.fit(X_train, y_train)

    # Faire des prédictions sur les données d'entraînement et de test
    y_train_predict = lr_model.predict(X_train)
    y_test_predict = lr_model.predict(X_test)

    # Calculer les métriques
    train_accuracy = accuracy_score(y_train, y_train_predict)
    test_accuracy = accuracy_score(y_test, y_test_predict)
    precision = precision_score(y_test, y_test_predict)
    recall = recall_score(y_test, y_test_predict)
    f1 = f1_score(y_test, y_test_predict)

    # Enregistrer les métriques dans MLflow
    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Enregistrer le rapport de classification comme artefact
    report = classification_report(y_test, y_test_predict)
    with open("classification_report.txt", "w") as f:
        f.write(report)

    mlflow.log_artifact("classification_report.txt")

    # Enregistrer le modèle
    signature = infer_signature(X_train, y_train_predict)
    mlflow.sklearn.log_model(lr_model, "linear_regression_model", signature=signature)

print("Modèle entraîné et enregistré avec succès.")

# Classification Report
print(
    "================================================================================================="
)
print("Classification Report for Logistic Regression Model (Train Set):")
print(classification_report(y_train, y_train_predict))

print(
    "================================================================================================="
)
print("Classification Report for Logistic Regression Model (Test Set):")
print(classification_report(y_test, y_test_predict))

with open("models/linear_regression_model_mlflow.pkl", "wb") as f:
    pickle.dump(lr_model, f)
print("Models trained and saved successfully.")

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
#     "Modèle entraîné et sauvegardé avec succès. Informations sur les features sauvegardées dans feature_info.json."
# )