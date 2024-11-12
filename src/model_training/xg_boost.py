import json
import os
import pickle
import sys

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models.signature import infer_signature
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing.preprocess_and_split import preprocess_data

mlflow.set_tracking_uri("http://localhost:5000")  # Ajustez l'URL si nécessaire
mlflow.set_experiment("XGBoost_Attrition")

# Charger et prétraiter les données
X_train, X_test, y_train, y_test, feature_names, feature_types, encoding_dict = (
    preprocess_data()
)

xgb_model = XGBClassifier(
    objective="binary:logistic",
    learning_rate=0.1,
    n_estimators=350,
    max_depth=3,
    random_state=42,
    max_features="sqrt",
)

with mlflow.start_run():
    # Entraîner le modèle
    xgb_model.fit(X_train, y_train)

    # Faire des prédictions sur les données d'entraînement et de test
    y_train_predict = xgb_model.predict(X_train)
    y_test_predict = xgb_model.predict(X_test)

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
    with open("classification_report_xgboost.txt", "w") as f:
        f.write(report)

    mlflow.log_artifact("classification_report_xgboost.txt")

print(
    "================================================================================================="
)
print("Classification Report for XGBoost Model (Train Set):")
print(classification_report(y_train, y_train_predict))

print(
    "================================================================================================="
)
print("Classification Report for XGBoost Model (Test Set):")
print(classification_report(y_test, y_test_predict))

with open("models/xgboost_model_mlflow.pkl", "wb") as f:
    pickle.dump(xgb_model, f)

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
