'''This module trains and choses the best model to use for prediction of employee attrition'''
import json
import os
import pickle
import sys

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_preprocessing.preprocess_and_split import preprocess_data
from model_training.features_in_json import save_feature_info


def train_and_log_model(model, model_name, X_train, y_train, X_test, y_test):
    '''Will train the 4 models chosen initially, 
    log the results in MLFLOW and identify the best one'''
    mlflow.set_experiment(model_name)  
    with mlflow.start_run():
        model.fit(X_train, y_train)

        y_train_predict = model.predict(X_train)
        y_test_predict = model.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_train_predict)
        test_accuracy = accuracy_score(y_test, y_test_predict)
        precision = precision_score(y_test, y_test_predict)
        recall = recall_score(y_test, y_test_predict)
        f1 = f1_score(y_test, y_test_predict)

        for param_name, param_value in model.get_params().items():
            mlflow.log_param(param_name, param_value)

        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        report = classification_report(y_test, y_test_predict)
        report_path = f"models/reports/classification_report_{model_name}.txt"
        with open(report_path, "w") as f:
            f.write(report)

        mlflow.log_artifact(report_path)


        signature = infer_signature(X_train, y_train_predict)
        mlflow.sklearn.log_model(model, f"{model_name}_model", signature=signature)

    print(f"Modèle {model_name} entraîné et enregistré avec succès.")
    return test_accuracy



mlflow.set_tracking_uri("http://localhost:5000")


X_train, X_test, y_train, y_test, feature_names, feature_types, encoding_dict = (
    preprocess_data()
)

models = {
    "KNN": KNeighborsClassifier(
        n_neighbors=10,
        weights="uniform",
        algorithm="auto",
        leaf_size=150,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=50, max_features="sqrt", random_state=70
    ),
    "XGBoost": XGBClassifier(
        objective="binary:logistic",
        learning_rate=0.1,
        n_estimators=350,
        max_depth=3,
        random_state=42,
        max_features="sqrt",
    ),
    "Logistic Regression": LogisticRegression(),
}

best_model_name = None
best_model = None
best_accuracy = 0.0

for name, model in models.items():
    accuracy = train_and_log_model(model, name, X_train, y_train, X_test, y_test)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = name
        best_model = model

best_model_details = {
    "best_model_name": best_model_name,
    "test_accuracy": best_accuracy,
    "hyperparameters": best_model.get_params(),
}

with open("models/best_model_detail.json", "w") as json_file:
    json.dump(best_model_details, json_file, indent=2)

with open(f"models/best_model_{best_model_name}.pkl", "wb") as f:
    pickle.dump(best_model, f)
    save_feature_info(
        feature_names,
        feature_types,
        encoding_dict,
        output_file="models/feature_info.json",
    )

print(
    f"Le meilleur modèle est {best_model_name} avec une précision de {best_accuracy:.4f}."
)
