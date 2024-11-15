"""This module trains and chooses the best model to use for prediction of employee attrition."""

import json
import os
import pickle
import sys
import logging
from datetime import datetime

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_and_log_model(model, model_name, x_train, y_train, x_test, y_test):
    """
    Train a model, log the results in MLflow, and return test accuracy.
    """
    logger.info("Starting training for model %s", model_name)
    mlflow.set_experiment(model_name)
    with mlflow.start_run():
        model.fit(x_train, y_train)

        y_train_predict = model.predict(x_train)
        y_test_predict = model.predict(x_test)

        train_accuracy = accuracy_score(y_train, y_train_predict)
        test_accuracy = accuracy_score(y_test, y_test_predict)
        precision = precision_score(y_test, y_test_predict)
        recall = recall_score(y_test, y_test_predict)
        f1 = f1_score(y_test, y_test_predict)

        logger.info("Metrics for %s: Test Accuracy = %.4f, F1 Score = %.4f", model_name, test_accuracy, f1)

        for param_name, param_value in model.get_params().items():
            mlflow.log_param(param_name, param_value)

        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        report = classification_report(y_test, y_test_predict)
        report_path = f"models/reports/classification_report_{model_name}.txt"
        with open(report_path, "w", encoding='utf-8') as report_file:
            report_file.write(report)

        mlflow.log_artifact(report_path)

        signature = infer_signature(x_train, y_train_predict)
        mlflow.sklearn.log_model(model, f"{model_name}_model", signature=signature)

    logger.info("Model %s trained and logged successfully with test accuracy of %.4f.", model_name, test_accuracy)
    return test_accuracy

def update_model_history(model_name, accuracy):
    """
    Update the history of model accuracies.
    """
    try:
        with open('models/model_history.json', 'r', encoding='utf-8') as history_file:
            history = json.load(history_file)
    except FileNotFoundError:
        history = {}

    if model_name not in history:
        history[model_name] = []

    history[model_name].append(accuracy)

    with open('models/model_history.json', 'w', encoding='utf-8') as history_file:
        json.dump(history, history_file)

def load_previous_best_model():
    """
    Load the test accuracy and name of the previously best model.
    """
    try:
        with open("models/best_model_detail.json", "r", encoding='utf-8') as detail_file:
            best_model_details = json.load(detail_file)
            return best_model_details["best_model_name"], best_model_details["test_accuracy"]
    except FileNotFoundError:
        return None, 0

mlflow.set_tracking_uri("http://localhost:5000")

x_train, x_test, y_train, y_test, feature_names, feature_types, encoding_dict = preprocess_data()

models = {
    "KNN": KNeighborsClassifier(
        n_neighbors=8,
        weights="uniform",
        algorithm="auto",
        leaf_size=300,
        p=2,
        metric="minkowski",
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=20,
        max_depth=100,
        max_features="sqrt",
        random_state=10
    ),
    "XGBoost": XGBClassifier(
        objective="binary:logistic",
        learning_rate=0.1,
        n_estimators=500,
        max_depth=3,
        random_state=82,
    ),
    "Logistic Regression": LogisticRegression(),
}

previous_best_model_name, previous_best_accuracy = load_previous_best_model()
best_model_name = previous_best_model_name
best_model = None
best_accuracy = previous_best_accuracy

for name, model in models.items():
    accuracy = train_and_log_model(model, name, x_train, y_train, x_test, y_test)
    update_model_history(name, accuracy)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = name
        best_model = model
        logger.info("New best model found: %s with test accuracy of %.4f", name, accuracy)

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
if best_accuracy > previous_best_accuracy:
    logger.info("[%s] New best model recorded: %s with a precision of %.4f", current_time, best_model_name, best_accuracy)

    best_model_details = {
        "best_model_name": best_model_name,
        "test_accuracy": best_accuracy,
        "hyperparameters": best_model.get_params(),
    }

    with open("models/best_model_detail.json", "w", encoding='utf-8') as json_file:
        json.dump(best_model_details, json_file, indent=2)

    with open(f"models/best_model_{best_model_name}.pkl", "wb") as model_file:
        pickle.dump(best_model, model_file)

else:
    logger.info("[%s] No new best model found. The previous model %s remains the best with a precision of %.4f", 
                current_time, previous_best_model_name, previous_best_accuracy)

save_feature_info(
    feature_names,
    feature_types,
    encoding_dict,
    output_file="models/feature_info.json",
)

print(f"The best model is {best_model_name} with a precision of {best_accuracy:.4f}.")