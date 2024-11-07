import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing.preprocess_and_split import preprocess_data
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# Charger et prétraiter les données
X_train, X_test, y_train, y_test = preprocess_data()

model = XGBClassifier(
    objective='binary:logistic',
    learning_rate=0.01,
    n_estimators=350,
    max_depth=3,
    random_state=42,
    max_features='sqrt',
)

model.fit(X_train, y_train)

y_train_predict = model.predict(X_train)
y_test_predict = model.predict(X_test)

print("=================================================================================================")
print("Classification Report for XGBoost Model (Train Set):")
print(classification_report(y_train, y_train_predict))

print("=================================================================================================")
print("Classification Report for XGBoost Model (Test Set):")
print(classification_report(y_test, y_test_predict))