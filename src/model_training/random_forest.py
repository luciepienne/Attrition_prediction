import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing.preprocess_and_split import preprocess_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import pickle

rf_model = RandomForestClassifier(n_estimators=200, max_depth=5, max_features='sqrt', random_state=42)

# Charger et prétraiter les données
X_train, X_test, y_train, y_test = preprocess_data()

rf_model.fit(X_train, y_train)

y_train_predict = rf_model.predict(X_train)

# Make predictions on the test data

y_test_predict = rf_model.predict(X_test)

# Classification Report for Train Set
print("=================================================================================================")
print("Classification Report for Random Forest Model (Train Set):")
print(classification_report(y_train, y_train_predict))

# Classification Report for Test Set
print("=================================================================================================")
print("Classification Report for Random Forest Model (Test Set):")
print(classification_report(y_test, y_test_predict))

with open('models/random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("Models trained and saved successfully.")