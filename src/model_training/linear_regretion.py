from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing.preprocess_and_split import preprocess_data

# Charger et prétraiter les données
X_train, X_test, y_train, y_test = preprocess_data()

# Create a Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)
y_train_predict = model.predict(X_train)

# Make predictions on the test data

y_test_predict = model.predict(X_test)

# Classification Report
print("=================================================================================================")
print("Classification Report for Logistic Regression Model (Train Set):")
print(classification_report(y_train, y_train_predict))

print("=================================================================================================")
print("Classification Report for Logistic Regression Model (Test Set):")
print(classification_report(y_test, y_test_predict))