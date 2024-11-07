import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing.preprocess_and_split import preprocess_data


from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

import pickle

# Charger et prétraiter les données
X_train, X_test, y_train, y_test = preprocess_data()

# Create a K-Nearest Neighbors model (with 5 neighbors, you can adjust this value)
knn_model = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto', 
                             leaf_size=50, p=2, metric='minkowski', metric_params=None, n_jobs=None)

# Train the model
knn_model.fit(X_train, y_train)

# Make predictions on the training data
y_train_predict = knn_model.predict(X_train)

# Make predictions on the test data
y_test_predict = knn_model.predict(X_test)

# Classification Report for Train Set
print("=================================================================================================")
print("Classification Report for KNN Model (Train Set):")
print(classification_report(y_train, y_train_predict))

# Classification Report for Test Set
print("=================================================================================================")
print("Classification Report for KNN Model (Test Set):")
print(classification_report(y_test, y_test_predict))

with open('models/knn_model.pkl', 'wb') as f:
    pickle.dump(knn_model, f)
print("Models trained and saved successfully.")