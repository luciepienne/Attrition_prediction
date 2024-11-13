# Employee Attrition Prediction API

## Description

Ce projet fournit une API FastAPI pour prédire le risque d'attrition des employés à l'aide de modèles d'apprentissage automatique. L'API inclut des endpoints pour l'authentification et la prédiction, permettant aux utilisateurs d'obtenir des résultats basés sur les données des employés.

## Structure du projet :

`
prediction_attrition/
│
├── data/                          # Dossier pour les données
│   └── IBM_data.csv               # Kaggle : https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/data
│
├── models/                        # Dossier pour les modèles de machine learning
│   ├── best_model.pkl             # Meilleur modèle choisi suite aux entraînements (actuellement Random Forest test_accuracy 0.87)
│   ├── best_model_details.json    # donne le nom, le résultats de test_accuracy et les hyperparamètres du meilleur modèle
│   └── feature_info.json          # Informations sur les features
│
├── notebooks/                     # Dossier pour les notebooks Jupyter
│   └── data_analysis.ipynb        # Notebook d'exploration des données
│
├── src/                           # Dossier source pour le code de l'application
│   ├── api/                       # Dossier pour l'API FastAPI
│   │   ├── app.py                 # Application FastAPI principale => localhost:8001
│   │   ├── auth.py                # Gestion de l'authentification
│   │   └── prediction_attrition.py# Client API avec Streamlit
│   │
│   ├── data_processing/            # Dossier pour le traitement des données
│   │   └── preprocess_and_split.py # Script de nettoyage des données et de split en train and test dataset
│   │
│   ├── model_training/            # Dossier pour l'entraînement des modèles
│   │   ├── feature_in_json.json   # fonction pour générer les features des modèles en .json
│   │   └── train_and_best.py      # fonction d'entraînement des modèles et de sélection du meilleur
│   ├── monitoring/
│   │   ├── prometheus_config.yml
│   │   ├── grafana_dashboard.json
│   │   └── mlflow_tracking.py
│   │
│   └── utils/                     # Dossier pour les utilitaires
│       └── helpers.py             # Fonctions utilitaires diverses
│ 
├── tests/                         # Dossier pour les tests unitaires et d'intégration
│   ├── test_app.py                # Tests pour l'application FastAPI
│   ├── test_preprocess.py         # Tests pour les fonctions de cleansing des data
│   ├── test_train_best_model.py   # Tests pour la fonction d'entraînement
│   └── test_api_client.py         # Tests pour le client API Streamlit
│ 
├── requirements.txt               # Fichier listant les dépendances Python
└── README.md                      # Documentation du projet
`

## Table des matières

- [Installation](#installation)
- [Utilisation](#utilisation)
- [Endpoints](#endpoints)
  - [Authentification](#authentification)
  - [Prédiction](#prédiction)
- [Tests](#tests)
- [Contributions](#contributions)
- [License](#license)

## Installation

1. Clonez le dépôt :

   ```bash
   git clone git@github.com:luciepienne/Attrition_prediction.git (linux)
   git clone https://github.com/luciepienne/Attrition_prediction.git
   cd your-repo
   ```

2. Créez un environnement virtuel et activez-le :
   python -m venv venv
   source venv/bin/activate # Sur Windows, utilisez `venv\Scripts\activate`

3. Installez les dépendances :
   `pip install -r requirements.txt`

## Utilisation

### Entrainer les modèles et choisir le meilleur modèle

Il faut créer le meilleur modèle de prédiction en pickle afin d'obtenir les prédictions et les features permettant la prédiction.

Pour executer l'entrainement sur les 4 modèles allez dans :
src/model_training/train_and_best.py
faites varier les hyper paramètres de chaque modèle et lancer l'entraînement :
`python src/model_training/train_and_best.py`

le meilleur modèle : best_model.pkl
ses features : feature_info.json
ainsi que les détails de ce modèle best_model_detail.json
à ce stade :
{
"best_model_name": "Random Forest",
"test_accuracy": 0.8741496598639455,
"hyperparameters": {
"bootstrap": true,
"ccp_alpha": 0.0,
"class_weight": null,
"criterion": "gini",
"max_depth": 50,
"max_features": "sqrt",
"max_leaf_nodes": null,
"max_samples": null,
"min_impurity_decrease": 0.0,
"min_samples_leaf": 1,
"min_samples_split": 2,
"min_weight_fraction_leaf": 0.0,
"monotonic_cst": null,
"n_estimators": 200,
"n_jobs": null,
"oob_score": false,
"random_state": 70,
"verbose": 0,
"warm_start": false
}
}

### Exposer les modèles de prédictions

Lancer MLFLOW :
`mlflow ui --port 5000`

Lancer l'API FastAPI :
`python src.api.app.py`

Lancer l'application Streamlit :
Ouvrez un autre terminal et exécutez :
`streamlit run src/api/attrition_prediction.py`
Accédez à l'application Streamlit à l'adresse : http://localhost:8501.

Ouvrir Prometeus pour la collecte des logs :
http://localhost:8000

### Endpoints

1. Authentification
   POST /token
   Description : Authentifie un utilisateur et retourne un token JWT.
   Paramètres :
   username: Nom d'utilisateur.
   password: Mot de passe.
   Réponse :
   json
   {
   "access_token": "string",
   "token_type": "bearer"
   }

2. Prédiction
   POST /predict
   Description : Prédit le risque d'attrition pour un employé donné.
   Paramètres : Les champs suivants sont requis dans le corps de la requête :
   Gender
   Age
   MaritalStatus
   Education
   EducationField
   NumCompaniesWorked
   WorkExperience
   Department
   JobRole
   JobLevel
   DistanceFromHome
   BusinessTravel
   OverTime
   JobInvolvement
   PercentSalaryHike
   MonthlyIncome
   MonthlyRate
   DailyRate
   HourlyRate
   StockOptionLevel
   PerformanceRating
   TrainingTimesLastYear
   YearsWithCurrManager
   OverallSatisfaction
   Réponse :
   json
   {
   "predictions": [
   {
   "model_name": "string",
   "prediction": float,
   "attrition_risk": "string"
   },
   ...
   ]
   }

## Tests

Pour exécuter les tests, assurez-vous que votre API est en cours d'exécution, puis exécutez :
bash
pytest tests/

## Contributions

Les contributions sont les bienvenues ! Veuillez soumettre une demande de tirage (pull request) pour toute amélioration ou correction.

## License

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.
