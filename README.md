# Employee Attrition Prediction API

## Description

Ce projet fournit une API FastAPI pour prédire le risque d'attrition des employés à l'aide de modèles d'apprentissage automatique. L'API inclut des endpoints pour l'authentification et la prédiction, permettant aux utilisateurs d'obtenir des résultats basés sur les données des employés.

## Structure du projet :

`E3_attrition/
│
├── data/                          # Dossier pour les données
│   └── IBM_data.csv               # Kaggle : https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/data
│
├── models/                        # Dossier pour les modèles de machine learning
│   ├── knn_model.pkl              # Modèle KNN
│   ├── random_forest_model.pkl    # Modèle Random Forest
│   ├── linear_regression_model.pkl # Modèle de régression linéaire
│   ├── xgboost_model.pkl          # Modèle XGBoost
│   └── feature_info.json          # Informations sur les features
│
├── notebooks/                     # Dossier pour les notebooks Jupyter
│   └── data_analysis.ipynb          # Notebook d'exploration des données
│
├── src/                           # Dossier source pour le code de l'application
│   ├── api/                       # Dossier pour l'API FastAPI
│   │   ├── app.py                 # Application FastAPI principale
│   │   ├── auth.py                # Gestion de l'authentification
│   │   └── api_client.py          # Client API pour Streamlit
│   │
│   ├── data_processing/            # Dossier pour le traitement des données
│   │   └── preprocess_and_split.py # Script de nettoyage des données
│   │
│   ├── model_training/            # Dossier pour l'entraînement des modèles
│   │   └── train_models.py         # Script d'entraînement des modèles
│   │
│   └── utils/                     # Dossier pour les utilitaires
│       └── helpers.py             # Fonctions utilitaires diverses
│ 
├── tests/                         # Dossier pour les tests unitaires et d'intégration
│   ├── test_app.py                # Tests pour l'application FastAPI
│   └── test_api_client.py         # Tests pour le client API Streamlit
│ 
├── requirements.txt               # Fichier listant les dépendances Python
└── README.md                      # Documentation du projet`

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

### Entrainer les modèles

Il faut créer les modèlesen pickle afin d'obtenir les prédictions et les features permettant la prédiction.

### Exposer les modèles de prédictions

Lancer l'API FastAPI :
bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

Lancer l'application Streamlit :
Ouvrez un autre terminal et exécutez :
bash
streamlit run src/api/app_client.py

Accédez à l'application Streamlit à l'adresse : http://localhost:8501.
Endpoints
Authentification
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

## Prédiction

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
