import mlflow


def get_best_model(experiment_name):
    
    mlflow.set_tracking_uri("http://localhost:5000")  
    # Récupérer l'expérience par son nom
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    # Rechercher tous les runs de cette expérience
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    # Vérifiez quelles colonnes sont disponibles
    print("Colonnes disponibles :", runs.columns)

    # Trouver le run avec la meilleure précision de test
    if "metrics.test_accuracy" in runs.columns:
        best_run = runs.loc[runs["metrics.test_accuracy"].idxmax()]
        
        model_uri = f"runs:/{best_run.run_id}/xgboost_best_model"  # Changez selon le modèle utilisé
        
        # Charger le meilleur modèle
        best_model = mlflow.sklearn.load_model(model_uri)
        
        return best_model
    else:
        raise KeyError("La colonne 'metrics.test_accuracy' n'existe pas dans les résultats des runs.")

# Usage example:
# try:
#     best_rf_model = get_best_model("XGBoost_Attrition")  # Assurez-vous que le nom est correct
#     print("Meilleur modèle récupéré avec succès.")
# except Exception as e:
#     print(f"Erreur lors de la récupération du meilleur modèle : {e}")

best_rf_model = get_best_model("zz_Random_Forest_Attrition")