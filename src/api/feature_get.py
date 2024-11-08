import pickle

# Charger le modèle
with open('models/knn_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Afficher les noms des features dans l'ordre
if hasattr(model, 'feature_names_in_'):
    print("Features order:")
    for i, feature in enumerate(model.feature_names_in_):
        print(f"{i}: {feature}")
else:
    print("Feature names not available in the model.")