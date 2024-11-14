# Démarrer MLflow en arrière-plan
mlflow ui --port 5000 &

# Démarrer FastAPI en arrière-plan
python src/api/app.py --host 0.0.0.0 --port ${PORT:-8001} &

# Démarrer Streamlit
streamlit run src/api/attrition_prediction.py --server.port 8501 --server.address 0.0.0.0
