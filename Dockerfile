
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .


EXPOSE 8001 8501

CMD ["sh", "-c", "uvicorn src/api/app.py --host 0.0.0.0 --port 8001 & streamlit run src/api/prediction_attrition.py --server.port 8501"]