FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000 8001 8501

# Rendre le script start.sh exécutable
RUN chmod +x start.sh

CMD ["./start.sh"]