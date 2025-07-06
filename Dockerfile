FROM python:3.11-slim

# Crée un dossier de travail
WORKDIR /app

# Installe les dépendances système nécessaires
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copie les fichiers requirements
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copie tout le projet
COPY . .

# Expose le port utilisé par FastAPI
EXPOSE 8000

# Lance le serveur FastAPI avec Uvicorn sur le port Railway (utilise la variable $PORT automatiquement injectée)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
