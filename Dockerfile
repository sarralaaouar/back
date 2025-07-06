# Étape 1 : Build all dependencies in a slim container
FROM python:3.11-slim AS builder

WORKDIR /app

# Install OS deps
RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Crée un venv et installe tout dedans
RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Étape 2 : Final image, plus légère
FROM python:3.11-slim

# Copie juste le venv déjà prêt
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
