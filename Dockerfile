# syntax=docker/dockerfile:1

FROM python:3.11-slim

# 1) Install system deps
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
    torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-1.13.1+cpu.html \
    && pip install --no-cache-dir torch-geometric

# 3) Copy your application code
COPY . .

# 4) Expose port and default command
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
