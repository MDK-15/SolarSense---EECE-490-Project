FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create base_models directory (you will mount or copy your .pkl files here)
RUN mkdir -p /app/base_models

# Gunicorn serves the Flask app on Cloud Run's expected port
ENV PORT=8080
CMD exec gunicorn --bind :$PORT --workers 2 --threads 4 --timeout 120 app:app
