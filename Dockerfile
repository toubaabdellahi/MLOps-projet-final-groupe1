FROM python:3.11-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# API code and UI
COPY api/ ./api/
COPY static/ ./static/

# Preprocessing artifacts produced by dvc repro (must be present at build time)
COPY data/processed/scaler.pkl ./data/processed/scaler.pkl
COPY data/processed/label_encoders.pkl ./data/processed/label_encoders.pkl
COPY data/processed/feature_names.pkl ./data/processed/feature_names.pkl

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
