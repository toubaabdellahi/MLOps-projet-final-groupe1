# MLOps Projet Final – Pipeline Churn Prediction | Groupe 1

**SupNum – Master DEML M2 | MLOps II | Prof. Yehdhih ANNA**

## Membres du groupe

| Nom | Matricule | Rôle |
|-----|-----------|------|
| (ML Engineer) | dev-21003 | Pipeline DVC, MLflow, entraînement |
| Rakia Dehah | dev-21008 | API FastAPI, Dockerfile, endpoint /predict |
| (DevOps) | dev-21027 | EC2, S3, CI/CD GitHub Actions |
| (Frontend) | dev-21032 | Interface utilisateur |

## Sujet

Prédiction du churn client (départ d'abonnés) à partir du dataset Telco Customer Churn.  
Modèle : Random Forest — Accuracy : 79.6% — ROC-AUC : 83.6%

## Architecture

```
GitHub Repository (main / staging / dev-MATRICULE)
         │
         ▼
   GitHub Actions
   ┌─────────────────────┬──────────────────────┐
   │ Workflow 1 (Code CI)│ Workflow 2 (Data CI) │
   │ merge → main        │ dvc repro            │
   │ SSH → docker build  │ MLflow log           │
   │ restart API         │ restart API          │
   └─────────────────────┴──────────────────────┘
         │ SSH
         ▼
   AWS EC2 (mlops-churn-server_G1) — 15.224.61.131
   ┌──────────────────┬──────────────────────────┐
   │ MLflow :5000     │ Docker API :8000          │
   │ Registry +       │ FastAPI + Interface Web   │
   │ Experiments      │ POST /predict             │
   └──────────────────┴──────────────────────────┘
         │
         ▼
   AWS S3 — s3://mlops-churn-groupe-1/
   ├── data/raw/churn.csv
   ├── data/processed/
   ├── mlflow-artifacts/
   └── dvc-store/
```

## Structure du repository

```
├── api/
│   └── main.py              # API FastAPI (health check + /predict + UI)
├── src/
│   ├── preprocess.py        # Stage DVC : nettoyage, encodage, normalisation
│   ├── train.py             # Stage DVC : entraînement + MLflow logging
│   ├── evaluate.py          # Stage DVC : métriques finales
│   └── register.py          # Stage DVC : enregistrement MLflow Registry
├── static/
│   └── index.html           # Interface web utilisateur
├── .github/
│   └── workflows/
│       ├── deploy.yml       # Workflow 1 : déploiement sur merge main
│       └── retrain.yml      # Workflow 2 : retraining automatique
├── data/
│   └── processed/           # Artifacts DVC (scaler, encoders, features)
├── Dockerfile               # Image API, port 8000, python 3.11
├── requirements-api.txt     # Dépendances API
├── dvc.yaml                 # Pipeline DVC (4 stages)
└── params.yaml              # Hyperparamètres (exclu du git)
```

## Services en production

| Service | URL |
|---------|-----|
| MLflow UI | http://15.224.61.131:5000 |
| API Prédiction | http://15.224.61.131:8000 |
| Interface Web | http://15.224.61.131:8000/ui |
| Health Check | http://15.224.61.131:8000/ |

## Instructions de déploiement

### Prérequis

- AWS credentials configurées (EC2 + S3)
- Clé SSH : `Group1.pem`
- Docker installé sur l'EC2

### 1. Démarrer le serveur MLflow (EC2)

```bash
ssh -i Group1.pem ubuntu@15.224.61.131

nohup mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --default-artifact-root s3://mlops-churn-groupe-1/mlflow-artifacts \
  --allowed-hosts "*" \
  --cors-allowed-origins "*" \
  > mlflow.log 2>&1 &
```

### 2. Lancer le pipeline DVC (ML Engineer)

```bash
dvc repro
```

### 3. Builder et lancer l'API Docker (EC2)

```bash
docker build -t churn-api .

docker run -d -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://15.224.61.131:5000 \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  -e AWS_DEFAULT_REGION=eu-west-3 \
  --name churn-api churn-api
```

### 4. Tester l'API

```bash
# Health check
curl http://15.224.61.131:8000/

# Prédiction
curl -X POST http://15.224.61.131:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
    "TotalCharges": 70.0
  }'
```

## API — Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Health check — retourne statut + modèle chargé |
| GET | `/ui` | Interface web pour prédictions manuelles |
| POST | `/predict` | Prédiction churn à partir des features client |

### Exemple de réponse `/predict`

```json
{
  "prediction": 1,
  "prediction_label": "Churn",
  "probability_churn": 0.73,
  "probability_no_churn": 0.27
}
```

## Pipeline DVC

```bash
dvc repro        # Relancer tout le pipeline
dvc dag          # Visualiser le graphe de dépendances
dvc push         # Pousser les artifacts vers S3
dvc pull         # Récupérer les artifacts depuis S3
```

Stages : `preprocess` → `train` → `evaluate` → `register`
