# MLOps Projet Final – Churn Prediction Pipeline | Groupe 1

**SupNum – Master DEML M2 | MLOps II | Prof. Yehdhih ANNA**

---

## Idée du projet

Ce projet implémente un **pipeline MLOps complet et automatisé** autour d'un problème de prédiction de churn client (résiliation d'abonnement télécom).

L'objectif est de simuler ce qu'un ingénieur MLOps fait en entreprise :
- Les données arrivent depuis S3 (simulation d'un ETL externe)
- Un pipeline DVC automatise le preprocessing → entraînement → évaluation → enregistrement du modèle
- Un serveur MLflow tracke les expériences et gère le Model Registry
- Une API FastAPI sert les prédictions en production via Docker
- Deux workflows GitHub Actions assurent le déploiement continu et le retraining automatique

**Dataset :** Telco Customer Churn — 6 562 clients, 19 features (contrat, services, charges...)
**Modèle :** Random Forest — Accuracy : 79.6% — ROC-AUC : 83.6%

---

## Membres du groupe

| Nom | Matricule | Rôle |
|-----|-----------|------|
| Touba Abdellahi | dev-21003 | ML Engineer — Pipeline DVC, MLflow, entraînement |
| Rakia Dehah | dev-21008 | Backend Dev — API FastAPI, Dockerfile, /predict |
| (DevOps) | dev-21027 | DevOps/Cloud — EC2, S3, GitHub Actions CI/CD |
| (Frontend) | dev-21032 | Frontend — Interface utilisateur, README |

---

## Architecture complète

```
GitHub Repository
  branches: main / staging / dev-21003 / dev-21008 / dev-21027 / dev-21032
         │
         │  push → main
         ▼
   GitHub Actions
   ┌──────────────────────────┬─────────────────────────────┐
   │  Workflow 1 — Deploy     │  Workflow 2 — Retrain        │
   │  Déclencheur: merge main │  Déclencheur: manuel/cron   │
   │  → SSH → EC2             │  → SSH → EC2                │
   │  → git pull              │  → aws s3 cp churn.csv      │
   │  → docker build          │  → dvc repro --force        │
   │  → docker run            │  → nouveau modèle MLflow    │
   │  → health check          │  → docker rebuild           │
   └──────────────────────────┴─────────────────────────────┘
         │ SSH (Group1.pem)
         ▼
   AWS EC2 t2.medium — mlops-churn-server_G1
   IP : 15.224.61.131
   ┌─────────────────────┬──────────────────────────────────┐
   │  MLflow :5000       │  Docker Container :8000           │
   │  - Experiments      │  - FastAPI                       │
   │  - Model Registry   │  - GET /  (health check)         │
   │  - Artifacts S3     │  - POST /predict                 │
   │                     │  - GET /ui (interface web)       │
   │                     │  - GET /model/info               │
   └─────────────────────┴──────────────────────────────────┘
         │
         ▼
   AWS S3 — s3://mlops-churn-groupe-1/
   ├── data/raw/churn.csv          ← source de données primaire
   ├── data/processed/             ← artifacts preprocessing (DVC)
   ├── mlflow-artifacts/           ← runs, modèles, métriques
   └── dvc-store/                  ← cache DVC versioning
```

---

## Structure du repository

```
MLOps-projet-final-groupe1/
│
├── api/
│   ├── __init__.py
│   └── main.py              # API FastAPI — charge modèle MLflow au démarrage
│
├── src/
│   ├── preprocess.py        # Stage 1 — nettoyage, encodage, normalisation, split
│   ├── train.py             # Stage 2 — RandomForest + logging MLflow
│   ├── evaluate.py          # Stage 3 — métriques (accuracy, F1, ROC-AUC)
│   └── register.py          # Stage 4 — enregistrement MLflow Registry si acc≥0.75
│
├── static/
│   └── index.html           # Interface web dark theme (formulaire + résultat)
│
├── .github/
│   └── workflows/
│       ├── deploy.yml       # Workflow 1 — déploiement automatique sur merge main
│       └── retrain.yml      # Workflow 2 — retraining + redéploiement
│
├── data/
│   ├── raw/                 # churn.csv (DVC-tracked, source S3)
│   └── processed/           # scaler.pkl, label_encoders.pkl, feature_names.pkl
│
├── models/                  # model.pkl, run_id.txt, metrics.json (DVC-tracked)
├── Dockerfile               # Image Python 3.11, port 8000
├── docker-compose.yml       # Orchestration container
├── requirements-api.txt     # Dépendances API
├── dvc.yaml                 # Définition pipeline DVC (4 stages)
├── dvc.lock                 # Hashes des artifacts (versionné)
├── .gitignore               # Exclut .env, params.yaml, data/, models/
└── params.yaml              # Hyperparamètres + config MLflow (NON commité)
```

---

## Pipeline DVC — 4 étapes

```
data/raw/churn.csv (S3)
        │
        ▼
  [preprocess]
  - Supprime customerID
  - Impute TotalCharges manquants
  - LabelEncoder sur colonnes catégorielles
  - StandardScaler sur toutes les features
  - Split train/test 80/20 stratifié
  → data/processed/ (X_train, X_test, y_train, y_test, scaler.pkl, encoders.pkl)
        │
        ▼
  [train]
  - RandomForestClassifier (n_estimators=100, max_depth=10)
  - Log hyperparamètres + train_accuracy dans MLflow
  → models/model.pkl, models/run_id.txt
        │
        ▼
  [evaluate]
  - Calcule : accuracy, F1, precision, recall, ROC-AUC
  - Log métriques dans MLflow
  → models/metrics.json
        │
        ▼
  [register]
  - Si accuracy ≥ 0.75 → enregistre dans MLflow Model Registry
  - Promeut en stage "Production" automatiquement
  → models:/churn-prediction-model/Production
```

Commandes DVC :
```bash
dvc repro          # Relancer le pipeline (stages modifiés uniquement)
dvc repro --force  # Forcer le retraining complet
dvc dag            # Visualiser le graphe de dépendances
dvc push           # Pousser artifacts vers S3
dvc pull           # Récupérer artifacts depuis S3
```

---

## API FastAPI — Endpoints

| Méthode | Endpoint | Description | Points grille |
|---------|----------|-------------|---------------|
| GET | `/` | Health check — HTTP 200 + statut modèle | C2 |
| GET | `/ui` | Interface web interactive | C5 |
| POST | `/predict` | Prédiction churn depuis features JSON | C1 |
| GET | `/model/info` | Version + métriques du modèle en production | — |
| POST | `/reload` | Recharge le modèle depuis MLflow Registry | — |
| GET | `/docs` | Swagger UI auto-généré | — |

### Exemple POST `/predict`

```bash
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

Réponse :
```json
{
  "prediction": 1,
  "prediction_label": "Churn",
  "probability_churn": 0.85,
  "probability_no_churn": 0.15
}
```

---

## Services en production

| Service | URL |
|---------|-----|
| MLflow UI | http://15.224.61.131:5000 |
| API Health Check | http://15.224.61.131:8000/ |
| Interface Web | http://15.224.61.131:8000/ui |
| Swagger Docs | http://15.224.61.131:8000/docs |
| Infos Modèle | http://15.224.61.131:8000/model/info |

---

## Instructions de déploiement

### Prérequis

- Clé SSH : `Group1.pem`
- AWS credentials avec accès S3 + EC2
- Docker installé sur l'EC2
- GitHub Secrets configurés : `EC2_HOST`, `EC2_USER`, `EC2_SSH_KEY`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, `MLFLOW_TRACKING_URI`, `S3_BUCKET`

### 1. Connexion EC2

```bash
ssh -i Group1.pem ubuntu@15.224.61.131
```

### 2. Démarrer MLflow

```bash
nohup mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --default-artifact-root s3://mlops-churn-groupe-1/mlflow-artifacts \
  --allowed-hosts "*" \
  --cors-allowed-origins "*" \
  > mlflow.log 2>&1 &
```

### 3. Lancer le pipeline ML

```bash
cd ~/mlops-churn-prediction/MLOps-projet-final-groupe1

# Récupérer les données
aws s3 cp s3://mlops-churn-groupe-1/data/raw/churn.csv data/raw/churn.csv

# Lancer le pipeline complet
dvc repro
```

### 4. Déployer l'API

```bash
# Créer le .env (jamais commité)
cat > .env << 'EOF'
MLFLOW_TRACKING_URI=http://15.224.61.131:5000
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=eu-west-3
EOF

# Builder et lancer
docker-compose up -d --build

# Vérifier
curl http://localhost:8000/
```

### 5. Déploiement automatique (CI/CD)

```bash
# Depuis ton PC — merge vers main déclenche Workflow 1
git checkout main
git merge staging
git push origin main
# → GitHub Actions déploie automatiquement sur EC2
```

---

## Workflows CI/CD

### Workflow 1 — Deploy API (déclencheur : push sur `main`)

```
merge staging → main
      ↓
GitHub Actions
      ↓ SSH
EC2 : git pull → docker build → docker run → health check
      ↓
API accessible sur :8000
```

### Workflow 2 — Retrain & Deploy (déclencheur : manuel / cron hebdo)

```
GitHub Actions → Run workflow
      ↓ SSH
EC2 : aws s3 cp churn.csv → dvc repro --force → nouveau modèle MLflow
      ↓
docker rebuild → API rechargée avec nouveau modèle
```

---

## Sécurité

- Aucun secret dans le code ou l'historique git
- `.env`, `params.yaml`, `*.pem` dans `.gitignore`
- Credentials AWS uniquement dans GitHub Secrets et variables d'environnement

---

## Checklist soutenance

- [x] MLflow accessible sur http://15.224.61.131:5000
- [x] Modèle `churn-prediction-model` en stage Production
- [x] API Docker accessible sur http://15.224.61.131:8000
- [x] `POST /predict` retourne une prédiction correcte
- [x] Interface web fonctionnelle sur `/ui`
- [x] Workflow 1 se déclenche sur merge main
- [x] Workflow 2 déclenchable manuellement
- [x] Aucun secret dans le repo
