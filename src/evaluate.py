import pandas as pd
import yaml
import mlflow
import json
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, classification_report
)
import pickle

def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    # Charger les données de test
    print("📥 Chargement du jeu de test...")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    # Charger le modèle
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    # Charger le run_id du training
    with open("models/run_id.txt", "r") as f:
        run_id = f.read().strip()

    print(f"📡 Évaluation du run : {run_id}")

    # Prédictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calcul des métriques
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4)
    }

    # Affichage détaillé
    print("\n📊 === MÉTRIQUES D'ÉVALUATION ===")
    print(f"   Accuracy  : {metrics['accuracy']}")
    print(f"   F1 Score  : {metrics['f1_score']}")
    print(f"   Precision : {metrics['precision']}")
    print(f"   Recall    : {metrics['recall']}")
    print(f"   ROC AUC   : {metrics['roc_auc']}")
    print(f"\n   Classification Report :")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    # Logger les métriques dans MLflow (même run que le training)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
        print("   Métriques loggées dans MLflow ✓")

    # Sauvegarder localement
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Évaluation terminée !")
    print(f"   Seuil requis pour registration : {params['evaluate']['threshold']}")
    print(f"   Accuracy obtenue : {metrics['accuracy']}")
    if metrics['accuracy'] >= params['evaluate']['threshold']:
        print(f"   → ✅ Seuil atteint, le modèle sera enregistré")
    else:
        print(f"   → ❌ Seuil NON atteint")

if __name__ == "__main__":
    main()