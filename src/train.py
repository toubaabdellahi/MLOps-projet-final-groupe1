import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle
import os

def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    # ============================
    # CONFIGURATION MLFLOW
    # ============================
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    print(f"📡 MLflow tracking : {params['mlflow']['tracking_uri']}")
    print(f"📁 Expérience : {params['mlflow']['experiment_name']}")

    # ============================
    # CHARGEMENT DES DONNÉES
    # ============================
    print("📥 Chargement des données preprocessées...")
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    print(f"   X_train : {X_train.shape}, y_train : {y_train.shape}")

    # ============================
    # ENTRAÎNEMENT AVEC MLFLOW
    # ============================
    with mlflow.start_run(run_name="churn-training") as run:
        run_id = run.info.run_id
        print(f"🏃 MLflow Run ID : {run_id}")

        # Log des hyperparamètres
        train_params = params["train"]
        mlflow.log_params({
            "model_type": train_params["model_type"],
            "n_estimators": train_params.get("n_estimators", "N/A"),
            "max_depth": train_params.get("max_depth", "N/A"),
            "random_state": train_params["random_state"],
            "train_samples": X_train.shape[0],
            "n_features": X_train.shape[1]
        })

        # Choix et création du modèle
        if train_params["model_type"] == "random_forest":
            model = RandomForestClassifier(
                n_estimators=train_params["n_estimators"],
                max_depth=train_params["max_depth"],
                random_state=train_params["random_state"],
                n_jobs=-1
            )
            print(f"🌲 RandomForest (n_estimators={train_params['n_estimators']}, max_depth={train_params['max_depth']})")
        elif train_params["model_type"] == "logistic_regression":
            model = LogisticRegression(
                random_state=train_params["random_state"],
                max_iter=1000
            )
            print("📈 LogisticRegression")
        else:
            raise ValueError(f"Modèle inconnu: {train_params['model_type']}")

        # Entraînement
        print("🏋️ Entraînement en cours...")
        model.fit(X_train, y_train)

        # Log du training accuracy
        train_accuracy = model.score(X_train, y_train)
        mlflow.log_metric("train_accuracy", train_accuracy)
        print(f"   Train accuracy : {train_accuracy:.4f}")

        # Log du modèle dans MLflow (artefact sur S3)
        mlflow.sklearn.log_model(model, "model")
        print("   Modèle loggé dans MLflow ✓")

        # Sauvegarder localement aussi
        os.makedirs("models", exist_ok=True)
        with open("models/model.pkl", "wb") as f:
            pickle.dump(model, f)

        # Sauvegarder le run_id pour evaluate et register
        with open("models/run_id.txt", "w") as f:
            f.write(run_id)

        print(f"\n✅ Entraînement terminé !")
        print(f"   Run ID : {run_id}")
        print(f"   Vérifier sur : {params['mlflow']['tracking_uri']}")

if __name__ == "__main__":
    main()