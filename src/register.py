import yaml
import json
import mlflow
from mlflow.tracking import MlflowClient

def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])

    # Charger les métriques et le run_id
    with open("models/metrics.json", "r") as f:
        metrics = json.load(f)
    with open("models/run_id.txt", "r") as f:
        run_id = f.read().strip()

    threshold = params["evaluate"]["threshold"]
    accuracy = metrics["accuracy"]

    print(f"📊 Accuracy obtenue : {accuracy:.4f}")
    print(f"📊 Seuil minimum   : {threshold}")

    if accuracy >= threshold:
        print("\n✅ Seuil atteint ! Enregistrement dans le Model Registry...")

        model_uri = f"runs:/{run_id}/model"
        model_name = "churn-prediction-model"

        # Enregistrer le modèle dans le Registry
        result = mlflow.register_model(model_uri, model_name)
        version = result.version
        print(f"   Modèle : {model_name}")
        print(f"   Version : {version}")

        # Promouvoir en Production
        client = MlflowClient(tracking_uri=params["mlflow"]["tracking_uri"])
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        print(f"   Stage : Production ✅")
        print(f"\n🎉 Le modèle est prêt à être utilisé par l'API !")
        print(f"   L'API chargera : models:/{model_name}/Production")
    else:
        print(f"\n❌ Accuracy ({accuracy:.4f}) < seuil ({threshold})")
        print("   Modèle NON enregistré.")
        print("   Ajuste les hyperparamètres dans params.yaml et relance dvc repro")

if __name__ == "__main__":
    main()