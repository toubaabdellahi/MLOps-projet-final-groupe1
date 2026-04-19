import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import yaml
import os
import pickle

def main():
    # Charger les paramètres
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    # ============================
    # CHARGEMENT DES DONNÉES
    # ============================
    print("📥 Chargement des données depuis data/raw/churn.csv...")
    df = pd.read_csv("data/raw/churn.csv")
    print(f"   Shape initial : {df.shape}")
    print(f"   Colonnes : {list(df.columns)}")

    # ============================
    # NETTOYAGE
    # ============================
    print("🧹 Nettoyage des données...")
    
    # Supprimer customerID (identifiant, pas une feature)
    df = df.drop("customerID", axis=1)

    # TotalCharges contient des espaces vides → convertir en numérique
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    # Remplir les valeurs manquantes par la médiane
    nb_missing = df["TotalCharges"].isna().sum()
    print(f"   Valeurs manquantes TotalCharges : {nb_missing}")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # ============================
    # ENCODAGE
    # ============================
    print("🔄 Encodage des variables catégorielles...")
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Retirer la colonne cible
    if "Churn" in categorical_cols:
        categorical_cols.remove("Churn")

    print(f"   Colonnes catégorielles à encoder : {categorical_cols}")

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"   ✓ {col} encodé ({len(le.classes_)} valeurs)")

    # Encoder la target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    print(f"   Distribution Churn : {df['Churn'].value_counts().to_dict()}")

    # ============================
    # SPLIT TRAIN/TEST
    # ============================
    print("✂️ Split train/test...")
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params["preprocess"]["test_size"],
        random_state=params["preprocess"]["random_state"],
        stratify=y  # Garder la même proportion de churn dans train et test
    )
    print(f"   Train : {X_train.shape}, Test : {X_test.shape}")

    # ============================
    # NORMALISATION
    # ============================
    print("📐 Normalisation (StandardScaler)...")
    scaler = StandardScaler()
    feature_names = X_train.columns.tolist()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_names
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_names
    )

    # ============================
    # SAUVEGARDE
    # ============================
    os.makedirs("data/processed", exist_ok=True)

    X_train_scaled.to_csv("data/processed/X_train.csv", index=False)
    X_test_scaled.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    # Sauvegarder scaler et encoders (l'API en aura besoin)
    with open("data/processed/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("data/processed/label_encoders.pkl", "wb") as f:
        pickle.dump(label_encoders, f)
    with open("data/processed/feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)

    print(f"\n✅ Preprocessing terminé !")
    print(f"   Features : {len(feature_names)}")
    print(f"   Fichiers sauvegardés dans data/processed/")

if __name__ == "__main__":
    main()