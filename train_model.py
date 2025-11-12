"""
Retrain model using RandomForest + binary symptom encoding.
Uses DiseaseAndSymptoms.csv and Disease precaution.csv.
Saves artifact with model, encoder, all_symptoms list, and precautions dict.
"""
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

BASE = os.path.dirname(__file__)
SYM_CSV = os.path.join(BASE, "DiseaseAndSymptoms.csv")
PREC_CSV = os.path.join(BASE, "Disease precaution.csv")
MODEL_PATH = os.path.join(BASE, "disease_model.pkl")


def train_model():
    print("Loading datasets...")
    sym_df = pd.read_csv(SYM_CSV)
    prec_df = pd.read_csv(PREC_CSV)
    
    print(f"Symptoms dataset: {sym_df.shape}")
    print(f"Precautions dataset: {prec_df.shape}")
    
    # Get symptom columns (columns 1-17, assuming column 0 is disease)
    symptom_cols = [col for col in sym_df.columns if col.startswith("Symptom")]
    print(f"Symptom columns: {len(symptom_cols)}")
    
    # Normalize symptoms
    for col in symptom_cols:
        sym_df[col] = sym_df[col].str.lower().str.strip()
    
    # Merge symptoms into list (drop NaN values)
    sym_df["symptoms"] = sym_df[symptom_cols].apply(
        lambda row: [s for s in row if pd.notna(s) and s != ""], axis=1
    )
    
    # Get all unique symptoms
    all_symptoms = sorted(set([s for lst in sym_df["symptoms"] for s in lst]))
    print(f"Total unique symptoms: {len(all_symptoms)}")
    
    # Binary encode symptoms
    def encode_symptoms(symptom_list):
        return [1 if s in symptom_list else 0 for s in all_symptoms]
    
    sym_df["symptom_vector"] = sym_df["symptoms"].apply(encode_symptoms)
    
    # Prepare training data
    X = np.array(sym_df['symptom_vector'].tolist())
    y = sym_df['Disease']
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    
    print("\nTraining RandomForest...")
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
    
    # Build precautions dict
    prec_dict = {}
    for _, row in prec_df.iterrows():
        disease = row['Disease']
        precautions = [
            str(row[col]).strip() for col in prec_df.columns 
            if col.startswith("Precaution") and pd.notna(row[col])
        ]
        prec_dict[disease.lower()] = precautions
    
    print(f"\nLoaded precautions for {len(prec_dict)} diseases")
    
    # Save artifact
    artifact = {
        'model': model,
        'le': le,
        'all_symptoms': all_symptoms,
        'prec_dict': prec_dict
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    
    return model, le, all_symptoms, prec_dict


if __name__ == "__main__":
    train_model()
