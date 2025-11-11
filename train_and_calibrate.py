"""Train (or load) a classifier and calibrate its probabilities using a held-out set.
Saves updated model artifact to disease_model.pkl (dict with keys 'mlb','le','clf').

Usage:
    python train_and_calibrate.py

If a pre-trained disease_model.pkl exists, it will be loaded and re-calibrated on a validation split from
`cleaned_disease_symptoms.csv`. Otherwise, the script will train a new LogisticRegression model and calibrate it.
"""
import os
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

BASE = os.path.dirname(__file__)
CLEANED = os.path.join(BASE, "cleaned_disease_symptoms.csv")
MODEL_PATH = os.path.join(BASE, "disease_model.pkl")


def read_cleaned(path=CLEANED):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    disease_col = cols.get("disease") or list(df.columns)[0]
    symptoms_col = cols.get("symptoms") or cols.get("symptom") or None
    if symptoms_col is None:
        for c in df.columns:
            sample = df[c].astype(str).iloc[0:10].tolist()
            if any(";" in s or "," in s or s.startswith("[") for s in sample):
                symptoms_col = c
                break
    if symptoms_col is None:
        raise ValueError("Could not find symptoms column")

    def parse_sym(s):
        if pd.isna(s):
            return []
        s = str(s).strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = json.loads(s)
                return [str(x).strip() for x in arr if x]
            except Exception:
                pass
        if ";" in s:
            return [p.strip() for p in s.split(";") if p.strip()]
        if "," in s:
            return [p.strip() for p in s.split(",") if p.strip()]
        return [s] if s else []

    df["_sym_list"] = df[symptoms_col].apply(parse_sym)
    df = df[df["_sym_list"].map(len) > 0].reset_index(drop=True)
    return df, disease_col, "_sym_list"


def train_and_calibrate(test_size=0.2, calib_size=0.2, random_state=42):
    df, disease_col, sym_col = read_cleaned()
    X_list = df[sym_col].tolist()
    y = df[disease_col].astype(str).values

    # Split into train+calibration+test
    X_temp, X_test, y_temp, y_test = train_test_split(X_list, y, test_size=test_size, stratify=y, random_state=random_state)
    # from X_temp create train and calib
    calib_relative = calib_size / (1 - test_size)
    X_train, X_calib, y_train, y_calib = train_test_split(X_temp, y_temp, test_size=calib_relative, stratify=y_temp, random_state=random_state)

    mlb = MultiLabelBinarizer()
    X_train_mat = mlb.fit_transform(X_train)
    X_calib_mat = mlb.transform(X_calib)
    X_test_mat = mlb.transform(X_test)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_calib_enc = le.transform(y_calib)
    y_test_enc = le.transform(y_test)

    # Train base classifier
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train_mat, y_train_enc)

    # Calibrate using sigmoid (Platt scaling) on calibration set
    calibrated = CalibratedClassifierCV(estimator=clf, method='sigmoid', cv='prefit')
    calibrated.fit(X_calib_mat, y_calib_enc)

    # Evaluate simple metrics
    y_pred = calibrated.predict(X_test_mat)
    from sklearn.metrics import accuracy_score, classification_report
    acc = accuracy_score(y_test_enc, y_pred)
    print(f"Calibration: test accuracy (post-calibration) = {acc:.4f}")

    # Save artifact
    artifact = {
        'mlb': mlb,
        'le': le,
        'clf': calibrated
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved calibrated model to {MODEL_PATH}")


if __name__ == '__main__':
    train_and_calibrate()
