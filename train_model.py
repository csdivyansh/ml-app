"""Train disease prediction model from a CSV of diseases and symptoms.

Usage:
    python train_model.py --csv path/to/data.csv --out disease_model.pkl

Assumptions and behavior:
- The CSV should contain a column named 'disease' (case-insensitive) or 'label' (or the last column will be used).
- Symptom data can be provided either as:
  1) A single column named 'symptoms' containing comma-separated symptom strings, OR
  2) Multiple binary columns (0/1) where each column is a symptom indicator.

The script will detect format automatically and use an appropriate feature extractor.
It trains a scikit-learn pipeline and saves it using joblib.
"""

import argparse
import os
import sys
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, FunctionTransformer


def detect_columns(df: pd.DataFrame):
    # Detect label column
    candidates = [c for c in df.columns if c.lower() in ("disease", "label", "target")] 
    if candidates:
        label_col = candidates[0]
    else:
        # fallback to last column
        label_col = df.columns[-1]

    # Detect symptom representations
    has_symptoms_col = any(c.lower() == "symptoms" for c in df.columns)

    # Detect binary indicator columns (0/1)
    symptom_like_cols = []
    for c in df.columns:
        if c == label_col:
            continue
        # treat small number of unique values (0/1) as indicators
        uniq = df[c].dropna().unique()
        if len(uniq) <= 3 and set(uniq).issubset({0, 1, "0", "1", True, False}):
            symptom_like_cols.append(c)

    return label_col, has_symptoms_col, symptom_like_cols


def parse_symptoms_col(series: pd.Series) -> List[List[str]]:
    # Expect comma-separated strings like "fever, cough"
    def split_symptoms(x):
        if pd.isna(x):
            return []
        if isinstance(x, list):
            return [s.strip().lower() for s in x if s]
        s = str(x)
        parts = [p.strip().lower() for p in s.split(",") if p.strip()]
        return parts

    return series.map(split_symptoms).tolist()


def build_pipeline_for_symptoms_list():
    # Convert list-of-symptoms to one-hot vector using MultiLabelBinarizer inside a FunctionTransformer
    def mlb_transform(X):
        # X is a pandas Series of lists
        mlb = MultiLabelBinarizer()
        arr = mlb.fit_transform(X)
        # store the mlb for later by returning a tuple (arr, mlb) -- but pipeline can't return that
        return arr

    # We'll handle MLb outside pipeline for clarity
    clf = LogisticRegression(max_iter=1000)
    return clf


def train_from_symptoms_col(df: pd.DataFrame, label_col: str, symptoms_col: str, out_path: str, classifier: str = "logreg"):
    print(f"Training from symptoms column: {symptoms_col} -> label: {label_col}")
    X_lists = parse_symptoms_col(df[symptoms_col])
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(X_lists)
    y = df[label_col].astype(str).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    if classifier == "logreg":
        clf = LogisticRegression(max_iter=1000)
    elif classifier == "rf":
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    else:
        clf = LogisticRegression(max_iter=1000)

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    # Save both the classifier and the mlb together
    model_artifact = {
        "type": "multilabel_list",
        "mlb": mlb,
        "classifier": clf
    }
    joblib.dump(model_artifact, out_path)
    print(f"Saved model to {out_path}")


def train_from_indicator_cols(df: pd.DataFrame, label_col: str, feature_cols: List[str], out_path: str, classifier: str = "logreg"):
    print(f"Training from indicator columns: {len(feature_cols)} cols -> label: {label_col}")
    X = df[feature_cols].fillna(0).astype(float).values
    y = df[label_col].astype(str).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    if classifier == "logreg":
        clf = LogisticRegression(max_iter=1000)
    elif classifier == "rf":
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    else:
        clf = LogisticRegression(max_iter=1000)

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    model_artifact = {
        "type": "indicator_matrix",
        "feature_columns": feature_cols,
        "classifier": clf
    }
    joblib.dump(model_artifact, out_path)
    print(f"Saved model to {out_path}")


def train_from_text_col(df: pd.DataFrame, label_col: str, text_col: str, out_path: str, classifier: str = "logreg"):
    print(f"Training from text column: {text_col} -> label: {label_col}")
    X_text = df[text_col].fillna("").astype(str).values
    y = df[label_col].astype(str).values

    X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42, stratify=y)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    if classifier == "logreg":
        clf = LogisticRegression(max_iter=1000)
    elif classifier == "rf":
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    else:
        clf = LogisticRegression(max_iter=1000)

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    model_artifact = {
        "type": "text",
        "vectorizer": vectorizer,
        "classifier": clf
    }
    joblib.dump(model_artifact, out_path)
    print(f"Saved model to {out_path}")


def main(args):
    df = pd.read_csv(args.csv)
    print("Loaded CSV with shape:", df.shape)

    label_col, has_symptoms_col, symptom_like_cols = detect_columns(df)
    print("Detected label column:", label_col)
    print("Has 'symptoms' column:", has_symptoms_col)
    print("Indicator-like columns:", len(symptom_like_cols))

    # Prefer explicit symptoms column
    if has_symptoms_col:
        symptoms_col = [c for c in df.columns if c.lower() == "symptoms"][0]
        train_from_symptoms_col(df, label_col, symptoms_col, args.out, classifier=args.classifier)
        return

    # If many indicator-like columns exist, use them
    if len(symptom_like_cols) >= 3:
        train_from_indicator_cols(df, label_col, symptom_like_cols, args.out, classifier=args.classifier)
        return

    # Fall back to using a text column (other than label)
    text_cols = [c for c in df.columns if c != label_col and df[c].dtype == object]
    if text_cols:
        # pick the longest-average column
        text_col = sorted(text_cols, key=lambda c: df[c].dropna().astype(str).map(len).mean(), reverse=True)[0]
        train_from_text_col(df, label_col, text_col, args.out, classifier=args.classifier)
        return

    # If nothing matched, raise
    print("Could not auto-detect symptom columns. Please provide a CSV with either:")
    print(" - a 'symptoms' column containing comma-separated symptoms, or")
    print(" - several 0/1 indicator columns where each column is a symptom")
    sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train disease prediction model from CSV")
    parser.add_argument("--csv", required=True, help="Path to CSV file containing symptoms and disease label")
    parser.add_argument("--out", default="disease_model.pkl", help="Output path for saved model (joblib)")
    parser.add_argument("--classifier", choices=["logreg", "rf"], default="logreg", help="Classifier to use (logreg or rf)")
    args = parser.parse_args()
    main(args)
