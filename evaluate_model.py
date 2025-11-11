import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support


def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    if isinstance(model, dict):
        mlb = model.get("mlb")
        le = model.get("le")
        clf = model.get("clf")
    else:
        mlb = le = None
        clf = model
    return mlb, le, clf


def read_cleaned_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cleaned dataset not found: {path}")
    df = pd.read_csv(path)
    # expected columns: Disease, Symptoms (semicolon-separated)
    # Accept different capitalizations
    cols = {c.lower(): c for c in df.columns}
    disease_col = cols.get("disease") or cols.get("diagnosis") or list(df.columns)[0]
    # symptoms column might be 'Symptoms' or 'symptoms'
    symptoms_col = cols.get("symptoms") or cols.get("symptom") or None
    if symptoms_col is None:
        # Try to find a column that contains a semicolon in many rows
        for c in df.columns:
            sample = df[c].astype(str).iloc[0:10].tolist()
            if any(";" in s for s in sample):
                symptoms_col = c
                break
    if symptoms_col is None:
        raise ValueError("Could not find symptoms column in cleaned dataset")

    # Parse symptoms into lists
    def parse_sym(s):
        if pd.isna(s):
            return []
        if isinstance(s, (list, tuple)):
            return list(s)
        s = str(s).strip()
        # If already JSON-like list
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = json.loads(s)
                return [str(x).strip() for x in arr if x]
            except Exception:
                pass
        # split on semicolon or comma
        if ";" in s:
            parts = [p.strip() for p in s.split(";") if p.strip()]
            return parts
        if "," in s:
            parts = [p.strip() for p in s.split(",") if p.strip()]
            return parts
        return [s] if s else []

    df["_sym_list"] = df[symptoms_col].apply(parse_sym)
    df = df[df["_sym_list"].map(len) > 0].reset_index(drop=True)
    return df, disease_col, "_sym_list"


def evaluate(model_path, cleaned_csv_path, out_json="eval_report.json", top_k=3):
    mlb, le, clf = load_model(model_path)
    df, disease_col, sym_col = read_cleaned_dataset(cleaned_csv_path)

    # true labels as strings
    y_true = df[disease_col].astype(str).tolist()

    # Build feature matrix
    if mlb is not None:
        X = mlb.transform(df[sym_col].tolist())
    else:
        # fallback: simple binary bag-of-symptom using union of all tokens
        all_tokens = sorted({t for L in df[sym_col].tolist() for t in L})
        tok_index = {t: i for i, t in enumerate(all_tokens)}
        X = np.zeros((len(df), len(all_tokens)), dtype=int)
        for i, L in enumerate(df[sym_col].tolist()):
            for t in L:
                if t in tok_index:
                    X[i, tok_index[t]] = 1

    # If label encoder available, transform y_true to indices for top-k checks
    if le is not None:
        try:
            y_true_idx = le.transform(y_true)
        except Exception:
            # fit_transform might fail if unseen labels; map via dict
            name_to_idx = {n: i for i, n in enumerate(le.classes_)}
            y_true_idx = np.array([name_to_idx.get(n, -1) for n in y_true])
    else:
        y_true_idx = None

    # Predictions
    y_pred = clf.predict(X)
    # Convert to strings if encoded
    if le is not None and np.issubdtype(type(y_pred[0]), np.integer):
        try:
            y_pred_labels = le.inverse_transform(y_pred)
        except Exception:
            y_pred_labels = [str(p) for p in y_pred]
    else:
        y_pred_labels = [str(p) for p in y_pred]

    # Metrics
    acc = accuracy_score(y_true, y_pred_labels)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred_labels, average='macro', zero_division=0)
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred_labels, average='micro', zero_division=0)

    report = {
        "n_samples": len(df),
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_micro": float(prec_micro),
        "recall_micro": float(rec_micro),
        "f1_micro": float(f1_micro),
        "classification_report": classification_report(y_true, y_pred_labels, zero_division=0, output_dict=True)
    }

    # Top-k accuracy if predict_proba available and label encoder present
    if hasattr(clf, "predict_proba") and y_true_idx is not None:
        try:
            probs = clf.predict_proba(X)
            topk_hits = 0
            total = len(df)
            for i in range(len(df)):
                row = probs[i]
                topk = np.argsort(row)[-top_k:][::-1]
                if y_true_idx[i] in topk:
                    topk_hits += 1
            report[f"top_{top_k}_accuracy"] = float(topk_hits / total)
        except Exception:
            report[f"top_{top_k}_accuracy"] = None
    else:
        report[f"top_{top_k}_accuracy"] = None

    # Save report
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print concise summary
    print(f"Evaluated {report['n_samples']} samples")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Precision (macro): {report['precision_macro']:.4f}")
    print(f"Recall (macro): {report['recall_macro']:.4f}")
    print(f"F1 (macro): {report['f1_macro']:.4f}")
    if report.get(f"top_{top_k}_accuracy") is not None:
        print(f"Top-{top_k} accuracy: {report[f'top_{top_k}_accuracy']:.4f}")
    else:
        print(f"Top-{top_k} accuracy: not available")
    print(f"Saved full report to: {out_json}")


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    model_path = os.path.join(base, "disease_model.pkl")
    cleaned_csv = os.path.join(base, "cleaned_disease_symptoms.csv")
    out = os.path.join(base, "eval_report.json")
    try:
        evaluate(model_path, cleaned_csv, out_json=out, top_k=3)
    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise
