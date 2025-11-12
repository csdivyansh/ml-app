import os
import json
import joblib
import numpy as np
import pandas as pd
import re
from sklearn.metrics import (
    accuracy_score, classification_report, precision_recall_fscore_support,
    confusion_matrix
)
from sklearn.model_selection import train_test_split, cross_val_score


def normalize_token(tok):
    """Normalize symptom token"""
    if not tok or pd.isna(tok):
        return ""
    t = str(tok).strip().lower()
    t = t.strip(' ,.')
    t = re.sub(r"\s*_\s*", "_", t)
    t = re.sub(r"\s+", " ", t)
    t = t.replace(' ', '_')
    t = re.sub(r"_+", "_", t)
    return t


def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    if isinstance(model, dict):
        # New format with RandomForest
        rf_model = model.get("model")
        le = model.get("le")
        all_symptoms = model.get("all_symptoms", [])
        return rf_model, le, all_symptoms
    else:
        # Old format
        mlb = model.get("mlb")
        le = model.get("le")
        clf = model.get("clf")
        return clf, le, None


def evaluate(model_path, dataset_csv_path, out_json="eval_report.json", top_k=3):
    print("=" * 80)
    print("MODEL EVALUATION: Accuracy, Precision, Recall, F1-Score")
    print("=" * 80)
    
    # Load model
    print("\n📦 Loading model...")
    rf_model, le, all_symptoms = load_model(model_path)
    print(f"✅ Model loaded successfully")
    print(f"   - Model type: {type(rf_model).__name__}")
    print(f"   - Features: {len(all_symptoms)}")
    print(f"   - Classes: {len(le.classes_)}")
    
    # Load dataset
    print("\n📊 Loading dataset...")
    sym_df = pd.read_csv(dataset_csv_path)
    print(f"✅ Dataset loaded: {sym_df.shape}")
    
    # Get symptom columns
    symptom_cols = [col for col in sym_df.columns if col.startswith("Symptom")]
    
    # Normalize symptoms
    for col in symptom_cols:
        sym_df[col] = sym_df[col].apply(normalize_token)
    
    # Build feature matrix
    print("\n🔧 Building feature matrix...")
    X = []
    y = []
    
    for _, row in sym_df.iterrows():
        symptoms = [row[col] for col in symptom_cols if row[col] and row[col] != ""]
        # Binary encode
        binary_vector = [1 if s in symptoms else 0 for s in all_symptoms]
        X.append(binary_vector)
        y.append(row['Disease'])
    
    X = np.array(X)
    y_encoded = le.transform(y)
    
    print(f"✅ Feature matrix built: {X.shape}")
    print(f"   - Samples: {len(X)}")
    print(f"   - Features: {X.shape[1]}")
    print(f"   - Feature density: {X.mean():.3f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\n📊 Train/Test Split:")
    print(f"   - Train: {len(X_train)} samples")
    print(f"   - Test: {len(X_test)} samples")
    
    # Make predictions
    print("\n🔮 Making predictions on test set...")
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)
    
    # Decode labels
    y_test_labels = le.inverse_transform(y_test)
    y_pred_labels = le.inverse_transform(y_pred)
    
    # Calculate metrics
    print("\n" + "=" * 80)
    print("EVALUATION METRICS")
    print("=" * 80)
    
    acc = accuracy_score(y_test_labels, y_pred_labels)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test_labels, y_pred_labels, average='macro', zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test_labels, y_pred_labels, average='weighted', zero_division=0
    )
    
    print(f"\n🎯 Overall Metrics (Test Set):")
    print(f"   Accuracy:            {acc:.4f} ({acc*100:.2f}%)")
    print(f"\n   Precision (Macro):   {prec_macro:.4f}")
    print(f"   Precision (Weighted):{prec_weighted:.4f}")
    print(f"\n   Recall (Macro):      {rec_macro:.4f}")
    print(f"   Recall (Weighted):   {rec_weighted:.4f}")
    print(f"\n   F1-Score (Macro):    {f1_macro:.4f}")
    print(f"   F1-Score (Weighted): {f1_weighted:.4f}")
    
    # Top-K accuracy
    print(f"\n📈 Top-{top_k} Accuracy:")
    topk_hits = 0
    for i in range(len(y_test)):
        probs = y_pred_proba[i]
        topk_indices = np.argsort(probs)[-top_k:][::-1]
        if y_test[i] in topk_indices:
            topk_hits += 1
    topk_acc = topk_hits / len(y_test)
    print(f"   Top-{top_k}: {topk_acc:.4f} ({topk_acc*100:.2f}%)")
    
    # Cross-validation
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION (5-Fold)")
    print("=" * 80)
    print("\n⏳ Running cross-validation...")
    
    cv_acc = cross_val_score(rf_model, X, y_encoded, cv=5, scoring='accuracy')
    cv_prec = cross_val_score(rf_model, X, y_encoded, cv=5, scoring='precision_macro')
    cv_rec = cross_val_score(rf_model, X, y_encoded, cv=5, scoring='recall_macro')
    cv_f1 = cross_val_score(rf_model, X, y_encoded, cv=5, scoring='f1_macro')
    
    print(f"\n📊 Cross-Validation Results:")
    print(f"   Accuracy:  {cv_acc.mean():.4f} (±{cv_acc.std():.4f})")
    print(f"   Precision: {cv_prec.mean():.4f} (±{cv_prec.std():.4f})")
    print(f"   Recall:    {cv_rec.mean():.4f} (±{cv_rec.std():.4f})")
    print(f"   F1-Score:  {cv_f1.mean():.4f} (±{cv_f1.std():.4f})")
    
    # Confidence analysis
    print("\n" + "=" * 80)
    print("CONFIDENCE ANALYSIS")
    print("=" * 80)
    
    max_probs = np.max(y_pred_proba, axis=1)
    print(f"\n📊 Prediction Confidence:")
    print(f"   Mean:   {max_probs.mean():.4f}")
    print(f"   Median: {np.median(max_probs):.4f}")
    print(f"   Min:    {max_probs.min():.4f}")
    print(f"   Max:    {max_probs.max():.4f}")
    print(f"   Std:    {max_probs.std():.4f}")
    
    # Confidence bins
    bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
    labels_conf = ['Very Low (<0.3)', 'Low (0.3-0.5)', 'Medium (0.5-0.7)', 'High (0.7-0.9)', 'Very High (>0.9)']
    confidence_bins = pd.cut(max_probs, bins=bins, labels=labels_conf)
    
    print(f"\n📊 Confidence Distribution:")
    for label in labels_conf:
        count = (confidence_bins == label).sum()
        pct = count / len(max_probs) * 100
        print(f"   {label:<20}: {count:>4} ({pct:>5.1f}%)")
    
    # Per-class report
    report_dict = classification_report(
        y_test_labels, y_pred_labels, zero_division=0, output_dict=True
    )
    
    # Save results
    report = {
        "n_samples_total": len(sym_df),
        "n_test_samples": len(X_test),
        "test_set_metrics": {
            "accuracy": float(acc),
            "precision_macro": float(prec_macro),
            "precision_weighted": float(prec_weighted),
            "recall_macro": float(rec_macro),
            "recall_weighted": float(rec_weighted),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            f"top_{top_k}_accuracy": float(topk_acc)
        },
        "cross_validation_metrics": {
            "accuracy_mean": float(cv_acc.mean()),
            "accuracy_std": float(cv_acc.std()),
            "precision_mean": float(cv_prec.mean()),
            "precision_std": float(cv_prec.std()),
            "recall_mean": float(cv_rec.mean()),
            "recall_std": float(cv_rec.std()),
            "f1_mean": float(cv_f1.mean()),
            "f1_std": float(cv_f1.std())
        },
        "confidence_stats": {
            "mean": float(max_probs.mean()),
            "median": float(np.median(max_probs)),
            "min": float(max_probs.min()),
            "max": float(max_probs.max()),
            "std": float(max_probs.std())
        },
        "classification_report": report_dict
    }
    
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"✅ Evaluation complete! Results saved to: {out_json}")
    print("=" * 80)


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    model_path = os.path.join(base, "disease_model.pkl")
    dataset_csv = os.path.join(base, "DiseaseAndSymptoms.csv")
    out = os.path.join(base, "eval_report.json")
    try:
        evaluate(model_path, dataset_csv, out_json=out, top_k=5)
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        raise
