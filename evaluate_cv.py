import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def read_cleaned(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    disease_col = cols.get("disease") or cols.get("diagnosis") or list(df.columns)[0]
    symptoms_col = cols.get("symptoms") or cols.get("symptom") or None
    if symptoms_col is None:
        for c in df.columns:
            sample = df[c].astype(str).iloc[0:10].tolist()
            if any(";" in s for s in sample):
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
                return [str(x).strip() for x in json.loads(s)]
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


def run_cv(cleaned_csv, n_splits=5, random_state=42, out_json="cv_report.json"):
    df, disease_col, sym_col = read_cleaned(cleaned_csv)
    y = df[disease_col].astype(str).values
    S = df[sym_col].tolist()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    accs = []
    precs = []
    recs = []
    f1s = []
    top3 = []

    fold = 0
    for train_idx, test_idx in skf.split(np.zeros(len(y)), y):
        fold += 1
        y_train = y[train_idx]
        y_test = y[test_idx]
        S_train = [S[i] for i in train_idx]
        S_test = [S[i] for i in test_idx]

        mlb = MultiLabelBinarizer()
        X_train = mlb.fit_transform(S_train)
        X_test = mlb.transform(S_test)

        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        clf = LogisticRegression(max_iter=2000)
        clf.fit(X_train, y_train_enc)

        y_pred_enc = clf.predict(X_test)
        y_pred = le.inverse_transform(y_pred_enc)

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)

        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)

        # top-3
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_test)
            hits = 0
            for i in range(len(y_test)):
                topk = np.argsort(probs[i])[-3:][::-1]
                if y_test_enc[i] in topk:
                    hits += 1
            top3.append(hits / len(y_test))
        else:
            top3.append(None)

        print(f"Fold {fold}: acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}, top3={(top3[-1] if top3[-1] is None else round(top3[-1],4))}")

    report = {
        "n_samples": len(df),
        "n_splits": n_splits,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "precision_mean": float(np.mean(precs)),
        "precision_std": float(np.std(precs)),
        "recall_mean": float(np.mean(recs)),
        "recall_std": float(np.std(recs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "top3_mean": float(np.mean([t for t in top3 if t is not None])) if any(t is not None for t in top3) else None,
        "top3_std": float(np.std([t for t in top3 if t is not None])) if any(t is not None for t in top3) else None,
        "per_fold": {
            f"fold_{i+1}": {
                "acc": float(accs[i]),
                "prec": float(precs[i]),
                "rec": float(recs[i]),
                "f1": float(f1s[i]),
                "top3": (float(top3[i]) if top3[i] is not None else None)
            } for i in range(len(accs))
        }
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nCross-validation summary:")
    print(f"Accuracy: {report['accuracy_mean']:.4f} ± {report['accuracy_std']:.4f}")
    print(f"Precision (macro): {report['precision_mean']:.4f} ± {report['precision_std']:.4f}")
    print(f"Recall (macro): {report['recall_mean']:.4f} ± {report['recall_std']:.4f}")
    print(f"F1 (macro): {report['f1_mean']:.4f} ± {report['f1_std']:.4f}")
    if report['top3_mean'] is not None:
        print(f"Top-3 accuracy: {report['top3_mean']:.4f} ± {report['top3_std']:.4f}")
    else:
        print("Top-3 accuracy: not available")
    print(f"Saved CV report to: {out_json}")


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    cleaned = os.path.join(base, "cleaned_disease_symptoms.csv")
    out = os.path.join(base, "cv_report.json")
    run_cv(cleaned, n_splits=5, random_state=42, out_json=out)
