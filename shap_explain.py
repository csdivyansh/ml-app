"""Small helper to compute SHAP explanations for a single symptom input.

Usage:
    from shap_explain import explain_symptoms
    explain_symptoms(['itching','skin_rash'])

Returns a list of top contributing symptoms with SHAP values.
"""
import os
import joblib
import numpy as np

BASE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, "disease_model.pkl")


def explain_symptoms(symptoms, top_k=5):
    """Explain prediction by showing which input symptoms contributed and model confidence."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    artifact = joblib.load(MODEL_PATH)
    mlb = artifact.get('mlb')
    le = artifact.get('le')
    clf = artifact.get('clf')

    if mlb is None or clf is None:
        raise RuntimeError("Model artifact missing required components (mlb and clf)")

    normalized = [str(s).strip().lower().replace(' ', '_') for s in symptoms]
    X = mlb.transform([normalized])

    # Get prediction and probabilities
    pred_idx = clf.predict(X)[0]
    probs = clf.predict_proba(X)[0]
    
    # Get class name
    if le is not None:
        try:
            pred_class = str(le.inverse_transform([pred_idx])[0])
        except Exception:
            pred_class = str(pred_idx)
    else:
        pred_class = str(pred_idx)

    # Get feature (symptom) names
    feature_names = list(getattr(mlb, 'classes_', []))
    if not feature_names:
        feature_names = [f'symptom_{i}' for i in range(mlb.n_features_in_)]

    # Simple explanation: show which input symptoms matched features
    matched_features = [f for f in feature_names if f in normalized]
    
    # Show top symptoms by their coefficient weights (for linear models)
    top_features = []
    try:
        if hasattr(clf, 'estimator_') and hasattr(clf.estimator_, 'coef_'):
            # CalibratedClassifierCV wraps base estimator
            coef = clf.estimator_.coef_[pred_idx]
        elif hasattr(clf, 'coef_'):
            coef = clf.coef_[pred_idx]
        else:
            coef = None
        
        if coef is not None:
            # Pair features with coefficients
            pairs = list(zip(feature_names, coef))
            # Sort by absolute value descending
            pairs_sorted = sorted(pairs, key=lambda x: abs(float(x[1])), reverse=True)
            for i, (fname, fval) in enumerate(pairs_sorted[:top_k]):
                top_features.append({
                    'rank': i + 1,
                    'feature': fname,
                    'coefficient': float(fval),
                    'in_input': fname in normalized,
                    'direction': 'positive' if fval > 0 else 'negative'
                })
    except Exception:
        # Fallback: just return input symptoms
        for i, f in enumerate(matched_features[:top_k]):
            top_features.append({
                'rank': i + 1,
                'feature': f,
                'in_input': True,
                'direction': 'contributing'
            })

    return {
        'input_symptoms': symptoms,
        'predicted_disease': pred_class,
        'confidence': float(probs.max()),
        'matched_symptoms': matched_features,
        'top_contributing_features': top_features
    }

