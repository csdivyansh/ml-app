import os
import joblib
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import warnings
import importlib
import logging

warnings.filterwarnings('ignore')

# Configure simple logging instead of print statements
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("ml_api")

# Initialize FastAPI app
app = FastAPI(title="AI Health ML API", version="1.0.0")

# Add CORS middleware to allow requests from MERN frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",      # Frontend development
        "http://localhost:3000",      # Alternative frontend port
        "http://localhost:5000",      # Node backend
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained model
model = None
model_loaded = False
# Expect the artifact saved as a dict: {'mlb': MultiLabelBinarizer, 'le': LabelEncoder, 'clf': estimator}
mlb = None
le = None
clf = None

try:
    model_path = os.path.join(os.path.dirname(__file__), "disease_model.pkl")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        # Support both plain estimator and saved dict artifact
        if isinstance(model, dict):
            mlb = model.get('mlb')
            le = model.get('le')
            clf = model.get('clf')
            model_loaded = clf is not None
        else:
            # legacy: model is an estimator
            clf = model
            model_loaded = True
        logger.info("Disease model loaded successfully")
    else:
        logger.warning("Model file not found at %s", model_path)
        logger.warning("Please ensure disease_model.pkl is in the ml-api directory")
except Exception as e:
    logger.exception("Error loading model: %s", e)
    model = None
    mlb = le = clf = None


# Define request schema
class SymptomInput(BaseModel):
    """Schema for symptom data input"""
    symptoms: List[str]
    metadata: Dict[str, Any] = {}


class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    predicted_disease: str
    confidence: float = None
    symptoms_input: List[str]
    success: bool = True
    message: str = None


# Health check endpoint
@app.get("/health")
async def health_check():
    """Check if API is running and model is loaded"""
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "message": "ML API is ready" if model_loaded else "Model not loaded - place disease_model.pkl in ml-api folder"
    }


# Main prediction endpoint
@app.post("/predict")
async def predict_disease(input_data: SymptomInput):
    """
    Predict disease based on symptoms
    
    Args:
        input_data: SymptomInput with symptoms list and optional metadata
        
    Returns:
        PredictionResponse with predicted disease
    """
    try:
        if not model_loaded or model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please ensure disease_model.pkl exists."
            )
        
        if not input_data.symptoms or len(input_data.symptoms) == 0:
            raise HTTPException(
                status_code=400,
                detail="No symptoms provided. Please provide at least one symptom."
            )
        
        # Convert symptoms list to features using the saved MultiLabelBinarizer if present
        try:
            if mlb is not None and clf is not None:
                # normalize tokens similarly to training (lower/strip/underscores)
                normalized = [normalize_token(s) for s in input_data.symptoms]
                X = mlb.transform([normalized])
                pred_idx = clf.predict(X)[0]
                # build probabilities / top-k
                probs = None
                top_k_list = []
                try:
                    if hasattr(clf, 'predict_proba'):
                        probs = clf.predict_proba(X)[0]
                        # get class labels
                        classes = None
                        if le is not None:
                            classes = list(le.inverse_transform(list(range(len(le.classes_)))))
                        else:
                            # try to get classes_ attribute
                            try:
                                classes = [str(c) for c in clf.classes_]
                            except Exception:
                                classes = None

                        if classes is not None:
                            # sort by probability descending
                            idxs = list(reversed(np.argsort(probs)))
                            for rank, i in enumerate(idxs[:5], start=1):
                                label = str(classes[i])
                                score = float(probs[i])
                                # explain by listing overlapping symptoms
                                matched = [t for t in normalized if t in (mlb.classes_.tolist() if hasattr(mlb, 'classes_') else [])]
                                short_expl = f"Based on presence of: {', '.join(matched)}" if matched else "Based on symptom patterns in training data"
                                top_k_list.append({
                                    "rank": rank,
                                    "text": label,
                                    "score": round(score, 3),
                                    "short_explanation": short_expl,
                                })
                except Exception:
                    probs = None

                # top-1 prediction and confidence
                if le is not None:
                    try:
                        predicted_disease = str(le.inverse_transform([pred_idx])[0])
                    except Exception:
                        predicted_disease = str(pred_idx)
                else:
                    predicted_disease = str(pred_idx)

                confidence = None
                if probs is not None:
                    confidence = float(probs.max())
                else:
                    # fallback: set confidence to None
                    confidence = None

                # urgency detection (simple rule-based)
                urgency_score, urgency_level, urgency_reason, recommended_action = 0.0, "none", "No immediate danger detected", None
                try:
                    red_flags = {"loss_of_consciousness","unconscious","severe_headache","sudden","paralysis","weakness","chest_pain","shortness_of_breath","severe","fainting"}
                    norm_set = set(normalized)
                    intersect = norm_set.intersection(red_flags)
                    if intersect:
                        urgency_score = 0.95
                        urgency_level = "critical"
                        urgency_reason = f"Contains red-flag symptoms: {', '.join(intersect)}"
                        recommended_action = "Seek immediate emergency medical care (call emergency services)."
                    else:
                        # moderate urgency if severe present in tokens
                        if any(t for t in normalized if "severe" in t or "high" in t):
                            urgency_score = 0.6
                            urgency_level = "high"
                            urgency_reason = "Symptoms indicate possible serious condition; clinical review advised"
                            recommended_action = "Consult a clinician promptly"
                except Exception:
                    pass

                overall_confidence = confidence

                # compatibility fields for existing clients/tests
                result = {
                    "predicted_disease": predicted_disease,
                    "confidence": overall_confidence,
                    "symptoms_input": input_data.symptoms,
                    "success": True,
                    "message": "Prediction successful",
                    # new structured output
                    "top_k": top_k_list,
                    "overall_confidence": overall_confidence,
                    "confidence_calibration": "unknown",
                    "urgency_flag": {
                        "level": urgency_level,
                        "score": round(float(urgency_score), 3),
                        "reason": urgency_reason,
                        "recommended_action": recommended_action,
                    },
                    "safety_actions": [],
                    "model_id": getattr(clf, '__class__', str(type(clf))).__name__,
                }

                return result
            elif clf is not None:
                # fallback: if only estimator saved, try previous prepare_features
                features = prepare_features(input_data.symptoms)
                prediction = clf.predict([features])
                predicted_disease = str(prediction[0])
                confidence = None
                try:
                    if hasattr(clf, 'predict_proba'):
                        probabilities = clf.predict_proba([features])
                        confidence = float(np.max(probabilities))
                except:
                    pass
                return PredictionResponse(
                    predicted_disease=predicted_disease,
                    confidence=confidence,
                    symptoms_input=input_data.symptoms,
                    success=True,
                    message="Prediction successful"
                )
            else:
                raise HTTPException(status_code=503, detail="Model not available for prediction")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction endpoint error: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


def prepare_features(symptoms: List[str]) -> np.ndarray:
    """
    Convert symptom list to feature vector for model
    
    Adjust this function based on how your model was trained.
    Common approaches:
    1. One-hot encoding of symptoms
    2. Symptom presence (0/1) vector
    3. TF-IDF vectorization
    
    Args:
        symptoms: List of symptom strings
        
    Returns:
        Feature vector as numpy array
    """
    try:
        # Create a DataFrame from symptoms for compatibility
        # This assumes your model was trained with specific symptom features
        
        # Simple approach: create a dictionary with symptom presence
        feature_dict = {}
        for symptom in symptoms:
            feature_dict[symptom.lower().strip()] = 1
        
        # Convert to pandas Series and then to numpy array
        df = pd.DataFrame([feature_dict])
        features = df.fillna(0).values.flatten()
        
        return features
    except Exception as e:
        print(f"Error preparing features: {e}")
        # Return a default feature vector if preparation fails
        return np.array([1.0] * len(symptoms))


def normalize_token(tok: str) -> str:
    """Normalize a symptom token to match training normalization.
    Lowercase, strip whitespace, collapse spaces, convert spaces to underscores,
    and remove stray punctuation around tokens.
    """
    if not tok:
        return tok
    t = str(tok).strip()
    t = t.strip(' ,.')
    t = t.lower()
    t = re.sub(r"\s*_\s*", "_", t)
    t = re.sub(r"\s+", " ", t)
    t = t.replace(' ', '_')
    t = re.sub(r"_+", "_", t)
    return t


# Batch prediction endpoint
@app.post("/predict-batch")
async def predict_batch(inputs: List[SymptomInput]):
    """
    Predict diseases for multiple patients
    
    Args:
        inputs: List of SymptomInput objects
        
    Returns:
        List of predictions
    """
    try:
        if not model_loaded or model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded"
            )
        
        results = []
        for input_data in inputs:
            try:
                result = await predict_disease(input_data)
                results.append(result)
            except HTTPException as e:
                results.append({
                    "success": False,
                    "error": e.detail,
                    "symptoms_input": input_data.symptoms
                })
        
        return {
            "success": True,
            "count": len(results),
            "predictions": results
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API documentation"""
    return {
        "message": "AI Health ML API",
        "version": "1.0.0",
        "model_loaded": model_loaded,
        "endpoints": {
            "health": "/health (GET)",
            "predict": "/predict (POST)",
            "predict_batch": "/predict-batch (POST)",
            "docs": "/docs (interactive documentation)"
        },
        "usage": {
            "predict": {
                "method": "POST",
                "url": "/predict",
                "body": {
                    "symptoms": ["symptom1", "symptom2"],
                    "metadata": {}
                }
            }
        }
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Run on app startup"""
    logger.info("AI Health ML API starting")
    if model_loaded:
        logger.info("Model loaded and ready for predictions")
    else:
        logger.warning("Model not loaded - predictions will fail")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Run on app shutdown"""
    logger.info("AI Health ML API shutting down")


# Explainability endpoint (optional - requires shap installed)
@app.post("/explain")
async def explain(input_data: SymptomInput):
    """Return SHAP-like top contributing symptoms for the top prediction.
    This endpoint requires the `shap` package to be installed in the environment.
    """
    try:
        shap_spec = importlib.util.find_spec("shap")
        if shap_spec is None:
            raise HTTPException(status_code=501, detail="SHAP is not installed on the server")
        # lazy import
        import shap
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error importing shap: {e}")

    if not model_loaded or mlb is None or clf is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        normalized = [normalize_token(s) for s in input_data.symptoms]
        X = mlb.transform([normalized])

        # Choose explainer for linear models
        try:
            if hasattr(clf, 'coef_'):
                explainer = shap.LinearExplainer(clf, mlb.transform([[]]), feature_dependence="independent")
            else:
                explainer = shap.Explainer(clf.predict_proba, mlb.transform([[]]))
        except Exception:
            explainer = shap.KernelExplainer(clf.predict_proba, mlb.transform([[]]))

        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            probs = clf.predict_proba(X)[0]
            top_class = int(np.argmax(probs))
            vals = sv[top_class][0]
        else:
            vals = sv[0]

        feature_names = list(getattr(mlb, 'classes_', []))
        if not feature_names:
            feature_names = [f'feature_{i}' for i in range(len(vals))]

        pairs = list(zip(feature_names, vals))
        pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)
        top = []
        for name, val in pairs_sorted[:10]:
            top.append({
                'feature': name,
                'shap_value': float(val),
                'direction': 'positive' if val > 0 else 'negative'
            })

        return {
            'input_symptoms': input_data.symptoms,
            'top_features': top
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("ML_API_PORT", 8000))
    logger.info("Starting ML API on http://localhost:%d", port)
    logger.info("API Docs available at http://localhost:%d/docs", port)
    # Use import string so reload works correctly when requested
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
