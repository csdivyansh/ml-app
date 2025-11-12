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
import logging

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("ml_api")

# Initialize FastAPI app
app = FastAPI(title="AI Health ML API", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:5000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained RandomForest model
model = None
model_loaded = False
rf_model = None
le = None
all_symptoms = []
prec_dict = {}

try:
    model_path = os.path.join(os.path.dirname(__file__), "disease_model.pkl")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        # Extract RandomForest artifact components
        if isinstance(model, dict):
            rf_model = model.get('model')
            le = model.get('le')
            all_symptoms = model.get('all_symptoms', [])
            prec_dict = model.get('prec_dict', {})
            model_loaded = rf_model is not None and le is not None
        logger.info("Disease model loaded successfully")
        logger.info(f"Total symptoms in vocabulary: {len(all_symptoms)}")
        logger.info(f"Total diseases: {len(le.classes_) if le else 0}")
        logger.info(f"Precautions loaded for {len(prec_dict)} diseases")
    else:
        logger.warning("Model file not found at %s", model_path)
except Exception as e:
    logger.exception("Error loading model: %s", e)
    model = None
    rf_model = le = None
    all_symptoms = []
    prec_dict = {}


# High-risk diseases requiring immediate emergency attention
HIGH_RISK_DISEASES = {
    "heart attack": "critical",
    "stroke": "critical",
    "myocardial infarction": "critical",
    "severe asthma": "critical",
    "pneumonia": "high",
    "sepsis": "critical",
    "pulmonary embolism": "critical",
    "meningitis": "critical",
    "seizure": "critical",
    "anaphylaxis": "critical",
    "paralysis (brain hemorrhage)": "high",
    "dengue": "high",
    "typhoid fever": "high",
    "aids": "high",
    "tuberculosis": "high",
    "hepatitis b": "high",
    "hepatitis c": "high",
    "malaria": "high"
}


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
    extracted_symptoms: List[str] = []
    precautions: List[str] = []
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


def normalize_token(tok: str) -> str:
    """Normalize symptom token to match training normalization"""
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


def extract_symptoms_fuzzy(user_symptoms: List[str], threshold: float = 0.8) -> List[str]:
    """
    Extract and match user symptoms to canonical vocabulary using fuzzy matching.
    Returns list of matched canonical symptoms.
    """
    try:
        from rapidfuzz import process, fuzz
    except ImportError:
        logger.warning("rapidfuzz not installed; using exact matching")
        process = None
    
    matched = []
    for user_symptom in user_symptoms:
        user_symptom = normalize_token(user_symptom)
        if not user_symptom:
            continue
        
        if process is not None:
            # Fuzzy match against all_symptoms vocabulary
            result = process.extractOne(
                user_symptom,
                all_symptoms,
                scorer=fuzz.partial_ratio
            )
            if result and result[1] >= threshold * 100:  # threshold is percentage
                matched.append(result[0])
            else:
                # Fall back to exact match
                if user_symptom in all_symptoms:
                    matched.append(user_symptom)
        else:
            # Exact match only
            if user_symptom in all_symptoms:
                matched.append(user_symptom)
    
    return list(set(matched))  # Remove duplicates


def encode_symptoms_binary(matched_symptoms: List[str]) -> np.ndarray:
    """Convert matched symptoms list to binary feature vector"""
    binary_vector = [1 if s in matched_symptoms else 0 for s in all_symptoms]
    return np.array(binary_vector).reshape(1, -1)


def get_urgency_flag(predicted_disease: str, confidence: float) -> Dict[str, Any]:
    """
    Determine urgency level based on disease type and confidence.
    Prioritizes high-risk diseases over confidence scores.
    
    Args:
        predicted_disease: The predicted disease name
        confidence: Confidence score from model (0-1)
        
    Returns:
        Dictionary with urgency level, score, reason, and recommended action
    """
    disease_lower = predicted_disease.lower().strip()
    
    # Check if it's a high-risk disease
    if disease_lower in HIGH_RISK_DISEASES:
        risk_level = HIGH_RISK_DISEASES[disease_lower]
        return {
            "level": risk_level,
            "score": round(float(confidence), 3),
            "reason": f"High-risk condition detected: {predicted_disease}",
            "recommended_action": "Seek emergency medical care immediately" if risk_level == "critical" else "Consult a clinician promptly"
        }
    
    # Otherwise use confidence-based urgency
    if confidence > 0.7:
        return {
            "level": "medium",
            "score": round(float(confidence), 3),
            "reason": "Moderate confidence in prediction; clinical review advised",
            "recommended_action": "Consult a clinician"
        }
    
    return {
        "level": "low",
        "score": round(float(confidence), 3),
        "reason": "Low confidence prediction; insufficient symptoms",
        "recommended_action": None
    }


# Main prediction endpoint
@app.post("/predict")
async def predict_disease(input_data: SymptomInput):
    """
    Predict disease based on symptoms using RandomForest model
    
    Args:
        input_data: SymptomInput with symptoms list
        
    Returns:
        PredictionResponse with predicted disease, confidence, extracted symptoms, and precautions
    """
    try:
        if not model_loaded or rf_model is None or le is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please ensure disease_model.pkl exists."
            )
        
        if not input_data.symptoms or len(input_data.symptoms) == 0:
            raise HTTPException(
                status_code=400,
                detail="No symptoms provided. Please provide at least one symptom."
            )
        
        # Extract and match symptoms using fuzzy matching
        extracted_symptoms = extract_symptoms_fuzzy(input_data.symptoms, threshold=0.8)
        
        if not extracted_symptoms:
            logger.warning("No symptoms matched from input: %s", input_data.symptoms)
            raise HTTPException(
                status_code=400,
                detail=f"Could not match any symptoms. Input: {input_data.symptoms}. Available symptoms: {all_symptoms[:10]}..."
            )
        
        # Encode matched symptoms to binary vector
        X = encode_symptoms_binary(extracted_symptoms)
        
        # Predict disease
        pred_idx = rf_model.predict(X)[0]
        predicted_disease = le.inverse_transform([pred_idx])[0]
        
        # Get confidence from predict_proba
        confidence = None
        top_k_list = []
        try:
            if hasattr(rf_model, 'predict_proba'):
                probs = rf_model.predict_proba(X)[0]
                confidence = float(np.max(probs))
                
                # Get top-5 predictions
                idxs = list(reversed(np.argsort(probs)))
                for rank, i in enumerate(idxs[:5], start=1):
                    disease_label = le.inverse_transform([i])[0]
                    score = float(probs[i])
                    top_k_list.append({
                        "rank": rank,
                        "disease": disease_label,
                        "confidence": round(score, 3)
                    })
        except Exception as e:
            logger.warning("Error computing probabilities: %s", e)
        
        # Look up precautions
        precautions = prec_dict.get(predicted_disease.lower(), [])
        
        # Get urgency flag based on disease type and confidence
        urgency_flag = get_urgency_flag(predicted_disease, confidence if confidence else 0.0)
        
        result = {
            "predicted_disease": predicted_disease,
            "confidence": confidence,
            "symptoms_input": input_data.symptoms,
            "extracted_symptoms": extracted_symptoms,
            "precautions": precautions,
            "success": True,
            "message": "Prediction successful",
            "top_k": top_k_list,
            "urgency_flag": urgency_flag,
            "model_version": "RandomForest + Binary Encoding v2"
        }
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction error: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


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
        if not model_loaded or rf_model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
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
        raise HTTPException(status_code=500, detail=str(e))


# Feature importance endpoint (RandomForest-specific)
@app.post("/explain")
async def explain(input_data: SymptomInput):
    """
    Return top contributing symptoms for prediction based on RandomForest feature importance
    """
    try:
        if not model_loaded or rf_model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Extract and match symptoms
        extracted_symptoms = extract_symptoms_fuzzy(input_data.symptoms, threshold=0.8)
        
        if not extracted_symptoms:
            raise HTTPException(status_code=400, detail="Could not match any symptoms")
        
        # Get feature importances from RandomForest
        if hasattr(rf_model, 'feature_importances_'):
            importances = rf_model.feature_importances_
            feature_names = all_symptoms
            
            # Create pairs and sort by importance
            pairs = list(zip(feature_names, importances))
            pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
            
            # Return top contributing symptoms
            top_features = []
            for name, importance in pairs_sorted[:10]:
                top_features.append({
                    'symptom': name,
                    'importance': float(importance),
                    'matched': name in extracted_symptoms
                })
            
            return {
                'input_symptoms': input_data.symptoms,
                'extracted_symptoms': extracted_symptoms,
                'top_contributing_features': top_features,
                'explanation': 'Top symptoms by feature importance in the RandomForest model'
            }
        else:
            raise HTTPException(status_code=501, detail="Feature importance not available for this model")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Explain error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API documentation"""
    return {
        "message": "AI Health ML API v2.0 (RandomForest + Precautions)",
        "version": "2.0.0",
        "model_loaded": model_loaded,
        "endpoints": {
            "health": "/health (GET)",
            "predict": "/predict (POST)",
            "predict_batch": "/predict-batch (POST)",
            "explain": "/explain (POST)",
            "docs": "/docs (interactive documentation)"
        },
        "model_info": {
            "type": "RandomForest",
            "symptoms_vocabulary": len(all_symptoms),
            "diseases": len(le.classes_) if le else 0,
            "precautions_mapped": len(prec_dict)
        }
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Run on app startup"""
    logger.info("AI Health ML API v2.0 starting")
    if model_loaded:
        logger.info("RandomForest model loaded and ready")
    else:
        logger.warning("Model not loaded - predictions will fail")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Run on app shutdown"""
    logger.info("AI Health ML API shutting down")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("ML_API_PORT", 8000))
    logger.info("Starting ML API on http://localhost:%d", port)
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
