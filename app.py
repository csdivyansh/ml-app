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
@app.post("/predict", response_model=PredictionResponse)
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
                # if label encoder present, inverse transform
                if le is not None:
                    predicted_disease = str(le.inverse_transform([pred_idx])[0])
                else:
                    predicted_disease = str(pred_idx)

                confidence = None
                try:
                    if hasattr(clf, 'predict_proba'):
                        probs = clf.predict_proba(X)[0]
                        confidence = float(probs.max())
                except Exception:
                    pass

                return PredictionResponse(
                    predicted_disease=predicted_disease,
                    confidence=confidence,
                    symptoms_input=input_data.symptoms,
                    success=True,
                    message="Prediction successful"
                )
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


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("ML_API_PORT", 8000))
    logger.info("Starting ML API on http://localhost:%d", port)
    logger.info("API Docs available at http://localhost:%d/docs", port)
    # Use import string so reload works correctly when requested
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
