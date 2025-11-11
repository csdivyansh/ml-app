// Example React Component using the ML API
// Place this in your AI-Health-frontend/src/services/

import axios from "axios";

const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";

/**
 * Service for ML predictions
 */
export const mlPredictionService = {
  /**
   * Predict disease from symptoms
   * @param {Array<string>} symptoms - List of symptoms
   * @param {Object} metadata - Optional metadata (age, gender, etc)
   * @returns {Promise} Prediction result
   */
  predictDisease: async (symptoms, metadata = {}) => {
    try {
      const response = await axios.post(
        `${API_BASE_URL}/api/ml/disease-predict`,
        { symptoms, metadata },
        {
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${localStorage.getItem("token")}`,
          },
        }
      );
      return response.data;
    } catch (error) {
      console.error("Disease prediction error:", error);
      throw error;
    }
  },

  /**
   * Batch predict diseases
   * @param {Array} predictions - Array of {symptoms, metadata}
   * @returns {Promise} Batch predictions
   */
  batchPredict: async (predictions) => {
    try {
      const response = await axios.post(
        `${API_BASE_URL}/api/ml/batch-predict`,
        { predictions },
        {
          headers: {
            Authorization: `Bearer ${localStorage.getItem("token")}`,
          },
        }
      );
      return response.data;
    } catch (error) {
      console.error("Batch prediction error:", error);
      throw error;
    }
  },

  /**
   * Check ML API health
   * @returns {Promise} API status
   */
  checkHealth: async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/ml/health`, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem("token")}`,
        },
      });
      return response.data;
    } catch (error) {
      console.error("Health check error:", error);
      return { success: false, error: error.message };
    }
  },
};

/**
 * Example React Component using predictions
 */
import React, { useState } from "react";
import { toast } from "sonner";

export function DiseasePredictor() {
  const [symptoms, setSymptoms] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAddSymptom = (symptom) => {
    if (!symptoms.includes(symptom)) {
      setSymptoms([...symptoms, symptom]);
    }
  };

  const handleRemoveSymptom = (symptom) => {
    setSymptoms(symptoms.filter((s) => s !== symptom));
  };

  const handlePredict = async () => {
    if (symptoms.length === 0) {
      toast.error("Please select at least one symptom");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const result = await mlPredictionService.predictDisease(symptoms);

      if (result.success) {
        setPrediction(result.data);
        toast.success("Disease predicted successfully!");
      } else {
        setError(result.error || "Prediction failed");
        toast.error(result.error || "Prediction failed");
      }
    } catch (err) {
      setError(err.message);
      toast.error("Error predicting disease: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-4">Disease Predictor</h2>

      {/* Symptoms Input */}
      <div className="mb-6">
        <label className="block text-sm font-medium mb-2">
          Select Symptoms:
        </label>
        <div className="space-y-2">
          {["fever", "cough", "headache", "fatigue", "rash"].map((symptom) => (
            <button
              key={symptom}
              onClick={() => handleAddSymptom(symptom)}
              className={`px-4 py-2 rounded mr-2 ${
                symptoms.includes(symptom)
                  ? "bg-emerald-500 text-white"
                  : "bg-gray-200 dark:bg-gray-700"
              }`}
            >
              {symptom}
            </button>
          ))}
        </div>
      </div>

      {/* Selected Symptoms */}
      {symptoms.length > 0 && (
        <div className="mb-6">
          <p className="text-sm font-medium mb-2">
            Selected: {symptoms.join(", ")}
          </p>
          <button
            onClick={() => setSymptoms([])}
            className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
          >
            Clear All
          </button>
        </div>
      )}

      {/* Predict Button */}
      <button
        onClick={handlePredict}
        disabled={loading || symptoms.length === 0}
        className="w-full px-6 py-3 bg-emerald-500 text-white rounded-lg font-semibold hover:bg-emerald-600 disabled:opacity-50"
      >
        {loading ? "Predicting..." : "Predict Disease"}
      </button>

      {/* Error Message */}
      {error && (
        <div className="mt-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded">
          {error}
        </div>
      )}

      {/* Prediction Result */}
      {prediction && (
        <div className="mt-6 p-6 bg-emerald-50 dark:bg-emerald-900/20 rounded-lg border border-emerald-200 dark:border-emerald-700">
          <h3 className="text-lg font-semibold mb-2">Prediction Result:</h3>
          <p className="text-2xl font-bold text-emerald-600 dark:text-emerald-400 mb-2">
            {prediction.predicted_disease}
          </p>
          {prediction.confidence && (
            <p className="text-sm text-gray-600 dark:text-gray-300">
              Confidence: {(prediction.confidence * 100).toFixed(2)}%
            </p>
          )}
          <p className="text-xs text-gray-500 mt-2">
            Symptoms analyzed: {prediction.symptoms_input.join(", ")}
          </p>
        </div>
      )}
    </div>
  );
}

export default DiseasePredictor;
