"""Quick test of RandomForest model predictions"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app import predict_disease, SymptomInput
import json
import asyncio

async def test_predict():
    """Test predictions"""
    test_cases = [
        {
            "name": "Cold/Flu-like",
            "symptoms": ["fever", "cough", "sore_throat"]
        },
        {
            "name": "Dengue",
            "symptoms": ["high_fever", "severe_headache", "body_aches", "joint_pain"]
        },
        {
            "name": "Fungal Infection",
            "symptoms": ["itching", "skin_rash", "redness"]
        },
        {
            "name": "Diabetes",
            "symptoms": ["increased_thirst", "increased_urination", "fatigue"]
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {test_case['name']}")
        print(f"Input symptoms: {test_case['symptoms']}")
        print('-'*60)
        
        try:
            input_data = SymptomInput(symptoms=test_case["symptoms"])
            result = await predict_disease(input_data)
            print(f"Predicted Disease: {result['predicted_disease']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Extracted Symptoms: {result['extracted_symptoms']}")
            print(f"Precautions: {result['precautions']}")
            if result.get('top_k'):
                print(f"\nTop 5 predictions:")
                for pred in result['top_k'][:5]:
                    print(f"  {pred['rank']}. {pred['disease']}: {pred['confidence']:.4f}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_predict())
