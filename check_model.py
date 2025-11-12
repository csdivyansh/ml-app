import joblib
import os

model_path = 'disease_model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print('Model loaded:', model is not None)
    print('Model type:', type(model))
    if isinstance(model, dict):
        print('Keys:', list(model.keys()))
        for key in model.keys():
            print(f'  {key}: {type(model[key]).__name__}')
    else:
        print('Model is not a dictionary!')
else:
    print('Model file not found')
