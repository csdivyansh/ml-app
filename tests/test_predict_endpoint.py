from fastapi.testclient import TestClient
import importlib.util
import os
import sys

# Import the app module from app.py using a path that's correct regardless of cwd
here = os.path.dirname(__file__)
app_path = os.path.abspath(os.path.join(here, '..', 'app.py'))
spec = importlib.util.spec_from_file_location("ml_api_app", app_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
app = mod.app

client = TestClient(app)


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "model_loaded" in body


def test_predict_endpoint():
    payload = {"symptoms": ["itching", "skin_rash"]}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body.get("success") is True
    assert "predicted_disease" in body
