# ML API Integration Guide

This guide shows how to integrate the FastAPI ML service with your MERN stack application.

## Architecture Overview

```
┌─────────────────────┐
│  React Frontend     │
│  (port 5173)        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Node.js Backend    │  ◄── Proxy requests
│  (port 5000)        │
│  /api/ml/*          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  FastAPI ML API     │
│  (port 8000)        │
│  /predict           │
└─────────────────────┘
```

## Setup Instructions

### 1. ML API Setup (Python FastAPI)

#### Prerequisites

- Python 3.8+
- Your trained model file: `disease_model.pkl`

#### Installation

```bash
cd ml-api

# Windows
start.bat

# macOS/Linux
bash start.sh
```

Or manually:

```bash
# Create virtual environment
python -m venv venv

# Activate
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run
python app.py
```

The API will be available at: `http://localhost:8000`

**Important**: Place your `disease_model.pkl` file in the `ml-api` directory before running.

### 2. Backend Node.js Integration

The ML prediction routes are already integrated in:

- Route file: `AI-Health-Backend/routes/mlPredictionRoutes.js`
- Added to: `AI-Health-Backend/server.js`

No additional setup needed! The backend will automatically proxy requests to the ML API.

### 3. Environment Variables

#### ml-api/.env

```
ML_API_PORT=8000
FRONTEND_URL=http://localhost:5173
NODE_BACKEND_URL=http://localhost:5000
DEBUG=True
```

#### AI-Health-Backend/.env

```
ML_API_URL=http://localhost:8000
```

## API Usage

### From React Frontend

```typescript
// 1. Call your Node backend (recommended for production)
const predictDisease = async (symptoms: string[]) => {
  try {
    const response = await fetch(
      "http://localhost:5000/api/ml/disease-predict",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`, // If using authentication
        },
        body: JSON.stringify({ symptoms }),
      }
    );

    const result = await response.json();
    if (result.success) {
      return result.data.predicted_disease;
    }
  } catch (error) {
    console.error("Prediction error:", error);
  }
};

// Usage
const disease = await predictDisease(["fever", "cough", "headache"]);
console.log("Predicted disease:", disease);
```

### Direct calls to ML API (Development only)

```typescript
const predictDirect = async (symptoms: string[]) => {
  try {
    const response = await fetch("http://localhost:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        symptoms,
        metadata: { age: 30, gender: "male" },
      }),
    });

    const result = await response.json();
    return result;
  } catch (error) {
    console.error("Error:", error);
  }
};
```

## Backend Routes

### 1. Disease Prediction

```
POST /api/ml/disease-predict
Authentication: Required (token-based)

Request:
{
  "symptoms": ["fever", "cough", "headache"],
  "metadata": { "age": 30 }
}

Response:
{
  "success": true,
  "data": {
    "predicted_disease": "Common Cold",
    "confidence": 0.89,
    "symptoms_input": ["fever", "cough", "headache"],
    "success": true,
    "message": "Prediction successful"
  }
}
```

### 2. Batch Prediction

```
POST /api/ml/batch-predict
Authentication: Required

Request:
{
  "predictions": [
    { "symptoms": ["fever", "cough"], "metadata": {} },
    { "symptoms": ["rash", "itching"], "metadata": {} }
  ]
}
```

### 3. Health Check

```
GET /api/ml/health
Response: ML API status
```

## Troubleshooting

### Issue: "Model not loaded"

```
Solution:
1. Ensure disease_model.pkl is in ml-api/ directory
2. Check file is not corrupted
3. Restart the ML API
```

### Issue: CORS Error

```
Solution:
1. Check ML_API_PORT in .env
2. Verify FRONTEND_URL matches your frontend URL
3. Check Node backend CORS settings
```

### Issue: Connection Refused

```
Solution:
1. Ensure ML API is running: http://localhost:8000
2. Check Node backend is running: http://localhost:5000
3. Check port numbers in .env files
```

### Issue: Prediction Returns Empty

```
Solution:
1. Verify symptoms array is not empty
2. Check model's feature requirements
3. Review app.py prepare_features() function
```

## Development Workflow

### Terminal 1: Start ML API

```bash
cd ml-api
python app.py
# http://localhost:8000/docs
```

### Terminal 2: Start Node Backend

```bash
cd AI-Health-Backend
npm start
# http://localhost:5000
```

### Terminal 3: Start React Frontend

```bash
cd AI-Health-frontend
npm run dev
# http://localhost:5173
```

## API Documentation

### ML API Interactive Docs

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Test Prediction

Visit: http://localhost:8000/docs

1. Click on `/predict`
2. Click "Try it out"
3. Enter symptoms:

```json
{
  "symptoms": ["fever", "cough"],
  "metadata": {}
}
```

4. Click "Execute"

## Performance Tips

1. **Caching**: Add Redis caching layer for common symptoms
2. **Batching**: Use `/predict-batch` for multiple predictions
3. **Async**: Node backend uses async/await for non-blocking calls
4. **Monitoring**: Check `/health` endpoints regularly

## Production Deployment

### Environment Variables

```
# .env for production
ML_API_PORT=8000
FRONTEND_URL=https://yourdomain.com
NODE_BACKEND_URL=https://api.yourdomain.com
DEBUG=False
```

### ML API Deployment (Python)

- Use `gunicorn` or `Uvicorn` with production server
- Add proper error logging
- Use environment variables for secrets
- Run behind reverse proxy (Nginx, Apache)

### Node Backend Deployment

- Add `ML_API_URL` pointing to production ML API
- Enable request logging
- Set up monitoring

## File Structure

```
Your Project/
├── ml-api/                           # Python FastAPI
│   ├── app.py                        # Main application
│   ├── requirements.txt              # Python dependencies
│   ├── disease_model.pkl             # Your model (add this)
│   ├── .env                          # Environment config
│   ├── start.sh / start.bat          # Quick start scripts
│   └── README.md
│
├── AI-Health-Backend/                # Node.js Express
│   ├── routes/
│   │   └── mlPredictionRoutes.js      # ML integration routes
│   ├── server.js                     # Already configured
│   └── .env                          # Backend config
│
└── AI-Health-frontend/               # React
    ├── src/
    │   ├── pages/
    │   ├── components/
    │   └── services/
    │       └── api.ts                # API calls
    └── .env
```

## Support

For issues or questions:

1. Check troubleshooting section
2. Review API documentation: http://localhost:8000/docs
3. Check Node backend logs
4. Verify all services are running
