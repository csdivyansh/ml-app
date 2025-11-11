# ML API Setup Complete ✅

Your FastAPI ML service is now ready to integrate with your MERN stack!

## 📁 Files Created

### Python FastAPI Application

- ✅ `app.py` - Main FastAPI application with disease prediction endpoint
- ✅ `requirements.txt` - Python dependencies
- ✅ `.env` - Environment configuration
- ✅ `.gitignore` - Git ignore rules
- ✅ `start.bat` - Windows quick start script
- ✅ `start.sh` - macOS/Linux quick start script
- ✅ `README.md` - ML API documentation

### Node.js Backend Integration

- ✅ `routes/mlPredictionRoutes.js` - Express routes for ML API proxy
- ✅ Updated `server.js` - Added ML routes to the backend

### Documentation

- ✅ `INTEGRATION_GUIDE.md` - Complete integration guide
- ✅ `example-usage.tsx` - React component example (reference only)

## 🚀 Quick Start

### Step 1: Add Your Model

Place your trained model file in the ml-api directory:

```
ml-api/disease_model.pkl  ← Add your model here
```

### Step 2: Install & Run ML API

**Windows:**

```bash
cd ml-api
start.bat
```

**macOS/Linux:**

```bash
cd ml-api
bash start.sh
```

**Manual:**

```bash
cd ml-api
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python app.py
```

ML API will run at: **http://localhost:8000**

### Step 3: Start Node Backend

```bash
cd AI-Health-Backend
npm start
```

Backend will run at: **http://localhost:5000**

### Step 4: Start React Frontend

```bash
cd AI-Health-frontend
npm run dev
```

Frontend will run at: **http://localhost:5173**

## 📊 API Architecture

```
React Frontend (5173)
        ↓
Node Backend (5000)  ← Handles auth, proxies to ML API
        ↓
FastAPI ML API (8000) ← Runs predictions
```

## 🔌 Available Endpoints

### ML API (Python - Direct)

- `GET /health` - Health check
- `POST /predict` - Single disease prediction
- `POST /predict-batch` - Batch predictions
- `GET /docs` - Interactive API documentation

### Backend (Node.js - Use these from Frontend)

- `POST /api/ml/disease-predict` - Predict disease (requires auth)
- `POST /api/ml/batch-predict` - Batch predictions (requires auth)
- `GET /api/ml/health` - ML API status

## 📝 Example Usage (React)

```typescript
// Call from React component
const response = await fetch("http://localhost:5000/api/ml/disease-predict", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    Authorization: `Bearer ${token}`,
  },
  body: JSON.stringify({
    symptoms: ["fever", "cough", "headache"],
  }),
});

const result = await response.json();
console.log(result.data.predicted_disease);
```

## 🛠️ Configuration

### ML API (.env)

```
ML_API_PORT=8000
FRONTEND_URL=http://localhost:5173
DEBUG=True
```

### Node Backend (.env)

```
ML_API_URL=http://localhost:8000
```

## 📖 Documentation

See detailed guides:

- `ml-api/README.md` - ML API documentation
- `INTEGRATION_GUIDE.md` - Complete integration guide
- `ml-api/example-usage.tsx` - React component example

## ✨ Features

✅ Disease prediction from symptoms
✅ Confidence scores
✅ Batch prediction support
✅ CORS enabled for frontend
✅ Authentication integration
✅ Error handling
✅ Interactive API docs
✅ Health check endpoints
✅ Production-ready structure

## 🔍 Testing

### Test ML API directly

Visit: http://localhost:8000/docs

1. Click `/predict`
2. Click "Try it out"
3. Enter JSON:

```json
{
  "symptoms": ["fever", "cough"],
  "metadata": {}
}
```

4. Click "Execute"

### Test via Backend

```bash
curl -X POST http://localhost:5000/api/ml/disease-predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"symptoms": ["fever", "cough"]}'
```

## ⚠️ Troubleshooting

**Model not loading?**

- Ensure `disease_model.pkl` is in `ml-api/` folder
- Check file is not corrupted

**CORS errors?**

- Verify frontend URL in `.env`
- Check CORS settings in `app.py`

**Connection refused?**

- Make sure all services are running
- Check port numbers: API (8000), Backend (5000), Frontend (5173)

## 📦 Project Structure

```
Healwell/
├── ml-api/                          # Python FastAPI
│   ├── app.py                       # Main app
│   ├── requirements.txt             # Dependencies
│   ├── .env                         # Config
│   ├── disease_model.pkl            # Your model
│   ├── start.bat / start.sh         # Quick start
│   └── README.md
│
├── AI-Health-Backend/               # Node.js
│   ├── routes/
│   │   └── mlPredictionRoutes.js    # ML integration
│   ├── server.js                    # Updated with ML routes
│   └── .env
│
├── AI-Health-frontend/              # React
│   ├── src/
│   └── .env
│
└── INTEGRATION_GUIDE.md             # This guide
```

## 🎉 You're All Set!

Everything is configured. Just:

1. Add your `disease_model.pkl`
2. Run `start.bat` (or `bash start.sh`) in ml-api folder
3. Start your backend and frontend
4. Your disease prediction API is live!

For detailed integration info, see: `INTEGRATION_GUIDE.md`
