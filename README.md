# AI Health ML API

FastAPI-based machine learning API for disease prediction integrated with MERN stack.

## Setup

### Prerequisites

- Python 3.8+
- `disease_model.pkl` (your trained model file)

### Installation

1. Navigate to the ml-api directory:

```bash
cd ml-api
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place your `disease_model.pkl` file in the ml-api directory

### Running the API

```bash
python app.py
```

The API will start at `http://localhost:8000`

## API Endpoints

### 1. Health Check

```
GET /health
```

Returns the status of the API and model.

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "message": "ML API is ready"
}
```

### 2. Single Prediction

```
POST /predict
```

**Request Body:**

```json
{
  "symptoms": ["fever", "cough", "headache"],
  "metadata": {
    "age": 30,
    "gender": "male"
  }
}
```

**Response:**

```json
{
  "predicted_disease": "Common Cold",
  "confidence": 0.89,
  "symptoms_input": ["fever", "cough", "headache"],
  "success": true,
  "message": "Prediction successful"
}
```

### 3. Batch Prediction

```
POST /predict-batch
```

**Request Body:**

```json
[
  {
    "symptoms": ["fever", "cough"],
    "metadata": {}
  },
  {
    "symptoms": ["rash", "itching"],
    "metadata": {}
  }
]
```

**Response:**

```json
{
  "success": true,
  "count": 2,
  "predictions": [...]
}
```

## Integration with MERN Backend

Add this route to your Node.js backend:

```javascript
// mlPredictionRoutes.js
import express from "express";
import axios from "axios";

const router = express.Router();
const ML_API_URL = process.env.ML_API_URL || "http://localhost:8000";

router.post("/disease-predict", async (req, res) => {
  try {
    const { symptoms } = req.body;

    if (!symptoms || !Array.isArray(symptoms)) {
      return res.status(400).json({
        success: false,
        error: "Symptoms array required",
      });
    }

    const response = await axios.post(`${ML_API_URL}/predict`, {
      symptoms,
      metadata: req.body.metadata || {},
    });

    res.status(200).json({
      success: true,
      data: response.data,
    });
  } catch (error) {
    console.error("ML prediction error:", error.message);
    res.status(500).json({
      success: false,
      error: error.message,
    });
  }
});

export default router;
```

Add to `server.js`:

```javascript
import mlPredictionRoutes from "./routes/mlPredictionRoutes.js";
app.use("/api/ml", mlPredictionRoutes);
```

## Frontend Integration

Call the Node backend from your React frontend:

```typescript
// In your React component
const predictDisease = async (symptoms: string[]) => {
  try {
    const response = await fetch(
      "http://localhost:5000/api/ml/disease-predict",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ symptoms }),
      }
    );

    const result = await response.json();
    return result.data;
  } catch (error) {
    console.error("Prediction error:", error);
  }
};
```

## Interactive Documentation

Once the API is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Configuration

Edit `.env` file:

```
ML_API_PORT=8000
FRONTEND_URL=http://localhost:5173
NODE_BACKEND_URL=http://localhost:5000
DEBUG=True
```

## Troubleshooting

### Model not found

- Ensure `disease_model.pkl` is in the ml-api directory
- Check the file path in the error message

### CORS errors

- Verify your frontend URL is in the `allow_origins` list in `app.py`
- Add your production domain when deploying

### Import errors

- Run `pip install -r requirements.txt` again
- Ensure you're using Python 3.8+

## File Structure

```
ml-api/
├── app.py                 # Main FastAPI application
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables
├── disease_model.pkl      # Your trained model (add this)
└── README.md             # This file
```

## License

AI Health & Wellness Analyzer
