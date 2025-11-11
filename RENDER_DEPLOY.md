# RENDER DEPLOYMENT GUIDE for AI Health ML API

## Prerequisites

1. **GitHub Repository**: Push your code to GitHub (AI-Health-Backend repo, ml-api folder).
2. **Render Account**: Sign up at https://render.com (free tier available).
3. **Model & Data Files**: Ensure `disease_model.pkl`, `cleaned_disease_symptoms.csv`, and `symptom_map.json` are committed to the repository.

## Deployment Steps

### Step 1: Connect GitHub to Render

1. Go to https://dashboard.render.com
2. Click "New +" → "Web Service"
3. Select "Deploy an existing Git repository"
4. Connect your GitHub account and authorize Render
5. Select the `AI-Health-Backend` repository

### Step 2: Configure the Service

- **Name**: `ai-health-ml-api` (or your preferred name)
- **Root Directory**: `ml-api`
- **Runtime**: `Docker`
- **Branch**: `Divyansh` (or your deployment branch)
- **Build Command**: Leave empty (Dockerfile handles it)
- **Start Command**: Leave empty (Dockerfile CMD runs)

### Step 3: Environment Variables (Optional)

Add these in Render dashboard under "Environment":

```
PYTHONUNBUFFERED=1
LOG_LEVEL=info
ML_API_PORT=8000
```

### Step 4: Deploy

1. Click "Create Web Service"
2. Render will pull your repo, build the Docker image, and deploy
3. View logs in the "Logs" tab
4. Once deployed, you'll get a public URL like: `https://ai-health-ml-api.onrender.com`

### Step 5: Test the Deployment

Replace `YOUR_RENDER_URL` with your actual Render URL:

```bash
# Health check
curl https://YOUR_RENDER_URL/health

# Predict endpoint
curl -X POST https://YOUR_RENDER_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"symptoms": ["itching", "skin_rash"]}'

# Docs (interactive)
https://YOUR_RENDER_URL/docs
```

## Troubleshooting

### Build Fails

- Check "Build" tab logs for errors
- Ensure all required files (model, CSV, etc.) are committed
- Verify Dockerfile path is correct

### Service Won't Start

- Check "Logs" tab for startup errors
- Ensure port is set to 8000
- Verify `disease_model.pkl` exists and is readable

### Slow Response Time

- Consider upgrading from "starter" to "standard" instance
- Add more gunicorn workers in Dockerfile if needed

### Model File Not Found

- Ensure `disease_model.pkl` is in the `ml-api/` folder
- Commit and push to GitHub before deploying

## Optional: Set Up Auto-Deploy on Push

1. In Render dashboard, go to your service settings
2. Under "Auto-Deploy", select your branch (e.g., `Divyansh`)
3. Any push to that branch will trigger a redeploy

## Production Considerations

- **Scaling**: Start with "starter" instance; upgrade to "standard" or "pro" for higher traffic
- **Cost**: Starter instances may have cold starts (5-10s delay); upgrade for always-on
- **Monitoring**: Use Render's built-in metrics and logs
- **CORS**: Update `app.py` CORS origins to include your Render URL
- **Secrets**: Use Render environment variables for API keys, model paths, etc.

## Local Docker Testing

Before deploying to Render, test locally:

```bash
# Build image
docker build -t ai-health-ml-api .

# Run container
docker run -p 8000:8000 ai-health-ml-api

# Test
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"symptoms": ["itching"]}'
```

Or use docker-compose:

```bash
docker-compose up --build
```

## Integration with Node Backend

Update your Node backend's ML API proxy to use the Render URL:

```javascript
const ML_API_BASE = "https://YOUR_RENDER_URL";
// e.g., https://ai-health-ml-api.onrender.com

fetch(`${ML_API_BASE}/predict`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ symptoms: ["itching", "skin_rash"] }),
})
  .then((r) => r.json())
  .then((data) => console.log(data))
  .catch((err) => console.error(err));
```

## Support

- Render Docs: https://docs.render.com
- FastAPI Docs: https://fastapi.tiangolo.com
- Docker Docs: https://docs.docker.com
