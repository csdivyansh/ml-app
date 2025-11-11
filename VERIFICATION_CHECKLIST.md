# ML API Setup Verification Checklist ✅

Complete this checklist to ensure everything is configured correctly.

## Pre-Setup
- [ ] Python 3.8+ installed (`python --version`)
- [ ] You have your trained `disease_model.pkl` file ready
- [ ] Node.js and npm installed for backend/frontend

## Step 1: ML API Folder (/ml-api)
- [ ] Folder created at: `Healwell/ml-api`
- [ ] Files present:
  - [ ] `app.py` (main FastAPI application)
  - [ ] `requirements.txt` (Python dependencies)
  - [ ] `.env` (environment variables)
  - [ ] `.gitignore` (git ignore rules)
  - [ ] `start.bat` (Windows quick start)
  - [ ] `start.sh` (macOS/Linux quick start)
  - [ ] `README.md` (documentation)
  - [ ] `config.json` (configuration reference)

## Step 2: Model File
- [ ] `disease_model.pkl` placed in `ml-api/` folder
- [ ] File is not corrupted (check file size > 0 MB)
- [ ] File is readable and accessible

## Step 3: Python Environment
- [ ] Virtual environment created: `ml-api/venv`
- [ ] All dependencies installed from `requirements.txt`:
  - [ ] fastapi
  - [ ] uvicorn
  - [ ] pydantic
  - [ ] joblib
  - [ ] pandas
  - [ ] numpy

## Step 4: Node Backend Integration
- [ ] File created: `AI-Health-Backend/routes/mlPredictionRoutes.js`
- [ ] `server.js` updated with:
  - [ ] Import: `import mlPredictionRoutes from "./routes/mlPredictionRoutes.js";`
  - [ ] Route: `app.use("/api/ml", mlPredictionRoutes);`
- [ ] Backend `.env` includes: `ML_API_URL=http://localhost:8000`

## Step 5: Configuration
- [ ] ML API `.env` configured:
  - [ ] `ML_API_PORT=8000`
  - [ ] `FRONTEND_URL=http://localhost:5173`
  - [ ] `DEBUG=True`
- [ ] Backend `.env` configured:
  - [ ] `ML_API_URL=http://localhost:8000`
- [ ] All ports are available (8000, 5000, 5173)

## Step 6: Services Running (In Different Terminals)

### Terminal 1: ML API
```bash
cd ml-api
# Windows: start.bat
# macOS/Linux: bash start.sh
```
- [ ] API starts without errors
- [ ] Message: "🚀 Starting ML API on http://localhost:8000"
- [ ] Message: "✅ Disease model loaded successfully" OR "⚠️ Model file not found"
- [ ] API accessible at: http://localhost:8000

### Terminal 2: Node Backend
```bash
cd AI-Health-Backend
npm start
```
- [ ] Backend starts without errors
- [ ] Message: "🚀 Server running on port 5000"
- [ ] Backend accessible at: http://localhost:5000

### Terminal 3: React Frontend
```bash
cd AI-Health-frontend
npm run dev
```
- [ ] Frontend builds and starts
- [ ] Frontend accessible at: http://localhost:5173

## Step 7: API Testing

### Test ML API Directly
- [ ] Visit: http://localhost:8000/health
  - [ ] Response shows: `"model_loaded": true` (if model.pkl exists)
- [ ] Visit: http://localhost:8000/docs
  - [ ] Interactive documentation loads
  - [ ] Can see `/predict` endpoint

### Test via Backend Proxy
- [ ] Backend health check: GET `/api/ml/health`
- [ ] Make prediction:
  ```bash
  curl -X POST http://localhost:5000/api/ml/disease-predict \
    -H "Content-Type: application/json" \
    -d '{"symptoms": ["fever", "cough"]}'
  ```
  - [ ] Returns prediction response

### Test via Frontend
- [ ] Frontend loads at http://localhost:5173
- [ ] Can make prediction request (if component implemented)
- [ ] No CORS errors in browser console

## Step 8: Documentation
- [ ] Read `ml-api/README.md` for API details
- [ ] Read `INTEGRATION_GUIDE.md` for full integration instructions
- [ ] Reviewed `config.json` for reference architecture
- [ ] Checked `ml-api/SETUP_COMPLETE.md` for summary

## Troubleshooting

### If ML API Won't Start:
- [ ] Check Python version: `python --version` (needs 3.8+)
- [ ] Reinstall dependencies: `pip install -r requirements.txt`
- [ ] Check for port conflicts: `netstat -ano | findstr 8000` (Windows)

### If Model Won't Load:
- [ ] Verify `disease_model.pkl` exists in `ml-api/` folder
- [ ] Check file isn't corrupted: `ls -lh ml-api/disease_model.pkl`
- [ ] Ensure joblib can load it: `python -c "import joblib; joblib.load('disease_model.pkl')"`

### If CORS Errors Occur:
- [ ] Check frontend URL in `ml-api/.env`
- [ ] Verify backend is receiving requests correctly
- [ ] Check logs in both frontend and backend consoles

### If No Connection to ML API:
- [ ] Ensure ML API is running on port 8000
- [ ] Check firewall settings
- [ ] Verify `ML_API_URL` in backend `.env`

## File Structure Verification

```
Healwell/
├── ml-api/                      ✓ Created
│   ├── app.py                   ✓ Created
│   ├── requirements.txt         ✓ Created
│   ├── .env                     ✓ Created
│   ├── .gitignore              ✓ Created
│   ├── start.bat               ✓ Created
│   ├── start.sh                ✓ Created
│   ├── README.md               ✓ Created
│   ├── SETUP_COMPLETE.md       ✓ Created
│   ├── config.json             ✓ Created
│   ├── disease_model.pkl       ⚠ Add this!
│   └── venv/                   ⚠ Create when running
│
├── AI-Health-Backend/
│   ├── routes/
│   │   ├── mlPredictionRoutes.js    ✓ Created
│   │   └── [other routes]           ✓ Existing
│   ├── server.js               ✓ Updated
│   └── .env                    ✓ Update this
│
├── AI-Health-frontend/         ✓ Existing
│   └── .env                    ✓ Existing
│
└── INTEGRATION_GUIDE.md        ✓ Created
```

## Success Criteria ✅

All of the following must be true:

- [ ] ML API runs without errors
- [ ] Model loads successfully: "✅ Disease model loaded successfully"
- [ ] Backend receives ML routes
- [ ] All three services run on correct ports (8000, 5000, 5173)
- [ ] API endpoints respond correctly
- [ ] No CORS errors in frontend
- [ ] Can make successful predictions
- [ ] Documentation is complete

## Next Steps

Once all checks pass:

1. **Integrate into your React components** - Use the example in `example-usage.tsx`
2. **Test with real symptom data** - Use your app's symptom input features
3. **Monitor performance** - Track prediction times and accuracy
4. **Set up logging** - Add monitoring for production
5. **Deploy** - Follow production deployment guidelines in INTEGRATION_GUIDE.md

## Contact & Support

For issues:
1. Check troubleshooting section above
2. Review error messages in console/logs
3. Visit http://localhost:8000/docs for API documentation
4. Check INTEGRATION_GUIDE.md for detailed help

---

**Setup Date:** 2025-11-11
**Version:** 1.0.0
**Status:** Ready to use! 🎉
