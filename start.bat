@echo off
REM Quick start script for ML API development on Windows

echo Starting AI Health ML API...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Check if model file exists
if not exist "disease_model.pkl" (
    echo.
    echo WARNING: disease_model.pkl not found!
    echo Please place your trained model file ^(disease_model.pkl^) in this directory
    echo.
)

REM Start the API
echo.
echo Starting FastAPI server...
echo API Docs: http://localhost:8000/docs
echo Backend Root: http://localhost:8000
echo.
REM Start with uvicorn using import string so --reload works correctly
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
