#!/bin/bash
# Quick start script for ML API development

echo "🚀 Starting AI Health ML API..."
echo ""

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "✅ Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check if model file exists
if [ ! -f "disease_model.pkl" ]; then
    echo ""
    echo "⚠️  WARNING: disease_model.pkl not found!"
    echo "📁 Please place your trained model file (disease_model.pkl) in this directory"
    echo ""
fi

# Start the API
echo ""
echo "🌐 Starting FastAPI server..."
echo "📚 API Docs: http://localhost:8000/docs"
echo "🔗 Backend Root: http://localhost:8000"
echo ""
python app.py
