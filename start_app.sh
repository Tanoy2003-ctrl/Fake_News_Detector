#!/bin/bash

# Fake News Detector - Streamlit App Launcher
echo "🔍 Fake News Detector - Starting Application..."
echo "================================================"

# Check if we're in the right directory
if [ ! -f "streamlit_app.py" ]; then
    echo "❌ Error: streamlit_app.py not found in current directory"
    echo "💡 Please run this script from the project directory"
    exit 1
fi

# Check if virtual environment exists
if [ -d "studysession" ]; then
    echo "🐍 Activating virtual environment..."
    source studysession/bin/activate
fi

# Install requirements if needed
if [ -f "requirements.txt" ]; then
    echo "📦 Checking requirements..."
    pip install -r requirements.txt > /dev/null 2>&1
fi

echo "🚀 Starting Streamlit app..."
echo "🌐 App will be available at: http://localhost:8501"
echo "⏹️  Press Ctrl+C to stop the server"
echo "================================================"

# Start the app
streamlit run streamlit_app.py --server.headless true --server.port 8501