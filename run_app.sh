#!/bin/bash

# Quick start script for DAE Question Similarity App

echo "=== DAE Question Similarity Finder ==="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements if needed
if ! python -c "import flask" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Run the app
echo ""
echo "Starting web application..."
echo "Open your browser to: http://localhost:5000"
echo ""
python app.py

