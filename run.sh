#!/bin/bash

echo "Starting Streamlit Article Generator..."
echo

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Run the application
echo
echo "Starting application..."
echo "Open your browser and go to: http://localhost:8501"
echo
streamlit run app.py
