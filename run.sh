#!/bin/bash

# Check if python3-venv is installed
dpkg -s python3-venv &> /dev/null

if [ $? -ne 0 ]; then
    echo "python3-venv is not installed."
    echo "Attempting to create venv using local virtualenv if available..."
    
    # Try to install virtualenv locally if not present
    pip install --user virtualenv
    ~/.local/bin/virtualenv venv
else
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Running application..."
python -m uvicorn main:app --reload
