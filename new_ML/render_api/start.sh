#!/bin/bash

echo "=== Moisture Meter API Startup ==="
echo "Current directory: $(pwd)"
echo "Files in current directory:"
ls -la

if [ -d "models" ]; then
    echo "Models directory found:"
    ls -la models/
else
    echo "Models directory not found"
fi

echo "=== Testing model loading ==="
python test_model_loading.py

echo "=== Starting API server ==="
exec gunicorn app:app --bind 0.0.0.0:8080 --workers 1 --timeout 120 