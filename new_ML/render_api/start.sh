#!/bin/bash
# Start script for Render deployment

# Activate virtual environment if needed
# source fvenv/bin/activate

# Install dependencies (if not already installed)
pip install -r requirements.txt

# Run database migrations if needed
# python migrate.py

# Remove test_models.py run (was here)

# Start the API
python app.py 