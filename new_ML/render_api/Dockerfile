FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p models

# Copy Phase 2 scikit-learn models and scalers from parent directory
COPY ../models/groundnut_best_model.pkl models/
COPY ../models/mustard_best_model.pkl models/
COPY ../models/groundnut_scaler.pkl models/
COPY ../models/mustard_scaler.pkl models/

# Expose port
EXPOSE 5000

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"] 