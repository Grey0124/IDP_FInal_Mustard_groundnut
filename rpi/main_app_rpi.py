#!/usr/bin/env python3
"""
Main Flask Application for Raspberry Pi
Handles sensor readings and moisture predictions using TensorFlow models
"""

from flask import Flask, request, jsonify, render_template_string
import numpy as np
import joblib
import tensorflow as tf
import os
import time
import json
from datetime import datetime
from sensor_interface import SensorInterface
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global sensor interface
sensor_interface = None

# Load models and scalers
models = {}
scalers = {}

def load_models():
    """Load TensorFlow models and scalers"""
    global models, scalers
    
    try:
        # Load models
        models = {
            'groundnut': tf.keras.models.load_model('new_groundnut_model.h5', compile=False),
            'mustard': tf.keras.models.load_model('new_mustard_model.h5', compile=False)
        }
        
        # Load scalers
        scalers = {
            'groundnut': joblib.load('scaler_groundnut.pkl'),
            'mustard': joblib.load('scaler_mustard.pkl')
        }
        
        # Compile models
        for crop in models:
            models[crop].compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss='mse',
                metrics=['mae']
            )
        
        logger.info("Models and scalers loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return False

def init_sensors():
    """Initialize sensor interface"""
    global sensor_interface
    try:
        sensor_interface = SensorInterface()
        logger.info("Sensor interface initialized")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize sensors: {e}")
        return False

@app.route('/')
def home():
    """Home page with sensor readings and predictions"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Soil Moisture Monitor</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { text-align: center; color: #333; margin-bottom: 30px; }
            .sensor-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
            .sensor-card { background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff; }
            .prediction-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .prediction-card { background: #e8f5e8; padding: 20px; border-radius: 8px; border-left: 4px solid #28a745; }
            .value { font-size: 24px; font-weight: bold; color: #007bff; }
            .moisture-value { font-size: 28px; font-weight: bold; color: #28a745; }
            .unit { font-size: 14px; color: #666; }
            .timestamp { font-size: 12px; color: #999; margin-top: 10px; }
            .refresh-btn { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 10px 0; }
            .refresh-btn:hover { background: #0056b3; }
            .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
            .status.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .status.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🌱 Soil Moisture Monitor</h1>
                <p>Real-time sensor readings and moisture predictions</p>
            </div>
            
            <button class="refresh-btn" onclick="location.reload()">🔄 Refresh Readings</button>
            
            <div class="sensor-grid">
                <div class="sensor-card">
                    <h3>🌡️ Temperature</h3>
                    <div class="value">{{ "%.1f"|format(sensors.temperature) if sensors.temperature else "N/A" }}</div>
                    <div class="unit">°C</div>
                </div>
                <div class="sensor-card">
                    <h3>💧 Humidity</h3>
                    <div class="value">{{ "%.1f"|format(sensors.humidity) if sensors.humidity else "N/A" }}</div>
                    <div class="unit">%</div>
                </div>
                <div class="sensor-card">
                    <h3>📊 ADC Value</h3>
                    <div class="value">{{ "%.0f"|format(sensors.adc) if sensors.adc else "N/A" }}</div>
                    <div class="unit">Raw ADC</div>
                </div>
            </div>
            
            <div class="prediction-grid">
                <div class="prediction-card">
                    <h3>🥜 Groundnut Moisture</h3>
                    <div class="moisture-value">{{ "%.2f"|format(predictions.groundnut) if predictions.groundnut else "N/A" }}</div>
                    <div class="unit">% Moisture</div>
                </div>
                <div class="prediction-card">
                    <h3>🌿 Mustard Moisture</h3>
                    <div class="moisture-value">{{ "%.2f"|format(predictions.mustard) if predictions.mustard else "N/A" }}</div>
                    <div class="unit">% Moisture</div>
                </div>
            </div>
            
            <div class="timestamp">
                Last updated: {{ timestamp }}
            </div>
            
            {% if status %}
            <div class="status {{ 'success' if status.success else 'error' }}">
                {{ status.message }}
            </div>
            {% endif %}
        </div>
        
        <script>
            // Auto-refresh every 30 seconds
            setTimeout(function() {
                location.reload();
            }, 30000);
        </script>
    </body>
    </html>
    """
    
    # Get current sensor readings
    sensors = {'temperature': None, 'humidity': None, 'adc': None}
    predictions = {'groundnut': None, 'mustard': None}
    status = {'success': True, 'message': 'All systems operational'}
    
    if sensor_interface:
        try:
            readings = sensor_interface.read_all_sensors()
            sensors = readings
            
            # Make predictions if we have valid sensor data
            if all(v is not None for v in readings.values()):
                predictions = predict_moisture(readings['adc'], readings['temperature'], readings['humidity'])
        except Exception as e:
            status = {'success': False, 'message': f'Sensor error: {str(e)}'}
    else:
        status = {'success': False, 'message': 'Sensor interface not initialized'}
    
    return render_template_string(html_template, 
                                sensors=sensors, 
                                predictions=predictions, 
                                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                status=status)

@app.route('/api/sensors')
def get_sensors():
    """API endpoint to get current sensor readings"""
    if not sensor_interface:
        return jsonify({'error': 'Sensor interface not initialized'}), 500
    
    try:
        readings = sensor_interface.read_all_sensors()
        return jsonify({
            'success': True,
            'data': readings,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/<crop>')
def predict_endpoint(crop):
    """API endpoint for moisture prediction"""
    if crop not in models:
        return jsonify({'error': f'Unknown crop: {crop}'}), 400
    
    try:
        # Get sensor readings
        if not sensor_interface:
            return jsonify({'error': 'Sensor interface not initialized'}), 500
        
        readings = sensor_interface.read_all_sensors()
        
        # Check if we have valid readings
        if any(v is None for v in readings.values()):
            return jsonify({'error': 'Invalid sensor readings'}), 500
        
        # Make prediction
        moisture = predict_moisture(readings['adc'], readings['temperature'], readings['humidity'], crop)
        
        return jsonify({
            'success': True,
            'crop': crop,
            'moisture': moisture,
            'sensor_data': readings,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def predict_moisture(adc: float, temp: float, hum: float, crop: str = None) -> dict:
    """
    Predict moisture for given sensor readings
    
    Args:
        adc: ADC value from moisture sensor
        temp: Temperature in Celsius
        hum: Humidity percentage
        crop: Specific crop to predict (if None, predicts for both)
    
    Returns:
        Dictionary with moisture predictions
    """
    X = np.array([[adc, temp, hum]])
    
    if crop:
        # Predict for specific crop
        if crop not in models:
            raise ValueError(f'Unknown crop: {crop}')
        
        scaler = scalers[crop]
        model = models[crop]
        
        Xs = scaler.transform(X)
        prediction = model.predict(Xs, verbose=0)[0, 0]
        
        return {crop: round(float(prediction), 2)}
    else:
        # Predict for both crops
        predictions = {}
        for crop_name in models:
            scaler = scalers[crop_name]
            model = models[crop_name]
            
            Xs = scaler.transform(X)
            prediction = model.predict(Xs, verbose=0)[0, 0]
            predictions[crop_name] = round(float(prediction), 2)
        
        return predictions

@app.route('/api/status')
def system_status():
    """Get system status"""
    status = {
        'models_loaded': len(models) > 0,
        'sensors_initialized': sensor_interface is not None,
        'available_crops': list(models.keys()),
        'timestamp': datetime.now().isoformat()
    }
    
    if sensor_interface:
        status['sensor_status'] = sensor_interface.get_sensor_status()
    
    return jsonify(status)

@app.route('/api/update/<crop>', methods=['POST'])
def update_model(crop):
    """Update model with new data"""
    if crop not in models:
        return jsonify({'error': f'Unknown crop: {crop}'}), 400
    
    try:
        data = request.get_json(force=True)
        adc = float(data['adc'])
        temp = float(data['temp'])
        hum = float(data['hum'])
        moisture = float(data['moisture'])
        
        X_new = np.array([[adc, temp, hum]])
        y_new = np.array([moisture])
        
        scaler = scalers[crop]
        model = models[crop]
        
        Xs = scaler.transform(X_new)
        model.fit(Xs, y_new, epochs=1, verbose=0)
        
        # Save updated model
        model.save(f'new_{crop}_model.h5')
        
        return jsonify({'status': f'{crop} model updated successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize system
    logger.info("Starting Raspberry Pi Soil Moisture Monitor...")
    
    # Load models
    if not load_models():
        logger.error("Failed to load models. Exiting.")
        exit(1)
    
    # Initialize sensors
    if not init_sensors():
        logger.warning("Failed to initialize sensors. Running in sensor-less mode.")
    
    # Start Flask app
    logger.info("Starting Flask application...")
    app.run(host='0.0.0.0', port=5000, debug=False) 