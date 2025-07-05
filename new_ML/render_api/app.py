#!/usr/bin/env python3
"""
Moisture Meter API for Render Deployment
Flask-based REST API for moisture prediction using trained models
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for models
groundnut_model = None
mustard_model = None
groundnut_scaler = None
mustard_scaler = None

def load_models():
    """Load the trained models from Phase 2"""
    global groundnut_model, mustard_model, groundnut_scaler, mustard_scaler
    
    try:
        logger.info("Starting model loading process...")
        
        # Check current working directory
        current_dir = os.getcwd()
        logger.info(f"Current working directory: {current_dir}")
        
        # List files in current directory
        logger.info(f"Files in current directory: {os.listdir('.')}")
        
        # Check if models directory exists
        if os.path.exists('models'):
            logger.info(f"Models directory found: {os.listdir('models')}")
        
        # Import warnings filter to suppress version warnings
        import warnings
        from sklearn.exceptions import InconsistentVersionWarning
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        
        # Try loading scalers first (they're smaller)
        try:
            logger.info("Attempting to load groundnut scaler...")
            # Try multiple possible paths
            groundnut_scaler_paths = [
                'groundnut_scaler.pkl',
                'models/groundnut_scaler.pkl'
            ]
            
            groundnut_scaler_path = None
            for path in groundnut_scaler_paths:
                if os.path.exists(path):
                    groundnut_scaler_path = path
                    break
            
            if groundnut_scaler_path:
                try:
                    groundnut_scaler = joblib.load(groundnut_scaler_path)
                    logger.info(f"✓ Groundnut scaler loaded successfully from {groundnut_scaler_path}")
                except Exception as e:
                    logger.error(f"❌ Error loading groundnut scaler from {groundnut_scaler_path}: {e}")
                    # Try with different joblib settings
                    try:
                        groundnut_scaler = joblib.load(groundnut_scaler_path, mmap_mode='r')
                        logger.info(f"✓ Groundnut scaler loaded successfully with mmap_mode from {groundnut_scaler_path}")
                    except Exception as e2:
                        logger.error(f"❌ Failed to load groundnut scaler with mmap_mode: {e2}")
            else:
                logger.error("❌ Groundnut scaler file not found in any expected location")
        except Exception as e:
            logger.error(f"❌ Error loading groundnut scaler: {e}")
        
        try:
            logger.info("Attempting to load mustard scaler...")
            # Try multiple possible paths
            mustard_scaler_paths = [
                'mustard_scaler.pkl',
                'models/mustard_scaler.pkl'
            ]
            
            mustard_scaler_path = None
            for path in mustard_scaler_paths:
                if os.path.exists(path):
                    mustard_scaler_path = path
                    break
            
            if mustard_scaler_path:
                try:
                    mustard_scaler = joblib.load(mustard_scaler_path)
                    logger.info(f"✓ Mustard scaler loaded successfully from {mustard_scaler_path}")
                except Exception as e:
                    logger.error(f"❌ Error loading mustard scaler from {mustard_scaler_path}: {e}")
                    # Try with different joblib settings
                    try:
                        mustard_scaler = joblib.load(mustard_scaler_path, mmap_mode='r')
                        logger.info(f"✓ Mustard scaler loaded successfully with mmap_mode from {mustard_scaler_path}")
                    except Exception as e2:
                        logger.error(f"❌ Failed to load mustard scaler with mmap_mode: {e2}")
            else:
                logger.error("❌ Mustard scaler file not found in any expected location")
        except Exception as e:
            logger.error(f"❌ Error loading mustard scaler: {e}")
        
        # Try loading models (they're larger)
        try:
            logger.info("Attempting to load groundnut model...")
            # Try multiple possible paths
            groundnut_model_paths = [
                'groundnut_best_model.pkl',
                'models/groundnut_best_model.pkl'
            ]
            
            groundnut_model_path = None
            for path in groundnut_model_paths:
                if os.path.exists(path):
                    groundnut_model_path = path
                    break
            
            if groundnut_model_path:
                # Check file size
                file_size = os.path.getsize(groundnut_model_path) / (1024 * 1024)  # MB
                logger.info(f"Groundnut model file size: {file_size:.2f} MB")
                
                try:
                    groundnut_model = joblib.load(groundnut_model_path)
                    logger.info(f"✓ Groundnut model loaded successfully from {groundnut_model_path}")
                except Exception as e:
                    logger.error(f"❌ Error loading groundnut model from {groundnut_model_path}: {e}")
                    # Try with different joblib settings
                    try:
                        groundnut_model = joblib.load(groundnut_model_path, mmap_mode='r')
                        logger.info(f"✓ Groundnut model loaded successfully with mmap_mode from {groundnut_model_path}")
                    except Exception as e2:
                        logger.error(f"❌ Failed to load groundnut model with mmap_mode: {e2}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
            else:
                logger.error("❌ Groundnut model file not found in any expected location")
        except Exception as e:
            logger.error(f"❌ Error loading groundnut model: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        try:
            logger.info("Attempting to load mustard model...")
            # Try multiple possible paths
            mustard_model_paths = [
                'mustard_best_model.pkl',
                'models/mustard_best_model.pkl'
            ]
            
            mustard_model_path = None
            for path in mustard_model_paths:
                if os.path.exists(path):
                    mustard_model_path = path
                    break
            
            if mustard_model_path:
                # Check file size
                file_size = os.path.getsize(mustard_model_path) / (1024 * 1024)  # MB
                logger.info(f"Mustard model file size: {file_size:.2f} MB")
                
                try:
                    mustard_model = joblib.load(mustard_model_path)
                    logger.info(f"✓ Mustard model loaded successfully from {mustard_model_path}")
                except Exception as e:
                    logger.error(f"❌ Error loading mustard model from {mustard_model_path}: {e}")
                    # Try with different joblib settings
                    try:
                        mustard_model = joblib.load(mustard_model_path, mmap_mode='r')
                        logger.info(f"✓ Mustard model loaded successfully with mmap_mode from {mustard_model_path}")
                    except Exception as e2:
                        logger.error(f"❌ Failed to load mustard model with mmap_mode: {e2}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
            else:
                logger.error("❌ Mustard model file not found in any expected location")
        except Exception as e:
            logger.error(f"❌ Error loading mustard model: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
    except Exception as e:
        logger.error(f"Error in load_models: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

@app.route('/')
def home():
    """API home endpoint"""
    return jsonify({
        'message': '🌾 Moisture Meter API',
        'version': '1.0.0',
        'status': 'active',
        'models_loaded': {
            'groundnut': groundnut_model is not None,
            'mustard': mustard_model is not None
        },
        'endpoints': {
            'predict': '/predict',
            'health': '/health',
            'models': '/models'
        }
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_available': {
            'groundnut': groundnut_model is not None,
            'mustard': mustard_model is not None
        }
    })

@app.route('/models')
def get_models():
    """Get information about available models"""
    return jsonify({
        'available_models': {
            'groundnut': {
                'available': groundnut_model is not None,
                'type': 'Phase 2 Best Model (scikit-learn)',
                'features': ['ADC', 'Temperature', 'Humidity']
            },
            'mustard': {
                'available': mustard_model is not None,
                'type': 'Phase 2 Best Model (scikit-learn)',
                'features': ['ADC', 'Temperature', 'Humidity']
            }
        }
    })

@app.route('/predict', methods=['POST'])
def predict_moisture():
    """Predict moisture content based on sensor readings"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided',
                'message': 'Please provide ADC, temperature, humidity, and crop_type'
            }), 400
        
        # Extract parameters
        adc = data.get('adc')
        temperature = data.get('temperature')
        humidity = data.get('humidity')
        crop_type = data.get('crop_type', 'auto').lower()
        
        # Validate required parameters
        if adc is None or temperature is None or humidity is None:
            return jsonify({
                'error': 'Missing required parameters',
                'message': 'Please provide ADC, temperature, and humidity values'
            }), 400
        
        # Validate data types
        try:
            adc = float(adc)
            temperature = float(temperature)
            humidity = float(humidity)
        except (ValueError, TypeError):
            return jsonify({
                'error': 'Invalid data types',
                'message': 'ADC, temperature, and humidity must be numeric values'
            }), 400
        
        # Auto-detect crop type based on ADC range if not specified
        if crop_type == 'auto':
            # Simple heuristic: Groundnut typically has higher ADC values
            if adc > 2950:
                crop_type = 'groundnut'
            else:
                crop_type = 'mustard'
        
        # Select appropriate model and scaler
        if crop_type == 'groundnut':
            if groundnut_model is None or groundnut_scaler is None:
                return jsonify({
                    'error': 'Model not available',
                    'message': 'Groundnut model is not loaded'
                }), 503
            model = groundnut_model
            scaler = groundnut_scaler
            model_name = "Groundnut"
        elif crop_type == 'mustard':
            if mustard_model is None or mustard_scaler is None:
                return jsonify({
                    'error': 'Model not available',
                    'message': 'Mustard model is not loaded'
                }), 503
            model = mustard_model
            scaler = mustard_scaler
            model_name = "Mustard"
        else:
            return jsonify({
                'error': 'Invalid crop type',
                'message': 'Crop type must be "groundnut", "mustard", or "auto"'
            }), 400
        
        # Prepare features for scikit-learn model
        features_array = np.array([[adc, temperature, humidity]])
        
        # Scale features
        scaled_features = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        
        if prediction is None or np.isnan(prediction):
            return jsonify({
                'error': 'Prediction failed',
                'message': 'Model could not make a prediction with the provided data'
            }), 500
        
        # Prepare response
        response = {
            'status': 'success',
            'prediction': {
                'moisture_percentage': round(float(prediction), 2),
                'crop_type': crop_type,
                'model_used': model_name,
                'confidence': 'high'
            },
            'input_data': {
                'adc': adc,
                'temperature': temperature,
                'humidity': humidity,
                'crop_type': crop_type
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction made: {crop_type} - {prediction:.2f}% moisture")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/predict/groundnut', methods=['POST'])
def predict_groundnut():
    """Predict moisture specifically for groundnut"""
    try:
        data = request.get_json()
        if data:
            data['crop_type'] = 'groundnut'
        return predict_moisture()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/mustard', methods=['POST'])
def predict_mustard():
    """Predict moisture specifically for mustard"""
    try:
        data = request.get_json()
        if data:
            data['crop_type'] = 'mustard'
        return predict_moisture()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/debug')
def debug_info():
    """Debug endpoint to check file system and model loading"""
    import os
    
    debug_info = {
        'current_working_directory': os.getcwd(),
        'files_in_current_dir': os.listdir('.'),
        'models_dir_exists': os.path.exists('models'),
        'models_dir_contents': os.listdir('models') if os.path.exists('models') else [],
        'models_loaded': {
            'groundnut_model': groundnut_model is not None,
            'mustard_model': mustard_model is not None,
            'groundnut_scaler': groundnut_scaler is not None,
            'mustard_scaler': mustard_scaler is not None
        },
        'file_checks': {
            'groundnut_scaler.pkl': os.path.exists('groundnut_scaler.pkl'),
            'mustard_scaler.pkl': os.path.exists('mustard_scaler.pkl'),
            'groundnut_best_model.pkl': os.path.exists('groundnut_best_model.pkl'),
            'mustard_best_model.pkl': os.path.exists('mustard_best_model.pkl'),
            'models/groundnut_scaler.pkl': os.path.exists('models/groundnut_scaler.pkl'),
            'models/mustard_scaler.pkl': os.path.exists('models/mustard_scaler.pkl'),
            'models/groundnut_best_model.pkl': os.path.exists('models/groundnut_best_model.pkl'),
            'models/mustard_best_model.pkl': os.path.exists('models/mustard_best_model.pkl')
        }
    }
    
    return jsonify(debug_info)

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load models on startup
    logger.info("Loading moisture prediction models...")
    load_models()
    
    # Get port from environment variable (for Render)
    port = int(os.environ.get('PORT', 8080))
    
    logger.info(f"Starting Moisture Meter API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False) 