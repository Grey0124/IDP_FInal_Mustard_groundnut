#!/usr/bin/env python3
"""
Moisture Meter API for Render Deployment
Flask-based REST API for moisture prediction using trained models
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import numpy as np
import os
from datetime import datetime
import logging
import tensorflow as tf
import pandas as pd
import time
import json

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
model_type = None  # 'sklearn' or 'tensorflow'

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
                'models/groundnut_scaler.pkl',
                'groundnut_scaler_new.pkl',
                'models/groundnut_scaler_new.pkl'
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
                'models/mustard_scaler.pkl',
                'mustard_scaler_new.pkl',
                'models/mustard_scaler_new.pkl'
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
            # Try multiple possible paths for .h5 models, prioritizing *_new.h5
            groundnut_model_paths = [
                'models/model_groundnut_new.h5',
                'model_groundnut_new.h5',
                'models/model_groundnut.h5',
                'model_groundnut.h5'
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
                
                # Load TensorFlow model
                try:
                    groundnut_model = tf.keras.models.load_model(groundnut_model_path, compile=False)
                    groundnut_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                    model_type = 'tensorflow'
                    logger.info(f"✓ Groundnut TensorFlow model loaded successfully from {groundnut_model_path}")
                except Exception as e:
                    logger.error(f"❌ Error loading groundnut TensorFlow model: {e}")
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
            # Try multiple possible paths for .h5 models, prioritizing *_new.h5
            mustard_model_paths = [
                'models/model_mustard_new.h5',
                'model_mustard_new.h5',
                'models/model_mustard.h5',
                'model_mustard.h5'
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
                
                # Load TensorFlow model
                try:
                    mustard_model = tf.keras.models.load_model(mustard_model_path, compile=False)
                    mustard_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                    model_type = 'tensorflow'
                    logger.info(f"✓ Mustard TensorFlow model loaded successfully from {mustard_model_path}")
                except Exception as e:
                    logger.error(f"❌ Error loading mustard TensorFlow model: {e}")
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
                'type': f'Phase 2 Best Model ({model_type or "unknown"})',
                'features': ['ADC', 'Temperature', 'Humidity']
            },
            'mustard': {
                'available': mustard_model is not None,
                'type': f'Phase 2 Best Model ({model_type or "unknown"})',
                'features': ['ADC', 'Temperature', 'Humidity']
            }
        },
        'model_type': model_type
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
        
        # Prepare features for model
        features_df = pd.DataFrame([[adc, temperature, humidity]], columns=['ADC', 'Temperature', 'Humidity'])
        
        # Scale features - handle feature names warning
        try:
            scaled_features = scaler.transform(features_df)
        except Exception as e:
            # If scaler has feature names, try without them
            if hasattr(scaler, 'feature_names_in_'):
                # Create DataFrame with proper feature names
                feature_df = pd.DataFrame([[adc, temperature, humidity]], columns=scaler.feature_names_in_)
                scaled_features = scaler.transform(feature_df)
            else:
                raise e
        
        # Make prediction (TensorFlow model only)
        prediction_array = model.predict(scaled_features, verbose=0)
        # Extract scalar value properly
        if hasattr(prediction_array, 'flatten'):
            prediction = float(prediction_array.flatten()[0])
        else:
            prediction = float(prediction_array[0, 0])
        
        if prediction is None or np.isnan(prediction):
            return jsonify({
                'error': 'Prediction failed',
                'message': 'Model could not make a prediction with the provided data'
            }), 500
        
        # Prepare response
        response = {
            'status': 'success',
            'prediction': {
                'moisture_percentage': round(prediction, 2),
                'crop_type': crop_type,
                'model_used': model_name,
                'model_type': model_type,
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
        
        logger.info(f"Prediction made: {crop_type} - {prediction:.2f}% moisture (using {model_type} model)")
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

@app.route('/update/<crop>', methods=['POST'])
def update_model(crop):
    """
    POST /update/<crop>
    JSON: { "adc": …, "temp": …, "hum": …, "moisture": … }
    Update .h5 model directly with new data
    """
    try:
        # Validate crop type
        if crop not in ['groundnut', 'mustard']:
            return jsonify({'error': f'Unknown crop: {crop}'}), 400
        
        # Get JSON data
        data = request.get_json(force=True)
        
        # Extract and validate parameters
        try:
            adc = float(data['adc'])
            temp = float(data['temp'])
            hum = float(data['hum'])
            moisture = float(data['moisture'])
        except (KeyError, TypeError, ValueError):
            return jsonify({'error': 'Invalid JSON body. Required: adc, temp, hum, moisture'}), 400
        
        # Get the appropriate model and scaler
        if crop == 'groundnut':
            if groundnut_model is None or groundnut_scaler is None:
                return jsonify({'error': 'Groundnut model not loaded'}), 503
            model = groundnut_model
            scaler = groundnut_scaler
        else:  # mustard
            if mustard_model is None or mustard_scaler is None:
                return jsonify({'error': 'Mustard model not loaded'}), 503
            model = mustard_model
            scaler = mustard_scaler
        
        # Prepare data
        X_new = pd.DataFrame([[adc, temp, hum]], columns=['ADC', 'Temperature', 'Humidity'])
        y_new = np.array([moisture])
        
        # Scale features
        try:
            Xs = scaler.transform(X_new)
        except Exception as e:
            # Handle feature names if present
            if hasattr(scaler, 'feature_names_in_'):
                feature_df = pd.DataFrame(X_new, columns=scaler.feature_names_in_)
                Xs = scaler.transform(feature_df)
            else:
                raise e
        
        # Update model with new data
        model.fit(Xs, y_new, epochs=1, verbose=0)
        
        # Save updated model
        model_path = f'model_{crop}_updated.h5'
        model.save(model_path)
        
        # Log the update
        logger.info(f"✓ {crop.capitalize()} model updated and saved to {model_path}")
        
        return jsonify({
            'status': 'success',
            'message': f'{crop} model updated successfully',
            'model_saved': model_path,
            'input_data': {
                'adc': adc,
                'temp': temp,
                'hum': hum,
                'moisture': moisture
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error updating {crop} model: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/update/groundnut', methods=['POST'])
def update_groundnut():
    """Update groundnut model specifically"""
    return update_model('groundnut')

@app.route('/update/mustard', methods=['POST'])
def update_mustard():
    """Update mustard model specifically"""
    return update_model('mustard')

@app.route('/models/status', methods=['GET'])
def get_models_status():
    """Get status of all models"""
    try:
        status = {
            'models_loaded': {
                'groundnut': groundnut_model is not None,
                'mustard': mustard_model is not None
            },
            'scalers_loaded': {
                'groundnut': groundnut_scaler is not None,
                'mustard': mustard_scaler is not None
            },
            'updated_model_files': {
                'groundnut': os.path.exists('model_groundnut_updated.h5'),
                'mustard': os.path.exists('model_mustard_updated.h5')
            },
            'model_type': model_type,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error getting models status: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

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
            # Scalers only
            'groundnut_scaler.pkl': os.path.exists('groundnut_scaler.pkl'),
            'mustard_scaler.pkl': os.path.exists('mustard_scaler.pkl'),
            'models/groundnut_scaler.pkl': os.path.exists('models/groundnut_scaler.pkl'),
            'models/mustard_scaler.pkl': os.path.exists('models/mustard_scaler.pkl'),
            'groundnut_scaler_new.pkl': os.path.exists('groundnut_scaler_new.pkl'),
            'mustard_scaler_new.pkl': os.path.exists('mustard_scaler_new.pkl'),
            'models/groundnut_scaler_new.pkl': os.path.exists('models/groundnut_scaler_new.pkl'),
            'models/mustard_scaler_new.pkl': os.path.exists('models/mustard_scaler_new.pkl'),
            # TensorFlow models only
            'model_groundnut.h5': os.path.exists('model_groundnut.h5'),
            'model_mustard.h5': os.path.exists('model_mustard.h5'),
            'models/model_groundnut.h5': os.path.exists('models/model_groundnut.h5'),
            'models/model_mustard.h5': os.path.exists('models/model_mustard.h5'),
            'models/model_groundnut_new.h5': os.path.exists('models/model_groundnut_new.h5'),
            'models/model_mustard_new.h5': os.path.exists('models/model_mustard_new.h5')
        }
    }
    
    return jsonify(debug_info)

@app.route('/models/<crop_type>/download')
def download_model(crop_type):
    """Download TFLite model for specified crop"""
    if crop_type not in ['mustard', 'groundnut']:
        return jsonify({'error': 'Invalid crop type'}), 400
    
    # Check for TFLite models in multiple locations
    model_paths = [
        f"models/{crop_type}_model.tflite",
        f"{crop_type}_model.tflite",
        f"new_ML/models/{crop_type}_model.tflite"
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        return jsonify({'error': 'TFLite model not found'}), 404
    
    try:
        return send_file(
            model_path,
            as_attachment=True,
            download_name=f"{crop_type}_model.tflite",
            mimetype='application/octet-stream'
        )
    except Exception as e:
        logger.error(f"Error downloading {crop_type} model: {e}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/models/status')
def model_status():
    """Get model status and version information"""
    try:
        models_info = {
            "available_models": {
                "mustard": {
                    "available": any(os.path.exists(path) for path in [
                        "models/mustard_model.tflite",
                        "mustard_model.tflite", 
                        "new_ML/models/mustard_model.tflite"
                    ]),
                    "version": "1.0",
                    "size": 0,
                    "last_updated": time.time()
                },
                "groundnut": {
                    "available": any(os.path.exists(path) for path in [
                        "models/groundnut_model.tflite",
                        "groundnut_model.tflite",
                        "new_ML/models/groundnut_model.tflite"
                    ]),
                    "version": "1.0", 
                    "size": 0,
                    "last_updated": time.time()
                }
            },
            "server_version": "1.0",
            "last_updated": time.time()
        }
        
        # Update sizes if models exist
        for crop in ["mustard", "groundnut"]:
            for path in [f"models/{crop}_model.tflite", f"{crop}_model.tflite", f"new_ML/models/{crop}_model.tflite"]:
                if os.path.exists(path):
                    models_info["available_models"][crop]["size"] = os.path.getsize(path)
                    break
        
        return jsonify(models_info)
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        return jsonify({'error': f'Status check failed: {str(e)}'}), 500

@app.route('/models/update', methods=['POST'])
def update_model_version():
    """Update model version (for version control)"""
    try:
        data = request.get_json()
        crop_type = data.get('crop_type')
        new_version = data.get('version', '1.0')
        
        if crop_type not in ['mustard', 'groundnut']:
            return jsonify({'error': 'Invalid crop type'}), 400
        
        # Update model metadata
        metadata_path = "models/model_metadata.json"
        metadata = {}
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        if 'models' not in metadata:
            metadata['models'] = {}
        
        # Find model size
        model_size = 0
        for path in [f"models/{crop_type}_model.tflite", f"{crop_type}_model.tflite", f"new_ML/models/{crop_type}_model.tflite"]:
            if os.path.exists(path):
                model_size = os.path.getsize(path)
                break
        
        metadata['models'][crop_type] = {
            'version': new_version,
            'last_updated': time.time(),
            'size': model_size
        }
        
        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return jsonify({
            'message': f'{crop_type} model updated to version {new_version}',
            'crop_type': crop_type,
            'version': new_version
        })
        
    except Exception as e:
        logger.error(f"Error updating model version: {e}")
        return jsonify({'error': f'Update failed: {str(e)}'}), 500

@app.route('/models/health')
def model_health():
    """Health check endpoint for model server"""
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': time.time(),
            'models': {
                'mustard': {
                    'available': any(os.path.exists(path) for path in [
                        "models/mustard_model.tflite",
                        "mustard_model.tflite",
                        "new_ML/models/mustard_model.tflite"
                    ]),
                    'size': 0
                },
                'groundnut': {
                    'available': any(os.path.exists(path) for path in [
                        "models/groundnut_model.tflite", 
                        "groundnut_model.tflite",
                        "new_ML/models/groundnut_model.tflite"
                    ]),
                    'size': 0
                }
            }
        }
        
        # Update sizes
        for crop in ["mustard", "groundnut"]:
            for path in [f"models/{crop}_model.tflite", f"{crop}_model.tflite", f"new_ML/models/{crop}_model.tflite"]:
                if os.path.exists(path):
                    health_status['models'][crop]['size'] = os.path.getsize(path)
                    break
        
        return jsonify(health_status)
    except Exception as e:
        logger.error(f"Error in model health check: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

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