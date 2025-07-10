#!/usr/bin/env python3
"""
Add model download endpoints to existing Flask API
for MicroPython ESP32 model updates
"""

import os
import time
import json
from flask import send_file, jsonify, request

def add_model_download_endpoints(app):
    """
    Add model download and status endpoints to Flask app
    
    Args:
        app: Flask application instance
    """
    
    @app.route('/models/<crop_type>/download')
    def download_model(crop_type):
        """Download TFLite model for specified crop"""
        if crop_type not in ['mustard', 'groundnut']:
            return jsonify({'error': 'Invalid crop type'}), 400
        
        model_path = f"new_ML/models/{crop_type}_model.tflite"
        
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model not found'}), 404
        
        try:
            return send_file(
                model_path,
                as_attachment=True,
                download_name=f"{crop_type}_model.tflite",
                mimetype='application/octet-stream'
            )
        except Exception as e:
            return jsonify({'error': f'Download failed: {str(e)}'}), 500
    
    @app.route('/models/status')
    def model_status():
        """Get model status and version information"""
        try:
            models_info = {
                "available_models": {
                    "mustard": {
                        "available": os.path.exists("new_ML/models/mustard_model.tflite"),
                        "version": "1.0",
                        "size": os.path.getsize("new_ML/models/mustard_model.tflite") if os.path.exists("new_ML/models/mustard_model.tflite") else 0,
                        "last_updated": time.time()
                    },
                    "groundnut": {
                        "available": os.path.exists("new_ML/models/groundnut_model.tflite"),
                        "version": "1.0",
                        "size": os.path.getsize("new_ML/models/groundnut_model.tflite") if os.path.exists("new_ML/models/groundnut_model.tflite") else 0,
                        "last_updated": time.time()
                    }
                },
                "server_version": "1.0",
                "last_updated": time.time()
            }
            
            return jsonify(models_info)
        except Exception as e:
            return jsonify({'error': f'Status check failed: {str(e)}'}), 500
    
    @app.route('/models/update', methods=['POST'])
    def update_model():
        """Update model version (for version control)"""
        try:
            data = request.get_json()
            crop_type = data.get('crop_type')
            new_version = data.get('version', '1.0')
            
            if crop_type not in ['mustard', 'groundnut']:
                return jsonify({'error': 'Invalid crop type'}), 400
            
            # Update model metadata
            metadata_path = "new_ML/models/model_metadata.json"
            metadata = {}
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            if 'models' not in metadata:
                metadata['models'] = {}
            
            metadata['models'][crop_type] = {
                'version': new_version,
                'last_updated': time.time(),
                'size': os.path.getsize(f"new_ML/models/{crop_type}_model.tflite") if os.path.exists(f"new_ML/models/{crop_type}_model.tflite") else 0
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return jsonify({
                'message': f'{crop_type} model updated to version {new_version}',
                'crop_type': crop_type,
                'version': new_version
            })
            
        except Exception as e:
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
                        'available': os.path.exists("new_ML/models/mustard_model.tflite"),
                        'size': os.path.getsize("new_ML/models/mustard_model.tflite") if os.path.exists("new_ML/models/mustard_model.tflite") else 0
                    },
                    'groundnut': {
                        'available': os.path.exists("new_ML/models/groundnut_model.tflite"),
                        'size': os.path.getsize("new_ML/models/groundnut_model.tflite") if os.path.exists("new_ML/models/groundnut_model.tflite") else 0
                    }
                }
            }
            
            return jsonify(health_status)
        except Exception as e:
            return jsonify({'status': 'unhealthy', 'error': str(e)}), 500
    
    print("Model download endpoints added successfully!")
    return app

def create_model_metadata():
    """Create initial model metadata file"""
    metadata = {
        "models": {
            "mustard": {
                "version": "1.0",
                "last_updated": time.time(),
                "size": 0,
                "description": "Mustard grain moisture prediction model"
            },
            "groundnut": {
                "version": "1.0",
                "last_updated": time.time(),
                "size": 0,
                "description": "Groundnut grain moisture prediction model"
            }
        },
        "server_version": "1.0",
        "last_updated": time.time()
    }
    
    # Update sizes if models exist
    if os.path.exists("new_ML/models/mustard_model.tflite"):
        metadata["models"]["mustard"]["size"] = os.path.getsize("new_ML/models/mustard_model.tflite")
    
    if os.path.exists("new_ML/models/groundnut_model.tflite"):
        metadata["models"]["groundnut"]["size"] = os.path.getsize("new_ML/models/groundnut_model.tflite")
    
    # Save metadata
    os.makedirs("new_ML/models", exist_ok=True)
    with open("new_ML/models/model_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Model metadata created: new_ML/models/model_metadata.json")

def test_endpoints():
    """Test the new endpoints"""
    print("Testing model endpoints...")
    
    # Test model status
    try:
        import requests
        base_url = "http://localhost:5000"  # Adjust to your server URL
        
        # Test status endpoint
        response = requests.get(f"{base_url}/models/status")
        if response.status_code == 200:
            print("✅ Status endpoint working")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Status endpoint failed: {response.status_code}")
        
        # Test health endpoint
        response = requests.get(f"{base_url}/models/health")
        if response.status_code == 200:
            print("✅ Health endpoint working")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
        
        # Test download endpoints
        for crop in ["mustard", "groundnut"]:
            response = requests.get(f"{base_url}/models/{crop}/download")
            if response.status_code == 200:
                print(f"✅ {crop} download endpoint working")
                print(f"Model size: {len(response.content)} bytes")
            else:
                print(f"❌ {crop} download endpoint failed: {response.status_code}")
                
    except ImportError:
        print("requests library not available - skipping endpoint tests")
    except Exception as e:
        print(f"Error testing endpoints: {e}")

def main():
    """Main function to setup model download endpoints"""
    print("=== Adding Model Download Endpoints ===")
    
    # Create model metadata
    create_model_metadata()
    
    # Instructions for adding to existing Flask app
    print("\n=== Integration Instructions ===")
    print("1. Add this to your existing Flask app:")
    print("""
from add_model_download_endpoints import add_model_download_endpoints

# Add after creating your Flask app
app = Flask(__name__)
add_model_download_endpoints(app)
""")
    
    print("\n2. New endpoints available:")
    print("   - GET /models/status - Get model status")
    print("   - GET /models/health - Health check")
    print("   - GET /models/{crop}/download - Download model")
    print("   - POST /models/update - Update model version")
    
    print("\n3. Test endpoints:")
    print("   curl http://localhost:5000/models/status")
    print("   curl http://localhost:5000/models/mustard/download")
    
    # Test endpoints if possible
    test_endpoints()
    
    print("\n=== Setup Complete ===")
    print("Your Flask API now supports MicroPython model updates!")

if __name__ == "__main__":
    main() 