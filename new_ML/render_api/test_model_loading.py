#!/usr/bin/env python3
"""
Test script to verify model loading in render_api directory
"""

import os
import joblib
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress scikit-learn version warnings
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

def test_model_loading():
    """Test loading all models and scalers"""
    
    logger.info("=== Model Loading Test ===")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Files in current directory: {os.listdir('.')}")
    
    if os.path.exists('models'):
        logger.info(f"Models directory found: {os.listdir('models')}")
    
    # Test scalers
    scaler_files = [
        'groundnut_scaler.pkl',
        'mustard_scaler.pkl',
        'models/groundnut_scaler.pkl',
        'models/mustard_scaler.pkl'
    ]
    
    for scaler_file in scaler_files:
        if os.path.exists(scaler_file):
            try:
                scaler = joblib.load(scaler_file)
                logger.info(f"✓ Successfully loaded {scaler_file}")
            except Exception as e:
                logger.error(f"❌ Failed to load {scaler_file}: {e}")
                # Try with mmap_mode
                try:
                    scaler = joblib.load(scaler_file, mmap_mode='r')
                    logger.info(f"✓ Successfully loaded {scaler_file} with mmap_mode")
                except Exception as e2:
                    logger.error(f"❌ Failed to load {scaler_file} with mmap_mode: {e2}")
        else:
            logger.warning(f"⚠️  File not found: {scaler_file}")
    
    # Test models
    model_files = [
        'groundnut_best_model.pkl',
        'mustard_best_model.pkl',
        'models/groundnut_best_model.pkl',
        'models/mustard_best_model.pkl'
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                file_size = os.path.getsize(model_file) / (1024 * 1024)  # MB
                logger.info(f"Found {model_file} ({file_size:.2f} MB)")
                
                model = joblib.load(model_file)
                logger.info(f"✓ Successfully loaded {model_file}")
            except Exception as e:
                logger.error(f"❌ Failed to load {model_file}: {e}")
                # Try with mmap_mode
                try:
                    model = joblib.load(model_file, mmap_mode='r')
                    logger.info(f"✓ Successfully loaded {model_file} with mmap_mode")
                except Exception as e2:
                    logger.error(f"❌ Failed to load {model_file} with mmap_mode: {e2}")
        else:
            logger.warning(f"⚠️  File not found: {model_file}")

if __name__ == "__main__":
    test_model_loading() 