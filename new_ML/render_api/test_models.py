#!/usr/bin/env python3
"""
Test script to debug model loading and prediction issues
"""

import joblib
import os
import sys

def test_model_loading():
    """Test if models can be loaded properly"""
    print("🔍 Testing Model Loading...")
    
    # Check if model files exist (now directly in current directory)
    groundnut_path = "groundnut_best_model.pkl"
    mustard_path = "mustard_best_model.pkl"
    
    print(f"Groundnut model exists: {os.path.exists(groundnut_path)}")
    print(f"Mustard model exists: {os.path.exists(mustard_path)}")
    
    if os.path.exists(groundnut_path):
        print(f"Groundnut model size: {os.path.getsize(groundnut_path)} bytes")
    if os.path.exists(mustard_path):
        print(f"Mustard model size: {os.path.getsize(mustard_path)} bytes")
    
    # Try to load models
    try:
        groundnut_model = joblib.load(groundnut_path)
        print("✅ Groundnut model loaded successfully")
        print(f"   Model type: {type(groundnut_model)}")
        
        # Test prediction
        features = {
            'ADC': 2920,
            'Temperature': 30.5,
            'Humidity': 53.0
        }
        
        prediction = groundnut_model.predict_one(features)
        print(f"   Test prediction: {prediction}")
        
    except Exception as e:
        print(f"❌ Error loading groundnut model: {e}")
    
    try:
        mustard_model = joblib.load(mustard_path)
        print("✅ Mustard model loaded successfully")
        print(f"   Model type: {type(mustard_model)}")
        
        # Test prediction
        features = {
            'ADC': 2850,
            'Temperature': 31.0,
            'Humidity': 58.0
        }
        
        prediction = mustard_model.predict_one(features)
        print(f"   Test prediction: {prediction}")
        
    except Exception as e:
        print(f"❌ Error loading mustard model: {e}")

def test_with_original_models():
    """Test with models from the main models directory"""
    print("\n🔍 Testing with Original Models...")
    
    # Try to load from main models directory
    original_groundnut_path = "../models/groundnut_best_model.pkl"
    original_mustard_path = "../models/mustard_best_model.pkl"
    
    print(f"Original groundnut model exists: {os.path.exists(original_groundnut_path)}")
    print(f"Original mustard model exists: {os.path.exists(original_mustard_path)}")
    
    if os.path.exists(original_groundnut_path):
        print(f"Original groundnut model size: {os.path.getsize(original_groundnut_path)} bytes")
    if os.path.exists(original_mustard_path):
        print(f"Original mustard model size: {os.path.getsize(original_mustard_path)} bytes")
    
    # Try to load original models
    try:
        groundnut_model = joblib.load(original_groundnut_path)
        print("✅ Original groundnut model loaded successfully")
        
        # Test prediction
        features = {
            'ADC': 2920,
            'Temperature': 30.5,
            'Humidity': 53.0
        }
        
        prediction = groundnut_model.predict_one(features)
        print(f"   Test prediction: {prediction}")
        
    except Exception as e:
        print(f"❌ Error loading original groundnut model: {e}")

def check_model_training():
    """Check if models need to be retrained"""
    print("\n🔍 Checking Model Training Status...")
    
    # Check if we have the training data
    training_files = [
        "../Groundnut_merged_data.csv",
        "../Mustard_merged_data.csv"
    ]
    
    for file_path in training_files:
        if os.path.exists(file_path):
            print(f"✅ Training data exists: {file_path}")
        else:
            print(f"❌ Training data missing: {file_path}")

if __name__ == "__main__":
    print("🌾 Model Debug Test")
    print("=" * 50)
    
    test_model_loading()
    test_with_original_models()
    check_model_training()
    
    print("\n" + "=" * 50)
    print("💡 If models are returning 0.0, they might need to be retrained.")
    print("   Run the training scripts again to ensure proper model training.")
    print("=" * 50) 