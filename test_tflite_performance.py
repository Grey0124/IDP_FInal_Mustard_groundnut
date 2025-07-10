#!/usr/bin/env python3
"""
Test TFLite model performance against original H5 models
"""

import os
import numpy as np
import tensorflow as tf
import time
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

def load_test_data():
    """Load test data from CSV files"""
    print("Loading test data...")
    
    # Load original datasets
    mustard_data = pd.read_csv('Mustard_Data.csv')
    groundnut_data = pd.read_csv('GroundNut_Data.csv')
    
    # Prepare test data (use last 20% of data for testing)
    mustard_test = mustard_data.tail(int(len(mustard_data) * 0.2))
    groundnut_test = groundnut_data.tail(int(len(groundnut_data) * 0.2))
    
    print(f"Mustard test samples: {len(mustard_test)}")
    print(f"Groundnut test samples: {len(groundnut_test)}")
    
    return mustard_test, groundnut_test

def load_models_and_scalers():
    """Load both H5 and TFLite models with scalers"""
    print("Loading models and scalers...")
    
    # Load scalers
    mustard_scaler = joblib.load('new_ML/models/mustard_scaler_new.pkl')
    groundnut_scaler = joblib.load('new_ML/models/groundnut_scaler_new.pkl')
    
    # Load H5 models
    mustard_h5 = tf.keras.models.load_model('new_ML/models/model_mustard_new.h5', compile=False)
    groundnut_h5 = tf.keras.models.load_model('new_ML/models/model_groundnut_new.h5', compile=False)
    
    # Compile H5 models
    mustard_h5.compile(optimizer='adam', loss='mse', metrics=['mae'])
    groundnut_h5.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Load TFLite models
    mustard_tflite = tf.lite.Interpreter(model_path='new_ML/models/mustard_model.tflite')
    groundnut_tflite = tf.lite.Interpreter(model_path='new_ML/models/groundnut_model.tflite')
    
    # Allocate tensors
    mustard_tflite.allocate_tensors()
    groundnut_tflite.allocate_tensors()
    
    return {
        'mustard': {
            'h5': mustard_h5,
            'tflite': mustard_tflite,
            'scaler': mustard_scaler
        },
        'groundnut': {
            'h5': groundnut_h5,
            'tflite': groundnut_tflite,
            'scaler': groundnut_scaler
        }
    }

def predict_h5(model, scaler, X):
    """Make predictions using H5 model"""
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled, verbose=0)
    return predictions.flatten()

def predict_tflite(interpreter, scaler, X):
    """Make predictions using TFLite model"""
    X_scaled = scaler.transform(X)
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    predictions = []
    
    for i in range(len(X_scaled)):
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], X_scaled[i:i+1].astype(np.float32))
        
        # Run inference
        interpreter.invoke()
        
        # Get output tensor
        prediction = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(prediction[0][0])
    
    return np.array(predictions)

def test_model_performance(models, test_data, crop_type):
    """Test performance of both H5 and TFLite models"""
    print(f"\n=== Testing {crop_type.capitalize()} Models ===")
    
    # Prepare features and target
    X = test_data[['ADC', 'Temperature', 'Humidity']].values
    y_true = test_data['Moisture'].values
    
    model_info = models[crop_type]
    
    # Test H5 model
    print("Testing H5 model...")
    start_time = time.time()
    y_pred_h5 = predict_h5(model_info['h5'], model_info['scaler'], X)
    h5_time = time.time() - start_time
    
    # Test TFLite model
    print("Testing TFLite model...")
    start_time = time.time()
    y_pred_tflite = predict_tflite(model_info['tflite'], model_info['scaler'], X)
    tflite_time = time.time() - start_time
    
    # Calculate metrics
    h5_mse = mean_squared_error(y_true, y_pred_h5)
    h5_mae = mean_absolute_error(y_true, y_pred_h5)
    h5_r2 = r2_score(y_true, y_pred_h5)
    
    tflite_mse = mean_squared_error(y_true, y_pred_tflite)
    tflite_mae = mean_absolute_error(y_true, y_pred_tflite)
    tflite_r2 = r2_score(y_true, y_pred_tflite)
    
    # Calculate prediction differences
    prediction_diff = np.abs(y_pred_h5 - y_pred_tflite)
    max_diff = np.max(prediction_diff)
    mean_diff = np.mean(prediction_diff)
    
    # Print results
    print(f"\n{crop_type.capitalize()} Model Performance:")
    print("=" * 50)
    print(f"{'Metric':<15} {'H5 Model':<15} {'TFLite Model':<15} {'Difference':<15}")
    print("-" * 60)
    print(f"{'MSE':<15} {h5_mse:<15.4f} {tflite_mse:<15.4f} {abs(h5_mse-tflite_mse):<15.4f}")
    print(f"{'MAE':<15} {h5_mae:<15.4f} {tflite_mae:<15.4f} {abs(h5_mae-tflite_mae):<15.4f}")
    print(f"{'R²':<15} {h5_r2:<15.4f} {tflite_r2:<15.4f} {abs(h5_r2-tflite_r2):<15.4f}")
    print(f"{'Time (s)':<15} {h5_time:<15.4f} {tflite_time:<15.4f} {abs(h5_time-tflite_time):<15.4f}")
    print("-" * 60)
    print(f"Max prediction difference: {max_diff:.4f}")
    print(f"Mean prediction difference: {mean_diff:.4f}")
    
    # Performance assessment
    print(f"\nPerformance Assessment:")
    if max_diff < 0.1:  # Less than 0.1% difference
        print("✅ Excellent: TFLite model maintains high accuracy")
    elif max_diff < 0.5:  # Less than 0.5% difference
        print("✅ Good: TFLite model maintains good accuracy")
    elif max_diff < 1.0:  # Less than 1% difference
        print("⚠️  Acceptable: TFLite model has acceptable accuracy")
    else:
        print("❌ Poor: TFLite model accuracy significantly degraded")
    
    if tflite_time < h5_time:
        print(f"✅ TFLite is {h5_time/tflite_time:.2f}x faster than H5")
    else:
        print(f"⚠️  TFLite is {tflite_time/h5_time:.2f}x slower than H5")
    
    return {
        'h5': {'mse': h5_mse, 'mae': h5_mae, 'r2': h5_r2, 'time': h5_time},
        'tflite': {'mse': tflite_mse, 'mae': tflite_mae, 'r2': tflite_r2, 'time': tflite_time},
        'max_diff': max_diff,
        'mean_diff': mean_diff
    }

def test_single_predictions(models):
    """Test single predictions for both models"""
    print("\n=== Single Prediction Test ===")
    
    # Test data points
    test_points = [
        {'adc': 2800, 'temp': 25, 'hum': 60, 'crop': 'mustard'},
        {'adc': 3200, 'temp': 30, 'hum': 70, 'crop': 'groundnut'},
        {'adc': 2900, 'temp': 28, 'hum': 65, 'crop': 'mustard'},
        {'adc': 3100, 'temp': 27, 'hum': 68, 'crop': 'groundnut'}
    ]
    
    for i, point in enumerate(test_points, 1):
        print(f"\nTest Point {i}: {point}")
        
        crop = point['crop']
        X = np.array([[point['adc'], point['temp'], point['hum']]])
        
        model_info = models[crop]
        
        # H5 prediction
        y_h5 = predict_h5(model_info['h5'], model_info['scaler'], X)[0]
        
        # TFLite prediction
        y_tflite = predict_tflite(model_info['tflite'], model_info['scaler'], X)[0]
        
        diff = abs(y_h5 - y_tflite)
        
        print(f"  H5 Prediction: {y_h5:.2f}%")
        print(f"  TFLite Prediction: {y_tflite:.2f}%")
        print(f"  Difference: {diff:.4f}%")
        
        if diff < 0.1:
            print("  ✅ Excellent match")
        elif diff < 0.5:
            print("  ✅ Good match")
        else:
            print("  ⚠️  Significant difference")

def main():
    """Main performance test function"""
    print("=== TFLite Model Performance Test ===")
    
    # Load test data
    mustard_test, groundnut_test = load_test_data()
    
    # Load models
    models = load_models_and_scalers()
    
    # Test performance
    mustard_results = test_model_performance(models, mustard_test, 'mustard')
    groundnut_results = test_model_performance(models, groundnut_test, 'groundnut')
    
    # Test single predictions
    test_single_predictions(models)
    
    # Overall assessment
    print("\n=== Overall Assessment ===")
    print("TFLite Model Quality:")
    
    max_diff_mustard = mustard_results['max_diff']
    max_diff_groundnut = groundnut_results['max_diff']
    
    if max_diff_mustard < 0.1 and max_diff_groundnut < 0.1:
        print("✅ EXCELLENT: Both models maintain high accuracy")
    elif max_diff_mustard < 0.5 and max_diff_groundnut < 0.5:
        print("✅ GOOD: Both models maintain good accuracy")
    elif max_diff_mustard < 1.0 and max_diff_groundnut < 1.0:
        print("⚠️  ACCEPTABLE: Models have acceptable accuracy")
    else:
        print("❌ POOR: Model accuracy significantly degraded")
    
    print(f"\nModel sizes:")
    print(f"Mustard TFLite: {os.path.getsize('new_ML/models/mustard_model.tflite')/1024:.1f} KB")
    print(f"Groundnut TFLite: {os.path.getsize('new_ML/models/groundnut_model.tflite')/1024:.1f} KB")
    
    print("\n✅ TFLite models are ready for ESP32 deployment!")

if __name__ == "__main__":
    main() 