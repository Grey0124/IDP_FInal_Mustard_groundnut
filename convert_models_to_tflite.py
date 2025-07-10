#!/usr/bin/env python3
"""
Convert trained Keras models to TensorFlow Lite format for ESP32 deployment
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json

def convert_model_to_tflite(model_path, output_path, crop_type):
    """
    Convert a Keras model to TensorFlow Lite format
    
    Args:
        model_path: Path to the saved Keras model
        output_path: Path to save the TFLite model
        crop_type: Type of crop (mustard/groundnut) for logging
    """
    print(f"Converting {crop_type} model...")
    
    try:
        # Load the Keras model with custom objects to handle metrics
        custom_objects = {
            'mse': 'mse',  # Use string reference instead of function
            'mae': 'mae'   # Use string reference instead of function
        }
        
        model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        print(f"Loaded {crop_type} model from {model_path}")
        
        # Recompile with standard metrics to avoid conversion issues
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss='mse',
            metrics=['mae']
        )
        
        # Print model summary
        print(f"\n{crop_type.capitalize()} Model Summary:")
        model.summary()
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Optimize for size and speed
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert to quantized model (optional - reduces size but may affect accuracy)
        # converter.representative_dataset = representative_dataset_gen
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.inference_input_type = tf.int8
        # converter.inference_output_type = tf.int8
        
        # Convert the model
        tflite_model = converter.convert()
        
        # Save the TFLite model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Get model size
        model_size = len(tflite_model)
        print(f"TFLite model saved to {output_path}")
        print(f"Model size: {model_size:,} bytes ({model_size/1024:.1f} KB)")
        
        return True, model_size
        
    except Exception as e:
        print(f"Error converting {crop_type} model: {str(e)}")
        return False, 0

def create_header_file(model_path, header_path, model_name):
    """
    Create a C header file containing the TFLite model as a byte array
    
    Args:
        model_path: Path to the TFLite model file
        header_path: Path to save the header file
        model_name: Name for the model variable
    """
    print(f"Creating header file for {model_name}...")
    
    try:
        # Read the TFLite model
        with open(model_path, 'rb') as f:
            model_data = f.read()
        
        # Create header file content
        header_content = f"""#ifndef {model_name.upper()}_MODEL_H
#define {model_name.upper()}_MODEL_H

// Auto-generated header file for {model_name} TFLite model
// Model size: {len(model_data):,} bytes

extern const unsigned char {model_name}_model_tflite[];
extern const unsigned int {model_name}_model_tflite_len;

#endif // {model_name.upper()}_MODEL_H
"""
        
        # Write header file
        with open(header_path, 'w') as f:
            f.write(header_content)
        
        print(f"Header file created: {header_path}")
        
    except Exception as e:
        print(f"Error creating header file: {str(e)}")

def test_tflite_model(tflite_path, crop_type, test_data):
    """
    Test the converted TFLite model with sample data
    
    Args:
        tflite_path: Path to the TFLite model
        crop_type: Type of crop
        test_data: Sample input data for testing
    """
    print(f"\nTesting {crop_type} TFLite model...")
    
    try:
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"Input details: {input_details}")
        print(f"Output details: {output_details}")
        
        # Prepare test data
        test_input = np.array(test_data, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        
        # Run inference
        interpreter.invoke()
        
        # Get prediction
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"Test input: {test_data}")
        print(f"TFLite prediction: {prediction[0][0]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"Error testing TFLite model: {str(e)}")
        return False

def main():
    """Main conversion function"""
    print("=== TensorFlow Lite Model Conversion ===")
    
    # Define paths
    models_dir = "new_ML/models"
    tflite_dir = "esp32_tflite_models"
    
    # Create output directory
    os.makedirs(tflite_dir, exist_ok=True)
    
    # Model paths - Updated to use correct model names
    mustard_model_path = os.path.join(models_dir, "model_mustard_new.h5")
    groundnut_model_path = os.path.join(models_dir, "model_groundnut_new.h5")
    
    # Check if models exist
    if not os.path.exists(mustard_model_path):
        print(f"Error: Mustard model not found at {mustard_model_path}")
        return
    
    if not os.path.exists(groundnut_model_path):
        print(f"Error: Groundnut model not found at {groundnut_model_path}")
        return
    
    # Convert models
    results = {}
    
    # Convert mustard model
    mustard_tflite_path = os.path.join(tflite_dir, "mustard_model.tflite")
    success, size = convert_model_to_tflite(mustard_model_path, mustard_tflite_path, "mustard")
    results["mustard"] = {"success": success, "size": size, "path": mustard_tflite_path}
    
    # Convert groundnut model
    groundnut_tflite_path = os.path.join(tflite_dir, "groundnut_model.tflite")
    success, size = convert_model_to_tflite(groundnut_model_path, groundnut_tflite_path, "groundnut")
    results["groundnut"] = {"success": success, "size": size, "path": groundnut_tflite_path}
    
    # Create header files
    if results["mustard"]["success"]:
        create_header_file(
            mustard_tflite_path,
            os.path.join(tflite_dir, "mustard_model.h"),
            "mustard_model_tflite"
        )
    
    if results["groundnut"]["success"]:
        create_header_file(
            groundnut_tflite_path,
            os.path.join(tflite_dir, "groundnut_model.h"),
            "groundnut_model_tflite"
        )
    
    # Test models with sample data
    print("\n=== Testing Converted Models ===")
    
    # Sample test data (normalized values)
    test_data = np.array([[0.5, 0.3, 0.6]], dtype=np.float32)  # [ADC, temp, humidity]
    
    if results["mustard"]["success"]:
        test_tflite_model(mustard_tflite_path, "mustard", test_data)
    
    if results["groundnut"]["success"]:
        test_tflite_model(groundnut_tflite_path, "groundnut", test_data)
    
    # Copy TFLite models to new_ML/models directory for API access
    print("\n=== Copying Models to API Directory ===")
    try:
        import shutil
        api_models_dir = "new_ML/models"
        os.makedirs(api_models_dir, exist_ok=True)
        
        if results["mustard"]["success"]:
            shutil.copy2(mustard_tflite_path, os.path.join(api_models_dir, "mustard_model.tflite"))
            print(f"✓ Copied mustard model to {api_models_dir}/mustard_model.tflite")
        
        if results["groundnut"]["success"]:
            shutil.copy2(groundnut_tflite_path, os.path.join(api_models_dir, "groundnut_model.tflite"))
            print(f"✓ Copied groundnut model to {api_models_dir}/groundnut_model.tflite")
        
    except Exception as e:
        print(f"Error copying models: {e}")
    
    # Summary
    print("\n=== Conversion Summary ===")
    for crop, result in results.items():
        if result["success"]:
            print(f"✅ {crop.capitalize()}: {result['size']:,} bytes ({result['size']/1024:.1f} KB)")
        else:
            print(f"❌ {crop.capitalize()}: Conversion failed")
    
    print(f"\nTFLite models saved to: {tflite_dir}/")
    print(f"Models copied to: {api_models_dir}/")
    print("\nYou can now use these models with ESP32 MicroPython!")

if __name__ == "__main__":
    main() 