#!/usr/bin/env python3
"""
Generate C header files containing TFLite model data as byte arrays
"""

import os
import struct

def generate_model_header(model_path, header_path, model_name):
    """
    Generate a C header file containing the TFLite model as a byte array
    
    Args:
        model_path: Path to the TFLite model file
        header_path: Path to save the header file
        model_name: Name for the model variable
    """
    print(f"Generating header file for {model_name}...")
    
    try:
        # Read the TFLite model
        with open(model_path, 'rb') as f:
            model_data = f.read()
        
        # Create header file content
        header_content = f"""#ifndef {model_name.upper()}_MODEL_H
#define {model_name.upper()}_MODEL_H

// Auto-generated header file for {model_name} TFLite model
// Model size: {len(model_data):,} bytes
// Generated automatically - DO NOT EDIT

#include <stdint.h>

// Model data as byte array
extern const unsigned char {model_name}_model_tflite[] asm("_binary_{model_name}_model_tflite_start");

// Model size
extern const unsigned int {model_name}_model_tflite_len asm("{model_name}_model_tflite_size");

// Model metadata
#define {model_name.upper()}_MODEL_SIZE {len(model_data)}
#define {model_name.upper()}_MODEL_NAME "{model_name}"

#endif // {model_name.upper()}_MODEL_H
"""
        
        # Write header file
        with open(header_path, 'w') as f:
            f.write(header_content)
        
        print(f"Header file created: {header_path}")
        print(f"Model size: {len(model_data):,} bytes ({len(model_data)/1024:.1f} KB)")
        
        return True, len(model_data)
        
    except Exception as e:
        print(f"Error generating header file: {str(e)}")
        return False, 0

def create_arduino_data_files(model_path, data_dir):
    """
    Create Arduino data files for SPIFFS upload
    
    Args:
        model_path: Path to the TFLite model
        data_dir: Directory to create data files
    """
    try:
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Copy model to data directory
        import shutil
        filename = os.path.basename(model_path)
        dest_path = os.path.join(data_dir, filename)
        shutil.copy2(model_path, dest_path)
        
        print(f"Data file created: {dest_path}")
        return True
        
    except Exception as e:
        print(f"Error creating data file: {str(e)}")
        return False

def main():
    """Main function to generate all header files"""
    print("=== Generating TFLite Model Headers ===")
    
    # Define paths
    tflite_dir = "esp32_tflite_models"
    arduino_dir = "esp32_tflite_models/Arduino"
    
    # Create directories
    os.makedirs(arduino_dir, exist_ok=True)
    os.makedirs(os.path.join(arduino_dir, "data"), exist_ok=True)
    
    # Model paths
    mustard_tflite = os.path.join(tflite_dir, "mustard_model.tflite")
    groundnut_tflite = os.path.join(tflite_dir, "groundnut_model.tflite")
    
    # Check if TFLite models exist
    if not os.path.exists(mustard_tflite):
        print(f"Error: Mustard TFLite model not found at {mustard_tflite}")
        print("Please run convert_models_to_tflite.py first")
        return
    
    if not os.path.exists(groundnut_tflite):
        print(f"Error: Groundnut TFLite model not found at {groundnut_tflite}")
        print("Please run convert_models_to_tflite.py first")
        return
    
    # Generate header files
    results = {}
    
    # Mustard model
    mustard_header = os.path.join(arduino_dir, "mustard_model.h")
    success, size = generate_model_header(mustard_tflite, mustard_header, "mustard_model_tflite")
    results["mustard"] = {"success": success, "size": size, "header": mustard_header}
    
    # Groundnut model
    groundnut_header = os.path.join(arduino_dir, "groundnut_model.h")
    success, size = generate_model_header(groundnut_tflite, groundnut_header, "groundnut_model_tflite")
    results["groundnut"] = {"success": success, "size": size, "header": groundnut_header}
    
    # Create Arduino data files
    data_dir = os.path.join(arduino_dir, "data")
    create_arduino_data_files(mustard_tflite, data_dir)
    create_arduino_data_files(groundnut_tflite, data_dir)
    
    # Create main header file
    main_header = os.path.join(arduino_dir, "models.h")
    main_header_content = """#ifndef MODELS_H
#define MODELS_H

// Main header file for all TFLite models
// Include this file in your Arduino sketch

#include "mustard_model.h"
#include "groundnut_model.h"

// Model selection helper
enum CropType {
    CROP_MUSTARD,
    CROP_GROUNDNUT
};

// Model information structure
struct ModelInfo {
    const unsigned char* data;
    unsigned int size;
    const char* name;
};

// Model registry
extern const ModelInfo models[];

// Get model by crop type
const ModelInfo* getModel(CropType crop);

#endif // MODELS_H
"""
    
    with open(main_header, 'w') as f:
        f.write(main_header_content)
    
    # Create models.cpp implementation
    models_cpp = os.path.join(arduino_dir, "models.cpp")
    models_cpp_content = """#include "models.h"

// Model registry implementation
const ModelInfo models[] = {
    {mustard_model_tflite, mustard_model_tflite_len, "mustard"},
    {groundnut_model_tflite, groundnut_model_tflite_len, "groundnut"}
};

const ModelInfo* getModel(CropType crop) {
    if (crop == CROP_MUSTARD) {
        return &models[0];
    } else if (crop == CROP_GROUNDNUT) {
        return &models[1];
    }
    return nullptr;
}
"""
    
    with open(models_cpp, 'w') as f:
        f.write(models_cpp_content)
    
    # Create Arduino sketch template
    sketch_template = os.path.join(arduino_dir, "GrainMoistureAnalyzer.ino")
    sketch_content = """#include <WiFi.h>
#include "DHT.h"
#include <TFT_eSPI.h>
#include <ArduinoJson.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "models.h"

// === CONFIGURATION ===
#define DHTPIN 4
#define DHTTYPE DHT11
#define SOIL_ADC 36

// === TFLite Objects ===
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// === Model Configuration ===
constexpr int kTensorArenaSize = 100 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// === Objects ===
DHT dht(DHTPIN, DHTTYPE);
TFT_eSPI tft = TFT_eSPI();

CropType currentCrop = CROP_GROUNDNUT;
bool modelLoaded = false;

void setup() {
    Serial.begin(115200);
    
    // Initialize TFLite
    error_reporter->Report("Initializing TFLite...");
    
    // Load default model
    loadModel(currentCrop);
    
    dht.begin();
    tft.init();
    tft.setRotation(1);
    
    Serial.println("Grain Moisture Analyzer - TFLite Version");
    Serial.println("Type 'mustard' or 'groundnut' to switch models");
    Serial.println("Type 'read' to predict moisture");
}

void loop() {
    if (Serial.available()) {
        String input = Serial.readStringUntil('\\n');
        input.trim();
        
        if (input == "mustard") {
            currentCrop = CROP_MUSTARD;
            loadModel(currentCrop);
        } else if (input == "groundnut") {
            currentCrop = CROP_GROUNDNUT;
            loadModel(currentCrop);
        } else if (input == "read") {
            predictMoisture();
        }
    }
    
    delay(100);
}

bool loadModel(CropType crop) {
    const ModelInfo* model = getModel(crop);
    if (!model) {
        Serial.println("Invalid crop type");
        return false;
    }
    
    // Load model implementation here
    // (Same as in esp32_tflite_gma.ino)
    
    Serial.println("Model loaded: " + String(model->name));
    return true;
}

void predictMoisture() {
    // Read sensors and predict
    // (Implementation from esp32_tflite_gma.ino)
}
"""
    
    with open(sketch_template, 'w') as f:
        f.write(sketch_content)
    
    # Summary
    print("\n=== Header Generation Summary ===")
    total_size = 0
    for crop, result in results.items():
        if result["success"]:
            print(f"{crop.capitalize()}: ✅ {result['size']:,} bytes ({result['size']/1024:.1f} KB)")
            total_size += result["size"]
        else:
            print(f"{crop.capitalize()}: ❌ Failed")
    
    print(f"\nTotal model size: {total_size:,} bytes ({total_size/1024:.1f} KB)")
    print(f"\nFiles created in: {arduino_dir}")
    print("- mustard_model.h")
    print("- groundnut_model.h")
    print("- models.h")
    print("- models.cpp")
    print("- GrainMoistureAnalyzer.ino (template)")
    print("- data/ (for SPIFFS upload)")
    
    print("\n=== Next Steps ===")
    print("1. Copy the Arduino folder contents to your Arduino project")
    print("2. Include 'models.h' in your main sketch")
    print("3. Upload data files to ESP32 SPIFFS")
    print("4. Compile and upload the sketch")

if __name__ == "__main__":
    main() 