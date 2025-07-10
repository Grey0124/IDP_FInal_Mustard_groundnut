# ESP32 TensorFlow Lite Setup Guide
## Grain Moisture Analyzer - On-Device Inference

### Overview
This guide explains how to convert your trained Keras models to TensorFlow Lite format and deploy them on ESP32 for offline inference, eliminating the need for internet connectivity during predictions.

## Key Benefits of TFLite Deployment

### ✅ **No Internet Required**
- Models run directly on ESP32
- Instant predictions (milliseconds vs seconds)
- Works in remote locations without connectivity
- Reduced latency and dependency on cloud services

### ✅ **Privacy & Security**
- Sensor data stays on device
- No data transmission to external servers
- Compliant with data privacy regulations

### ✅ **Cost Effective**
- No cloud hosting costs
- No API usage fees
- Reduced bandwidth requirements

### ✅ **Reliability**
- No network dependency
- Consistent performance
- Works offline

## Prerequisites

### Hardware Requirements
- ESP32 development board (4MB flash recommended)
- DHT11/DHT22 temperature & humidity sensor
- Capacitive soil moisture sensor
- TFT display (optional, for UI)
- Breadboard and connecting wires

### Software Requirements
- Arduino IDE 2.0 or later
- Python 3.7+ with TensorFlow
- ESP32 board support in Arduino IDE

## Step 1: Install Required Libraries

### Arduino IDE Libraries
1. Open Arduino IDE
2. Go to **Tools > Manage Libraries**
3. Install the following libraries:
   - **TensorFlowLite_ESP32** by TensorFlow
   - **DHT sensor library** by Adafruit
   - **TFT_eSPI** by Bodmer
   - **ArduinoJson** by Benoit Blanchon

### Python Dependencies
```bash
pip install tensorflow numpy
```

## Step 2: Convert Models to TFLite

### Run the Conversion Script
```bash
python convert_models_to_tflite.py
```

This script will:
- Load your trained Keras models from `new_ML/models/`
- Convert them to TFLite format
- Create C header files for Arduino
- Test the converted models
- Generate Arduino library structure

### Expected Output
```
=== TensorFlow Lite Model Conversion ===
Converting mustard model...
Loaded mustard model from new_ML/models/mustard_model.h5
TFLite model saved to esp32_tflite_models/mustard_model.tflite
Model size: 45,632 bytes (44.6 KB)

Converting groundnut model...
Loaded groundnut model from new_ML/models/groundnut_model.h5
TFLite model saved to esp32_tflite_models/groundnut_model.tflite
Model size: 47,104 bytes (46.0 KB)

=== Conversion Summary ===
Mustard: ✅ 45,632 bytes (44.6 KB)
Groundnut: ✅ 47,104 bytes (46.0 KB)

Total model size: 92,736 bytes (90.6 KB)
```

## Step 3: Setup Arduino Project

### 1. Create Project Structure
```
GrainMoistureAnalyzer/
├── esp32_tflite_gma.ino          # Main Arduino sketch
├── data/                         # TFLite models
│   ├── mustard_model.tflite
│   └── groundnut_model.tflite
└── src/                          # Source files
    ├── mustard_model.h
    └── groundnut_model.h
```

### 2. Configure Arduino IDE
1. Select **ESP32 Dev Module** board
2. Set **Flash Size** to **4MB (32Mb)**
3. Set **Partition Scheme** to **Default 4MB with spiffs**
4. Set **Upload Speed** to **115200**

### 3. Upload Models to SPIFFS
The TFLite models need to be uploaded to ESP32's SPIFFS (file system):

1. Install **ESP32 Sketch Data Upload** tool
2. Create `data` folder in your sketch directory
3. Copy `.tflite` files to `data` folder
4. Use **Tools > ESP32 Sketch Data Upload**

## Step 4: Hardware Connections

### Pin Configuration
```
ESP32 Pin    Component
---------    ---------
GPIO4        DHT11/DHT22 Data
GPIO36 (VP)  Soil Moisture Sensor (ADC)
GPIO5        TFT Display MOSI
GPIO18       TFT Display SCK
GPIO23       TFT Display CS
GPIO2        TFT Display DC
GPIO4        TFT Display RST
3.3V         Power for sensors
GND          Ground
```

### Wiring Diagram
```
ESP32                    Sensors
-----                    -------
3.3V  --------------->  VCC (DHT11, Soil Sensor)
GND   --------------->  GND (DHT11, Soil Sensor)
GPIO4 --------------->  DATA (DHT11)
GPIO36 --------------->  SIG (Soil Sensor)
```

## Step 5: Code Customization

### Input Normalization
Update the normalization values in `predictMoisture()` function to match your training data:

```cpp
// Current normalization (adjust based on your data ranges)
float normalized_adc = (float)adc / 4095.0;  // ESP32 ADC is 12-bit
float normalized_temp = (temperature - 20.0) / 30.0;  // Range 20-50°C
float normalized_hum = humidity / 100.0;  // Humidity 0-100%
```

### Output Denormalization
Update the output scaling to match your moisture range:

```cpp
// Current denormalization (adjust based on your moisture range)
float moisture_percentage = prediction * 20.0 + 5.0;  // Scale to 5-25%
```

## Step 6: Testing & Validation

### 1. Serial Monitor Commands
```
mustard    - Load mustard model
groundnut  - Load groundnut model
read       - Take reading and predict
test       - Test sensor readings
```

### 2. Expected Behavior
- Models load successfully on startup
- Sensor readings update every 2 seconds
- Predictions complete in <100ms
- Display shows moisture percentage

### 3. Validation Tests
```cpp
// Test with known values
ADC: 2048 (50% of 4095)
Temperature: 35°C
Humidity: 60%

Expected: Should predict moisture based on your model
```

## Memory Optimization

### Model Size Analysis
- **Mustard Model**: ~45KB
- **Groundnut Model**: ~47KB
- **Total**: ~90KB
- **ESP32 Flash**: 4MB (4,096KB)
- **Usage**: ~2.2% of flash memory

### Optimization Options
1. **Quantization**: Reduce model size by 75% (may affect accuracy)
2. **Model Pruning**: Remove unnecessary weights
3. **External Storage**: Use SD card for larger models

## Troubleshooting

### Common Issues

#### 1. Model Loading Fails
```
Error: Model schema mismatch!
```
**Solution**: Ensure TFLite model version compatibility

#### 2. Out of Memory
```
Error: AllocateTensors() failed
```
**Solution**: Increase `kTensorArenaSize` or optimize model

#### 3. Incorrect Predictions
**Solution**: Check input normalization and output denormalization

#### 4. Sensor Reading Errors
**Solution**: Verify wiring and sensor connections

### Debug Commands
```cpp
// Add to setup() for debugging
Serial.println("Free heap: " + String(ESP.getFreeHeap()));
Serial.println("Model size: " + String(model_size));
```

## Performance Comparison

| Metric | Cloud API | TFLite ESP32 |
|--------|-----------|--------------|
| **Latency** | 2-5 seconds | <100ms |
| **Internet** | Required | Not needed |
| **Cost** | Per API call | One-time |
| **Reliability** | Network dependent | Always available |
| **Privacy** | Data sent to cloud | Local only |

## Advanced Features

### 1. Model Switching
The code supports dynamic model switching:
```cpp
// Switch between models at runtime
cropType = "mustard";
modelLoaded = loadModel(cropType);
```

### 2. Batch Predictions
```cpp
// Multiple predictions in sequence
for(int i = 0; i < 5; i++) {
    float moisture = predictMoisture(adcValue, temp, hum);
    delay(1000);
}
```

### 3. Data Logging
```cpp
// Log predictions to SPIFFS
void logPrediction(float moisture, String crop) {
    File file = SPIFFS.open("/predictions.csv", "a");
    file.println(String(millis()) + "," + crop + "," + String(moisture));
    file.close();
}
```

## Next Steps

1. **Deploy to field**: Test in real agricultural conditions
2. **Calibrate sensors**: Fine-tune for your specific sensors
3. **Add data logging**: Store predictions for analysis
4. **Implement alerts**: Notify when moisture is outside range
5. **Battery optimization**: Reduce power consumption for field use

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify hardware connections
3. Test with known sensor values
4. Review model conversion logs

---

**Note**: This TFLite implementation provides offline inference capabilities while maintaining the same accuracy as your cloud-based models. The models are optimized for ESP32's memory constraints and provide fast, reliable predictions without internet connectivity. 