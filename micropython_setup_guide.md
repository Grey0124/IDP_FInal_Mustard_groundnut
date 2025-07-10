# MicroPython ESP32 Setup Guide
## Grain Moisture Analyzer with Model Updates

### Overview
This guide explains how to deploy your Grain Moisture Analyzer using MicroPython on ESP32, which enables **dynamic model updates** without reflashing firmware.

## Key Advantages of MicroPython

### ✅ **Dynamic Model Updates**
- Download new models over WiFi
- No firmware reflashing required
- Version control and rollback capability
- Automatic update checking

### ✅ **Flexible Model Management**
- Store multiple model versions
- Switch between models at runtime
- Model metadata tracking
- Over-the-air (OTA) updates

### ✅ **Easier Development**
- Python syntax (familiar and readable)
- Rich ecosystem of libraries
- Faster prototyping and debugging
- Better error handling

### ✅ **File System Access**
- Models stored as files
- Easy backup and restore
- Configuration management
- Data logging capabilities

## Prerequisites

### Hardware Requirements
- ESP32 development board (4MB flash recommended)
- DHT11/DHT22 temperature & humidity sensor
- Capacitive soil moisture sensor
- MicroSD card (optional, for additional storage)
- USB cable for programming

### Software Requirements
- MicroPython firmware for ESP32
- Thonny IDE or similar Python IDE
- Python 3.7+ (for development tools)

## Step 1: Install MicroPython on ESP32

### 1. Download MicroPython Firmware
```bash
# Download latest MicroPython firmware for ESP32
wget https://micropython.org/resources/firmware/esp32-20230426-v1.20.0.bin
```

### 2. Flash MicroPython
```bash
# Using esptool
esptool.py --chip esp32 --port COM3 erase_flash
esptool.py --chip esp32 --port COM3 write_flash -z 0x1000 esp32-20230426-v1.20.0.bin
```

### 3. Verify Installation
Connect to ESP32 via serial and test:
```python
>>> import machine
>>> machine.freq()
>>> import gc
>>> gc.mem_free()
```

## Step 2: Install Required Libraries

### Core Libraries (Built-in)
- `machine` - Hardware access
- `network` - WiFi connectivity
- `urequests` - HTTP requests
- `json` - JSON handling
- `uos` - File system operations

### Additional Libraries
```python
# Install using upip (if internet available)
import upip
upip.install('micropython-urequests')
upip.install('micropython-umqtt.simple')  # For MQTT if needed
```

### TFLite Micro Support
For TensorFlow Lite support, you'll need:
```python
# Install TFLite Micro for MicroPython
# This may require custom compilation
import tflite
```

## Step 3: Setup Project Structure

### File Organization
```
ESP32/
├── main.py                 # Main application
├── gma.py                  # Grain Moisture Analyzer class
├── config.py              # Configuration settings
├── models/                # Model storage directory
│   ├── mustard_model.tflite
│   ├── groundnut_model.tflite
│   └── model_metadata.json
└── lib/                   # Custom libraries
    ├── dht.py            # DHT sensor library
    └── display.py        # Display interface
```

### Upload Files to ESP32
Using Thonny IDE or similar:
1. Connect to ESP32
2. Upload `esp32_micropython_gma.py` as `main.py`
3. Create `models/` directory
4. Upload TFLite models to `models/` directory

## Step 4: Model Update Server Setup

### Add Model Download Endpoints
Add these endpoints to your Flask API:

```python
# Add to your existing Flask app
import os
from flask import send_file

@app.route('/models/<crop_type>/download')
def download_model(crop_type):
    """Download TFLite model for specified crop"""
    if crop_type not in ['mustard', 'groundnut']:
        return jsonify({'error': 'Invalid crop type'}), 400
    
    model_path = f"new_ML/models/{crop_type}_model.tflite"
    
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model not found'}), 404
    
    return send_file(
        model_path,
        as_attachment=True,
        download_name=f"{crop_type}_model.tflite",
        mimetype='application/octet-stream'
    )

@app.route('/models/status')
def model_status():
    """Get model status and version information"""
    models_info = {
        "available_models": {
            "mustard": {
                "available": os.path.exists("new_ML/models/mustard_model.tflite"),
                "version": "1.0",
                "size": os.path.getsize("new_ML/models/mustard_model.tflite") if os.path.exists("new_ML/models/mustard_model.tflite") else 0
            },
            "groundnut": {
                "available": os.path.exists("new_ML/models/groundnut_model.tflite"),
                "version": "1.0",
                "size": os.path.getsize("new_ML/models/groundnut_model.tflite") if os.path.exists("new_ML/models/groundnut_model.tflite") else 0
            }
        },
        "server_version": "1.0",
        "last_updated": time.time()
    }
    
    return jsonify(models_info)
```

## Step 5: Configuration

### Update Configuration
Edit the configuration in `esp32_micropython_gma.py`:

```python
# === CONFIGURATION ===
WIFI_SSID = "YourWiFiSSID"
WIFI_PASSWORD = "YourWiFiPassword"
MODEL_SERVER_URL = "https://your-server-url.com"

# Pin definitions (adjust for your setup)
DHT_PIN = const(4)
SOIL_ADC_PIN = const(36)
LED_PIN = const(2)
```

### Model Update Settings
```python
# Update intervals (in seconds)
UPDATE_CHECK_INTERVAL = 3600  # Check for updates every hour
AUTO_UPDATE = True           # Automatically download updates
BACKUP_MODELS = True         # Keep backup of previous models
```

## Step 6: Testing & Usage

### Basic Commands
```python
# Connect to ESP32 via serial/REPL
>>> import main
>>> gma = main.GrainMoistureAnalyzer()

# Take a reading
>>> gma.take_reading()

# Switch crop type
>>> gma.load_model("mustard")
>>> gma.take_reading()

# Check for updates
>>> gma.check_model_updates()

# Update models
>>> gma.update_models()

# Show status
>>> gma.show_status()
```

### Serial Commands
When running the main loop, send these commands via serial:
```
read      - Take moisture reading
mustard   - Switch to mustard model
groundnut - Switch to groundnut model
update    - Update all models
status    - Show system status
```

## Step 7: Advanced Features

### Automatic Updates
```python
# Enable automatic model updates
gma.auto_update = True
gma.update_check_interval = 3600  # 1 hour
```

### Model Versioning
```python
# Check model versions
metadata = gma.get_model_metadata()
print(f"Mustard version: {metadata['models']['mustard']['version']}")
print(f"Groundnut version: {metadata['models']['groundnut']['version']}")
```

### Data Logging
```python
# Log predictions to file
def log_prediction(gma, moisture, crop):
    timestamp = time.time()
    with open('/predictions.csv', 'a') as f:
        f.write(f"{timestamp},{crop},{moisture}\n")
```

### MQTT Integration
```python
# Send predictions to MQTT broker
from umqtt.simple import MQTTClient

def publish_prediction(client, moisture, crop):
    topic = f"gma/{crop}/moisture"
    message = json.dumps({
        "moisture": moisture,
        "timestamp": time.time()
    })
    client.publish(topic, message)
```

## Step 8: Model Update Workflow

### 1. Train New Model
```python
# Train improved model on your server
python train_model.py --crop mustard --epochs 100
```

### 2. Convert to TFLite
```python
# Convert and optimize
python convert_models_to_tflite.py
```

### 3. Update Server
```python
# Upload new model to server
# Update version in model metadata
```

### 4. ESP32 Updates Automatically
```python
# ESP32 checks for updates and downloads new model
# No firmware reflashing required!
```

## Performance Comparison

| Feature | Arduino C++ | MicroPython |
|---------|-------------|-------------|
| **Model Updates** | ❌ Static | ✅ Dynamic |
| **Development Speed** | Slow | Fast |
| **Memory Usage** | Low | Higher |
| **Execution Speed** | Fast | Slower |
| **Debugging** | Difficult | Easy |
| **File System** | Limited | Full Access |
| **OTA Updates** | ❌ No | ✅ Yes |

## Troubleshooting

### Common Issues

#### 1. Memory Errors
```python
# Free memory before loading model
import gc
gc.collect()
print(f"Free memory: {gc.mem_free()} bytes")
```

#### 2. WiFi Connection Issues
```python
# Reset WiFi connection
gma.wlan.disconnect()
gma.connect_wifi()
```

#### 3. Model Download Fails
```python
# Check server connectivity
import urequests
response = urequests.get("https://your-server.com/health")
print(f"Server status: {response.status_code}")
```

#### 4. File System Full
```python
# Check available space
import uos
stat = uos.statvfs('/')
free_space = stat[0] * stat[3]
print(f"Free space: {free_space} bytes")
```

### Debug Commands
```python
# Enable debug mode
gma.debug = True

# Check system info
gma.show_status()

# Test model loading
gma.load_model("mustard")
gma.test_prediction()
```

## Best Practices

### 1. Model Management
- Keep backup of working models
- Test new models before deployment
- Use version control for models
- Monitor model performance

### 2. Update Strategy
- Check for updates during low-usage periods
- Implement rollback mechanism
- Validate downloaded models
- Log update activities

### 3. Error Handling
- Implement retry mechanisms
- Handle network failures gracefully
- Validate sensor readings
- Monitor system health

### 4. Power Management
- Use deep sleep when possible
- Optimize WiFi usage
- Monitor battery levels
- Implement power-saving modes

## Next Steps

1. **Deploy to field**: Test in real agricultural conditions
2. **Implement monitoring**: Add remote monitoring capabilities
3. **Add alerts**: Notify when moisture is outside range
4. **Data analytics**: Collect and analyze prediction data
5. **Multi-device support**: Scale to multiple ESP32 devices

---

**Note**: MicroPython provides the flexibility for dynamic model updates while maintaining the accuracy and performance of your machine learning models. The ability to update models over-the-air makes this solution ideal for field deployment where manual firmware updates are impractical. 