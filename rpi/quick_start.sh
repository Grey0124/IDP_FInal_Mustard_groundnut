#!/bin/bash

# Quick Start Script for Raspberry Pi Soil Moisture Monitor
# This script provides a fast way to get the system running

echo "🚀 Quick Start - Raspberry Pi Soil Moisture Monitor"
echo "=================================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Function to print status
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo; then
    print_error "This script should be run on a Raspberry Pi"
    exit 1
fi

# Check if running as pi user
if [ "$USER" != "pi" ]; then
    print_warning "This script is designed to run as the 'pi' user"
fi

# Create application directory
APP_DIR="/home/pi/soil-moisture-monitor"
print_status "Setting up application directory: $APP_DIR"
mkdir -p "$APP_DIR"
cd "$APP_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Install dependencies if requirements file exists
if [ -f "requirements_rpi.txt" ]; then
    print_status "Installing dependencies..."
    pip install -r requirements_rpi.txt
else
    print_warning "requirements_rpi.txt not found, installing basic dependencies..."
    pip install flask tensorflow numpy pandas scikit-learn joblib RPi.GPIO adafruit-circuitpython-dht adafruit-circuitpython-ads1x15
fi

# Check for model files
print_status "Checking for model files..."
MODEL_FILES=("new_groundnut_model.h5" "new_mustard_model.h5" "scaler_groundnut.pkl" "scaler_mustard.pkl")
MISSING_MODELS=()

for file in "${MODEL_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_MODELS+=("$file")
        print_warning "Missing: $file"
    else
        print_status "Found: $file"
    fi
done

if [ ${#MISSING_MODELS[@]} -gt 0 ]; then
    print_warning "Some model files are missing. Please copy them to $APP_DIR"
    echo "Missing files: ${MISSING_MODELS[*]}"
fi

# Check for application files
print_status "Checking for application files..."
APP_FILES=("main_app_rpi.py" "sensor_interface.py")
MISSING_APPS=()

for file in "${APP_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_APPS+=("$file")
        print_warning "Missing: $file"
    else
        print_status "Found: $file"
    fi
done

if [ ${#MISSING_APPS[@]} -gt 0 ]; then
    print_error "Application files are missing. Please copy them to $APP_DIR"
    echo "Missing files: ${MISSING_APPS[*]}"
    exit 1
fi

# Create logs and data directories
print_status "Creating directories..."
mkdir -p logs data

# Test I2C
print_status "Testing I2C interface..."
if command -v i2cdetect &> /dev/null; then
    i2cdetect -y 1
else
    print_warning "i2cdetect not found. Install with: sudo apt install i2c-tools"
fi

# Test sensors
print_status "Testing sensors..."
if [ -f "test_sensors.py" ]; then
    python test_sensors.py
else
    print_warning "test_sensors.py not found. Creating basic test..."
    cat > test_sensors.py << 'EOF'
#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from sensor_interface import SensorInterface
    import time
    
    print("Testing sensors...")
    sensor = SensorInterface()
    print("Sensor interface created successfully")
    
    print("\nSensor status:")
    status = sensor.get_sensor_status()
    for sensor_name, is_working in status.items():
        print(f"  {sensor_name}: {'✓' if is_working else '✗'}")
    
    print("\nTaking readings...")
    for i in range(3):
        readings = sensor.read_all_sensors()
        print(f"Reading {i+1}: {readings}")
        time.sleep(1)
        
except Exception as e:
    print(f"Error: {e}")
finally:
    if 'sensor' in locals():
        sensor.cleanup()
EOF
    chmod +x test_sensors.py
    python test_sensors.py
fi

# Start web application
print_status "Starting web application..."
print_status "Web interface will be available at: http://$(hostname -I | awk '{print $1}'):5000"
print_status "Press Ctrl+C to stop the application"

# Start the application
python main_app_rpi.py 