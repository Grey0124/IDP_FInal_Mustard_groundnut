#!/bin/bash

# Raspberry Pi 4B Soil Moisture Monitor Setup Script
# This script sets up the complete environment for the soil moisture monitoring system

set -e  # Exit on any error

echo "🌱 Setting up Raspberry Pi 4B Soil Moisture Monitor..."
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "Please do not run this script as root. Use regular user (pi)."
    exit 1
fi

# Update system
print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
print_status "Installing system dependencies..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    i2c-tools \
    python3-dev \
    libatlas-base-dev \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libhdf5-103 \
    libqtgui4 \
    libqtwebkit4 \
    libqt4-test \
    python3-pyqt5 \
    libatlas-base-dev \
    libjasper-dev \
    libqtcore4 \
    libqt4-test

# Enable I2C interface
print_status "Enabling I2C interface..."
sudo raspi-config nonint do_i2c 0

# Enable SPI interface
print_status "Enabling SPI interface..."
sudo raspi-config nonint do_spi 0

# Create application directory
APP_DIR="/home/pi/soil-moisture-monitor"
print_status "Creating application directory: $APP_DIR"
mkdir -p "$APP_DIR"
cd "$APP_DIR"

# Create virtual environment
print_status "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
print_status "Installing Python dependencies..."
pip install -r requirements_rpi.txt

# Copy model files (assuming they're in the parent directory)
print_status "Copying model files..."
if [ -f "../new_groundnut_model.h5" ]; then
    cp ../new_groundnut_model.h5 .
    print_success "Groundnut model copied"
else
    print_warning "Groundnut model not found. Please copy new_groundnut_model.h5 to $APP_DIR"
fi

if [ -f "../new_mustard_model.h5" ]; then
    cp ../new_mustard_model.h5 .
    print_success "Mustard model copied"
else
    print_warning "Mustard model not found. Please copy new_mustard_model.h5 to $APP_DIR"
fi

if [ -f "../scaler_groundnut.pkl" ]; then
    cp ../scaler_groundnut.pkl .
    print_success "Groundnut scaler copied"
else
    print_warning "Groundnut scaler not found. Please copy scaler_groundnut.pkl to $APP_DIR"
fi

if [ -f "../scaler_mustard.pkl" ]; then
    cp ../scaler_mustard.pkl .
    print_success "Mustard scaler copied"
else
    print_warning "Mustard scaler not found. Please copy scaler_mustard.pkl to $APP_DIR"
fi

# Copy application files
print_status "Copying application files..."
cp ../rpi/main_app_rpi.py .
cp ../rpi/sensor_interface.py .

# Set up systemd service
print_status "Setting up systemd service..."
sudo cp ../rpi/soil-moisture-monitor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable soil-moisture-monitor.service

# Create data logging directory
print_status "Creating data logging directory..."
mkdir -p logs
mkdir -p data

# Set proper permissions
print_status "Setting proper permissions..."
chmod +x main_app_rpi.py
chmod +x sensor_interface.py

# Test I2C devices
print_status "Testing I2C devices..."
i2cdetect -y 1

# Create configuration file
print_status "Creating configuration file..."
cat > config.py << EOF
# Configuration file for Soil Moisture Monitor

# Sensor Configuration
DHT_PIN = 4  # GPIO pin for DHT22 sensor
MOISTURE_PIN = 17  # GPIO pin for capacitive moisture sensor

# Application Configuration
HOST = '0.0.0.0'
PORT = 5000
DEBUG = False

# Logging Configuration
LOG_LEVEL = 'INFO'
LOG_FILE = 'logs/app.log'

# Data Storage
DATA_DIR = 'data'
LOG_INTERVAL = 300  # 5 minutes
EOF

# Create startup script
print_status "Creating startup script..."
cat > start_monitor.sh << 'EOF'
#!/bin/bash
cd /home/pi/soil-moisture-monitor
source venv/bin/activate
python main_app_rpi.py
EOF

chmod +x start_monitor.sh

# Create test script
print_status "Creating test script..."
cat > test_sensors.py << 'EOF'
#!/usr/bin/env python3
"""
Test script for sensors
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sensor_interface import SensorInterface
import time

def main():
    print("Testing sensors...")
    try:
        sensor = SensorInterface()
        print("Sensor interface created successfully")
        
        print("\nSensor status:")
        status = sensor.get_sensor_status()
        for sensor_name, is_working in status.items():
            print(f"  {sensor_name}: {'✓' if is_working else '✗'}")
        
        print("\nTaking readings...")
        for i in range(5):
            readings = sensor.read_all_sensors()
            print(f"Reading {i+1}: {readings}")
            time.sleep(2)
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'sensor' in locals():
            sensor.cleanup()

if __name__ == "__main__":
    main()
EOF

chmod +x test_sensors.py

# Create README
print_status "Creating README file..."
cat > README.md << 'EOF'
# Soil Moisture Monitor - Raspberry Pi 4B

## Overview
This system monitors soil moisture using DHT22 (temperature/humidity) and capacitive soil moisture sensor V2.0, and provides predictions using TensorFlow models.

## Hardware Requirements
- Raspberry Pi 4B (2GB, 4GB, or 8GB RAM)
- DHT22 temperature and humidity sensor
- Capacitive soil moisture sensor V2.0
- Breadboard and jumper wires

## Connections
### DHT22 Sensor
- VCC → 3.3V
- GND → GND
- DATA → GPIO4

### Capacitive Soil Moisture Sensor V2.0
- VCC → 3.3V
- GND → GND
- AOUT → GPIO17

## Usage

### Manual Start
```bash
cd /home/pi/soil-moisture-monitor
source venv/bin/activate
python main_app_rpi.py
```

### Service Management
```bash
# Start service
sudo systemctl start soil-moisture-monitor

# Stop service
sudo systemctl stop soil-moisture-monitor

# Check status
sudo systemctl status soil-moisture-monitor

# View logs
sudo journalctl -u soil-moisture-monitor -f
```

### Testing Sensors
```bash
python test_sensors.py
```

## Web Interface
Access the web interface at: http://raspberry-pi-ip:5000

## API Endpoints
- `GET /` - Web interface
- `GET /api/sensors` - Get current sensor readings
- `GET /api/predict/<crop>` - Get moisture prediction for specific crop
- `GET /api/status` - Get system status
- `POST /api/update/<crop>` - Update model with new data

## Troubleshooting
1. Check I2C devices: `i2cdetect -y 1`
2. Check GPIO permissions: `groups pi`
3. View service logs: `sudo journalctl -u soil-moisture-monitor -f`
4. Test sensors manually: `python test_sensors.py`

## Files
- `main_app_rpi.py` - Main Flask application
- `sensor_interface.py` - Sensor interface module
- `config.py` - Configuration file
- `requirements_rpi.txt` - Python dependencies
- `test_sensors.py` - Sensor testing script
EOF

print_success "Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Connect your sensors according to the README"
echo "2. Test sensors: python test_sensors.py"
echo "3. Start the application: python main_app_rpi.py"
echo "4. Access web interface: http://$(hostname -I | awk '{print $1}'):5000"
echo "5. Enable auto-start: sudo systemctl start soil-moisture-monitor"
echo ""
echo "For troubleshooting, see README.md" 