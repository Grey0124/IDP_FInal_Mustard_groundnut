# Raspberry Pi Soil Moisture Monitor

This folder contains all the necessary files to deploy the soil moisture monitoring system on a Raspberry Pi with proper sensor connections and remote access.

## 📁 Files Overview

### Core Application Files
- **`main_app_rpi.py`** - Main Flask web application with real-time sensor readings and predictions
- **`sensor_interface.py`** - Hardware interface for DHT22 and ADS1115 sensors
- **`data_logger.py`** - Continuous data logging to SQLite database
- **`requirements_rpi.txt`** - Python dependencies optimized for Raspberry Pi

### Setup and Deployment
- **`setup_rpi.sh`** - Complete automated setup script
- **`quick_start.sh`** - Quick deployment for immediate testing
- **`soil-moisture-monitor.service`** - Systemd service for auto-start
- **`DEPLOYMENT_GUIDE.md`** - Comprehensive deployment instructions

## 🚀 Quick Start

### 1. Transfer Files to Raspberry Pi
```bash
# From your development machine
scp -r rpi/ pi@raspberry-pi-ip:/home/pi/
```

### 2. Copy Model Files
```bash
# Copy your trained models to the rpi folder
scp new_groundnut_model.h5 pi@raspberry-pi-ip:/home/pi/rpi/
scp new_mustard_model.h5 pi@raspberry-pi-ip:/home/pi/rpi/
scp scaler_groundnut.pkl pi@raspberry-pi-ip:/home/pi/rpi/
scp scaler_mustard.pkl pi@raspberry-pi-ip:/home/pi/rpi/
```

### 3. Run Quick Start
```bash
# SSH into Raspberry Pi
ssh pi@raspberry-pi-ip

# Navigate to rpi folder
cd rpi

# Make scripts executable
chmod +x *.sh

# Run quick start
./quick_start.sh
```

## 🔧 Hardware Connections

### DHT22 Temperature & Humidity Sensor
| Pin | Raspberry Pi | Description |
|-----|--------------|-------------|
| VCC | 3.3V (Pin 1) | Power       |
| GND | GND (Pin 6)  | Ground      |
| DATA| GPIO4 (Pin 7)| Data signal |

### ADS1115 ADC for Moisture Sensor
| Pin | Raspberry Pi | Description |
|-----|--------------|-------------|
| VDD | 3.3V (Pin 1) | Power       |
| GND | GND (Pin 6)  | Ground      |
| SCL | GPIO3 (Pin 5)| I2C Clock   |
| SDA | GPIO2 (Pin 3)| I2C Data    |
| A0  | Soil sensor  | Analog input|

## 🌐 Web Interface

Once running, access the web interface at:
```
http://raspberry-pi-ip:5000
```

Features:
- Real-time sensor readings
- Moisture predictions for groundnut and mustard
- System status monitoring
- Auto-refresh every 30 seconds

## 📊 API Endpoints

### Get Sensor Readings
```bash
curl http://raspberry-pi-ip:5000/api/sensors
```

### Get Moisture Prediction
```bash
# For groundnut
curl http://raspberry-pi-ip:5000/api/predict/groundnut

# For mustard
curl http://raspberry-pi-ip:5000/api/predict/mustard
```

### Get System Status
```bash
curl http://raspberry-pi-ip:5000/api/status
```

### Update Model
```bash
curl -X POST http://raspberry-pi-ip:5000/api/update/groundnut \
  -H "Content-Type: application/json" \
  -d '{"adc": 1234, "temp": 25.5, "hum": 60.2, "moisture": 15.8}'
```

## 🔄 Service Management

### Enable Auto-Start
```bash
# Copy service file
sudo cp soil-moisture-monitor.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start service
sudo systemctl enable soil-moisture-monitor
sudo systemctl start soil-moisture-monitor
```

### Service Commands
```bash
# Check status
sudo systemctl status soil-moisture-monitor

# View logs
sudo journalctl -u soil-moisture-monitor -f

# Restart service
sudo systemctl restart soil-moisture-monitor

# Stop service
sudo systemctl stop soil-moisture-monitor
```

## 📈 Data Logging

The system automatically logs all sensor readings and predictions to a SQLite database:

### View Recent Data
```bash
sqlite3 data/soil_moisture.db "SELECT * FROM sensor_readings ORDER BY timestamp DESC LIMIT 10;"
```

### Export Data
```bash
sqlite3 data/soil_moisture.db "SELECT * FROM sensor_readings;" > data_export.csv
```

### Start Data Logger
```bash
python data_logger.py
```

## 🔍 Testing

### Test Sensors
```bash
python test_sensors.py
```

### Test I2C Devices
```bash
i2cdetect -y 1
```

### Test Web Application
```bash
python main_app_rpi.py
```

## 🛠️ Troubleshooting

### Common Issues

1. **Sensors Not Detected**
   ```bash
   # Check I2C devices
   i2cdetect -y 1
   
   # Check GPIO permissions
   groups pi
   ```

2. **Permission Errors**
   ```bash
   # Add user to required groups
   sudo usermod -a -G gpio,i2c,spi pi
   sudo reboot
   ```

3. **Service Not Starting**
   ```bash
   # Check logs
   sudo journalctl -u soil-moisture-monitor -n 50
   
   # Test manual start
   cd /home/pi/soil-moisture-monitor
   source venv/bin/activate
   python main_app_rpi.py
   ```

4. **Model Loading Errors**
   ```bash
   # Check if model files exist
   ls -la *.h5 *.pkl
   
   # Check TensorFlow installation
   python -c "import tensorflow as tf; print(tf.__version__)"
   ```

## 📋 Requirements

### Hardware
- Raspberry Pi (3B+, 4B, or newer)
- DHT22 temperature and humidity sensor
- ADS1115 16-bit ADC module
- Soil moisture sensor (analog output)
- Breadboard and jumper wires

### Software
- Raspberry Pi OS (Bullseye or newer)
- Python 3.7+
- TensorFlow 2.10+
- Required Python packages (see requirements_rpi.txt)

## 🔐 Security

### Firewall Setup
```bash
# Install UFW
sudo apt install ufw

# Allow SSH and web interface
sudo ufw allow ssh
sudo ufw allow 5000

# Enable firewall
sudo ufw enable
```

### Change Default Password
```bash
passwd
```

## 📚 Documentation

- **`DEPLOYMENT_GUIDE.md`** - Complete deployment instructions
- **Hardware Setup** - Detailed connection diagrams
- **API Documentation** - All available endpoints
- **Troubleshooting** - Common issues and solutions

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the deployment guide
3. Check hardware connections
4. Verify software installation
5. Check system logs

## 📝 Notes

- The system uses full TensorFlow (not TensorFlow Lite) for better accuracy
- All sensor readings are logged to SQLite database
- Web interface auto-refreshes every 30 seconds
- System can run in sensor-less mode for testing
- Data logger runs independently of web application

---

**Happy Monitoring! 🌱** 