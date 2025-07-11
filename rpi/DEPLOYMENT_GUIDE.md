# Raspberry Pi Soil Moisture Monitor - Deployment Guide

## Overview
This guide will help you deploy the soil moisture monitoring system on a Raspberry Pi with proper sensor connections and remote access via SSH and VNC.

## Prerequisites

### Hardware Requirements
- **Raspberry Pi** (3B+, 4B, or newer recommended)
- **MicroSD Card** (16GB or larger, Class 10 recommended)
- **Power Supply** (5V/3A for Pi 4, 5V/2.5A for Pi 3)
- **DHT22** temperature and humidity sensor
- **ADS1115** 16-bit ADC module
- **Soil moisture sensor** (analog output)
- **Breadboard** and jumper wires
- **Case** for Raspberry Pi (optional but recommended)

### Software Requirements
- **Raspberry Pi OS** (Bullseye or newer)
- **SSH** enabled
- **VNC Server** (for remote desktop access)

## Step 1: Raspberry Pi Setup

### 1.1 Install Raspberry Pi OS
1. Download Raspberry Pi Imager from [raspberrypi.org](https://www.raspberrypi.org/software/)
2. Insert microSD card into your computer
3. Open Raspberry Pi Imager
4. Choose "Raspberry Pi OS (32-bit)" or "Raspberry Pi OS (64-bit)"
5. Select your microSD card
6. Click the gear icon to configure:
   - Set hostname: `soil-moisture-pi`
   - Enable SSH
   - Set username: `pi`
   - Set password
   - Configure wireless LAN (if using WiFi)
7. Click "Write" and wait for completion

### 1.2 Initial Boot and Configuration
1. Insert microSD card into Raspberry Pi
2. Connect power supply and boot
3. Wait for first boot to complete
4. Connect via SSH or directly with monitor/keyboard

### 1.3 Enable Required Interfaces
```bash
# Enable I2C
sudo raspi-config nonint do_i2c 0

# Enable SPI (if needed)
sudo raspi-config nonint do_spi 0

# Enable SSH (if not already enabled)
sudo raspi-config nonint do_ssh 0

# Enable VNC
sudo raspi-config nonint do_vnc 0
```

### 1.4 Update System
```bash
sudo apt update && sudo apt upgrade -y
sudo reboot
```

## Step 2: Hardware Connections

### 2.1 DHT22 Sensor Connections
| DHT22 Pin | Raspberry Pi Pin | Description |
|-----------|------------------|-------------|
| VCC       | 3.3V (Pin 1)     | Power       |
| GND       | GND (Pin 6)      | Ground      |
| DATA      | GPIO4 (Pin 7)    | Data signal |

### 2.2 ADS1115 ADC Connections
| ADS1115 Pin | Raspberry Pi Pin | Description |
|-------------|------------------|-------------|
| VDD         | 3.3V (Pin 1)     | Power       |
| GND         | GND (Pin 6)      | Ground      |
| SCL         | GPIO3 (Pin 5)    | I2C Clock   |
| SDA         | GPIO2 (Pin 3)    | I2C Data    |
| A0          | Soil sensor      | Analog input|

### 2.3 Soil Moisture Sensor
- Connect the analog output of your soil moisture sensor to A0 of ADS1115
- Connect power and ground as per sensor specifications

### 2.4 Connection Diagram
```
Raspberry Pi
┌─────────────────┐
│                 │
│ 3.3V ──┬─────── │
│        │        │
│ GND  ──┼─────── │
│        │        │
│ GPIO4 ─┼─────── │
│        │        │
│ GPIO3 ─┼─────── │
│        │        │
│ GPIO2 ─┼─────── │
└────────┼────────┘
         │
         └── Breadboard
             │
             ├── DHT22
             │   ├── VCC
             │   ├── GND
             │   └── DATA
             │
             └── ADS1115
                 ├── VDD
                 ├── GND
                 ├── SCL
                 ├── SDA
                 └── A0 ── Soil Sensor
```

## Step 3: Software Deployment

### 3.1 Transfer Files to Raspberry Pi
```bash
# From your development machine, copy the rpi folder
scp -r rpi/ pi@raspberry-pi-ip:/home/pi/

# Or use SCP to copy individual files
scp rpi/* pi@raspberry-pi-ip:/home/pi/soil-moisture-monitor/
```

### 3.2 Run Setup Script
```bash
# SSH into Raspberry Pi
ssh pi@raspberry-pi-ip

# Navigate to the rpi directory
cd rpi

# Make setup script executable
chmod +x setup_rpi.sh

# Run setup script
./setup_rpi.sh
```

### 3.3 Manual Setup (Alternative)
If the setup script fails, follow these manual steps:

```bash
# Create application directory
mkdir -p /home/pi/soil-moisture-monitor
cd /home/pi/soil-moisture-monitor

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements_rpi.txt

# Copy model files (from your development machine)
# new_groundnut_model.h5
# new_mustard_model.h5
# scaler_groundnut.pkl
# scaler_mustard.pkl

# Copy application files
cp main_app_rpi.py .
cp sensor_interface.py .
cp data_logger.py .
```

## Step 4: Testing and Verification

### 4.1 Test I2C Devices
```bash
# Check if I2C devices are detected
i2cdetect -y 1

# Expected output should show ADS1115 at address 0x48
```

### 4.2 Test Sensors
```bash
# Activate virtual environment
source venv/bin/activate

# Test sensors
python test_sensors.py
```

### 4.3 Test Web Application
```bash
# Start the web application
python main_app_rpi.py

# Access from browser: http://raspberry-pi-ip:5000
```

### 4.4 Test Data Logger
```bash
# Start data logger
python data_logger.py

# Check logs
tail -f logs/data_logger.log
```

## Step 5: Remote Access Setup

### 5.1 SSH Access
SSH should already be enabled. Connect using:
```bash
ssh pi@raspberry-pi-ip
```

### 5.2 VNC Server Setup
```bash
# Install VNC server
sudo apt install -y realvnc-vnc-server

# Enable VNC
sudo raspi-config nonint do_vnc 0

# Set VNC password
vncpasswd

# Start VNC server
vncserver :1 -geometry 1920x1080 -depth 24
```

### 5.3 VNC Client Connection
1. Download VNC Viewer from [realvnc.com](https://www.realvnc.com/en/connect/download/viewer/)
2. Connect to `raspberry-pi-ip:1`
3. Enter the VNC password you set

## Step 6: Service Management

### 6.1 Enable Auto-Start
```bash
# Start the service
sudo systemctl start soil-moisture-monitor

# Enable auto-start on boot
sudo systemctl enable soil-moisture-monitor

# Check status
sudo systemctl status soil-moisture-monitor
```

### 6.2 Service Commands
```bash
# Start service
sudo systemctl start soil-moisture-monitor

# Stop service
sudo systemctl stop soil-moisture-monitor

# Restart service
sudo systemctl restart soil-moisture-monitor

# View logs
sudo journalctl -u soil-moisture-monitor -f

# Check if service is running
sudo systemctl is-active soil-moisture-monitor
```

## Step 7: Monitoring and Maintenance

### 7.1 Web Interface
- Access: `http://raspberry-pi-ip:5000`
- Real-time sensor readings
- Moisture predictions
- System status

### 7.2 API Endpoints
```bash
# Get sensor readings
curl http://raspberry-pi-ip:5000/api/sensors

# Get moisture prediction for groundnut
curl http://raspberry-pi-ip:5000/api/predict/groundnut

# Get system status
curl http://raspberry-pi-ip:5000/api/status
```

### 7.3 Data Access
```bash
# View recent data
sqlite3 data/soil_moisture.db "SELECT * FROM sensor_readings ORDER BY timestamp DESC LIMIT 10;"

# Export data to CSV
sqlite3 data/soil_moisture.db "SELECT * FROM sensor_readings;" > data_export.csv
```

### 7.4 Log Monitoring
```bash
# Application logs
tail -f logs/app.log

# Data logger logs
tail -f logs/data_logger.log

# System logs
sudo journalctl -u soil-moisture-monitor -f
```

## Step 8: Troubleshooting

### 8.1 Common Issues

#### Sensors Not Detected
```bash
# Check I2C devices
i2cdetect -y 1

# Check GPIO permissions
groups pi

# Test individual sensors
python -c "import board; import busio; i2c = busio.I2C(board.SCL, board.SDA); print('I2C OK')"
```

#### Permission Errors
```bash
# Add user to required groups
sudo usermod -a -G gpio,i2c,spi pi

# Reboot to apply changes
sudo reboot
```

#### Model Loading Errors
```bash
# Check if model files exist
ls -la *.h5 *.pkl

# Check TensorFlow installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

#### Service Not Starting
```bash
# Check service logs
sudo journalctl -u soil-moisture-monitor -n 50

# Check if port is in use
sudo netstat -tlnp | grep :5000

# Test manual start
cd /home/pi/soil-moisture-monitor
source venv/bin/activate
python main_app_rpi.py
```

### 8.2 Performance Optimization
```bash
# Monitor system resources
htop

# Check disk usage
df -h

# Check memory usage
free -h

# Monitor temperature
vcgencmd measure_temp
```

## Step 9: Security Considerations

### 9.1 Firewall Setup
```bash
# Install UFW
sudo apt install ufw

# Allow SSH
sudo ufw allow ssh

# Allow web interface
sudo ufw allow 5000

# Enable firewall
sudo ufw enable
```

### 9.2 Change Default Password
```bash
# Change pi user password
passwd

# Or create new user
sudo adduser newuser
sudo usermod -a -G sudo newuser
```

### 9.3 Secure VNC
```bash
# Use SSH tunnel for VNC
ssh -L 5901:localhost:5901 pi@raspberry-pi-ip

# Then connect VNC to localhost:5901
```

## Step 10: Backup and Recovery

### 10.1 Data Backup
```bash
# Backup database
cp data/soil_moisture.db backup/soil_moisture_$(date +%Y%m%d).db

# Backup logs
tar -czf backup/logs_$(date +%Y%m%d).tar.gz logs/

# Backup models
cp *.h5 *.pkl backup/
```

### 10.2 System Backup
```bash
# Create system image
sudo dd if=/dev/mmcblk0 of=backup/pi_backup_$(date +%Y%m%d).img bs=4M status=progress
```

## Support and Maintenance

### Regular Maintenance Tasks
1. **Weekly**: Check logs for errors
2. **Monthly**: Update system packages
3. **Quarterly**: Clean sensor connections
4. **Annually**: Replace sensors if needed

### Monitoring Checklist
- [ ] Sensors responding correctly
- [ ] Web interface accessible
- [ ] Data logging working
- [ ] Predictions reasonable
- [ ] System resources adequate
- [ ] Logs free of errors

### Contact Information
For technical support or questions:
- Check the logs first
- Review this deployment guide
- Check hardware connections
- Verify software installation

---

**Note**: This deployment guide assumes you have basic familiarity with Raspberry Pi, Linux command line, and Python. If you encounter issues, start with the troubleshooting section and work through the common problems listed there. 