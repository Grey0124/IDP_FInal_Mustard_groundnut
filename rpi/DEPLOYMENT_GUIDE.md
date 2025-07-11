# Raspberry Pi 4B Soil Moisture Monitor Setup

This guide helps you set up your soil moisture monitoring system on a Raspberry Pi 4B with the capacitive soil moisture sensor V2.0.

## What You Need

### Hardware
- Raspberry Pi 4B (2GB, 4GB, or 8GB RAM)
- 16GB+ microSD card (Class 10 recommended)
- 5V/3A power supply
- DHT22 temperature and humidity sensor
- Capacitive soil moisture sensor V2.0
- Breadboard and jumper wires
- Pi case (optional but nice to have)

### Software
- Raspberry Pi OS (latest version)
- SSH access
- VNC for remote desktop (optional)

## Step 1: Get Your Pi Ready

### Install Raspberry Pi OS
1. Download Raspberry Pi Imager from raspberrypi.org
2. Put your microSD card in your computer
3. Open Raspberry Pi Imager
4. Choose "Raspberry Pi OS (64-bit)"
5. Pick your microSD card
6. Click the gear icon and set:
   - Hostname: soil-moisture-pi
   - Enable SSH
   - Username: pi
   - Set a password you'll remember
   - Add your WiFi details if using wireless
7. Click "Write" and wait

### First Boot
1. Put the microSD card in your Pi 4B
2. Connect power and wait for it to boot
3. Connect via SSH or use a monitor(preferrably use monitor because I faced lot of connectivity issues through ssh during first boot)

### Enable What You Need
```bash
# Turn on I2C
sudo raspi-config nonint do_i2c 0

# Turn on SPI
sudo raspi-config nonint do_spi 0

# Turn on SSH
sudo raspi-config nonint do_ssh 0 (ssh can aslo turned on during upload of OS through customization settings)

# Turn on VNC
sudo raspi-config nonint do_vnc 0
```

### Update Everything
```bash
sudo apt update && sudo apt upgrade -y
sudo reboot
```

## Step 2: Connect Your Sensors

### DHT22 Sensor
| DHT22 Pin | Pi 4B Pin | What It Does |
|-----------|-----------|--------------|
| VCC       | 3.3V (Pin 1) | Power       |
| GND       | GND (Pin 6)  | Ground      |
| DATA      | GPIO4 (Pin 7)| Data signal |

### Capacitive Soil Moisture Sensor V2.0
| Sensor Pin | Pi 4B Pin | What It Does |
|------------|-----------|--------------|
| VCC        | 3.3V (Pin 1) | Power       |
| GND        | GND (Pin 6)  | Ground      |
| AOUT       | GPIO17 (Pin 11) | Analog output |

### How to Wire It
```
Raspberry Pi 4B
┌─────────────────┐
│                 │
│ 3.3V ──┬─────── │
│        │        │
│ GND  ──┼─────── │
│        │        │
│ GPIO4 ─┼─────── │
│        │        │
│ GPIO17 ─┼────── │
└────────┼────────┘
         │
         └── Breadboard
             │
             ├── DHT22
             │   ├── VCC
             │   ├── GND
             │   └── DATA
             │
             └── Capacitive Sensor V2.0
                 ├── VCC
                 ├── GND
                 └── AOUT
```

## Step 3: Put the Software On

### Copy Files to Your Pi
```bash
# From your computer, copy the rpi folder
scp -r rpi/ pi@your-pi-ip:/home/pi/

# Copy your model files too (these are in the new_ML models folder in github)
scp new_groundnut_model.h5 pi@your-pi-ip:/home/pi/rpi/
scp new_mustard_model.h5 pi@your-pi-ip:/home/pi/rpi/
scp scaler_groundnut.pkl pi@your-pi-ip:/home/pi/rpi/
scp scaler_mustard.pkl pi@your-pi-ip:/home/pi/rpi/
```

### Run the Setup
```bash
# Connect to your Pi
ssh pi@your-pi-ip

# Go to the rpi folder
cd rpi

# Make the script work
chmod +x setup_rpi.sh

# Run it
./setup_rpi.sh
```

### If Setup Fails
```bash
# Make the folder
mkdir -p /home/pi/soil-moisture-monitor
cd /home/pi/soil-moisture-monitor

# Make Python environment
python3 -m venv venv
source venv/bin/activate

# Install what you need
pip install -r requirements_rpi.txt (First check python 3.11 is installed or not)

# Copy your files
cp main_app_rpi.py .
cp sensor_interface.py .
cp data_logger.py .
```

## Step 4: Test Everything

### Check I2C
```bash
i2cdetect -y 1
```

### Test Your Sensors
```bash
# Turn on the Python environment
source venv/bin/activate

# Test sensors
python test_sensors.py
```

### Test the Web App
```bash
# Start it
python main_app_rpi.py

# Open in browser: http://your-pi-ip:5000
```

### Test Data Logging
```bash
# Start logger
python data_logger.py

# Check logs
tail -f logs/data_logger.log
```

## Step 5: Remote Access

### SSH (Already Working)
```bash
ssh pi@your-pi-ip
```

### VNC for Desktop
```bash
# Install VNC
sudo apt install -y realvnc-vnc-server

# Turn it on
sudo raspi-config nonint do_vnc 0

# Set password
vncpasswd

# Start it
vncserver :1 -geometry 1920x1080 -depth 24
```

### Connect with VNC Viewer
1. Download VNC Viewer from realvnc.com
2. Connect to your-pi-ip:1
3. Use the password you set

## Step 6: Make It Start Automatically

### Turn On Auto-Start
```bash
# Start the service
sudo systemctl start soil-moisture-monitor

# Make it start when Pi boots
sudo systemctl enable soil-moisture-monitor

# Check if it's working
sudo systemctl status soil-moisture-monitor
```

### Control the Service
```bash
# Start it
sudo systemctl start soil-moisture-monitor

# Stop it
sudo systemctl stop soil-moisture-monitor

# Restart it
sudo systemctl restart soil-moisture-monitor

# See logs
sudo journalctl -u soil-moisture-monitor -f

# Check if running
sudo systemctl is-active soil-moisture-monitor
```

## Step 7: Use Your System

### Web Interface
Go to: http://your-pi-ip:5000

You'll see:
- Real-time sensor readings
- Moisture predictions
- System status

### API Commands
```bash
# Get sensor data
curl http://your-pi-ip:5000/api/sensors (i am specifying your-pi-ip because it depends on the name you give to your raspverry pi during upload itself, use proper name)

# Get groundnut moisture
curl http://your-pi-ip:5000/api/predict/groundnut

# Get mustard moisture
curl http://your-pi-ip:5000/api/predict/mustard

# Check system status
curl http://your-pi-ip:5000/api/status
```

### Look at Your Data
```bash
# See recent readings
sqlite3 data/soil_moisture.db "SELECT * FROM sensor_readings ORDER BY timestamp DESC LIMIT 10;"

# Export to CSV
sqlite3 data/soil_moisture.db "SELECT * FROM sensor_readings;" > data_export.csv
```

### Check Logs
```bash
# App logs
tail -f logs/app.log

# Data logger logs
tail -f logs/data_logger.log

# System logs
sudo journalctl -u soil-moisture-monitor -f
```

## Step 8: Fix Problems

### Sensors Not Working
```bash
# Check I2C
i2cdetect -y 1

# Check permissions
groups pi




```

### Models Not Loading
```bash
# Check files
ls -la *.h5 *.pkl

# Check TensorFlow
python -c "import tensorflow as tf; print(tf.__version__)"
```

### Service Won't Start
```bash
# Check logs
sudo journalctl -u soil-moisture-monitor -n 50

# Check port
sudo netstat -tlnp | grep :5000

# Test manually
cd /home/pi/soil-moisture-monitor
source venv/bin/activate
python main_app_rpi.py
```

### Check Performance
```bash
# System resources
htop

# Disk space
df -h

# Memory
free -h

# Temperature
vcgencmd measure_temp
```

## Step 9: Keep It Safe

### Firewall
```bash
# Install firewall
sudo apt install ufw

# Allow SSH and web
sudo ufw allow ssh
sudo ufw allow 5000

# Turn on firewall
sudo ufw enable
```

### Change Password
```bash
passwd
```

### Secure VNC
```bash
# Use SSH tunnel
ssh -L 5901:localhost:5901 pi@your-pi-ip

# Then connect VNC to localhost:5901
```

## Step 10: Backup Your Data

### Backup Data
```bash
# Database
cp data/soil_moisture.db backup/soil_moisture_$(date +%Y%m%d).db

# Logs
tar -czf backup/logs_$(date +%Y%m%d).tar.gz logs/

# Models
cp *.h5 *.pkl backup/
```

### Backup System
```bash
# Full system backup
sudo dd if=/dev/mmcblk0 of=backup/pi_backup_$(date +%Y%m%d).img bs=4M status=progress
```

## Keep It Running

### Regular Checks
- Weekly: Look at logs for errors
- Monthly: Update system
- Quarterly: Clean sensor connections
- Yearly: Replace sensors if needed

### What to Watch
- [ ] Sensors working
- [ ] Web interface accessible
- [ ] Data logging working
- [ ] Predictions make sense
- [ ] Pi has enough resources
- [ ] No errors in logs

### Get Help
If something's not working:
1. Check the logs first
2. Look at this guide
3. Check your connections
4. Make sure software is installed right

---

That's it! Your soil moisture monitor should be working on your Raspberry Pi 4B with the capacitive sensor V2.0. 