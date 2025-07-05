#!/usr/bin/env python3
"""
Web Dashboard for Moisture Meter
Flask-based real-time monitoring interface
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import threading
import time
import json
from datetime import datetime
from pathlib import Path
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from phase4_raspberry_pi_deployment import MoistureMeterDeployment

app = Flask(__name__)
app.config['SECRET_KEY'] = 'moisture_meter_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
moisture_meter = None
monitoring_active = False
latest_reading = None
readings_history = []

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    """Get current system status"""
    global moisture_meter, latest_reading
    
    status = {
        'monitoring_active': monitoring_active,
        'sensors_available': moisture_meter.sensors_initialized if moisture_meter else False,
        'models_loaded': {
            'groundnut': moisture_meter.groundnut_model is not None if moisture_meter else False,
            'mustard': moisture_meter.mustard_model is not None if moisture_meter else False
        },
        'latest_reading': latest_reading,
        'total_readings': len(readings_history),
        'uptime': get_uptime()
    }
    
    return jsonify(status)

@app.route('/api/start_monitoring', methods=['POST'])
def start_monitoring():
    """Start continuous monitoring"""
    global monitoring_active
    
    if not monitoring_active:
        monitoring_active = True
        thread = threading.Thread(target=monitoring_thread)
        thread.daemon = True
        thread.start()
        
        return jsonify({'status': 'success', 'message': 'Monitoring started'})
    else:
        return jsonify({'status': 'error', 'message': 'Monitoring already active'})

@app.route('/api/stop_monitoring', methods=['POST'])
def stop_monitoring():
    """Stop continuous monitoring"""
    global monitoring_active
    monitoring_active = False
    return jsonify({'status': 'success', 'message': 'Monitoring stopped'})

@app.route('/api/single_reading', methods=['POST'])
def get_single_reading():
    """Get a single moisture reading"""
    global moisture_meter
    
    if not moisture_meter:
        return jsonify({'status': 'error', 'message': 'Moisture meter not initialized'})
    
    try:
        # Read sensors
        sensor_data = moisture_meter.read_sensors()
        
        # Make prediction
        prediction_result = moisture_meter.predict_moisture(
            sensor_data['adc'],
            sensor_data['temperature'],
            sensor_data['humidity']
        )
        
        if prediction_result:
            reading = {
                'timestamp': sensor_data['timestamp'].isoformat(),
                'adc': sensor_data['adc'],
                'temperature': round(sensor_data['temperature'], 1),
                'humidity': round(sensor_data['humidity'], 1),
                'moisture': round(prediction_result['moisture'], 2),
                'crop_type': prediction_result['crop_type'],
                'model_used': prediction_result['model_used'],
                'confidence': prediction_result['confidence']
            }
            
            # Add to history
            readings_history.append(reading)
            if len(readings_history) > 100:  # Keep last 100 readings
                readings_history.pop(0)
            
            return jsonify({'status': 'success', 'reading': reading})
        else:
            return jsonify({'status': 'error', 'message': 'Prediction failed'})
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/history')
def get_history():
    """Get reading history"""
    return jsonify({'readings': readings_history})

@app.route('/api/export_data')
def export_data():
    """Export data as CSV"""
    if not readings_history:
        return jsonify({'status': 'error', 'message': 'No data to export'})
    
    try:
        import pandas as pd
        
        df = pd.DataFrame(readings_history)
        filename = f"moisture_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = Path("logs") / filename
        filepath.parent.mkdir(exist_ok=True)
        
        df.to_csv(filepath, index=False)
        
        return jsonify({
            'status': 'success', 
            'filename': filename,
            'filepath': str(filepath)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

def monitoring_thread():
    """Background thread for continuous monitoring"""
    global moisture_meter, monitoring_active, latest_reading, readings_history
    
    while monitoring_active:
        try:
            # Read sensors
            sensor_data = moisture_meter.read_sensors()
            
            # Make prediction
            prediction_result = moisture_meter.predict_moisture(
                sensor_data['adc'],
                sensor_data['temperature'],
                sensor_data['humidity']
            )
            
            if prediction_result:
                reading = {
                    'timestamp': sensor_data['timestamp'].isoformat(),
                    'adc': sensor_data['adc'],
                    'temperature': round(sensor_data['temperature'], 1),
                    'humidity': round(sensor_data['humidity'], 1),
                    'moisture': round(prediction_result['moisture'], 2),
                    'crop_type': prediction_result['crop_type'],
                    'model_used': prediction_result['model_used'],
                    'confidence': prediction_result['confidence']
                }
                
                latest_reading = reading
                readings_history.append(reading)
                
                # Keep only last 100 readings
                if len(readings_history) > 100:
                    readings_history.pop(0)
                
                # Emit to web clients
                socketio.emit('new_reading', reading)
                
                # Log to file
                moisture_meter.log_prediction(sensor_data, prediction_result)
            
            time.sleep(5)  # 5-second interval
            
        except Exception as e:
            print(f"Error in monitoring thread: {e}")
            time.sleep(5)

def get_uptime():
    """Get system uptime"""
    try:
        with open('/proc/uptime', 'r') as f:
            uptime_seconds = float(f.readline().split()[0])
        
        hours = int(uptime_seconds // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        
        return f"{hours}h {minutes}m"
    except:
        return "Unknown"

def initialize_moisture_meter():
    """Initialize the moisture meter system"""
    global moisture_meter
    
    try:
        moisture_meter = MoistureMeterDeployment()
        print("✓ Moisture meter initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Error initializing moisture meter: {e}")
        return False

if __name__ == '__main__':
    print("🌾 Moisture Meter Web Dashboard")
    print("="*40)
    
    # Initialize moisture meter
    if not initialize_moisture_meter():
        print("Failed to initialize moisture meter. Exiting.")
        sys.exit(1)
    
    print("🚀 Starting web dashboard...")
    print("📱 Access the dashboard at: http://raspberrypi.local:5000")
    print("🌐 Or use your Pi's IP address: http://[YOUR_PI_IP]:5000")
    
    # Create templates directory and HTML file
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    # Create the HTML template
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌾 Moisture Meter Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            color: white;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .status-bar {
            background: rgba(255,255,255,0.95);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .status-item {
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 4px solid #007bff;
        }
        
        .status-item h3 {
            color: #007bff;
            margin-bottom: 5px;
        }
        
        .status-item .value {
            font-size: 1.5rem;
            font-weight: bold;
        }
        
        .controls {
            display: flex;
            gap: 15px;
            justify-content: center;
            flex-wrap: wrap;
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .btn-primary {
            background: #007bff;
            color: white;
        }
        
        .btn-primary:hover {
            background: #0056b3;
            transform: translateY(-2px);
        }
        
        .btn-success {
            background: #28a745;
            color: white;
        }
        
        .btn-success:hover {
            background: #1e7e34;
            transform: translateY(-2px);
        }
        
        .btn-danger {
            background: #dc3545;
            color: white;
        }
        
        .btn-danger:hover {
            background: #c82333;
            transform: translateY(-2px);
        }
        
        .reading-display {
            background: rgba(255,255,255,0.95);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .moisture-value {
            font-size: 4rem;
            font-weight: bold;
            color: #007bff;
            margin: 20px 0;
        }
        
        .sensor-readings {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .sensor-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        
        .sensor-item h4 {
            color: #6c757d;
            margin-bottom: 5px;
        }
        
        .sensor-item .value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #333;
        }
        
        .chart-container {
            background: rgba(255,255,255,0.95);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        
        .chart-container h3 {
            text-align: center;
            margin-bottom: 20px;
            color: #333;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #6c757d;
        }
        
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        
        .success {
            background: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .moisture-value {
                font-size: 3rem;
            }
            
            .controls {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌾 Moisture Meter Dashboard</h1>
            <p>Real-time grain moisture monitoring system</p>
        </div>
        
        <div class="status-bar">
            <div class="status-grid">
                <div class="status-item">
                    <h3>System Status</h3>
                    <div class="value" id="system-status">Loading...</div>
                </div>
                <div class="status-item">
                    <h3>Monitoring</h3>
                    <div class="value" id="monitoring-status">Stopped</div>
                </div>
                <div class="status-item">
                    <h3>Total Readings</h3>
                    <div class="value" id="total-readings">0</div>
                </div>
                <div class="status-item">
                    <h3>Uptime</h3>
                    <div class="value" id="uptime">--</div>
                </div>
            </div>
            
            <div class="controls">
                <button class="btn btn-success" onclick="startMonitoring()">▶ Start Monitoring</button>
                <button class="btn btn-danger" onclick="stopMonitoring()">⏹ Stop Monitoring</button>
                <button class="btn btn-primary" onclick="getSingleReading()">📊 Single Reading</button>
                <button class="btn btn-primary" onclick="exportData()">📥 Export Data</button>
            </div>
        </div>
        
        <div class="reading-display">
            <h2>Current Moisture Reading</h2>
            <div class="moisture-value" id="moisture-value">--</div>
            <div id="crop-info">No reading available</div>
            
            <div class="sensor-readings">
                <div class="sensor-item">
                    <h4>Temperature</h4>
                    <div class="value" id="temperature">--°C</div>
                </div>
                <div class="sensor-item">
                    <h4>Humidity</h4>
                    <div class="value" id="humidity">--%</div>
                </div>
                <div class="sensor-item">
                    <h4>ADC Value</h4>
                    <div class="value" id="adc">--</div>
                </div>
                <div class="sensor-item">
                    <h4>Model Used</h4>
                    <div class="value" id="model-used">--</div>
                </div>
            </div>
        </div>
        
        <div class="chart-container">
            <h3>Moisture History (Last 20 Readings)</h3>
            <canvas id="moistureChart" width="400" height="200"></canvas>
        </div>
    </div>
    
    <script>
        // Initialize Socket.IO
        const socket = io();
        
        // Chart variables
        let moistureChart;
        let chartData = {
            labels: [],
            datasets: [{
                label: 'Moisture (%)',
                data: [],
                borderColor: '#007bff',
                backgroundColor: 'rgba(0, 123, 255, 0.1)',
                tension: 0.4
            }]
        };
        
        // Initialize chart
        function initChart() {
            const ctx = document.getElementById('moistureChart').getContext('2d');
            moistureChart = new Chart(ctx, {
                type: 'line',
                data: chartData,
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Moisture (%)'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Time'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    }
                }
            });
        }
        
        // Update status
        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('system-status').textContent = 
                        data.sensors_available ? 'Online' : 'Simulation';
                    document.getElementById('monitoring-status').textContent = 
                        data.monitoring_active ? 'Active' : 'Stopped';
                    document.getElementById('total-readings').textContent = 
                        data.total_readings;
                    document.getElementById('uptime').textContent = 
                        data.uptime;
                })
                .catch(error => console.error('Error updating status:', error));
        }
        
        // Start monitoring
        function startMonitoring() {
            fetch('/api/start_monitoring', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        showMessage('Monitoring started successfully!', 'success');
                        updateStatus();
                    } else {
                        showMessage(data.message, 'error');
                    }
                })
                .catch(error => {
                    showMessage('Error starting monitoring', 'error');
                    console.error('Error:', error);
                });
        }
        
        // Stop monitoring
        function stopMonitoring() {
            fetch('/api/stop_monitoring', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        showMessage('Monitoring stopped', 'success');
                        updateStatus();
                    } else {
                        showMessage(data.message, 'error');
                    }
                })
                .catch(error => {
                    showMessage('Error stopping monitoring', 'error');
                    console.error('Error:', error);
                });
        }
        
        // Get single reading
        function getSingleReading() {
            fetch('/api/single_reading', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        updateReadingDisplay(data.reading);
                        updateStatus();
                    } else {
                        showMessage(data.message, 'error');
                    }
                })
                .catch(error => {
                    showMessage('Error getting reading', 'error');
                    console.error('Error:', error);
                });
        }
        
        // Export data
        function exportData() {
            fetch('/api/export_data')
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        showMessage(`Data exported to ${data.filename}`, 'success');
                    } else {
                        showMessage(data.message, 'error');
                    }
                })
                .catch(error => {
                    showMessage('Error exporting data', 'error');
                    console.error('Error:', error);
                });
        }
        
        // Update reading display
        function updateReadingDisplay(reading) {
            document.getElementById('moisture-value').textContent = reading.moisture + '%';
            document.getElementById('temperature').textContent = reading.temperature + '°C';
            document.getElementById('humidity').textContent = reading.humidity + '%';
            document.getElementById('adc').textContent = reading.adc;
            document.getElementById('model-used').textContent = reading.model_used;
            document.getElementById('crop-info').textContent = 
                `${reading.crop_type.charAt(0).toUpperCase() + reading.crop_type.slice(1)} - ${reading.confidence} confidence`;
            
            // Update chart
            updateChart(reading);
        }
        
        // Update chart
        function updateChart(reading) {
            const time = new Date(reading.timestamp).toLocaleTimeString();
            
            chartData.labels.push(time);
            chartData.datasets[0].data.push(reading.moisture);
            
            // Keep only last 20 readings
            if (chartData.labels.length > 20) {
                chartData.labels.shift();
                chartData.datasets[0].data.shift();
            }
            
            moistureChart.update();
        }
        
        // Show message
        function showMessage(message, type) {
            const messageDiv = document.createElement('div');
            messageDiv.className = type;
            messageDiv.textContent = message;
            
            const container = document.querySelector('.container');
            container.insertBefore(messageDiv, container.firstChild);
            
            setTimeout(() => {
                messageDiv.remove();
            }, 5000);
        }
        
        // Socket.IO events
        socket.on('connect', () => {
            console.log('Connected to server');
        });
        
        socket.on('new_reading', (reading) => {
            updateReadingDisplay(reading);
            updateStatus();
        });
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            initChart();
            updateStatus();
            
            // Update status every 10 seconds
            setInterval(updateStatus, 10000);
        });
    </script>
</body>
</html>'''
    
    with open(templates_dir / "dashboard.html", "w") as f:
        f.write(html_template)
    
    print("✅ HTML template created")
    
    # Run the Flask app
    socketio.run(app, host='0.0.0.0', port=5000, debug=False) 