#!/usr/bin/env python3
"""
Data Logger for Soil Moisture Monitor
Continuously logs sensor readings and predictions to SQLite database
"""

import sqlite3
import time
import json
import logging
from datetime import datetime
from sensor_interface import SensorInterface
import numpy as np
import joblib
import tensorflow as tf
from typing import Dict, Optional
import threading
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_logger.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataLogger:
    def __init__(self, db_path: str = 'data/soil_moisture.db', log_interval: int = 300):
        """
        Initialize data logger
        
        Args:
            db_path: Path to SQLite database
            log_interval: Logging interval in seconds (default: 5 minutes)
        """
        self.db_path = db_path
        self.log_interval = log_interval
        self.running = False
        self.sensor_interface = None
        self.models = {}
        self.scalers = {}
        
        # Initialize database
        self._init_database()
        
        # Load models
        self._load_models()
        
        # Initialize sensors
        self._init_sensors()
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create sensor readings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sensor_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    temperature REAL,
                    humidity REAL,
                    adc_value REAL,
                    groundnut_moisture REAL,
                    mustard_moisture REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create system events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    event_type TEXT,
                    message TEXT,
                    details TEXT
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON sensor_readings(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_event_type ON system_events(event_type)')
            
            conn.commit()
            conn.close()
            
            logger.info(f"Database initialized: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _load_models(self):
        """Load TensorFlow models and scalers"""
        try:
            # Load models
            self.models = {
                'groundnut': tf.keras.models.load_model('new_groundnut_model.h5', compile=False),
                'mustard': tf.keras.models.load_model('new_mustard_model.h5', compile=False)
            }
            
            # Load scalers
            self.scalers = {
                'groundnut': joblib.load('scaler_groundnut.pkl'),
                'mustard': joblib.load('scaler_mustard.pkl')
            }
            
            # Compile models
            for crop in self.models:
                self.models[crop].compile(
                    optimizer=tf.keras.optimizers.Adam(),
                    loss='mse',
                    metrics=['mae']
                )
            
            logger.info("Models and scalers loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            self.models = {}
            self.scalers = {}
    
    def _init_sensors(self):
        """Initialize sensor interface"""
        try:
            self.sensor_interface = SensorInterface()
            logger.info("Sensor interface initialized")
        except Exception as e:
            logger.error(f"Failed to initialize sensors: {e}")
            self.sensor_interface = None
    
    def _predict_moisture(self, adc: float, temp: float, hum: float) -> Dict[str, float]:
        """Predict moisture for both crops"""
        predictions = {}
        
        if not self.models or not self.scalers:
            return predictions
        
        try:
            X = np.array([[adc, temp, hum]])
            
            for crop in self.models:
                scaler = self.scalers[crop]
                model = self.models[crop]
                
                Xs = scaler.transform(X)
                prediction = model.predict(Xs, verbose=0)[0, 0]
                predictions[f'{crop}_moisture'] = round(float(prediction), 2)
                
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
        
        return predictions
    
    def _log_sensor_data(self, readings: Dict, predictions: Dict):
        """Log sensor readings and predictions to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO sensor_readings 
                (temperature, humidity, adc_value, groundnut_moisture, mustard_moisture)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                readings.get('temperature'),
                readings.get('humidity'),
                readings.get('adc'),
                predictions.get('groundnut_moisture'),
                predictions.get('mustard_moisture')
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Data logged: {readings} | Predictions: {predictions}")
            
        except Exception as e:
            logger.error(f"Failed to log data: {e}")
    
    def _log_event(self, event_type: str, message: str, details: str = None):
        """Log system events"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO system_events (event_type, message, details)
                VALUES (?, ?, ?)
            ''', (event_type, message, details))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log event: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def start(self):
        """Start the data logging loop"""
        if self.running:
            logger.warning("Data logger is already running")
            return
        
        self.running = True
        self._log_event('START', 'Data logger started')
        logger.info("Data logger started")
        
        try:
            while self.running:
                # Read sensors
                if self.sensor_interface:
                    try:
                        readings = self.sensor_interface.read_all_sensors()
                        
                        # Make predictions if we have valid sensor data
                        predictions = {}
                        if all(v is not None for v in readings.values()):
                            predictions = self._predict_moisture(
                                readings['adc'], 
                                readings['temperature'], 
                                readings['humidity']
                            )
                        
                        # Log data
                        self._log_sensor_data(readings, predictions)
                        
                    except Exception as e:
                        logger.error(f"Error reading sensors: {e}")
                        self._log_event('ERROR', f'Sensor reading error: {str(e)}')
                else:
                    logger.warning("Sensor interface not available")
                    self._log_event('WARNING', 'Sensor interface not available')
                
                # Wait for next logging interval
                time.sleep(self.log_interval)
                
        except KeyboardInterrupt:
            logger.info("Data logger interrupted by user")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the data logger"""
        self.running = False
        self._log_event('STOP', 'Data logger stopped')
        
        if self.sensor_interface:
            self.sensor_interface.cleanup()
        
        logger.info("Data logger stopped")
    
    def get_recent_data(self, hours: int = 24) -> list:
        """Get recent sensor data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM sensor_readings 
                WHERE timestamp >= datetime('now', '-{} hours')
                ORDER BY timestamp DESC
            '''.format(hours))
            
            data = cursor.fetchall()
            conn.close()
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to get recent data: {e}")
            return []
    
    def get_statistics(self, hours: int = 24) -> Dict:
        """Get statistics for recent data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    AVG(temperature) as avg_temp,
                    AVG(humidity) as avg_humidity,
                    AVG(adc_value) as avg_adc,
                    AVG(groundnut_moisture) as avg_groundnut,
                    AVG(mustard_moisture) as avg_mustard,
                    COUNT(*) as total_readings
                FROM sensor_readings 
                WHERE timestamp >= datetime('now', '-{} hours')
            '''.format(hours))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'avg_temperature': result[0],
                    'avg_humidity': result[1],
                    'avg_adc': result[2],
                    'avg_groundnut_moisture': result[3],
                    'avg_mustard_moisture': result[4],
                    'total_readings': result[5]
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}

def main():
    """Main function"""
    # Create logs directory if it doesn't exist
    import os
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Create and start data logger
    logger_instance = DataLogger()
    
    try:
        logger_instance.start()
    except Exception as e:
        logger.error(f"Data logger error: {e}")
    finally:
        logger_instance.stop()

if __name__ == "__main__":
    main() 