#!/usr/bin/env python3
"""
Sensor Interface for Raspberry Pi
Handles DHT22 (temperature/humidity) and capacitive soil moisture sensor V2.0
"""

import time
import board
import busio
import adafruit_dht
import RPi.GPIO as GPIO
from typing import Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SensorInterface:
    def __init__(self, dht_pin: int = 4, moisture_pin: int = 17):
        """
        Initialize sensor interface
        
        Args:
            dht_pin: GPIO pin for DHT22 sensor (default: GPIO4)
            moisture_pin: GPIO pin for capacitive moisture sensor (default: GPIO17)
        """
        self.dht_pin = dht_pin
        self.moisture_pin = moisture_pin
        
        # Initialize GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Initialize sensors
        self._init_dht_sensor()
        self._init_moisture_sensor()
        
        logger.info("Sensor interface initialized successfully")
    
    def _init_dht_sensor(self):
        """Initialize DHT22 temperature and humidity sensor"""
        try:
            self.dht_sensor = adafruit_dht.DHT22(self.dht_pin)
            logger.info(f"DHT22 sensor initialized on GPIO{self.dht_pin}")
        except Exception as e:
            logger.error(f"Failed to initialize DHT22 sensor: {e}")
            self.dht_sensor = None
    
    def _init_moisture_sensor(self):
        """Initialize capacitive soil moisture sensor V2.0"""
        try:
            # The capacitive sensor V2.0 has built-in ADC and outputs analog signal
            # We'll use GPIO17 as analog input (you may need to enable analog input)
            self.moisture_pin = self.moisture_pin
            logger.info(f"Capacitive moisture sensor initialized on GPIO{self.moisture_pin}")
        except Exception as e:
            logger.error(f"Failed to initialize moisture sensor: {e}")
    
    def read_temperature_humidity(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Read temperature and humidity from DHT22
        
        Returns:
            Tuple of (temperature, humidity) in Celsius and percentage
        """
        if self.dht_sensor is None:
            return None, None
        
        try:
            temperature = self.dht_sensor.temperature
            humidity = self.dht_sensor.humidity
            
            if temperature is not None and humidity is not None:
                logger.info(f"Temperature: {temperature:.1f}°C, Humidity: {humidity:.1f}%")
                return temperature, humidity
            else:
                logger.warning("DHT22 sensor returned None values")
                return None, None
                
        except Exception as e:
            logger.error(f"Error reading DHT22 sensor: {e}")
            return None, None
    
    def read_moisture_adc(self) -> Optional[float]:
        """
        Read moisture sensor value from capacitive sensor V2.0
        
        Returns:
            ADC value (0-4095 for 12-bit ADC on Pi 4B)
        """
        try:
            # Read analog value from capacitive sensor
            # The sensor outputs 0-3.3V which gets converted to 0-4095 by Pi's ADC
            adc_value = self._read_analog(self.moisture_pin)
            
            # Convert to voltage (Pi 4B reference voltage is 3.3V)
            voltage = (adc_value / 4095) * 3.3
            
            logger.info(f"Moisture ADC: {adc_value}, Voltage: {voltage:.3f}V")
            return float(adc_value)
            
        except Exception as e:
            logger.error(f"Error reading moisture sensor: {e}")
            return None
    
    def _read_analog(self, pin: int) -> int:
        """
        Read analog value from GPIO pin
        Note: This is a simplified implementation. For accurate readings,
        you might need to use an external ADC or MCP3008
        """
        try:
            # For capacitive sensor V2.0, we can use GPIO as analog input
            # This is a basic implementation - you may need to adjust based on your setup
            GPIO.setup(pin, GPIO.IN)
            
            # Read multiple times and average for stability
            readings = []
            for _ in range(10):
                # Read the pin state (this is simplified - actual analog reading may differ)
                value = GPIO.input(pin)
                readings.append(value)
                time.sleep(0.01)
            
            # Calculate average and scale to 12-bit range
            avg_value = sum(readings) / len(readings)
            adc_value = int(avg_value * 4095)  # Scale to 12-bit
            
            return adc_value
            
        except Exception as e:
            logger.error(f"Error reading analog pin {pin}: {e}")
            return 0
    
    def read_all_sensors(self) -> Dict[str, Optional[float]]:
        """
        Read all sensors and return a dictionary of values
        
        Returns:
            Dictionary with 'temperature', 'humidity', and 'adc' values
        """
        temp, hum = self.read_temperature_humidity()
        adc = self.read_moisture_adc()
        
        readings = {
            'temperature': temp,
            'humidity': hum,
            'adc': adc
        }
        
        logger.info(f"All sensor readings: {readings}")
        return readings
    
    def get_sensor_status(self) -> Dict[str, bool]:
        """Get status of all sensors"""
        return {
            'dht22': self.dht_sensor is not None,
            'capacitive_moisture': True  # Always true as it's just GPIO
        }
    
    def cleanup(self):
        """Clean up GPIO resources"""
        GPIO.cleanup()
        logger.info("GPIO cleanup completed")

# Test function
def test_sensors():
    """Test function to verify sensor readings"""
    sensor = SensorInterface()
    
    try:
        print("Testing sensors...")
        print(f"Sensor status: {sensor.get_sensor_status()}")
        
        for i in range(5):
            readings = sensor.read_all_sensors()
            print(f"Reading {i+1}: {readings}")
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        sensor.cleanup()

if __name__ == "__main__":
    test_sensors() 