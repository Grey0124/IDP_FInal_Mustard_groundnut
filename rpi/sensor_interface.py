#!/usr/bin/env python3
"""
Sensor Interface for Raspberry Pi
Handles DHT22 (temperature/humidity) and ADS1115 (moisture sensor) readings
"""

import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import adafruit_dht
import RPi.GPIO as GPIO
from typing import Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SensorInterface:
    def __init__(self, dht_pin: int = 4, ads_address: int = 0x48):
        """
        Initialize sensor interface
        
        Args:
            dht_pin: GPIO pin for DHT22 sensor (default: GPIO4)
            ads_address: I2C address for ADS1115 (default: 0x48)
        """
        self.dht_pin = dht_pin
        self.ads_address = ads_address
        
        # Initialize GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Initialize I2C bus
        self.i2c = busio.I2C(board.SCL, board.SDA)
        
        # Initialize sensors
        self._init_dht_sensor()
        self._init_ads_sensor()
        
        logger.info("Sensor interface initialized successfully")
    
    def _init_dht_sensor(self):
        """Initialize DHT22 temperature and humidity sensor"""
        try:
            self.dht_sensor = adafruit_dht.DHT22(self.dht_pin)
            logger.info(f"DHT22 sensor initialized on GPIO{self.dht_pin}")
        except Exception as e:
            logger.error(f"Failed to initialize DHT22 sensor: {e}")
            self.dht_sensor = None
    
    def _init_ads_sensor(self):
        """Initialize ADS1115 ADC for moisture sensor"""
        try:
            self.ads = ADS.ADS1115(self.i2c, address=self.ads_address)
            # Configure for single-ended reading on A0
            self.moisture_channel = AnalogIn(self.ads, ADS.P0)
            logger.info(f"ADS1115 sensor initialized at address 0x{self.ads_address:02x}")
        except Exception as e:
            logger.error(f"Failed to initialize ADS1115 sensor: {e}")
            self.ads = None
            self.moisture_channel = None
    
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
        Read moisture sensor value from ADS1115
        
        Returns:
            ADC value (0-65535 for 16-bit ADC)
        """
        if self.moisture_channel is None:
            return None
        
        try:
            # Read raw ADC value
            adc_value = self.moisture_channel.value
            
            # Convert to voltage (ADS1115 reference voltage is 4.096V)
            voltage = (adc_value / 32767) * 4.096
            
            logger.info(f"Moisture ADC: {adc_value}, Voltage: {voltage:.3f}V")
            return float(adc_value)
            
        except Exception as e:
            logger.error(f"Error reading moisture sensor: {e}")
            return None
    
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
            'ads1115': self.ads is not None
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