#!/usr/bin/env python3
"""
MicroPython ESP32 Grain Moisture Analyzer with TFLite
Supports dynamic model loading and over-the-air updates
"""

import gc
import json
import urequests
import time
from machine import Pin, ADC, I2C
import network
import uos
from micropython import const

# TFLite Micro imports (if available)
try:
    import tflite
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False
    print("TFLite Micro not available - using simulated predictions")

# === CONFIGURATION ===
WIFI_SSID = "RMRaikar"
WIFI_PASSWORD = "RMRaikar@777"
MODEL_SERVER_URL = "https://idp-final-mustard-groundnut-1.onrender.com"

# Pin definitions
DHT_PIN = const(4)
SOIL_ADC_PIN = const(36)
LED_PIN = const(2)

# Model file paths
MODEL_DIR = "/models"
MUSTARD_MODEL_PATH = f"{MODEL_DIR}/mustard_model.tflite"
GROUNDNUT_MODEL_PATH = f"{MODEL_DIR}/groundnut_model.tflite"
MODEL_METADATA_PATH = f"{MODEL_DIR}/model_metadata.json"

class GrainMoistureAnalyzer:
    def __init__(self):
        """Initialize the Grain Moisture Analyzer"""
        self.current_crop = "groundnut"
        self.model_loaded = False
        self.interpreter = None
        self.wifi_connected = False
        
        # Initialize hardware
        self.setup_hardware()
        
        # Initialize filesystem
        self.setup_filesystem()
        
        # Connect to WiFi
        self.connect_wifi()
        
        print("Grain Moisture Analyzer initialized")
    
    def setup_hardware(self):
        """Initialize hardware components"""
        # ADC for soil moisture sensor
        self.adc = ADC(Pin(SOIL_ADC_PIN))
        self.adc.atten(ADC.ATTN_11DB)  # 0-3.3V range
        
        # LED for status indication
        self.led = Pin(LED_PIN, Pin.OUT)
        
        # DHT sensor (simplified - you'll need DHT library)
        # self.dht = DHT22(Pin(DHT_PIN))
        
        print("Hardware initialized")
    
    def setup_filesystem(self):
        """Setup filesystem for model storage"""
        try:
            # Create models directory if it doesn't exist
            if MODEL_DIR not in uos.listdir("/"):
                uos.mkdir(MODEL_DIR)
                print(f"Created directory: {MODEL_DIR}")
        except Exception as e:
            print(f"Error setting up filesystem: {e}")
    
    def connect_wifi(self):
        """Connect to WiFi network"""
        try:
            self.wlan = network.WLAN(network.STA_IF)
            self.wlan.active(True)
            
            if not self.wlan.isconnected():
                print(f"Connecting to WiFi: {WIFI_SSID}")
                self.wlan.connect(WIFI_SSID, WIFI_PASSWORD)
                
                # Wait for connection
                max_wait = 10
                while max_wait > 0:
                    if self.wlan.isconnected():
                        break
                    max_wait -= 1
                    print("Waiting for connection...")
                    time.sleep(1)
            
            if self.wlan.isconnected():
                self.wifi_connected = True
                print(f"WiFi connected: {self.wlan.ifconfig()}")
                self.led.on()  # Turn on LED to indicate WiFi connection
            else:
                print("WiFi connection failed")
                self.led.off()
                
        except Exception as e:
            print(f"WiFi connection error: {e}")
            self.wifi_connected = False
    
    def download_model(self, crop_type):
        """Download model from server"""
        if not self.wifi_connected:
            print("WiFi not connected - cannot download model")
            return False
        
        try:
            print(f"Downloading {crop_type} model...")
            
            # Get model from server
            url = f"{MODEL_SERVER_URL}/models/{crop_type}/download"
            response = urequests.get(url)
            
            if response.status_code == 200:
                # Save model to filesystem
                model_path = MUSTARD_MODEL_PATH if crop_type == "mustard" else GROUNDNUT_MODEL_PATH
                
                with open(model_path, 'wb') as f:
                    f.write(response.content)
                
                # Update metadata
                self.update_model_metadata(crop_type, len(response.content))
                
                print(f"{crop_type} model downloaded successfully")
                response.close()
                return True
            else:
                print(f"Failed to download model: {response.status_code}")
                response.close()
                return False
                
        except Exception as e:
            print(f"Error downloading model: {e}")
            return False
    
    def update_model_metadata(self, crop_type, model_size):
        """Update model metadata file"""
        try:
            metadata = {
                "last_updated": time.time(),
                "models": {
                    "mustard": {
                        "size": 0,
                        "version": "1.0",
                        "last_updated": 0
                    },
                    "groundnut": {
                        "size": 0,
                        "version": "1.0",
                        "last_updated": 0
                    }
                }
            }
            
            # Load existing metadata if available
            try:
                with open(MODEL_METADATA_PATH, 'r') as f:
                    metadata = json.load(f)
            except:
                pass
            
            # Update specific model metadata
            metadata["models"][crop_type]["size"] = model_size
            metadata["models"][crop_type]["last_updated"] = time.time()
            metadata["last_updated"] = time.time()
            
            # Save metadata
            with open(MODEL_METADATA_PATH, 'w') as f:
                json.dump(metadata, f)
                
        except Exception as e:
            print(f"Error updating metadata: {e}")
    
    def load_model(self, crop_type):
        """Load TFLite model from filesystem"""
        try:
            model_path = MUSTARD_MODEL_PATH if crop_type == "mustard" else GROUNDNUT_MODEL_PATH
            
            # Check if model file exists
            try:
                with open(model_path, 'rb') as f:
                    model_data = f.read()
                print(f"Model file found: {len(model_data)} bytes")
            except:
                print(f"Model file not found: {model_path}")
                return False
            
            if TFLITE_AVAILABLE:
                # Load TFLite model
                self.interpreter = tflite.Interpreter(model_data)
                self.interpreter.allocate_tensors()
                
                # Get input/output details
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                
                print(f"TFLite model loaded: {crop_type}")
                print(f"Input shape: {self.input_details[0]['shape']}")
                print(f"Output shape: {self.output_details[0]['shape']}")
                
            else:
                # Simulated model for testing
                print(f"Using simulated model for: {crop_type}")
            
            self.current_crop = crop_type
            self.model_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False
            return False
    
    def predict_moisture(self, adc_value, temperature, humidity):
        """Predict moisture content using loaded model"""
        if not self.model_loaded:
            print("Model not loaded")
            return None
        
        try:
            # Normalize inputs
            normalized_adc = adc_value / 4095.0  # ESP32 ADC is 12-bit
            normalized_temp = (temperature - 20.0) / 30.0  # Range 20-50°C
            normalized_hum = humidity / 100.0  # Humidity 0-100%
            
            if TFLITE_AVAILABLE and self.interpreter:
                # Prepare input data
                input_data = [[normalized_adc, normalized_temp, normalized_hum]]
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                
                # Run inference
                self.interpreter.invoke()
                
                # Get prediction
                prediction = self.interpreter.get_tensor(self.output_details[0]['index'])
                moisture_percentage = prediction[0][0] * 20.0 + 5.0  # Scale to 5-25%
                
            else:
                # Simulated prediction for testing
                moisture_percentage = 12.5 + (normalized_adc - 0.5) * 10 + (normalized_temp - 0.5) * 5
            
            return moisture_percentage
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def read_sensors(self):
        """Read sensor values"""
        try:
            # Read ADC (soil moisture)
            adc_value = self.adc.read()
            
            # Read DHT sensor (simplified)
            # temperature = self.dht.temperature()
            # humidity = self.dht.humidity()
            
            # Simulated sensor readings for testing
            temperature = 25.0 + (adc_value / 4095.0) * 10  # 25-35°C
            humidity = 50.0 + (adc_value / 4095.0) * 20     # 50-70%
            
            return {
                'adc': adc_value,
                'temperature': temperature,
                'humidity': humidity
            }
            
        except Exception as e:
            print(f"Sensor reading error: {e}")
            return None
    
    def check_model_updates(self):
        """Check for model updates from server"""
        if not self.wifi_connected:
            return False
        
        try:
            print("Checking for model updates...")
            
            # Get server model info
            url = f"{MODEL_SERVER_URL}/models/status"
            response = urequests.get(url)
            
            if response.status_code == 200:
                server_info = response.json()
                response.close()
                
                # Load local metadata
                try:
                    with open(MODEL_METADATA_PATH, 'r') as f:
                        local_metadata = json.load(f)
                except:
                    local_metadata = {"models": {}}
                
                # Check for updates
                updates_available = []
                for crop_type in ["mustard", "groundnut"]:
                    if crop_type in server_info.get("available_models", {}):
                        server_model = server_info["available_models"][crop_type]
                        local_model = local_metadata.get("models", {}).get(crop_type, {})
                        
                        # Compare versions or timestamps
                        if server_model.get("version") != local_model.get("version"):
                            updates_available.append(crop_type)
                
                if updates_available:
                    print(f"Updates available for: {updates_available}")
                    return updates_available
                else:
                    print("No updates available")
                    return False
            else:
                response.close()
                return False
                
        except Exception as e:
            print(f"Error checking updates: {e}")
            return False
    
    def update_models(self, crop_types=None):
        """Update models for specified crop types"""
        if crop_types is None:
            crop_types = ["mustard", "groundnut"]
        
        updated = []
        for crop_type in crop_types:
            if self.download_model(crop_type):
                updated.append(crop_type)
                # Reload current model if it was updated
                if crop_type == self.current_crop:
                    self.load_model(crop_type)
        
        return updated
    
    def run(self):
        """Main application loop"""
        print("Starting Grain Moisture Analyzer...")
        print("Commands: 'read', 'mustard', 'groundnut', 'update', 'status'")
        
        # Load default model
        if not self.load_model(self.current_crop):
            print("Failed to load default model")
        
        while True:
            try:
                # Check for serial commands
                if hasattr(self, 'serial') and self.serial.any():
                    command = self.serial.readline().decode().strip()
                    self.handle_command(command)
                
                # Periodic tasks
                self.periodic_tasks()
                
                time.sleep(1)
                
            except KeyboardInterrupt:
                print("Stopping...")
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(5)
    
    def handle_command(self, command):
        """Handle serial commands"""
        command = command.lower().strip()
        
        if command == "read":
            self.take_reading()
        elif command == "mustard":
            self.load_model("mustard")
        elif command == "groundnut":
            self.load_model("groundnut")
        elif command == "update":
            self.update_models()
        elif command == "status":
            self.show_status()
        else:
            print(f"Unknown command: {command}")
    
    def take_reading(self):
        """Take a moisture reading"""
        if not self.model_loaded:
            print("Model not loaded. Please select a crop first.")
            return
        
        print("Taking reading...")
        sensors = self.read_sensors()
        
        if sensors:
            moisture = self.predict_moisture(
                sensors['adc'], 
                sensors['temperature'], 
                sensors['humidity']
            )
            
            if moisture is not None:
                print(f"=== Moisture Reading ===")
                print(f"Crop: {self.current_crop}")
                print(f"ADC: {sensors['adc']}")
                print(f"Temperature: {sensors['temperature']:.1f}°C")
                print(f"Humidity: {sensors['humidity']:.1f}%")
                print(f"Moisture: {moisture:.2f}%")
                print("========================")
            else:
                print("Prediction failed")
        else:
            print("Sensor reading failed")
    
    def show_status(self):
        """Show system status"""
        print(f"=== System Status ===")
        print(f"WiFi: {'Connected' if self.wifi_connected else 'Disconnected'}")
        print(f"Current Crop: {self.current_crop}")
        print(f"Model Loaded: {'Yes' if self.model_loaded else 'No'}")
        print(f"Free Memory: {gc.mem_free()} bytes")
        
        # Show model metadata
        try:
            with open(MODEL_METADATA_PATH, 'r') as f:
                metadata = json.load(f)
                print(f"Last Updated: {metadata.get('last_updated', 'Never')}")
        except:
            print("No model metadata found")
        
        print("====================")
    
    def periodic_tasks(self):
        """Perform periodic tasks"""
        # Check for model updates every hour
        if hasattr(self, '_last_update_check'):
            if time.time() - self._last_update_check > 3600:  # 1 hour
                self.check_model_updates()
                self._last_update_check = time.time()
        else:
            self._last_update_check = time.time()

# Main execution
if __name__ == "__main__":
    gma = GrainMoistureAnalyzer()
    gma.run() 