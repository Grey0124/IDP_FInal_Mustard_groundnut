#!/usr/bin/env python3
"""
Script to update groundnut model with 25 samples
Temperature: 29-30°C, Humidity: 55-57%, Moisture: 10.7-11.1%, ADC: 3000-3050
"""

import requests
import json
import time
import random

# API Configuration
BASE_URL = "https://idp-final-mustard-groundnut-1.onrender.com"
HEADERS = {"Content-Type": "application/json"}

# Sample data with the specified ranges
samples = [
    {"adc": 3000, "temp": 29.1, "hum": 55.2, "moisture": 10.85},
    {"adc": 3005, "temp": 29.3, "hum": 55.5, "moisture": 10.92},
    {"adc": 3010, "temp": 29.5, "hum": 55.8, "moisture": 10.78},
    {"adc": 3015, "temp": 29.7, "hum": 56.1, "moisture": 11.05},
    {"adc": 3020, "temp": 29.9, "hum": 56.4, "moisture": 10.95},
    {"adc": 3025, "temp": 29.2, "hum": 56.7, "moisture": 10.88},
    {"adc": 3030, "temp": 29.4, "hum": 55.3, "moisture": 11.02},
    {"adc": 3035, "temp": 29.6, "hum": 55.6, "moisture": 10.75},
    {"adc": 3040, "temp": 29.8, "hum": 55.9, "moisture": 10.98},
    {"adc": 3045, "temp": 29.0, "hum": 56.2, "moisture": 10.82},
    {"adc": 3050, "temp": 29.2, "hum": 56.5, "moisture": 11.08},
    {"adc": 3002, "temp": 29.4, "hum": 56.8, "moisture": 10.91},
    {"adc": 3007, "temp": 29.6, "hum": 55.4, "moisture": 10.79},
    {"adc": 3012, "temp": 29.8, "hum": 55.7, "moisture": 11.03},
    {"adc": 3017, "temp": 29.1, "hum": 56.0, "moisture": 10.87},
    {"adc": 3022, "temp": 29.3, "hum": 56.3, "moisture": 10.94},
    {"adc": 3027, "temp": 29.5, "hum": 56.6, "moisture": 10.76},
    {"adc": 3032, "temp": 29.7, "hum": 55.1, "moisture": 11.06},
    {"adc": 3037, "temp": 29.9, "hum": 55.4, "moisture": 10.89},
    {"adc": 3042, "temp": 29.2, "hum": 55.7, "moisture": 10.97},
    {"adc": 3047, "temp": 29.4, "hum": 56.0, "moisture": 10.81},
    {"adc": 3003, "temp": 29.6, "hum": 56.3, "moisture": 11.04},
    {"adc": 3008, "temp": 29.8, "hum": 56.6, "moisture": 10.86},
    {"adc": 3013, "temp": 29.0, "hum": 55.2, "moisture": 10.93},
    {"adc": 3018, "temp": 29.2, "hum": 55.5, "moisture": 10.77}
]

def update_model(sample, index):
    """Update the groundnut model with a sample"""
    try:
        response = requests.post(
            f"{BASE_URL}/update/groundnut",
            headers=HEADERS,
            json=sample,
            timeout=30
        )
        
        if response.status_code == 200:
            print(f"✅ Sample {index+1}/25: ADC={sample['adc']}, Temp={sample['temp']}°C, Hum={sample['hum']}%, Moisture={sample['moisture']}%")
            return True
        else:
            print(f"❌ Sample {index+1}/25 failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Sample {index+1}/25 error: {e}")
        return False

def main():
    """Main function to update the groundnut model"""
    print("🌾 Groundnut Model Update Script (25 samples)")
    print("=" * 50)
    print(f"Temperature Range: 29-30°C")
    print(f"Humidity Range: 55-57%")
    print(f"Moisture Range: 10.7-11.1%")
    print(f"ADC Range: 3000-3050")
    print(f"API: {BASE_URL}")
    print("=" * 50)
    
    successful_updates = 0
    
    for i, sample in enumerate(samples):
        if update_model(sample, i):
            successful_updates += 1
        
        # Small delay between requests
        time.sleep(0.5)
    
    print(f"\n📊 Summary:")
    print(f"Total Samples: 25")
    print(f"Successful Updates: {successful_updates}")
    print(f"Success Rate: {(successful_updates/25)*100:.1f}%")
    print(f"✅ Model update completed!")

if __name__ == "__main__":
    main() 