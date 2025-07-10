#!/usr/bin/env python3
"""
Script to update groundnut model with new environmental conditions
Temperature: 30.0-31.0°C, Humidity: 48-54%, Moisture: 9.0-11.2%, ADC: 2900-3000
"""

import requests
import json
import time
import random
from datetime import datetime

# API Configuration
BASE_URL = "https://idp-final-mustard-groundnut-1.onrender.com"
HEADERS = {"Content-Type": "application/json"}

def generate_sample():
    """Generate a sample with the specified ranges"""
    # ADC range: 2900-3000
    adc = random.uniform(2900, 3000)
    
    # Temperature range: 30.0-31.0°C
    temp = random.uniform(30.0, 31.0)
    
    # Humidity range: 48-54%
    hum = random.uniform(48.0, 54.0)
    
    # Moisture range: 9.0-11.2%
    moisture = random.uniform(9.0, 11.2)
    
    return {
        "adc": round(adc, 1),
        "temp": round(temp, 1),
        "hum": round(hum, 1),
        "moisture": round(moisture, 2)
    }

def update_model(sample):
    """Update the groundnut model with a sample"""
    try:
        response = requests.post(
            f"{BASE_URL}/update/groundnut",
            headers=HEADERS,
            json=sample,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Sample {sample['adc']:.1f}°C, {sample['temp']:.1f}°C, {sample['hum']:.1f}%, {sample['moisture']:.2f}% → Success")
            return True
        else:
            print(f"❌ Sample failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error updating model: {e}")
        return False

def check_model_status():
    """Check the current model status"""
    try:
        response = requests.get(f"{BASE_URL}/models/status", timeout=10)
        if response.status_code == 200:
            status = response.json()
            print(f"\n📊 Model Status:")
            print(f"   Groundnut Model Loaded: {status['models_loaded']['groundnut']}")
            print(f"   Updated Model File: {status['updated_model_files']['groundnut']}")
            return status
        else:
            print(f"❌ Failed to get status: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        return None

def test_prediction():
    """Test a prediction with the updated model"""
    test_sample = {
        "adc": 2950,
        "temperature": 30.5,
        "humidity": 51.0
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/groundnut",
            headers=HEADERS,
            json=test_sample,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            prediction = result['prediction']['moisture_percentage']
            print(f"\n🧪 Test Prediction:")
            print(f"   Input: ADC={test_sample['adc']}, Temp={test_sample['temperature']}°C, Hum={test_sample['humidity']}%")
            print(f"   Predicted Moisture: {prediction}%")
            return prediction
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error testing prediction: {e}")
        return None

def main():
    """Main function to update the groundnut model"""
    print("🌾 Groundnut Model Update Script")
    print("=" * 50)
    print(f"Target Ranges:")
    print(f"  Temperature: 30.0-31.0°C")
    print(f"  Humidity: 48-54%")
    print(f"  Moisture: 9.0-11.2%")
    print(f"  ADC: 2900-3000")
    print(f"API: {BASE_URL}")
    print("=" * 50)
    
    # Check initial status
    print("\n🔍 Checking initial model status...")
    initial_status = check_model_status()
    
    # Test initial prediction
    print("\n🧪 Testing initial prediction...")
    initial_prediction = test_prediction()
    
    # Generate and send samples
    num_samples = 50
    successful_updates = 0
    
    print(f"\n🚀 Starting model updates ({num_samples} samples)...")
    print("-" * 50)
    
    for i in range(num_samples):
        sample = generate_sample()
        
        print(f"📝 Sample {i+1}/{num_samples}: ", end="")
        if update_model(sample):
            successful_updates += 1
        
        # Small delay to avoid overwhelming the server
        time.sleep(0.5)
        
        # Progress update every 10 samples
        if (i + 1) % 10 == 0:
            print(f"\n📈 Progress: {i+1}/{num_samples} samples completed ({successful_updates} successful)")
    
    # Final status check
    print(f"\n🔍 Checking final model status...")
    final_status = check_model_status()
    
    # Test final prediction
    print(f"\n🧪 Testing final prediction...")
    final_prediction = test_prediction()
    
    # Summary
    print(f"\n📊 Update Summary")
    print("=" * 50)
    print(f"Total Samples Sent: {num_samples}")
    print(f"Successful Updates: {successful_updates}")
    print(f"Success Rate: {(successful_updates/num_samples)*100:.1f}%")
    
    if initial_prediction and final_prediction:
        print(f"Initial Prediction: {initial_prediction}%")
        print(f"Final Prediction: {final_prediction}%")
        print(f"Prediction Change: {final_prediction - initial_prediction:+.2f}%")
    
    print(f"\n✅ Model update completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 