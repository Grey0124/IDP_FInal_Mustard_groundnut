#!/usr/bin/env python3
"""
Test script for deployed API at https://idp-final-mustard-groundnut-1.onrender.com
"""

import requests
import json

BASE_URL = "https://idp-final-mustard-groundnut-1.onrender.com"

def test_endpoint(endpoint, method="GET", data=None):
    """Test an API endpoint"""
    url = f"{BASE_URL}{endpoint}"
    headers = {"Content-Type": "application/json"} if data else {}
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers)
        
        print(f"\n=== {method} {endpoint} ===")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response
        
    except Exception as e:
        print(f"Error testing {endpoint}: {e}")
        return None

def main():
    """Run all API tests"""
    print("🧪 Testing Deployed API")
    print(f"Base URL: {BASE_URL}")
    
    # Basic status tests
    test_endpoint("/")
    test_endpoint("/health")
    test_endpoint("/models")
    test_endpoint("/debug")
    
    # Prediction tests
    test_endpoint("/predict", "POST", {
        "adc": 3000,
        "temperature": 25.5,
        "humidity": 60.0,
        "crop_type": "groundnut"
    })
    
    test_endpoint("/predict", "POST", {
        "adc": 2800,
        "temperature": 24.0,
        "humidity": 55.0,
        "crop_type": "mustard"
    })
    
    test_endpoint("/predict/groundnut", "POST", {
        "adc": 3100,
        "temperature": 27.0,
        "humidity": 62.0
    })
    
    test_endpoint("/predict/mustard", "POST", {
        "adc": 2750,
        "temperature": 23.5,
        "humidity": 52.0
    })
    
    test_endpoint("/predict", "POST", {
        "adc": 2900,
        "temperature": 26.0,
        "humidity": 58.0,
        "crop_type": "auto"
    })

if __name__ == "__main__":
    main() 