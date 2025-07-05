#!/usr/bin/env python3
"""
Test script for Moisture Meter API
Tests all endpoints and functionality locally
"""

import requests
import json
import time
from datetime import datetime

# API Configuration
BASE_URL = "http://localhost:5000"  # Local testing
# BASE_URL = "https://your-app-name.onrender.com"  # Production

def test_health_endpoint():
    """Test the health endpoint"""
    print("🔍 Testing Health Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed")
            print(f"   Status: {data['status']}")
            print(f"   Models: {data['models_available']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_models_endpoint():
    """Test the models endpoint"""
    print("\n🔍 Testing Models Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/models")
        if response.status_code == 200:
            data = response.json()
            print("✅ Models endpoint working")
            for crop, info in data['available_models'].items():
                status = "✅" if info['available'] else "❌"
                print(f"   {status} {crop}: {info['type']}")
            return True
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Models endpoint error: {e}")
        return False

def test_prediction_endpoint(crop_type, test_data):
    """Test the prediction endpoint"""
    print(f"\n🔍 Testing Prediction for {crop_type}...")
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            headers={"Content-Type": "application/json"},
            json=test_data
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Prediction successful")
            print(f"   Moisture: {data['prediction']['moisture_percentage']}%")
            print(f"   Model: {data['prediction']['model_used']}")
            print(f"   Crop: {data['prediction']['crop_type']}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False

def test_crop_specific_endpoints():
    """Test crop-specific endpoints"""
    print("\n🔍 Testing Crop-Specific Endpoints...")
    
    # Test groundnut endpoint
    groundnut_data = {
        "adc": 2920,
        "temperature": 30.5,
        "humidity": 53.0
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/groundnut",
            headers={"Content-Type": "application/json"},
            json=groundnut_data
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Groundnut endpoint working")
            print(f"   Moisture: {data['prediction']['moisture_percentage']}%")
        else:
            print(f"❌ Groundnut endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Groundnut endpoint error: {e}")
    
    # Test mustard endpoint
    mustard_data = {
        "adc": 2850,
        "temperature": 31.0,
        "humidity": 58.0
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/mustard",
            headers={"Content-Type": "application/json"},
            json=mustard_data
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Mustard endpoint working")
            print(f"   Moisture: {data['prediction']['moisture_percentage']}%")
        else:
            print(f"❌ Mustard endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Mustard endpoint error: {e}")

def test_error_handling():
    """Test error handling"""
    print("\n🔍 Testing Error Handling...")
    
    # Test missing parameters
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            headers={"Content-Type": "application/json"},
            json={"adc": 2920}  # Missing temperature and humidity
        )
        
        if response.status_code == 400:
            print("✅ Missing parameters handled correctly")
        else:
            print(f"❌ Missing parameters not handled: {response.status_code}")
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
    
    # Test invalid crop type
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            headers={"Content-Type": "application/json"},
            json={
                "adc": 2920,
                "temperature": 30.5,
                "humidity": 53.0,
                "crop_type": "invalid_crop"
            }
        )
        
        if response.status_code == 400:
            print("✅ Invalid crop type handled correctly")
        else:
            print(f"❌ Invalid crop type not handled: {response.status_code}")
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")

def test_performance():
    """Test API performance"""
    print("\n🔍 Testing Performance...")
    
    test_data = {
        "adc": 2920,
        "temperature": 30.5,
        "humidity": 53.0,
        "crop_type": "groundnut"
    }
    
    times = []
    for i in range(5):
        start_time = time.time()
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                headers={"Content-Type": "application/json"},
                json=test_data
            )
            if response.status_code == 200:
                end_time = time.time()
                times.append(end_time - start_time)
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            break
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"✅ Average response time: {avg_time:.3f} seconds")
        print(f"   Fastest: {min(times):.3f} seconds")
        print(f"   Slowest: {max(times):.3f} seconds")
    else:
        print("❌ Performance test failed")

def main():
    """Run all tests"""
    print("🌾 Moisture Meter API Test Suite")
    print("=" * 50)
    print(f"Testing API at: {BASE_URL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # Run tests
    tests = [
        test_health_endpoint,
        test_models_endpoint,
        lambda: test_prediction_endpoint("groundnut", {
            "adc": 2920,
            "temperature": 30.5,
            "humidity": 53.0,
            "crop_type": "groundnut"
        }),
        lambda: test_prediction_endpoint("mustard", {
            "adc": 2850,
            "temperature": 31.0,
            "humidity": 58.0,
            "crop_type": "mustard"
        }),
        lambda: test_prediction_endpoint("auto", {
            "adc": 2950,
            "temperature": 30.0,
            "humidity": 54.0
        }),
        test_crop_specific_endpoints,
        test_error_handling,
        test_performance
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed! API is ready for deployment.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    print("=" * 50)

if __name__ == "__main__":
    main() 