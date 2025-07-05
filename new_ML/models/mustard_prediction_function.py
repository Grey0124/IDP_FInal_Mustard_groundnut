
import joblib
import numpy as np

def predict_moisture(adc, temperature, humidity, crop_type=None):
    """
    Predict original moisture meter reading from custom meter data
    
    Args:
        adc (float): Custom meter ADC reading
        temperature (float): Temperature in Celsius
        humidity (float): Humidity percentage
        crop_type (str, optional): Crop type ('groundnut' or 'mustard') for combined model
    
    Returns:
        float: Predicted original moisture meter reading
    """
    # Load the trained River model
    model = joblib.load('models/mustard_final_river_model.pkl')
    
    # Prepare features
    if crop_type is not None:
        # Combined model with crop encoding
        crop_encoded = 0 if crop_type.lower() == 'groundnut' else 1
        features = {
            'ADC': adc,
            'Temperature': temperature,
            'Humidity': humidity,
            'crop_encoded': crop_encoded
        }
    else:
        features = {
            'ADC': adc,
            'Temperature': temperature,
            'Humidity': humidity
        }
    
    # Make prediction
    prediction = model.predict_one(features)
    return prediction if prediction is not None else 0.0

# Example usage:
# prediction = predict_moisture(adc=500, temperature=25.0, humidity=60.0, crop_type='groundnut')
# print(f"Predicted moisture: {prediction:.2f}%")
