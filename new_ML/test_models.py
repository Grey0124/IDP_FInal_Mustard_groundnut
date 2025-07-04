import joblib
import numpy as np
from pathlib import Path

class ModelTester:
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        
    def load_models(self):
        """Load all trained models"""
        try:
            # Load Groundnut model
            self.groundnut_model = joblib.load(self.models_dir / "groundnut_best_model.pkl")
            self.groundnut_scaler = joblib.load(self.models_dir / "groundnut_scaler.pkl")
            print("✅ Groundnut model loaded successfully")
            
            # Load Mustard model
            self.mustard_model = joblib.load(self.models_dir / "mustard_best_model.pkl")
            self.mustard_scaler = joblib.load(self.models_dir / "mustard_scaler.pkl")
            print("✅ Mustard model loaded successfully")
            
            # Load Combined model
            self.combined_model = joblib.load(self.models_dir / "combined_best_model.pkl")
            self.combined_scaler = joblib.load(self.models_dir / "combined_scaler.pkl")
            self.combined_encoder = joblib.load(self.models_dir / "combined_label_encoder.pkl")
            print("✅ Combined model loaded successfully")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def predict_groundnut(self, adc, temperature, humidity):
        """Predict original moisture meter reading for Groundnut"""
        try:
            # Prepare features
            features = np.array([[adc, temperature, humidity]])
            
            # Scale features
            features_scaled = self.groundnut_scaler.transform(features)
            
            # Make prediction
            prediction = self.groundnut_model.predict(features_scaled)[0]
            
            return prediction
            
        except Exception as e:
            print(f"❌ Error predicting Groundnut: {e}")
            return None
    
    def predict_mustard(self, adc, temperature, humidity):
        """Predict original moisture meter reading for Mustard"""
        try:
            # Prepare features
            features = np.array([[adc, temperature, humidity]])
            
            # Scale features
            features_scaled = self.mustard_scaler.transform(features)
            
            # Make prediction
            prediction = self.mustard_model.predict(features_scaled)[0]
            
            return prediction
            
        except Exception as e:
            print(f"❌ Error predicting Mustard: {e}")
            return None
    
    def predict_combined(self, adc, temperature, humidity, crop_type):
        """Predict original moisture meter reading using combined model"""
        try:
            # Encode crop type
            crop_encoded = self.combined_encoder.transform([crop_type])[0]
            
            # Prepare features
            features = np.array([[adc, temperature, humidity, crop_encoded]])
            
            # Scale features
            features_scaled = self.combined_scaler.transform(features)
            
            # Make prediction
            prediction = self.combined_model.predict(features_scaled)[0]
            
            return prediction
            
        except Exception as e:
            print(f"❌ Error predicting with combined model: {e}")
            return None
    
    def test_single_prediction(self, adc, temperature, humidity):
        """Test all models with the same inputs"""
        print(f"\n{'='*60}")
        print(f"TESTING MODELS")
        print(f"{'='*60}")
        print(f"Input Values:")
        print(f"  ADC: {adc}")
        print(f"  Temperature: {temperature}°C")
        print(f"  Humidity: {humidity}%")
        print(f"{'='*60}")
        
        # Test Groundnut model
        groundnut_pred = self.predict_groundnut(adc, temperature, humidity)
        if groundnut_pred is not None:
            print(f"🌰 Groundnut Model Prediction: {groundnut_pred:.3f}%")
        
        # Test Mustard model
        mustard_pred = self.predict_mustard(adc, temperature, humidity)
        if mustard_pred is not None:
            print(f"🌿 Mustard Model Prediction: {mustard_pred:.3f}%")
        
        # Test Combined model for both crops
        groundnut_combined = self.predict_combined(adc, temperature, humidity, "Groundnut")
        mustard_combined = self.predict_combined(adc, temperature, humidity, "Mustard")
        
        if groundnut_combined is not None:
            print(f"🌰 Combined Model (Groundnut): {groundnut_combined:.3f}%")
        if mustard_combined is not None:
            print(f"🌿 Combined Model (Mustard): {mustard_combined:.3f}%")
        
        print(f"{'='*60}")
        
        return {
            'groundnut': groundnut_pred,
            'mustard': mustard_pred,
            'groundnut_combined': groundnut_combined,
            'mustard_combined': mustard_combined
        }
    
    def interactive_test(self):
        """Interactive testing mode"""
        print("\n🧪 INTERACTIVE MODEL TESTING")
        print("Enter your sensor values to test the models")
        print("Type 'quit' to exit")
        
        while True:
            try:
                print(f"\n{'='*40}")
                
                # Get ADC value
                adc_input = input("Enter ADC value (or 'quit'): ").strip()
                if adc_input.lower() == 'quit':
                    break
                adc = float(adc_input)
                
                # Get Temperature
                temp_input = input("Enter Temperature (°C): ").strip()
                if temp_input.lower() == 'quit':
                    break
                temperature = float(temp_input)
                
                # Get Humidity
                humid_input = input("Enter Humidity (%): ").strip()
                if humid_input.lower() == 'quit':
                    break
                humidity = float(humid_input)
                
                # Test models
                self.test_single_prediction(adc, temperature, humidity)
                
            except ValueError:
                print("❌ Invalid input. Please enter numeric values.")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
    
    def test_sample_values(self):
        """Test with sample values from the training data"""
        print("\n📊 TESTING WITH SAMPLE VALUES")
        print("These are typical values from the training datasets")
        
        # Sample values from training data
        test_cases = [
            {"name": "Groundnut Sample 1", "adc": 2890, "temp": 30.5, "humid": 57, "crop": "Groundnut"},
            {"name": "Groundnut Sample 2", "adc": 2935, "temp": 30.8, "humid": 58, "crop": "Groundnut"},
            {"name": "Mustard Sample 1", "adc": 2899, "temp": 31.4, "humid": 59, "crop": "Mustard"},
            {"name": "Mustard Sample 2", "adc": 2946, "temp": 31.7, "humid": 60, "crop": "Mustard"},
        ]
        
        for case in test_cases:
            print(f"\n{case['name']}:")
            self.test_single_prediction(case['adc'], case['temp'], case['humid'])

def main():
    """Main function to run model testing"""
    print("🧪 MODEL TESTING UTILITY")
    print("Test your trained moisture meter models")
    
    # Initialize tester
    tester = ModelTester()
    
    # Load models
    if not tester.load_models():
        print("❌ Failed to load models. Make sure Phase 2 has been completed.")
        return
    
    print("\n✅ All models loaded successfully!")
    
    # Test with sample values
    tester.test_sample_values()
    
    # Interactive testing
    print("\n" + "="*60)
    print("INTERACTIVE TESTING MODE")
    print("="*60)
    tester.interactive_test()

if __name__ == "__main__":
    main() 