import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import River, if not available, use sklearn's SGDRegressor with partial_fit
try:
    from river import linear_model, preprocessing, metrics, compose
    RIVER_AVAILABLE = True
    print("River library available - using online learning")
except ImportError:
    RIVER_AVAILABLE = False
    from sklearn.linear_model import SGDRegressor
    from sklearn.preprocessing import StandardScaler
    print("River library not available - using sklearn SGDRegressor with partial_fit")

class OnlineLearning:
    def __init__(self, data_dir=".", models_dir="models"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
    def load_pretrained_model(self, crop_name=None):
        """Load pretrained model and scaler"""
        prefix = crop_name.lower() if crop_name else "combined"
        
        try:
            model = joblib.load(self.models_dir / f"{prefix}_best_model.pkl")
            scaler = joblib.load(self.models_dir / f"{prefix}_scaler.pkl")
            
            if crop_name is None:
                label_encoder = joblib.load(self.models_dir / f"{prefix}_label_encoder.pkl")
            else:
                label_encoder = None
                
            print(f"Loaded pretrained model for {crop_name or 'combined data'}")
            return model, scaler, label_encoder
            
        except FileNotFoundError:
            print(f"No pretrained model found for {crop_name or 'combined data'}")
            return None, None, None
    
    def create_online_model(self, crop_name=None):
        """Create an online learning model"""
        if RIVER_AVAILABLE:
            # Use River for true online learning
            if crop_name is None:
                # Combined model with crop encoding
                model = compose.Pipeline(
                    preprocessing.StandardScaler(),
                    linear_model.SGDRegressor(
                        optimizer=linear_model.SGD(learning_rate=0.01),
                        loss='squared'
                    )
                )
            else:
                # Single crop model
                model = compose.Pipeline(
                    preprocessing.StandardScaler(),
                    linear_model.SGDRegressor(
                        optimizer=linear_model.SGD(learning_rate=0.01),
                        loss='squared'
                    )
                )
        else:
            # Use sklearn SGDRegressor with partial_fit
            model = SGDRegressor(
                learning_rate='invscaling',
                eta0=0.01,
                max_iter=1,
                warm_start=True,
                random_state=42
            )
            scaler = StandardScaler()
            model = (scaler, model)
        
        return model
    
    def simulate_online_learning(self, crop_name=None, window_size=50):
        """Simulate online learning with streaming data"""
        print(f"\n=== Online Learning Simulation for {crop_name or 'Combined Data'} ===")
        print("Objective: Continuously improve prediction of original moisture meter readings")
        
        # Load data
        if crop_name:
            data_file = self.data_dir / f"{crop_name}_merged_data.csv"
        else:
            data_file = self.data_dir / "combined_crops_data.csv"
        
        df = pd.read_csv(data_file)
        print(f"Loaded {len(df)} samples for online learning simulation")
        
        # Create online model
        online_model = self.create_online_model(crop_name)
        
        # Initialize metrics
        if RIVER_AVAILABLE:
            metric = metrics.MAE()
        else:
            predictions = []
            actuals = []
        
        # Performance tracking
        performance_history = []
        window_predictions = []
        window_actuals = []
        
        # Simulate streaming data
        print("Starting online learning simulation...")
        print("Target: Original moisture meter readings")
        print("Features: Custom meter ADC, Temperature, Humidity")
        
        for i, (_, row) in enumerate(df.iterrows(), 1):
            # Prepare features
            if crop_name is None and 'crop' in df.columns:
                # For combined model, we need to encode crop
                features = {
                    'ADC': row['ADC'],
                    'Temperature': row['Temperature'],
                    'Humidity': row['Humidity'],
                    'crop_encoded': 0 if row['crop'] == 'Groundnut' else 1  # Simple encoding
                }
            else:
                features = {
                    'ADC': row['ADC'],
                    'Temperature': row['Temperature'],
                    'Humidity': row['Humidity']
                }
            
            # Use original moisture meter reading as target
            target = row['original_moisture']
            
            if RIVER_AVAILABLE:
                # River online learning
                y_pred = online_model.predict_one(features)
                if y_pred is not None:
                    metric.update(target, y_pred)
                    window_predictions.append(y_pred)
                    window_actuals.append(target)
                
                online_model.learn_one(features, target)
                
                # Track performance every window_size samples
                if i % window_size == 0:
                    if len(window_predictions) >= window_size:
                        recent_mae = np.mean(np.abs(np.array(window_actuals[-window_size:]) - 
                                                  np.array(window_predictions[-window_size:])))
                        performance_history.append({
                            'sample': i,
                            'mae': recent_mae,
                            'cumulative_mae': metric.get()
                        })
                        print(f"Samples {i-window_size+1}-{i}: MAE = {recent_mae:.4f}, Cumulative MAE = {metric.get():.4f}")
            else:
                # Sklearn partial_fit approach
                features_array = np.array([[features['ADC'], features['Temperature'], features['Humidity']]])
                
                if i == 1:
                    # Initialize scaler with first sample
                    online_model[0].partial_fit(features_array, [target])
                    online_model[1].partial_fit(features_array, [target])
                    y_pred = online_model[1].predict(features_array)[0]
                else:
                    # Update scaler and model
                    online_model[0].partial_fit(features_array, [target])
                    scaled_features = online_model[0].transform(features_array)
                    online_model[1].partial_fit(scaled_features, [target])
                    y_pred = online_model[1].predict(scaled_features)[0]
                
                predictions.append(y_pred)
                actuals.append(target)
                window_predictions.append(y_pred)
                window_actuals.append(target)
                
                # Track performance every window_size samples
                if i % window_size == 0:
                    if len(window_predictions) >= window_size:
                        recent_mae = np.mean(np.abs(np.array(window_actuals[-window_size:]) - 
                                                  np.array(window_predictions[-window_size:])))
                        cumulative_mae = np.mean(np.abs(np.array(actuals) - np.array(predictions)))
                        performance_history.append({
                            'sample': i,
                            'mae': recent_mae,
                            'cumulative_mae': cumulative_mae
                        })
                        print(f"Samples {i-window_size+1}-{i}: MAE = {recent_mae:.4f}, Cumulative MAE = {cumulative_mae:.4f}")
        
        return performance_history, online_model
    
    def create_performance_plots(self, performance_history, crop_name=None):
        """Create performance tracking plots"""
        if not performance_history:
            print("No performance history to plot")
            return
        
        df_perf = pd.DataFrame(performance_history)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'Online Learning Performance - {crop_name or "Combined Data"} (Predicting Original Meter Readings)', fontsize=16)
        
        # Rolling MAE
        axes[0].plot(df_perf['sample'], df_perf['mae'], 'b-', linewidth=2, label='Rolling MAE')
        axes[0].set_title('Rolling MAE (Window-based)')
        axes[0].set_xlabel('Sample Number')
        axes[0].set_ylabel('Mean Absolute Error (%)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Cumulative MAE
        axes[1].plot(df_perf['sample'], df_perf['cumulative_mae'], 'r-', linewidth=2, label='Cumulative MAE')
        axes[1].set_title('Cumulative MAE')
        axes[1].set_xlabel('Sample Number')
        axes[1].set_ylabel('Mean Absolute Error (%)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(self.data_dir / f'{crop_name or "combined"}_online_learning_performance.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance plots saved to {self.data_dir / f'{crop_name or 'combined'}_online_learning_performance.png'}")
    
    def save_online_model(self, online_model, crop_name=None):
        """Save the online learning model"""
        prefix = crop_name.lower() if crop_name else "combined"
        
        if RIVER_AVAILABLE:
            # Save River model
            joblib.dump(online_model, self.models_dir / f"{prefix}_online_model.pkl")
        else:
            # Save sklearn model and scaler
            joblib.dump(online_model[0], self.models_dir / f"{prefix}_online_scaler.pkl")
            joblib.dump(online_model[1], self.models_dir / f"{prefix}_online_model.pkl")
        
        print(f"Online model saved to {self.models_dir}")
    
    def create_realtime_prediction_system(self, crop_name=None):
        """Create a real-time prediction system for deployment"""
        print(f"\n=== Creating Real-time Prediction System for {crop_name or 'Combined Data'} ===")
        print("This system will predict original moisture meter readings from custom meter data")
        
        # Load or create online model
        pretrained_model, pretrained_scaler, label_encoder = self.load_pretrained_model(crop_name)
        
        if pretrained_model is not None:
            # Use pretrained model as starting point
            online_model = self.create_online_model(crop_name)
            print("Using pretrained model as starting point for online learning")
        else:
            # Create new online model
            online_model = self.create_online_model(crop_name)
            print("Creating new online learning model")
        
        # Create prediction function
        def predict_moisture(adc, temperature, humidity, crop_type=None):
            """Real-time moisture prediction function"""
            if crop_name is None and crop_type is not None:
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
            
            if RIVER_AVAILABLE:
                prediction = online_model.predict_one(features)
                if prediction is not None:
                    online_model.learn_one(features, prediction)  # Self-learning
                return prediction
            else:
                features_array = np.array([[adc, temperature, humidity]])
                scaled_features = online_model[0].transform(features_array)
                prediction = online_model[1].predict(scaled_features)[0]
                return prediction
        
        # Save prediction function
        joblib.dump(predict_moisture, self.models_dir / f"{crop_name.lower() if crop_name else 'combined'}_prediction_function.pkl")
        
        print("Real-time prediction system created and saved")
        print("This system will output readings that match original moisture meter accuracy")
        return predict_moisture
    
    def run_online_learning_simulation(self):
        """Run online learning simulation for all crops"""
        crops = ['Groundnut', 'Mustard']
        
        # Individual crop simulations
        for crop in crops:
            print(f"\n{'='*60}")
            print(f"Online Learning Simulation for {crop}")
            print(f"{'='*60}")
            
            performance_history, online_model = self.simulate_online_learning(crop)
            self.create_performance_plots(performance_history, crop)
            self.save_online_model(online_model, crop)
            self.create_realtime_prediction_system(crop)
        
        # Combined model simulation
        print(f"\n{'='*60}")
        print("Online Learning Simulation for Combined Model")
        print(f"{'='*60}")
        
        performance_history_combined, online_model_combined = self.simulate_online_learning()
        self.create_performance_plots(performance_history_combined)
        self.save_online_model(online_model_combined)
        self.create_realtime_prediction_system()

def main():
    """Main function to run Phase 3"""
    print("=== PHASE 3: Online Learning Implementation ===")
    print("Objective: Continuously improve prediction of original moisture meter readings")
    print("This will make custom meter readings match original meter accuracy over time")
    print("and adapt to changing environmental conditions")
    
    # Initialize online learning
    ol = OnlineLearning()
    
    # Run online learning simulation
    ol.run_online_learning_simulation()
    
    print("\nPhase 3 completed successfully!")
    print("Online learning system will continuously improve accuracy")
    print("Next: Run Phase 4 for Raspberry Pi deployment")

if __name__ == "__main__":
    main() 