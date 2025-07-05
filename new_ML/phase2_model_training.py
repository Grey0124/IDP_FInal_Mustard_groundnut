import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

class ModelTraining:
    def __init__(self, data_dir="new_ML", models_dir="models"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.label_encoder = LabelEncoder()
        
    def load_data(self, crop_name=None):
        """Load data for training"""
        if crop_name:
            # Load specific crop data
            data_file = self.data_dir / f"{crop_name}_merged_data.csv"
            df = pd.read_csv(data_file)
            print(f"Loaded {crop_name} data: {len(df)} samples")
        else:
            # Load combined data
            data_file = self.data_dir / "combined_crops_data.csv"
            df = pd.read_csv(data_file)
            print(f"Loaded combined data: {len(df)} samples")
            print(f"Crops: {df['crop'].value_counts().to_dict()}")
        
        # Check target source
        if 'target_source' in df.columns:
            print(f"Target source distribution:")
            print(df['target_source'].value_counts())
        
        return df
    
    def prepare_features(self, df, crop_name=None):
        """Prepare features and target for training"""
        # Select features (custom meter readings)
        feature_columns = ['ADC', 'Temperature', 'Humidity']
        X = df[feature_columns].copy()
        
        # Use original moisture meter readings as target
        y = df['original_moisture'].copy()
        
        print(f"Features: {feature_columns}")
        print(f"Target: original_moisture (Original moisture meter readings)")
        print(f"Target range: {y.min():.2f} - {y.max():.2f}%")
        
        # Add crop encoding if using combined data
        if crop_name is None and 'crop' in df.columns:
            X['crop_encoded'] = self.label_encoder.fit_transform(df['crop'])
            print(f"Crop encoding: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        return X, y
    
    def train_models(self, X, y, crop_name=None):
        """Train multiple models and compare performance"""
        print(f"\n=== Training Models for {crop_name or 'Combined Data'} ===")
        print("Objective: Predict original moisture meter readings from custom meter data")
        print("This will make custom meter readings match original meter accuracy")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'SGD Regressor': SGDRegressor(max_iter=1000, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf'),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        }
        
        # Train and evaluate models
        results = {}
        best_model = None
        best_score = float('inf')
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Train on full training set
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            
            # Metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'cv_mae': cv_mae,
                'cv_std': cv_std,
                'test_mae': mae,
                'test_rmse': rmse,
                'test_r2': r2,
                'y_pred': y_pred
            }
            
            print(f"  CV MAE: {cv_mae:.4f} ± {cv_std:.4f}")
            print(f"  Test MAE: {mae:.4f}")
            print(f"  Test RMSE: {rmse:.4f}")
            print(f"  Test R²: {r2:.4f}")
            
            # Track best model
            if mae < best_score:
                best_score = mae
                best_model = name
        
        print(f"\nBest model: {best_model} (MAE: {best_score:.4f})")
        print(f"This means the custom meter will be within ±{best_score:.2f}% of the original meter")
        
        return results, scaler, X_test, y_test
    
    def hyperparameter_tuning(self, X, y, crop_name=None):
        """Perform hyperparameter tuning for the best model"""
        print(f"\n=== Hyperparameter Tuning for {crop_name or 'Combined Data'} ===")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define parameter grids for different models
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'SVR': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'epsilon': [0.01, 0.1, 0.2]
            }
        }
        
        best_tuned_model = None
        best_tuned_score = float('inf')
        
        for model_name, param_grid in param_grids.items():
            print(f"\nTuning {model_name}...")
            
            if model_name == 'Random Forest':
                model = RandomForestRegressor(random_state=42)
            elif model_name == 'Gradient Boosting':
                model = GradientBoostingRegressor(random_state=42)
            elif model_name == 'SVR':
                model = SVR()
            
            # Grid search
            grid_search = GridSearchCV(
                model, param_grid, cv=5, scoring='neg_mean_absolute_error',
                n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = grid_search.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            
            print(f"  Best parameters: {grid_search.best_params_}")
            print(f"  Best CV score: {-grid_search.best_score_:.4f}")
            print(f"  Test MAE: {mae:.4f}")
            
            if mae < best_tuned_score:
                best_tuned_score = mae
                best_tuned_model = {
                    'name': model_name,
                    'model': grid_search.best_estimator_,
                    'params': grid_search.best_params_,
                    'score': mae
                }
        
        return best_tuned_model, scaler
    
    def convert_to_tensorflow_model(self, sklearn_model, scaler, X_train, y_train, model_name):
        """Convert scikit-learn model to TensorFlow model by training it to mimic sklearn predictions"""
        print(f"Converting {model_name} to TensorFlow model...")
        
        # Get sklearn model predictions on training data
        X_train_scaled = scaler.transform(X_train)
        sklearn_predictions = sklearn_model.predict(X_train_scaled)
        
        # Create TensorFlow model
        input_shape = (X_train.shape[1],)
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=input_shape),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Train TensorFlow model to mimic sklearn predictions
        # Use sklearn predictions as target (transfer learning)
        model.fit(
            X_train_scaled, sklearn_predictions,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        print(f"✓ TensorFlow model trained to mimic {model_name}")
        return model
    
    def save_scaler_for_tensorflow(self, scaler, prefix):
        """Save scaler in a format compatible with TensorFlow"""
        # Save scaler parameters for TensorFlow compatibility
        scaler_params = {
            'scale_': scaler.scale_,
            'mean_': scaler.mean_,
            'var_': scaler.var_,
            'feature_names_in_': scaler.feature_names_in_.tolist() if hasattr(scaler, 'feature_names_in_') else None
        }
        joblib.dump(scaler_params, self.models_dir / f"scaler_{prefix}_tf_new.pkl")
        print(f"✓ scaler_{prefix}_tf_new.pkl (TensorFlow compatible)")
    
    def save_models(self, results, scaler, best_tuned_model, X_train, y_train, crop_name=None):
        """Save trained models and scaler"""
        prefix = crop_name.lower() if crop_name else "combined"
        
        # Save best model from initial training
        best_model_name = min(results.keys(), key=lambda x: results[x]['test_mae'])
        best_model = results[best_model_name]['model']
        
        # Save scikit-learn model (for compatibility)
        joblib.dump(best_model, self.models_dir / f"{prefix}_best_model_new.pkl")
        joblib.dump(scaler, self.models_dir / f"{prefix}_scaler_new.pkl")
        
        # Convert and save as TensorFlow .h5 model
        tf_model = self.convert_to_tensorflow_model(best_model, scaler, X_train, y_train, best_model_name)
        
        # Save TensorFlow model in .h5 format
        tf_model.save(self.models_dir / f"model_{prefix}_new.h5")
        
        # Save tuned model if available
        if best_tuned_model:
            joblib.dump(best_tuned_model['model'], self.models_dir / f"{prefix}_tuned_model_new.pkl")
            joblib.dump(best_tuned_model['params'], self.models_dir / f"{prefix}_best_params_new.pkl")
            
            # Convert tuned model to TensorFlow
            tf_tuned_model = self.convert_to_tensorflow_model(
                best_tuned_model['model'], scaler, X_train, y_train, best_tuned_model['name']
            )
            tf_tuned_model.save(self.models_dir / f"model_{prefix}_tuned_new.h5")
        
        # Save label encoder if used
        if hasattr(self.label_encoder, 'classes_'):
            joblib.dump(self.label_encoder, self.models_dir / f"{prefix}_label_encoder_new.pkl")
        
        # Save results summary
        results_summary = {}
        for name, result in results.items():
            results_summary[name] = {
                'cv_mae': result['cv_mae'],
                'cv_std': result['cv_std'],
                'test_mae': result['test_mae'],
                'test_rmse': result['test_rmse'],
                'test_r2': result['test_r2']
            }
        
        joblib.dump(results_summary, self.models_dir / f"{prefix}_results_summary_new.pkl")
        
        # Save scaler for TensorFlow
        self.save_scaler_for_tensorflow(scaler, prefix)
        
        print(f"\nModels saved to {self.models_dir}")
        print(f"✓ {prefix}_best_model_new.pkl (scikit-learn)")
        print(f"✓ model_{prefix}_new.h5 (TensorFlow)")
        print(f"✓ {prefix}_scaler_new.pkl (scaler)")
        if best_tuned_model:
            print(f"✓ model_{prefix}_tuned_new.h5 (TensorFlow tuned)")
    
    def create_comparison_plots(self, results, X_test, y_test, crop_name=None):
        """Create comparison plots for model performance"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Model Comparison - {crop_name or "Combined Data"} (Predicting Original Meter Readings)', fontsize=16)
        
        # MAE comparison
        model_names = list(results.keys())
        mae_scores = [results[name]['test_mae'] for name in model_names]
        
        axes[0, 0].bar(model_names, mae_scores, color='skyblue')
        axes[0, 0].set_title('Test MAE Comparison')
        axes[0, 0].set_ylabel('Mean Absolute Error (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # R² comparison
        r2_scores = [results[name]['test_r2'] for name in model_names]
        
        axes[0, 1].bar(model_names, r2_scores, color='lightgreen')
        axes[0, 1].set_title('Test R² Comparison')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Predicted vs Actual (best model)
        best_model_name = min(results.keys(), key=lambda x: results[x]['test_mae'])
        y_pred_best = results[best_model_name]['y_pred']
        
        axes[1, 0].scatter(y_test, y_pred_best, alpha=0.6)
        axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('Actual Original Moisture (%)')
        axes[1, 0].set_ylabel('Predicted Original Moisture (%)')
        axes[1, 0].set_title(f'Predicted vs Actual ({best_model_name})')
        
        # Residuals plot
        residuals = y_test - y_pred_best
        axes[1, 1].scatter(y_pred_best, residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Predicted Original Moisture (%)')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title(f'Residuals Plot ({best_model_name})')
        
        plt.tight_layout()
        crop_name_str = crop_name or "combined"
        plt.savefig(self.data_dir / f'{crop_name_str}_model_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plots saved to {self.data_dir / f'{crop_name_str}_model_comparison.png'}")
    
    def train_all_models(self):
        """Train models for individual crops and combined data"""
        crops = ['Groundnut', 'Mustard']
        
        # Train individual crop models
        for crop in crops:
            print(f"\n{'='*60}")
            print(f"Training models for {crop}")
            print(f"{'='*60}")
            
            # Load data
            df = self.load_data(crop)
            X, y = self.prepare_features(df, crop)
            
            # Train models
            results, scaler, X_test, y_test = self.train_models(X, y, crop)
            
            # Hyperparameter tuning
            best_tuned, tuned_scaler = self.hyperparameter_tuning(X, y, crop)
            
            # Save models
            self.save_models(results, scaler, best_tuned, X, y, crop)
            
            # Create plots
            self.create_comparison_plots(results, X_test, y_test, crop)
        
        # Train combined model
        print(f"\n{'='*60}")
        print("Training combined model")
        print(f"{'='*60}")
        
        df_combined = self.load_data()
        X_combined, y_combined = self.prepare_features(df_combined)
        
        results_combined, scaler_combined, X_test_combined, y_test_combined = self.train_models(
            X_combined, y_combined
        )
        
        best_tuned_combined, tuned_scaler_combined = self.hyperparameter_tuning(
            X_combined, y_combined
        )
        
        self.save_models(results_combined, scaler_combined, best_tuned_combined, X_combined, y_combined)
        self.create_comparison_plots(results_combined, X_test_combined, y_test_combined)

def main():
    """Main function to run Phase 2"""
    print("=== PHASE 2: Model Training & Evaluation ===")
    print("Objective: Train models to predict original moisture meter readings")
    print("This will make custom meter readings match original meter accuracy")
    print("and be less affected by temperature and humidity fluctuations")
    
    # Initialize model training
    mt = ModelTraining()
    
    # Train all models
    mt.train_all_models()
    
    print("\nPhase 2 completed successfully!")
    print("Models now predict original moisture meter readings from custom meter data")
    print("Next: Run Phase 3 for online learning implementation")

if __name__ == "__main__":
    main() 