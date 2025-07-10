#!/usr/bin/env python3


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ResearchTablesGenerator:
    def __init__(self, data_dir="new_ML", models_dir="models"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.output_dir = Path("research_tables")
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def generate_all_tables(self):
        """Generate all research tables"""
        print("Generating research tables...")
        
        # Generate dataset statistics
        self.generate_dataset_statistics()
        
        # Generate model performance tables
        self.generate_model_performance_tables()
        
        print("✅ All tables generated successfully!")
    
    def generate_dataset_statistics(self):
        """Generate dataset statistics table"""
        # Load data
        groundnut_data = pd.read_csv(self.data_dir / "Groundnut_merged_data.csv")
        mustard_data = pd.read_csv(self.data_dir / "Mustard_merged_data.csv")
        
        # Create statistics table
        stats_data = [
            {
                'Crop': 'Groundnut',
                'Samples': len(groundnut_data),
                'Moisture Range (%)': f"{groundnut_data['original_moisture'].min():.2f} - {groundnut_data['original_moisture'].max():.2f}",
                'ADC Range': f"{groundnut_data['ADC'].min():.0f} - {groundnut_data['ADC'].max():.0f}",
                'Mean Moisture (%)': f"{groundnut_data['original_moisture'].mean():.2f}",
                'Std Moisture (%)': f"{groundnut_data['original_moisture'].std():.2f}"
            },
            {
                'Crop': 'Mustard',
                'Samples': len(mustard_data),
                'Moisture Range (%)': f"{mustard_data['original_moisture'].min():.2f} - {mustard_data['original_moisture'].max():.2f}",
                'ADC Range': f"{mustard_data['ADC'].min():.0f} - {mustard_data['ADC'].max():.0f}",
                'Mean Moisture (%)': f"{mustard_data['original_moisture'].mean():.2f}",
                'Std Moisture (%)': f"{mustard_data['original_moisture'].std():.2f}"
            }
        ]
        
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_csv(self.output_dir / "dataset_statistics.csv", index=False)
        print("✓ Dataset statistics table generated")
    
    def generate_model_performance_tables(self):
        """Generate model performance tables"""
        # Load results
        groundnut_results = joblib.load(self.models_dir / "groundnut_results_summary.pkl")
        mustard_results = joblib.load(self.models_dir / "mustard_results_summary.pkl")
        
        # Create performance tables
        for crop, results in [("Groundnut", groundnut_results), ("Mustard", mustard_results)]:
            table_data = []
            for model_name, metrics in results['all_results'].items():
                table_data.append({
                    'Model': model_name,
                    'MAE (%)': f"{metrics['mae']:.4f}",
                    'RMSE (%)': f"{metrics['rmse']:.4f}",
                    'R² Score': f"{metrics['r2']:.4f}",
                    'CV MAE': f"{metrics['cv_mae']:.4f} ± {metrics['cv_std']:.4f}"
                })
            
            table_df = pd.DataFrame(table_data)
            table_df.to_csv(self.output_dir / f"{crop.lower()}_model_performance.csv", index=False)
            print(f"✓ {crop} model performance table generated")

def main():
    """Main function to generate research tables"""
    generator = ResearchTablesGenerator()
    generator.generate_all_tables()
    
    print("\n📊 RESEARCH TABLES GENERATION COMPLETE")
    print("Use these tables in your research paper:")
    print("1. dataset_statistics.csv - Dataset characteristics")
    print("2. groundnut_model_performance.csv - Groundnut model comparison")
    print("3. mustard_model_performance.csv - Mustard model comparison")

if __name__ == "__main__":
    main() 