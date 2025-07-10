#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataPreparation:
    def __init__(self, data_dir="."):
        self.data_dir = Path(data_dir)
        self.output_dir = Path("new_ML")
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def clean_data(self, df, data_type):
        """Clean and validate data with enhanced outlier detection"""
        print(f"  Cleaning {data_type} data...")
        
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Convert numeric columns and handle errors
        numeric_columns = ['Moisture', 'ADC', 'Temperature', 'Humidity']
        
        for col in numeric_columns:
            if col in df_clean.columns:
                # Convert to numeric, errors='coerce' will turn invalid values to NaN
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                
                # Count and remove NaN values
                nan_count = df_clean[col].isna().sum()
                if nan_count > 0:
                    print(f"    Removed {nan_count} invalid values from {col}")
                    df_clean = df_clean.dropna(subset=[col])
        
        # Additional cleaning for specific issues
        if 'Humidity' in df_clean.columns:
            # Fix the specific corrupted value '53.0012.16'
            corrupted_mask = df_clean['Humidity'].astype(str).str.contains('53\.0012\.16', na=False)
            if corrupted_mask.any():
                print(f"    Found {corrupted_mask.sum()} corrupted humidity values, fixing...")
                # Replace with the correct humidity value (53)
                df_clean.loc[corrupted_mask, 'Humidity'] = 53
        
        # Enhanced outlier detection for Mustard data
        if 'Mustard' in data_type:
            df_clean = self.clean_mustard_outliers(df_clean)
        
        # Remove any remaining rows with invalid data
        initial_count = len(df_clean)
        df_clean = df_clean.dropna()
        final_count = len(df_clean)
        
        if initial_count != final_count:
            print(f"    Removed {initial_count - final_count} rows with invalid data")
        
        print(f"    Clean data: {len(df_clean)} samples")
        return df_clean
    
    def clean_mustard_outliers(self, df):
        """Enhanced outlier detection specifically for Mustard data"""
        print("    Applying enhanced Mustard outlier detection...")
        
        initial_count = len(df)
        
        # 1. Remove physically impossible moisture values
        # Moisture should be between 0% and 50% for agricultural products
        moisture_col = None
        if 'Moisture' in df.columns:
            moisture_col = 'Moisture'
        elif 'moisture' in df.columns:
            moisture_col = 'moisture'
        elif 'meter' in df.columns:
            moisture_col = 'meter'
            
        if moisture_col:
            before_moisture = len(df)
            df = df[df[moisture_col] >= 0]
            df = df[df[moisture_col] <= 50]
            after_moisture = len(df)
            if before_moisture != after_moisture:
                print(f"      Removed {before_moisture - after_moisture} impossible moisture values")
        
        # 2. Remove extreme ADC outliers (3 standard deviations)
        adc_col = None
        if 'ADC' in df.columns:
            adc_col = 'ADC'
        elif 'adc' in df.columns:
            adc_col = 'adc'
            
        if adc_col:
            adc_mean = df[adc_col].mean()
            adc_std = df[adc_col].std()
            adc_lower = adc_mean - 3 * adc_std
            adc_upper = adc_mean + 3 * adc_std
            
            before_adc = len(df)
            df = df[df[adc_col] >= adc_lower]
            df = df[df[adc_col] <= adc_upper]
            after_adc = len(df)
            if before_adc != after_adc:
                print(f"      Removed {before_adc - after_adc} extreme ADC outliers")
                print(f"      ADC range: {adc_lower:.0f} to {adc_upper:.0f}")
        
        # 3. Remove temperature outliers (reasonable room temperature)
        temp_col = None
        if 'Temperature' in df.columns:
            temp_col = 'Temperature'
        elif 'temp' in df.columns:
            temp_col = 'temp'
            
        if temp_col:
            before_temp = len(df)
            df = df[df[temp_col] >= 20]
            df = df[df[temp_col] <= 40]
            after_temp = len(df)
            if before_temp != after_temp:
                print(f"      Removed {before_temp - after_temp} temperature outliers")
        
        # 4. Remove humidity outliers (reasonable humidity range)
        humid_col = None
        if 'Humidity' in df.columns:
            humid_col = 'Humidity'
        elif 'humid' in df.columns:
            humid_col = 'humid'
            
        if humid_col:
            before_humid = len(df)
            df = df[df[humid_col] >= 30]
            df = df[df[humid_col] <= 90]
            after_humid = len(df)
            if before_humid != after_humid:
                print(f"      Removed {before_humid - after_humid} humidity outliers")
        
        # 5. Remove duplicate or near-duplicate readings (only if all required columns exist)
        required_cols = []
        if adc_col:
            required_cols.append(adc_col)
        if temp_col:
            required_cols.append(temp_col)
        if humid_col:
            required_cols.append(humid_col)
            
        if len(required_cols) >= 2:  # Need at least 2 columns for meaningful deduplication
            before_dupes = len(df)
            df = df.drop_duplicates(subset=required_cols, keep='first')
            after_dupes = len(df)
            if before_dupes != after_dupes:
                print(f"      Removed {before_dupes - after_dupes} duplicate readings")
        
        final_count = len(df)
        total_removed = initial_count - final_count
        
        if total_removed > 0:
            print(f"      Total outliers removed: {total_removed} ({total_removed/initial_count*100:.1f}% of data)")
            
            # Show final statistics
            if moisture_col:
                print(f"      Final moisture range: {df[moisture_col].min():.2f}% to {df[moisture_col].max():.2f}%")
            if adc_col:
                print(f"      Final ADC range: {df[adc_col].min():.0f} to {df[adc_col].max():.0f}")
            if temp_col:
                print(f"      Final temperature range: {df[temp_col].min():.1f}°C to {df[temp_col].max():.1f}°C")
            if humid_col:
                print(f"      Final humidity range: {df[humid_col].min():.0f}% to {df[humid_col].max():.0f}%")
        
        return df
    
    def load_custom_meter_data(self):
        """Load custom meter data for both crops"""
        print("Loading custom meter data...")
        
        # Load Groundnut custom meter data
        groundnut_custom_file = self.data_dir / "GroundNut_Data_custom_meter.csv"
        if groundnut_custom_file.exists():
            groundnut_custom = pd.read_csv(groundnut_custom_file)
            print(f"  Groundnut custom meter: {len(groundnut_custom)} samples")
            # Clean the data
            groundnut_custom = self.clean_data(groundnut_custom, "Groundnut custom meter")
        else:
            print("  Warning: Groundnut custom meter data not found")
            groundnut_custom = None
            
        # Load Mustard custom meter data
        mustard_custom_file = self.data_dir / "Mustard_Data_custom_meter.csv"
        if mustard_custom_file.exists():
            mustard_custom = pd.read_csv(mustard_custom_file)
            print(f"  Mustard custom meter: {len(mustard_custom)} samples")
            # Clean the data
            mustard_custom = self.clean_data(mustard_custom, "Mustard custom meter")
        else:
            print("  Warning: Mustard custom meter data not found")
            mustard_custom = None
            
        return groundnut_custom, mustard_custom
    
    def load_original_meter_data(self):
        """Load original moisture meter data for both crops with enhanced cleaning"""
        print("Loading original moisture meter data...")
        
        # Load Groundnut original meter data (2 samples)
        groundnut_original_files = [
            self.data_dir / "Groundut_Original_moisture_meter_sample_1.csv",
            self.data_dir / "Groundut_Original_moisture_meter_sample_2.csv"
        ]
        
        groundnut_original_data = []
        for i, file_path in enumerate(groundnut_original_files):
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['sample'] = i + 1
                # Clean original meter data
                df = self.clean_data(df, f"Groundnut original meter sample {i+1}")
                groundnut_original_data.append(df)
                print(f"  Groundnut original sample {i+1}: {len(df)} samples")
            else:
                print(f"  Warning: Groundnut original sample {i+1} not found")
        
        # Load Mustard original meter data (2 samples)
        mustard_original_files = [
            self.data_dir / "Mustard_original_moisture_meter_sample_1.csv",
            self.data_dir / "Mustard_original_moisture_meter_sample_2.csv"
        ]
        
        mustard_original_data = []
        for i, file_path in enumerate(mustard_original_files):
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['sample'] = i + 1
                # Clean original meter data with enhanced cleaning for Mustard
                df = self.clean_data(df, f"Mustard original meter sample {i+1}")
                mustard_original_data.append(df)
                print(f"  Mustard original sample {i+1}: {len(df)} samples")
            else:
                print(f"  Warning: Mustard original sample {i+1} not found")
        
        return groundnut_original_data, mustard_original_data
    
    def standardize_column_names(self, df, crop_type):
        """Standardize column names across datasets"""
        # Check what columns are actually present
        print(f"  Original columns in {crop_type} data: {list(df.columns)}")
        
        # Create a mapping based on what's actually in the data
        column_mapping = {}
        
        # Map temperature columns
        if 'temp' in df.columns:
            column_mapping['temp'] = 'Temperature'
        elif 'temperature' in df.columns:
            column_mapping['temperature'] = 'Temperature'
            
        # Map humidity columns
        if 'humid' in df.columns:
            column_mapping['humid'] = 'Humidity'
        elif 'humidity' in df.columns:
            column_mapping['humidity'] = 'Humidity'
            
        # Map ADC columns
        if 'adc' in df.columns:
            column_mapping['adc'] = 'ADC'
        elif 'ADC' in df.columns:
            column_mapping['ADC'] = 'ADC'
            
        # Map meter columns
        if 'meter' in df.columns:
            column_mapping['meter'] = 'original_moisture'
        elif 'moisture' in df.columns:
            column_mapping['moisture'] = 'original_moisture'
        
        print(f"  Column mapping: {column_mapping}")
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
            print(f"  Standardized columns: {list(df.columns)}")
        
        return df
    
    def merge_datasets(self, custom_data, original_data_list, crop_name):
        """Merge custom meter data with original meter data"""
        print(f"\nMerging {crop_name} datasets...")
        
        if custom_data is None or not original_data_list:
            print(f"  Warning: Missing data for {crop_name}")
            return None
        
        # Standardize custom meter data
        custom_data = custom_data.copy()
        custom_data['crop'] = crop_name
        custom_data['target_source'] = 'custom_meter'
        
        # Combine original meter data
        original_combined = []
        for i, original_df in enumerate(original_data_list):
            if original_df is not None:
                # Standardize column names
                original_df = self.standardize_column_names(original_df, crop_name.lower())
                original_df['crop'] = crop_name
                original_df['target_source'] = 'original_meter'
                original_df['sample'] = i + 1
                original_combined.append(original_df)
        
        if not original_combined:
            print(f"  Warning: No original meter data for {crop_name}")
            return custom_data
        
        original_combined = pd.concat(original_combined, ignore_index=True)
        
        # Print column names for debugging
        print(f"  Custom meter columns: {list(custom_data.columns)}")
        print(f"  Original meter columns: {list(original_combined.columns)}")
        
        # Merge based on matching ADC, Temperature, and Humidity values
        print(f"  Custom meter data: {len(custom_data)} samples")
        print(f"  Original meter data: {len(original_combined)} samples")
        
        # Create merged dataset
        merged_data = []
        
        for _, custom_row in custom_data.iterrows():
            # Find matching original meter readings
            matching_original = original_combined[
                (original_combined['ADC'] == custom_row['ADC']) &
                (original_combined['Temperature'] == custom_row['Temperature']) &
                (original_combined['Humidity'] == custom_row['Humidity'])
            ]
            
            if len(matching_original) > 0:
                # Use the first match (or average if multiple)
                original_moisture = matching_original['original_moisture'].iloc[0]
                merged_data.append({
                    'Moisture': custom_row['Moisture'],
                    'ADC': custom_row['ADC'],
                    'Temperature': custom_row['Temperature'],
                    'Humidity': custom_row['Humidity'],
                    'crop': crop_name,
                    'original_moisture': original_moisture,
                    'target_source': 'merged'
                })
            else:
                # No match found, use custom meter reading as target
                merged_data.append({
                    'Moisture': custom_row['Moisture'],
                    'ADC': custom_row['ADC'],
                    'Temperature': custom_row['Temperature'],
                    'Humidity': custom_row['Humidity'],
                    'crop': crop_name,
                    'original_moisture': custom_row['Moisture'],
                    'target_source': 'custom_meter'
                })
        
        merged_df = pd.DataFrame(merged_data)
        print(f"  Merged data: {len(merged_df)} samples")
        print(f"  Target source distribution:")
        print(merged_df['target_source'].value_counts())
        
        return merged_df
    
    def explore_data(self, groundnut_data, mustard_data):
        """Explore and visualize the data"""
        print("\n=== Data Exploration ===")
        
        # Create comprehensive visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Moisture Meter Data Exploration', fontsize=16, fontweight='bold')
        
        # Groundnut data analysis
        if groundnut_data is not None:
            # Moisture distribution
            axes[0, 0].hist(groundnut_data['Moisture'], bins=30, alpha=0.7, label='Custom Meter')
            axes[0, 0].hist(groundnut_data['original_moisture'], bins=30, alpha=0.7, label='Original Meter')
            axes[0, 0].set_title('Groundnut: Moisture Distribution')
            axes[0, 0].set_xlabel('Moisture (%)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
            
            # ADC vs Moisture
            axes[0, 1].scatter(groundnut_data['ADC'], groundnut_data['Moisture'], alpha=0.6, label='Custom')
            axes[0, 1].scatter(groundnut_data['ADC'], groundnut_data['original_moisture'], alpha=0.6, label='Original')
            axes[0, 1].set_title('Groundnut: ADC vs Moisture')
            axes[0, 1].set_xlabel('ADC Value')
            axes[0, 1].set_ylabel('Moisture (%)')
            axes[0, 1].legend()
            
            # Temperature vs Moisture
            axes[0, 2].scatter(groundnut_data['Temperature'], groundnut_data['Moisture'], alpha=0.6, label='Custom')
            axes[0, 2].scatter(groundnut_data['Temperature'], groundnut_data['original_moisture'], alpha=0.6, label='Original')
            axes[0, 2].set_title('Groundnut: Temperature vs Moisture')
            axes[0, 2].set_xlabel('Temperature (°C)')
            axes[0, 2].set_ylabel('Moisture (%)')
            axes[0, 2].legend()
        
        # Mustard data analysis
        if mustard_data is not None:
            # Moisture distribution
            axes[1, 0].hist(mustard_data['Moisture'], bins=30, alpha=0.7, label='Custom Meter')
            axes[1, 0].hist(mustard_data['original_moisture'], bins=30, alpha=0.7, label='Original Meter')
            axes[1, 0].set_title('Mustard: Moisture Distribution')
            axes[1, 0].set_xlabel('Moisture (%)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
            
            # ADC vs Moisture
            axes[1, 1].scatter(mustard_data['ADC'], mustard_data['Moisture'], alpha=0.6, label='Custom')
            axes[1, 1].scatter(mustard_data['ADC'], mustard_data['original_moisture'], alpha=0.6, label='Original')
            axes[1, 1].set_title('Mustard: ADC vs Moisture')
            axes[1, 1].set_xlabel('ADC Value')
            axes[1, 1].set_ylabel('Moisture (%)')
            axes[1, 1].legend()
            
            # Temperature vs Moisture
            axes[1, 2].scatter(mustard_data['Temperature'], mustard_data['Moisture'], alpha=0.6, label='Custom')
            axes[1, 2].scatter(mustard_data['Temperature'], mustard_data['original_moisture'], alpha=0.6, label='Original')
            axes[1, 2].set_title('Mustard: Temperature vs Moisture')
            axes[1, 2].set_xlabel('Temperature (°C)')
            axes[1, 2].set_ylabel('Moisture (%)')
            axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'combined_data_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print("\n=== Summary Statistics ===")
        
        if groundnut_data is not None:
            print(f"\nGroundnut Data:")
            print(f"  Total samples: {len(groundnut_data)}")
            print(f"  Custom meter range: {groundnut_data['Moisture'].min():.2f} - {groundnut_data['Moisture'].max():.2f}%")
            print(f"  Original meter range: {groundnut_data['original_moisture'].min():.2f} - {groundnut_data['original_moisture'].max():.2f}%")
            print(f"  Temperature range: {groundnut_data['Temperature'].min():.1f} - {groundnut_data['Temperature'].max():.1f}°C")
            print(f"  Humidity range: {groundnut_data['Humidity'].min()} - {groundnut_data['Humidity'].max()}%")
            print(f"  ADC range: {groundnut_data['ADC'].min()} - {groundnut_data['ADC'].max()}")
        
        if mustard_data is not None:
            print(f"\nMustard Data:")
            print(f"  Total samples: {len(mustard_data)}")
            print(f"  Custom meter range: {mustard_data['Moisture'].min():.2f} - {mustard_data['Moisture'].max():.2f}%")
            print(f"  Original meter range: {mustard_data['original_moisture'].min():.2f} - {mustard_data['original_moisture'].max():.2f}%")
            print(f"  Temperature range: {mustard_data['Temperature'].min():.1f} - {mustard_data['Temperature'].max():.1f}°C")
            print(f"  Humidity range: {mustard_data['Humidity'].min()} - {mustard_data['Humidity'].max()}%")
            print(f"  ADC range: {mustard_data['ADC'].min()} - {mustard_data['ADC'].max()}")
    
    def create_individual_crop_plots(self, groundnut_data, mustard_data):
        """Create detailed plots for each crop"""
        print("\nCreating individual crop analysis plots...")
        
        # Groundnut analysis
        if groundnut_data is not None:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Groundnut Data Analysis', fontsize=16, fontweight='bold')
            
            # Moisture comparison
            axes[0, 0].scatter(groundnut_data['Moisture'], groundnut_data['original_moisture'], alpha=0.6)
            axes[0, 0].plot([groundnut_data['Moisture'].min(), groundnut_data['Moisture'].max()], 
                           [groundnut_data['Moisture'].min(), groundnut_data['Moisture'].max()], 'r--', lw=2)
            axes[0, 0].set_xlabel('Custom Meter Moisture (%)')
            axes[0, 0].set_ylabel('Original Meter Moisture (%)')
            axes[0, 0].set_title('Custom vs Original Meter Readings')
            axes[0, 0].grid(True, alpha=0.3)
            
            # ADC vs Moisture
            axes[0, 1].scatter(groundnut_data['ADC'], groundnut_data['Moisture'], alpha=0.6, label='Custom')
            axes[0, 1].scatter(groundnut_data['ADC'], groundnut_data['original_moisture'], alpha=0.6, label='Original')
            axes[0, 1].set_xlabel('ADC Value')
            axes[0, 1].set_ylabel('Moisture (%)')
            axes[0, 1].set_title('ADC vs Moisture')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Temperature vs Moisture
            axes[1, 0].scatter(groundnut_data['Temperature'], groundnut_data['Moisture'], alpha=0.6, label='Custom')
            axes[1, 0].scatter(groundnut_data['Temperature'], groundnut_data['original_moisture'], alpha=0.6, label='Original')
            axes[1, 0].set_xlabel('Temperature (°C)')
            axes[1, 0].set_ylabel('Moisture (%)')
            axes[1, 0].set_title('Temperature vs Moisture')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Moisture distribution
            axes[1, 1].hist(groundnut_data['Moisture'], bins=30, alpha=0.7, label='Custom Meter')
            axes[1, 1].hist(groundnut_data['original_moisture'], bins=30, alpha=0.7, label='Original Meter')
            axes[1, 1].set_xlabel('Moisture (%)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Moisture Distribution')
            axes[1, 1].legend()
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'Groundnut_data_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Mustard analysis
        if mustard_data is not None:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Mustard Data Analysis', fontsize=16, fontweight='bold')
            
            # Moisture comparison
            axes[0, 0].scatter(mustard_data['Moisture'], mustard_data['original_moisture'], alpha=0.6)
            axes[0, 0].plot([mustard_data['Moisture'].min(), mustard_data['Moisture'].max()], 
                           [mustard_data['Moisture'].min(), mustard_data['Moisture'].max()], 'r--', lw=2)
            axes[0, 0].set_xlabel('Custom Meter Moisture (%)')
            axes[0, 0].set_ylabel('Original Meter Moisture (%)')
            axes[0, 0].set_title('Custom vs Original Meter Readings')
            axes[0, 0].grid(True, alpha=0.3)
            
            # ADC vs Moisture
            axes[0, 1].scatter(mustard_data['ADC'], mustard_data['Moisture'], alpha=0.6, label='Custom')
            axes[0, 1].scatter(mustard_data['ADC'], mustard_data['original_moisture'], alpha=0.6, label='Original')
            axes[0, 1].set_xlabel('ADC Value')
            axes[0, 1].set_ylabel('Moisture (%)')
            axes[0, 1].set_title('ADC vs Moisture')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Temperature vs Moisture
            axes[1, 0].scatter(mustard_data['Temperature'], mustard_data['Moisture'], alpha=0.6, label='Custom')
            axes[1, 0].scatter(mustard_data['Temperature'], mustard_data['original_moisture'], alpha=0.6, label='Original')
            axes[1, 0].set_xlabel('Temperature (°C)')
            axes[1, 0].set_ylabel('Moisture (%)')
            axes[1, 0].set_title('Temperature vs Moisture')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Moisture distribution
            axes[1, 1].hist(mustard_data['Moisture'], bins=30, alpha=0.7, label='Custom Meter')
            axes[1, 1].hist(mustard_data['original_moisture'], bins=30, alpha=0.7, label='Original Meter')
            axes[1, 1].set_xlabel('Moisture (%)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Moisture Distribution')
            axes[1, 1].legend()
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'Mustard_data_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def save_dataset_statistics_table(self, groundnut_data, mustard_data):
        """Save dataset statistics in table format for research paper"""
        print("\n=== Saving Dataset Statistics Table ===")
        
        # Create comprehensive dataset statistics
        stats_data = []
        
        # Groundnut statistics
        if groundnut_data is not None:
            stats_data.append({
                'Crop': 'Groundnut',
                'Total Samples': len(groundnut_data),
                'Moisture Range (%)': f"{groundnut_data['original_moisture'].min():.2f} - {groundnut_data['original_moisture'].max():.2f}",
                'ADC Range': f"{groundnut_data['ADC'].min():.0f} - {groundnut_data['ADC'].max():.0f}",
                'Temperature Range (°C)': f"{groundnut_data['Temperature'].min():.1f} - {groundnut_data['Temperature'].max():.1f}",
                'Humidity Range (%)': f"{groundnut_data['Humidity'].min():.1f} - {groundnut_data['Humidity'].max():.1f}",
                'Mean Moisture (%)': f"{groundnut_data['original_moisture'].mean():.2f}",
                'Std Moisture (%)': f"{groundnut_data['original_moisture'].std():.2f}",
                'Mean ADC': f"{groundnut_data['ADC'].mean():.1f}",
                'Std ADC': f"{groundnut_data['ADC'].std():.1f}"
            })
        
        # Mustard statistics
        if mustard_data is not None:
            stats_data.append({
                'Crop': 'Mustard',
                'Total Samples': len(mustard_data),
                'Moisture Range (%)': f"{mustard_data['original_moisture'].min():.2f} - {mustard_data['original_moisture'].max():.2f}",
                'ADC Range': f"{mustard_data['ADC'].min():.0f} - {mustard_data['ADC'].max():.0f}",
                'Temperature Range (°C)': f"{mustard_data['Temperature'].min():.1f} - {mustard_data['Temperature'].max():.1f}",
                'Humidity Range (%)': f"{mustard_data['Humidity'].min():.1f} - {mustard_data['Humidity'].max():.1f}",
                'Mean Moisture (%)': f"{mustard_data['original_moisture'].mean():.2f}",
                'Std Moisture (%)': f"{mustard_data['original_moisture'].std():.2f}",
                'Mean ADC': f"{mustard_data['ADC'].mean():.1f}",
                'Std ADC': f"{mustard_data['ADC'].std():.1f}"
            })
        
        # Combined statistics
        if groundnut_data is not None and mustard_data is not None:
            combined_data = pd.concat([groundnut_data, mustard_data], ignore_index=True)
            stats_data.append({
                'Crop': 'Combined',
                'Total Samples': len(combined_data),
                'Moisture Range (%)': f"{combined_data['original_moisture'].min():.2f} - {combined_data['original_moisture'].max():.2f}",
                'ADC Range': f"{combined_data['ADC'].min():.0f} - {combined_data['ADC'].max():.0f}",
                'Temperature Range (°C)': f"{combined_data['Temperature'].min():.1f} - {combined_data['Temperature'].max():.1f}",
                'Humidity Range (%)': f"{combined_data['Humidity'].min():.1f} - {combined_data['Humidity'].max():.1f}",
                'Mean Moisture (%)': f"{combined_data['original_moisture'].mean():.2f}",
                'Std Moisture (%)': f"{combined_data['original_moisture'].std():.2f}",
                'Mean ADC': f"{combined_data['ADC'].mean():.1f}",
                'Std ADC': f"{combined_data['ADC'].std():.1f}"
            })
        
        # Convert to DataFrame
        stats_df = pd.DataFrame(stats_data)
        
        # Save as CSV for easy import into research paper
        stats_filename = "dataset_statistics_table.csv"
        stats_path = self.output_dir / stats_filename
        stats_df.to_csv(stats_path, index=False)
        print(f"✓ Dataset statistics table saved to: {stats_path}")
        
        # Also save as LaTeX table format
        latex_table = stats_df.to_latex(index=False)
        latex_filename = "dataset_statistics_table.tex"
        latex_path = self.output_dir / latex_filename
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        print(f"✓ LaTeX table saved to: {latex_path}")
        
        # Print formatted table
        print(f"\nDataset Statistics:")
        print("=" * 120)
        print(stats_df.to_string(index=False))
        print("=" * 120)
        
        return stats_df
    
    def save_processed_data(self, groundnut_data, mustard_data):
        """Save processed datasets"""
        print("\n=== Saving Processed Data ===")
        
        # Save individual crop datasets
        if groundnut_data is not None:
            groundnut_file = self.output_dir / "Groundnut_merged_data.csv"
            groundnut_data.to_csv(groundnut_file, index=False)
            print(f"✓ Groundnut data saved: {len(groundnut_data)} samples")
        
        if mustard_data is not None:
            mustard_file = self.output_dir / "Mustard_merged_data.csv"
            mustard_data.to_csv(mustard_file, index=False)
            print(f"✓ Mustard data saved: {len(mustard_data)} samples")
        
        # Save combined dataset
        if groundnut_data is not None and mustard_data is not None:
            combined_data = pd.concat([groundnut_data, mustard_data], ignore_index=True)
            combined_file = self.output_dir / "combined_crops_data.csv"
            combined_data.to_csv(combined_file, index=False)
            print(f"✓ Combined data saved: {len(combined_data)} samples")
            
            # Save dataset statistics table
            stats_table = self.save_dataset_statistics_table(groundnut_data, mustard_data)
        
        print(f"✓ All processed data saved to {self.output_dir}")
        
        return stats_table if 'stats_table' in locals() else None
    
    def run_data_preparation(self):
        """Run complete data preparation pipeline"""
        print("="*80)
        print("PHASE 1: DATA PREPARATION & EXPLORATION")
        print("="*80)
        print("Objective: Prepare and merge custom meter data with original meter readings")
        print("This will create training datasets that predict original meter accuracy")
        print("="*80)
        
        # Load data
        groundnut_custom, mustard_custom = self.load_custom_meter_data()
        groundnut_original, mustard_original = self.load_original_meter_data()
        
        # Merge datasets
        groundnut_merged = self.merge_datasets(groundnut_custom, groundnut_original, "Groundnut")
        mustard_merged = self.merge_datasets(mustard_custom, mustard_original, "Mustard")
        
        # Explore data
        self.explore_data(groundnut_merged, mustard_merged)
        self.create_individual_crop_plots(groundnut_merged, mustard_merged)
        
        # Save processed data
        self.save_processed_data(groundnut_merged, mustard_merged)
        
        return groundnut_merged, mustard_merged

def main():
    """Main function to run data preparation"""
    data_prep = DataPreparation()
    groundnut_data, mustard_data = data_prep.run_data_preparation()
    
    print("\n" + "="*80)
    print("PHASE 1 COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nNext steps:")
    print("1. Review the generated visualizations")
    print("2. Run Phase 2: Model Training")
    print("   python new_ML/phase2_model_training.py")
    print("\nOr run all phases:")
    print("   python new_ML/run_all_phases.py")

if __name__ == "__main__":
    main()
