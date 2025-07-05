#!/usr/bin/env python3
"""
Script to copy .h5 models from models directory to render_api models directory
"""

import shutil
from pathlib import Path

def copy_h5_models():
    """Copy .h5 models and scalers to render_api models directory"""
    
    # Source and destination directories
    models_dir = Path("new_ML/models")
    render_api_models_dir = Path("new_ML/render_api/models")
    main_dir = Path(".")
    
    # Ensure render_api models directory exists
    render_api_models_dir.mkdir(exist_ok=True)
    
    # Models to copy to render_api
    render_api_models = [
        ("model_groundnut_new.h5", "model_groundnut.h5"),
        ("model_mustard_new.h5", "model_mustard.h5"),
        ("groundnut_scaler_new.pkl", "groundnut_scaler.pkl"),
        ("mustard_scaler_new.pkl", "mustard_scaler.pkl")
    ]
    
    # Models to copy to main directory
    main_models = [
        ("model_groundnut_new.h5", "model_groundnut.h5"),
        ("model_mustard_new.h5", "model_mustard.h5"),
        ("groundnut_scaler_new.pkl", "scaler_groundnut.pkl"),
        ("mustard_scaler_new.pkl", "scaler_mustard.pkl")
    ]
    
    print("=== Copying .h5 Models to Render API Models Directory ===")
    
    for src_name, dst_name in render_api_models:
        src_path = models_dir / src_name
        dst_path = render_api_models_dir / dst_name
        
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            print(f"✓ Copied {src_name} to render_api/models/{dst_name}")
        else:
            print(f"⚠️  {src_name} not found in {models_dir}")
    
    print("\n=== Copying .h5 Models to Main Directory ===")
    
    for src_name, dst_name in main_models:
        src_path = models_dir / src_name
        dst_path = main_dir / dst_name
        
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            print(f"✓ Copied {src_name} to main directory as {dst_name}")
        else:
            print(f"⚠️  {src_name} not found in {models_dir}")
    
    print("\nModels copied successfully!")
    print("✓ Render API models: new_ML/render_api/models/")
    print("✓ Main app models: root directory")
    print("You can now run main_app.py and deploy render_api with the new .h5 models")

if __name__ == "__main__":
    copy_h5_models() 