#!/usr/bin/env python3
"""
ML Moisture Meter - Complete Implementation
Runs all phases sequentially for the ML-based moisture meter system.
"""

import sys
import time
from pathlib import Path

def main():
    """Run all phases of the ML moisture meter implementation"""
    print("="*80)
    print("ML MOISTURE METER - COMPLETE IMPLEMENTATION")
    print("="*80)
    print("This script will run all phases sequentially:")
    print("Phase 1: Data Preparation & Exploration")
    print("Phase 2: Model Training & Evaluation")
    print("Phase 3: Online Learning Implementation")
    print("Phase 4: Raspberry Pi Deployment")
    print("="*80)
    
    # Check if we're in the right directory
    if not Path("new_ML").exists():
        print("Error: 'new_ML' directory not found!")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    # Phase 1: Data Preparation
    print("\n" + "="*60)
    print("PHASE 1: Data Preparation & Exploration")
    print("="*60)
    
    try:
        from phase1_data_preparation import main as phase1_main
        phase1_main()
        print("✓ Phase 1 completed successfully!")
    except Exception as e:
        print(f"✗ Phase 1 failed: {e}")
        sys.exit(1)
    
    # Phase 2: Model Training
    print("\n" + "="*60)
    print("PHASE 2: Model Training & Evaluation")
    print("="*60)
    
    try:
        from phase2_model_training import main as phase2_main
        phase2_main()
        print("✓ Phase 2 completed successfully!")
    except Exception as e:
        print(f"✗ Phase 2 failed: {e}")
        sys.exit(1)
    
    # Phase 3: Online Learning
    print("\n" + "="*60)
    print("PHASE 3: Online Learning Implementation")
    print("="*60)
    
    try:
        from phase3_online_learning import main as phase3_main
        phase3_main()
        print("✓ Phase 3 completed successfully!")
    except Exception as e:
        print(f"✗ Phase 3 failed: {e}")
        sys.exit(1)
    
    # Phase 4: Deployment
    print("\n" + "="*60)
    print("PHASE 4: Raspberry Pi Deployment")
    print("="*60)
    
    try:
        from phase4_raspberry_pi_deployment import main as phase4_main
        phase4_main()
        print("✓ Phase 4 completed successfully!")
    except Exception as e:
        print(f"✗ Phase 4 failed: {e}")
        sys.exit(1)
    
    # Summary
    print("\n" + "="*80)
    print("🎉 ALL PHASES COMPLETED SUCCESSFULLY! 🎉")
    print("="*80)
    print("\nGenerated Files:")
    print("📁 new_ML/ - All data and analysis files")
    print("📁 new_ML/render_api/ - Trained ML models")
    print("📁 new_ML/deployment/ - Raspberry Pi/Arduino code")
    print("\nNext Steps:")
    print("1. Review the generated visualizations in new_ML/")
    print("2. Check model performance in new_ML/render_api/")
    print("3. Deploy the Arduino code to your moisture meter")
    print("4. Monitor online learning performance")
    print("\nFor individual phase execution, run:")
    print("  python new_ML/phase1_data_preparation.py")
    print("  python new_ML/phase2_model_training.py")
    print("  python new_ML/phase3_online_learning.py")
    print("  python new_ML/phase4_raspberry_pi_deployment.py")

if __name__ == "__main__":
    main() 