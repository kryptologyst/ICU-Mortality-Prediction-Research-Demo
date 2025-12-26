#!/usr/bin/env python3
"""Quick start script for ICU mortality prediction demo."""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main quick start function."""
    print("🏥 ICU Mortality Prediction - Quick Start")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ required. Current version:", sys.version)
        return False
    
    print(f"✅ Python version: {sys.version.split()[0]}")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        return False
    
    # Create directories
    directories = ['data/raw', 'data/processed', 'models', 'checkpoints', 'logs', 'results', 'assets/plots']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✅ Directory structure created")
    
    # Train a sample model
    if not run_command("python3 scripts/train.py --config configs/default.yaml --model random_forest", "Training sample model"):
        return False
    
    # Run evaluation
    if not run_command("python3 scripts/evaluate.py --model_path models/random_forest_model.pkl --results_path results/random_forest_results.yaml --config configs/default.yaml", "Running evaluation"):
        return False
    
    print("\n🎉 Quick start completed successfully!")
    print("\nNext steps:")
    print("1. Launch the demo: streamlit run demo/app.py")
    print("2. Try different models: python3 scripts/train.py --model xgboost")
    print("3. Explore the code in src/ directory")
    print("\n⚠️  Remember: This is a research demo only, not for clinical use!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
