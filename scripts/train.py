#!/usr/bin/env python3
"""Training script for ICU mortality prediction."""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any
import yaml
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import set_seed, get_device, validate_config, create_directory_structure
from data import ICUDataProcessor
from models import create_model
from metrics import ICUEvaluator
from utils.explainability import ICUExplainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_model(config: Dict[str, Any]) -> Dict[str, Any]:
    """Train ICU mortality prediction model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Training results
    """
    # Set up reproducibility
    set_seed(config['seed'], config['deterministic'])
    
    # Create directory structure
    create_directory_structure('.')
    
    # Initialize data processor
    data_processor = ICUDataProcessor(config)
    
    # Generate synthetic data
    logger.info("Generating synthetic ICU data...")
    df = data_processor.generate_synthetic_data()
    
    # Preprocess data
    logger.info("Preprocessing data...")
    X, y, metadata = data_processor.preprocess_data(df)
    
    # Split data
    logger.info("Splitting data...")
    splits = data_processor.split_data(X, y, df)
    
    # Initialize model
    logger.info(f"Initializing {config['model']['name']} model...")
    model = create_model(config, input_dim=X.shape[1])
    
    # Train model
    logger.info("Training model...")
    model.fit(
        splits['X_train'], splits['y_train'],
        splits['X_val'], splits['y_val']
    )
    
    # Evaluate model
    logger.info("Evaluating model...")
    evaluator = ICUEvaluator(config)
    
    # Test set predictions
    y_pred_proba_full = model.predict_proba(splits['X_test'])
    y_pred_proba = y_pred_proba_full[:, 1]  # Get probability of positive class
    y_pred = model.predict(splits['X_test'])
    
    # Create evaluation report
    evaluation_report = evaluator.create_evaluation_report(
        splits['y_test'], y_pred_proba, config['model']['name']
    )
    
    # Generate explainability
    logger.info("Generating explanations...")
    explainer = ICUExplainer(config, metadata['feature_names'])
    
    # Fit explainers
    explainer.fit_shap_explainer(model, splits['X_train'])
    explainer.fit_lime_explainer(splits['X_train'])
    
    # Create explanation report
    explanation_report = explainer.create_explanation_report(
        splits['X_test'], model, splits['y_test']
    )
    
    # Save results
    results = {
        'config': config,
        'metadata': metadata,
        'evaluation': evaluation_report,
        'explanations': explanation_report,
        'predictions': {
            'y_true': splits['y_test'],
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba_full
        }
    }
    
    # Save model and results
    model_path = f"models/{config['model']['name']}_model.pkl"
    os.makedirs('models', exist_ok=True)
    
    if hasattr(model, 'model'):
        # For sklearn models
        import joblib
        joblib.dump(model.model, model_path)
    else:
        # For PyTorch models
        import torch
        torch.save(model.state_dict(), model_path)
    
    # Save results
    results_path = f"results/{config['model']['name']}_results.yaml"
    os.makedirs('results', exist_ok=True)
    
    # Convert numpy arrays and sklearn objects to lists for YAML serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif hasattr(obj, '__module__') and 'sklearn' in obj.__module__:
            # Skip sklearn objects in serialization
            return str(type(obj).__name__)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_numpy(results)
    
    with open(results_path, 'w') as f:
        yaml.dump(results_serializable, f, default_flow_style=False)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Results saved to {results_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Model: {config['model']['name']}")
    print(f"Dataset: {metadata['n_samples']} samples, {metadata['n_features']} features")
    print(f"Mortality rate: {metadata['mortality_rate']:.1%}")
    print(f"Test AUROC: {evaluation_report['best_metrics']['auroc']:.3f}")
    print(f"Test AUPRC: {evaluation_report['best_metrics']['auprc']:.3f}")
    print(f"Test Sensitivity: {evaluation_report['best_metrics']['sensitivity']:.3f}")
    print(f"Test Specificity: {evaluation_report['best_metrics']['specificity']:.3f}")
    print(f"Calibration Error: {evaluation_report['best_metrics']['calibration_error']:.4f}")
    print("="*50)
    
    return results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train ICU mortality prediction model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, default=None,
                       help='Override model name in config')
    parser.add_argument('--seed', type=int, default=None,
                       help='Override random seed')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.model:
        config['model']['name'] = args.model
    if args.seed:
        config['seed'] = args.seed
    
    # Validate configuration
    validate_config(config)
    
    # Train model
    try:
        results = train_model(config)
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
