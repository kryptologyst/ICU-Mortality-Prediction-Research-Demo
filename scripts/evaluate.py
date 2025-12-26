#!/usr/bin/env python3
"""Evaluation script for ICU mortality prediction."""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import set_seed, get_device
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
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model_and_results(model_path: str, results_path: str) -> tuple:
    """Load trained model and results."""
    # Load results
    with open(results_path, 'r') as f:
        results = yaml.safe_load(f)
    
    # Load model
    config = results['config']
    model = create_model(config, input_dim=results['metadata']['n_features'])
    
    if model_path.endswith('.pkl'):
        import joblib
        model.model = joblib.load(model_path)
        model.is_trained = True
    else:
        import torch
        model.load_state_dict(torch.load(model_path))
    
    return model, results


def evaluate_model(model_path: str, results_path: str, config_path: str) -> None:
    """Evaluate trained model."""
    # Load configuration
    config = load_config(config_path)
    
    # Load model and results
    model, results = load_model_and_results(model_path, results_path)
    
    # Set up reproducibility
    set_seed(config['seed'], config['deterministic'])
    
    # Initialize evaluator
    evaluator = ICUEvaluator(config)
    
    # Get test data
    y_true = np.array(results['predictions']['y_true'])
    y_pred_proba_full = np.array(results['predictions']['y_pred_proba'])
    y_pred_proba = y_pred_proba_full[:, 1]  # Get probability of positive class
    y_pred = np.array(results['predictions']['y_pred'])
    
    # Create plots directory
    os.makedirs('assets/plots', exist_ok=True)
    
    # Generate evaluation plots
    logger.info("Generating evaluation plots...")
    
    # ROC curve
    evaluator.plot_roc_curve(
        y_true, y_pred_proba, 
        save_path='assets/plots/roc_curve.png'
    )
    
    # Precision-Recall curve
    evaluator.plot_precision_recall_curve(
        y_true, y_pred_proba,
        save_path='assets/plots/pr_curve.png'
    )
    
    # Calibration curve
    evaluator.plot_calibration_curve(
        y_true, y_pred_proba,
        save_path='assets/plots/calibration_curve.png'
    )
    
    # Confusion matrix
    evaluator.plot_confusion_matrix(
        y_true, y_pred,
        save_path='assets/plots/confusion_matrix.png'
    )
    
    # Generate explainability plots
    logger.info("Generating explainability plots...")
    
    explainer = ICUExplainer(config, results['metadata']['feature_names'])
    
    # Feature importance
    if 'model_feature_importance' in results['explanations']:
        importance_scores = np.array(results['explanations']['model_feature_importance'])
        explainer.plot_feature_importance(
            importance_scores,
            title=f"Feature Importance - {config['model']['name']}",
            save_path='assets/plots/feature_importance.png'
        )
    
    # SHAP plots
    if 'shap_values' in results['explanations']:
        shap_values = np.array(results['explanations']['shap_values'])
        
        # SHAP summary plot
        explainer.plot_shap_summary(
            shap_values, y_pred_proba.reshape(-1, 1),
            save_path='assets/plots/shap_summary.png'
        )
        
        # SHAP waterfall plot for first sample
        explainer.plot_shap_waterfall(
            shap_values, y_pred_proba.reshape(-1, 1),
            sample_idx=0,
            save_path='assets/plots/shap_waterfall.png'
        )
    
    # Print detailed evaluation report
    print("\n" + "="*60)
    print("DETAILED EVALUATION REPORT")
    print("="*60)
    
    evaluation = results['evaluation']
    print(f"Model: {evaluation['model_name']}")
    print(f"Test samples: {evaluation['n_samples']}")
    print(f"Mortality rate: {evaluation['mortality_rate']:.1%}")
    print(f"Best threshold: {evaluation['best_threshold']:.3f}")
    print()
    
    print("Performance Metrics:")
    print("-" * 30)
    best_metrics = evaluation['best_metrics']
    for metric, value in best_metrics.items():
        if metric != 'threshold':
            print(f"{metric.upper():20s}: {value:.3f}")
    
    print("\nThreshold Analysis:")
    print("-" * 30)
    threshold_metrics = evaluation['threshold_metrics']
    df_thresholds = pd.DataFrame(threshold_metrics)
    
    # Find optimal thresholds for different criteria
    print("Optimal thresholds:")
    print(f"Max AUROC:     {df_thresholds.loc[df_thresholds['auroc'].idxmax(), 'threshold']:.3f}")
    print(f"Max Sensitivity: {df_thresholds.loc[df_thresholds['sensitivity'].idxmax(), 'threshold']:.3f}")
    print(f"Max Specificity: {df_thresholds.loc[df_thresholds['specificity'].idxmax(), 'threshold']:.3f}")
    print(f"Max F1:        {df_thresholds.loc[df_thresholds['f1'].idxmax(), 'threshold']:.3f}")
    
    print("\nFeature Importance (Top 5):")
    print("-" * 30)
    if 'model_feature_importance' in results['explanations']:
        importance_scores = np.array(results['explanations']['model_feature_importance'])
        feature_names = results['metadata']['feature_names']
        
        # Sort by importance
        sorted_idx = np.argsort(importance_scores)[::-1]
        for i in range(min(5, len(feature_names))):
            idx = sorted_idx[i]
            print(f"{feature_names[idx]:20s}: {importance_scores[idx]:.3f}")
    
    print("="*60)
    
    logger.info("Evaluation completed successfully!")
    logger.info("Plots saved to assets/plots/")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate ICU mortality prediction model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--results_path', type=str, required=True,
                       help='Path to results file')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    try:
        evaluate_model(args.model_path, args.results_path, args.config)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
