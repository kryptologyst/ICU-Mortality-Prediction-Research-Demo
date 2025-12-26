"""Utility functions for ICU mortality prediction project."""

import random
import numpy as np
import torch
from typing import Any, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}, deterministic={deterministic}")


def get_device(device: str = 'auto') -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Device specification ('auto', 'cpu', 'cuda', 'mps')
        
    Returns:
        torch.device: The selected device
    """
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    device_obj = torch.device(device)
    logger.info(f"Using device: {device_obj}")
    return device_obj


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if division by zero
        
    Returns:
        float: Result of division or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator


def format_metric(value: float, metric_name: str) -> str:
    """Format metric value for display.
    
    Args:
        value: Metric value
        metric_name: Name of the metric
        
    Returns:
        str: Formatted metric string
    """
    if metric_name in ['auroc', 'auprc', 'sensitivity', 'specificity', 'ppv', 'npv', 'f1']:
        return f"{value:.3f}"
    elif metric_name in ['calibration_error']:
        return f"{value:.4f}"
    else:
        return f"{value:.2f}"


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_sections = ['data', 'model', 'training', 'evaluation']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate model name
    valid_models = ['random_forest', 'xgboost', 'lightgbm', 'tabnet', 'ft_transformer']
    model_name = config['model']['name']
    if model_name not in valid_models:
        raise ValueError(f"Invalid model name: {model_name}. Must be one of {valid_models}")
    
    # Validate device
    valid_devices = ['auto', 'cpu', 'cuda', 'mps']
    device = config['training']['device']
    if device not in valid_devices:
        raise ValueError(f"Invalid device: {device}. Must be one of {valid_devices}")


def create_directory_structure(base_path: str) -> None:
    """Create necessary directory structure.
    
    Args:
        base_path: Base path for the project
    """
    import os
    
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'checkpoints',
        'logs',
        'results',
        'assets/plots',
        'assets/explanations'
    ]
    
    for directory in directories:
        full_path = os.path.join(base_path, directory)
        os.makedirs(full_path, exist_ok=True)
        logger.info(f"Created directory: {full_path}")


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """Check if training should stop.
        
        Args:
            score: Current validation score
            model: Model to potentially save weights from
            
        Returns:
            bool: True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        
        return False
    
    def save_checkpoint(self, model: torch.nn.Module) -> None:
        """Save model checkpoint.
        
        Args:
            model: Model to save
        """
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()
