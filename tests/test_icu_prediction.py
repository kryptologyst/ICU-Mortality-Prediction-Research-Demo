"""Tests for ICU mortality prediction project."""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import set_seed, get_device, safe_divide, format_metric
from data import ICUDataProcessor
from models import create_model
from metrics import ICUEvaluator


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # Test that random numbers are reproducible
        np.random.seed(42)
        val1 = np.random.random()
        set_seed(42)
        val2 = np.random.random()
        assert val1 == val2
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device('cpu')
        assert str(device) == 'cpu'
    
    def test_safe_divide(self):
        """Test safe division."""
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(10, 0) == 0.0
        assert safe_divide(10, 0, default=1.0) == 1.0
    
    def test_format_metric(self):
        """Test metric formatting."""
        assert format_metric(0.1234, 'auroc') == '0.123'
        assert format_metric(0.001234, 'calibration_error') == '0.0012'


class TestDataProcessor:
    """Test data processing."""
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        config = {
            'data': {
                'synthetic': {
                    'n_samples': 100,
                    'mortality_rate': 0.2,
                    'random_seed': 42
                },
                'features': {
                    'numerical': ['age', 'sofa_score', 'glucose'],
                    'categorical': ['gender'],
                    'target': 'mortality'
                }
            }
        }
        
        processor = ICUDataProcessor(config)
        df = processor.generate_synthetic_data()
        
        assert len(df) == 100
        assert 'mortality' in df.columns
        assert df['mortality'].mean() == pytest.approx(0.2, abs=0.1)
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        config = {
            'data': {
                'features': {
                    'numerical': ['age', 'sofa_score'],
                    'categorical': ['gender'],
                    'target': 'mortality'
                }
            }
        }
        
        # Create test data
        df = pd.DataFrame({
            'age': [30, 40, 50],
            'gender': ['Male', 'Female', 'Male'],
            'sofa_score': [5, 10, 15],
            'mortality': [0, 1, 0]
        })
        
        processor = ICUDataProcessor(config)
        X, y, metadata = processor.preprocess_data(df)
        
        assert X.shape[0] == 3
        assert X.shape[1] == 3  # age, gender, sofa_score
        assert len(y) == 3
        assert 'feature_names' in metadata


class TestModels:
    """Test model implementations."""
    
    def test_random_forest_model(self):
        """Test Random Forest model."""
        config = {
            'model': {
                'name': 'random_forest',
                'random_forest': {
                    'n_estimators': 10,
                    'max_depth': 5,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'random_state': 42
                }
            }
        }
        
        model = create_model(config)
        
        # Generate test data
        X = np.random.random((100, 5))
        y = np.random.randint(0, 2, 100)
        
        # Train model
        model.fit(X, y)
        
        # Test predictions
        predictions = model.predict(X[:10])
        probabilities = model.predict_proba(X[:10])
        
        assert len(predictions) == 10
        assert probabilities.shape == (10, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)


class TestEvaluator:
    """Test evaluation metrics."""
    
    def test_compute_metrics(self):
        """Test metric computation."""
        config = {
            'evaluation': {
                'metrics': ['auroc', 'auprc', 'sensitivity', 'specificity'],
                'thresholds': [0.1, 0.5, 0.9]
            }
        }
        
        evaluator = ICUEvaluator(config)
        
        # Generate test data
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
        
        metrics = evaluator.compute_metrics(y_true, y_pred_proba)
        
        assert 'auroc' in metrics
        assert 'auprc' in metrics
        assert 'sensitivity' in metrics
        assert 'specificity' in metrics
        assert 0 <= metrics['auroc'] <= 1
        assert 0 <= metrics['auprc'] <= 1


if __name__ == "__main__":
    pytest.main([__file__])
