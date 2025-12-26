"""Explainability utilities for ICU mortality prediction."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class ICUExplainer:
    """Explainability module for ICU mortality prediction."""
    
    def __init__(self, config: Dict[str, Any], feature_names: List[str]):
        """Initialize explainer.
        
        Args:
            config: Explainability configuration
            feature_names: List of feature names
        """
        self.config = config
        self.feature_names = feature_names
        self.shap_enabled = config['explainability']['shap']['enabled']
        self.lime_enabled = config['explainability']['lime']['enabled']
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        
        if self.shap_enabled:
            try:
                import shap
                self.shap = shap
                logger.info("SHAP explainer initialized")
            except ImportError:
                logger.warning("SHAP not available")
                self.shap_enabled = False
        
        if self.lime_enabled:
            try:
                from lime import lime_tabular
                self.lime_tabular = lime_tabular
                logger.info("LIME explainer initialized")
            except ImportError:
                logger.warning("LIME not available")
                self.lime_enabled = False
    
    def fit_shap_explainer(self, model: Any, X_train: np.ndarray, 
                          X_background: Optional[np.ndarray] = None) -> None:
        """Fit SHAP explainer.
        
        Args:
            model: Trained model
            X_train: Training data
            X_background: Background data for SHAP
        """
        if not self.shap_enabled:
            return
        
        try:
            if X_background is None:
                # Use subset of training data as background
                n_background = min(self.config['explainability']['shap']['background_samples'], len(X_train))
                background_idx = np.random.choice(len(X_train), n_background, replace=False)
                X_background = X_train[background_idx]
            
            # Create explainer based on model type
            if hasattr(model, 'predict_proba'):
                self.shap_explainer = self.shap.Explainer(model.predict_proba, X_background)
            else:
                self.shap_explainer = self.shap.Explainer(model, X_background)
            
            logger.info("SHAP explainer fitted successfully")
        except Exception as e:
            logger.error(f"Failed to fit SHAP explainer: {e}")
            self.shap_enabled = False
    
    def fit_lime_explainer(self, X_train: np.ndarray) -> None:
        """Fit LIME explainer.
        
        Args:
            X_train: Training data
        """
        if not self.lime_enabled:
            return
        
        try:
            self.lime_explainer = self.lime_tabular.LimeTabularExplainer(
                X_train,
                feature_names=self.feature_names,
                mode='classification',
                discretize_continuous=True
            )
            logger.info("LIME explainer fitted successfully")
        except Exception as e:
            logger.error(f"Failed to fit LIME explainer: {e}")
            self.lime_enabled = False
    
    def explain_shap(self, X: np.ndarray, max_samples: Optional[int] = None) -> Optional[np.ndarray]:
        """Generate SHAP explanations.
        
        Args:
            X: Data to explain
            max_samples: Maximum number of samples to explain
            
        Returns:
            SHAP values
        """
        if not self.shap_enabled or self.shap_explainer is None:
            return None
        
        try:
            if max_samples is None:
                max_samples = self.config['explainability']['shap']['max_samples']
            
            X_explain = X[:max_samples] if len(X) > max_samples else X
            shap_values = self.shap_explainer(X_explain)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Get positive class values
            
            return shap_values.values if hasattr(shap_values, 'values') else shap_values
        except Exception as e:
            logger.error(f"Failed to generate SHAP explanations: {e}")
            return None
    
    def explain_lime(self, X: np.ndarray, model: Any, max_samples: Optional[int] = None) -> List[Dict]:
        """Generate LIME explanations.
        
        Args:
            X: Data to explain
            model: Trained model
            max_samples: Maximum number of samples to explain
            
        Returns:
            List of LIME explanations
        """
        if not self.lime_enabled or self.lime_explainer is None:
            return []
        
        try:
            if max_samples is None:
                max_samples = self.config['explainability']['lime']['max_samples']
            
            explanations = []
            X_explain = X[:max_samples] if len(X) > max_samples else X
            
            for i, sample in enumerate(X_explain):
                explanation = self.lime_explainer.explain_instance(
                    sample, 
                    model.predict_proba,
                    num_features=len(self.feature_names)
                )
                explanations.append({
                    'sample_idx': i,
                    'explanation': explanation,
                    'feature_importance': explanation.as_list()
                })
            
            return explanations
        except Exception as e:
            logger.error(f"Failed to generate LIME explanations: {e}")
            return []
    
    def plot_feature_importance(self, importance_scores: np.ndarray, 
                               title: str = "Feature Importance", 
                               save_path: Optional[str] = None) -> None:
        """Plot feature importance.
        
        Args:
            importance_scores: Feature importance scores
            title: Plot title
            save_path: Path to save plot
        """
        # Sort features by importance
        sorted_idx = np.argsort(importance_scores)[::-1]
        sorted_scores = importance_scores[sorted_idx]
        sorted_names = [self.feature_names[i] for i in sorted_idx]
        
        plt.figure(figsize=(10, 6))
        bars = plt.barh(range(len(sorted_names)), sorted_scores)
        plt.yticks(range(len(sorted_names)), sorted_names)
        plt.xlabel('Importance Score')
        plt.title(title)
        plt.gca().invert_yaxis()
        
        # Color bars by importance
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_names)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_shap_summary(self, shap_values: np.ndarray, X: np.ndarray,
                         save_path: Optional[str] = None) -> None:
        """Plot SHAP summary.
        
        Args:
            shap_values: SHAP values
            X: Input data
            save_path: Path to save plot
        """
        if not self.shap_enabled:
            return
        
        try:
            plt.figure(figsize=(10, 8))
            
            # Create SHAP summary plot
            self.shap.summary_plot(
                shap_values, 
                X, 
                feature_names=self.feature_names,
                show=False
            )
            
            plt.title('SHAP Summary Plot')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            logger.error(f"Failed to create SHAP summary plot: {e}")
    
    def plot_shap_waterfall(self, shap_values: np.ndarray, X: np.ndarray, 
                           sample_idx: int = 0, save_path: Optional[str] = None) -> None:
        """Plot SHAP waterfall for a single sample.
        
        Args:
            shap_values: SHAP values
            X: Input data
            sample_idx: Index of sample to explain
            save_path: Path to save plot
        """
        if not self.shap_enabled:
            return
        
        try:
            plt.figure(figsize=(10, 6))
            
            # Create SHAP waterfall plot
            self.shap.waterfall_plot(
                shap_values[sample_idx],
                X[sample_idx],
                feature_names=self.feature_names,
                show=False
            )
            
            plt.title(f'SHAP Waterfall Plot - Sample {sample_idx}')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            logger.error(f"Failed to create SHAP waterfall plot: {e}")
    
    def create_explanation_report(self, X: np.ndarray, model: Any, 
                                y_true: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Create comprehensive explanation report.
        
        Args:
            X: Data to explain
            model: Trained model
            y_true: True labels (optional)
            
        Returns:
            Explanation report
        """
        report = {
            'feature_names': self.feature_names,
            'n_samples': len(X),
            'shap_enabled': self.shap_enabled,
            'lime_enabled': self.lime_enabled
        }
        
        # SHAP explanations
        if self.shap_enabled:
            shap_values = self.explain_shap(X)
            if shap_values is not None:
                report['shap_values'] = shap_values
                report['shap_mean_importance'] = np.abs(shap_values).mean(axis=0)
        
        # LIME explanations
        if self.lime_enabled:
            lime_explanations = self.explain_lime(X, model)
            report['lime_explanations'] = lime_explanations
        
        # Feature importance from model (if available)
        if hasattr(model, 'feature_importances_'):
            report['model_feature_importance'] = model.feature_importances_
        elif hasattr(model, 'coef_'):
            report['model_feature_importance'] = np.abs(model.coef_[0])
        
        return report
