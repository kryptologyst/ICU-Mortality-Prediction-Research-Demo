"""Evaluation metrics for ICU mortality prediction."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


class ICUEvaluator:
    """Evaluator for ICU mortality prediction models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.metrics = config['evaluation']['metrics']
        self.thresholds = config['evaluation']['thresholds']
        
    def compute_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                       threshold: float = 0.5) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold
            
        Returns:
            Dictionary of metrics
        """
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Basic metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics = {
            'auroc': roc_auc_score(y_true, y_pred_proba),
            'auprc': average_precision_score(y_true, y_pred_proba),
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0.0,
            'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0,
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'threshold': threshold
        }
        
        # Calibration error
        metrics['calibration_error'] = self._compute_calibration_error(y_true, y_pred_proba)
        
        return metrics
    
    def _compute_calibration_error(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                  n_bins: int = 10) -> float:
        """Compute calibration error (ECE).
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            n_bins: Number of bins for calibration
            
        Returns:
            Calibration error
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def evaluate_at_thresholds(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, List[float]]:
        """Evaluate model at multiple thresholds.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary with metrics at each threshold
        """
        results = {metric: [] for metric in self.metrics}
        results['threshold'] = self.thresholds
        
        for threshold in self.thresholds:
            metrics = self.compute_metrics(y_true, y_pred_proba, threshold)
            for metric in self.metrics:
                results[metric].append(metrics[metric])
        
        return results
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                      save_path: Optional[str] = None) -> None:
        """Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save plot
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auroc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUROC = {auroc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   save_path: Optional[str] = None) -> None:
        """Plot precision-recall curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save plot
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        auprc = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUPRC = {auprc:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_calibration_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                              save_path: Optional[str] = None) -> None:
        """Plot calibration curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save plot
        """
        from sklearn.calibration import calibration_curve
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=10
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             save_path: Optional[str] = None) -> None:
        """Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Survived', 'Deceased'],
                   yticklabels=['Survived', 'Deceased'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_evaluation_report(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                               model_name: str = "Model") -> Dict[str, Any]:
        """Create comprehensive evaluation report.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            
        Returns:
            Comprehensive evaluation report
        """
        # Best threshold based on Youden's index
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        youden_index = tpr - fpr
        best_threshold_idx = np.argmax(youden_index)
        best_threshold = thresholds[best_threshold_idx]
        
        # Metrics at best threshold
        best_metrics = self.compute_metrics(y_true, y_pred_proba, best_threshold)
        
        # Metrics at multiple thresholds
        threshold_metrics = self.evaluate_at_thresholds(y_true, y_pred_proba)
        
        report = {
            'model_name': model_name,
            'best_threshold': best_threshold,
            'best_metrics': best_metrics,
            'threshold_metrics': threshold_metrics,
            'n_samples': len(y_true),
            'mortality_rate': y_true.mean()
        }
        
        return report
