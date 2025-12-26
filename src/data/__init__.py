"""Data processing utilities for ICU mortality prediction."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class ICUDataProcessor:
    """Data processor for ICU mortality prediction."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize data processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic ICU patient data.
        
        Returns:
            pd.DataFrame: Synthetic ICU data
        """
        np.random.seed(self.config['data']['synthetic']['random_seed'])
        
        n_samples = self.config['data']['synthetic']['n_samples']
        mortality_rate = self.config['data']['synthetic']['mortality_rate']
        
        # Generate realistic ICU data
        data = {
            'patient_id': range(n_samples),
            'age': np.random.randint(18, 90, n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'sofa_score': np.random.randint(0, 20, n_samples),
            'glucose': np.random.normal(110, 20, n_samples),
            'heart_rate': np.random.normal(85, 10, n_samples),
            'systolic_bp': np.random.normal(120, 15, n_samples),
            'spo2': np.random.normal(96, 3, n_samples),
            'mortality': np.random.choice([0, 1], n_samples, p=[1-mortality_rate, mortality_rate])
        }
        
        # Add some realistic correlations
        df = pd.DataFrame(data)
        
        # Higher SOFA scores correlate with mortality
        mortality_mask = df['mortality'] == 1
        df.loc[mortality_mask, 'sofa_score'] = np.random.randint(8, 20, mortality_mask.sum())
        df.loc[~mortality_mask, 'sofa_score'] = np.random.randint(0, 12, (~mortality_mask).sum())
        
        # Lower SpO2 correlates with mortality
        df.loc[mortality_mask, 'spo2'] = np.random.normal(88, 5, mortality_mask.sum())
        df.loc[~mortality_mask, 'spo2'] = np.random.normal(97, 2, (~mortality_mask).sum())
        
        logger.info(f"Generated synthetic data with {n_samples} samples, {mortality_rate:.1%} mortality rate")
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Preprocess the data for training.
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (X, y, metadata)
        """
        # Separate features and target
        feature_cols = self.config['data']['features']['numerical'] + self.config['data']['features']['categorical']
        X = df[feature_cols].copy()
        y = df[self.config['data']['features']['target']].values
        
        # Encode categorical variables
        for col in self.config['data']['features']['categorical']:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                self.label_encoders[col] = le
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X)
        
        metadata = {
            'feature_names': self.feature_names,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'mortality_rate': y.mean(),
            'label_encoders': self.label_encoders
        }
        
        logger.info(f"Preprocessed data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Mortality rate: {metadata['mortality_rate']:.1%}")
        
        return X_scaled, y, metadata
    
    def split_data(self, X: np.ndarray, y: np.ndarray, df: Optional[pd.DataFrame] = None) -> Dict[str, np.ndarray]:
        """Split data into train/validation/test sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            df: Original dataframe (for patient-level splits)
            
        Returns:
            Dictionary containing train/val/test splits
        """
        random_state = self.config['data']['preprocessing']['random_seed']
        test_size = self.config['data']['preprocessing']['test_size']
        val_size = self.config['data']['preprocessing']['val_size']
        
        if self.config['data']['preprocessing']['patient_level_split'] and df is not None:
            # Patient-level split to avoid data leakage
            unique_patients = df['patient_id'].unique()
            n_patients = len(unique_patients)
            
            # Split patients
            train_patients, temp_patients = train_test_split(
                unique_patients, test_size=test_size + val_size, random_state=random_state
            )
            val_patients, test_patients = train_test_split(
                temp_patients, test_size=test_size/(test_size + val_size), random_state=random_state
            )
            
            # Get indices for each split
            train_idx = df[df['patient_id'].isin(train_patients)].index
            val_idx = df[df['patient_id'].isin(val_patients)].index
            test_idx = df[df['patient_id'].isin(test_patients)].index
            
            splits = {
                'X_train': X[train_idx], 'y_train': y[train_idx],
                'X_val': X[val_idx], 'y_val': y[val_idx],
                'X_test': X[test_idx], 'y_test': y[test_idx]
            }
            
            logger.info(f"Patient-level split: {len(train_patients)} train, {len(val_patients)} val, {len(test_patients)} test patients")
        else:
            # Random split
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=test_size + val_size, random_state=random_state, stratify=y
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=test_size/(test_size + val_size), random_state=random_state, stratify=y_temp
            )
            
            splits = {
                'X_train': X_train, 'y_train': y_train,
                'X_val': X_val, 'y_val': y_val,
                'X_test': X_test, 'y_test': y_test
            }
            
            logger.info(f"Random split: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test samples")
        
        return splits
    
    def get_class_weights(self, y: np.ndarray) -> List[float]:
        """Calculate class weights for imbalanced data.
        
        Args:
            y: Target vector
            
        Returns:
            List of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        class_weights = compute_class_weight(
            'balanced', classes=classes, y=y
        )
        
        # Convert to list format expected by PyTorch
        weight_dict = dict(zip(classes, class_weights))
        weights = [weight_dict[0], weight_dict[1]]  # [survived, deceased]
        
        logger.info(f"Class weights: {weights}")
        return weights
