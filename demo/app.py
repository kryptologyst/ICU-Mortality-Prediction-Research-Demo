"""Streamlit demo for ICU mortality prediction."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
import joblib
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import set_seed
from data import ICUDataProcessor
from models import create_model
from metrics import ICUEvaluator
from utils.explainability import ICUExplainer

# Page configuration
st.set_page_config(
    page_title="ICU Mortality Prediction - Research Demo",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Disclaimer banner
st.error("""
**⚠️ IMPORTANT DISCLAIMER: This is a research demonstration only. NOT FOR CLINICAL USE. NOT MEDICAL ADVICE.**
""")

# Title and description
st.title("🏥 ICU Mortality Prediction - Research Demo")
st.markdown("""
This interactive demo showcases machine learning models for predicting ICU patient mortality using clinical features.
**This is for research and educational purposes only - not for clinical decision-making.**
""")

# Sidebar
st.sidebar.title("Configuration")

# Model selection
model_options = ['random_forest', 'xgboost', 'lightgbm']
selected_model = st.sidebar.selectbox("Select Model", model_options, index=0)

# Load configuration
@st.cache_data
def load_config():
    with open('configs/default.yaml', 'r') as f:
        return yaml.safe_load(f)

config = load_config()
config['model']['name'] = selected_model

# Generate synthetic data
@st.cache_data
def generate_data():
    set_seed(42)
    data_processor = ICUDataProcessor(config)
    df = data_processor.generate_synthetic_data()
    return df, data_processor

df, data_processor = generate_data()

# Train model
@st.cache_data
def train_model():
    set_seed(42)
    X, y, metadata = data_processor.preprocess_data(df)
    splits = data_processor.split_data(X, y, df)
    
    model = create_model(config, input_dim=X.shape[1])
    model.fit(splits['X_train'], splits['y_train'], splits['X_val'], splits['y_val'])
    
    return model, splits, metadata

model, splits, metadata = train_model()

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🔮 Predictions", "📈 Model Performance", "🔍 Explainability"])

with tab1:
    st.header("Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Patients", len(df))
    with col2:
        st.metric("Mortality Rate", f"{df['mortality'].mean():.1%}")
    with col3:
        st.metric("Features", len(metadata['feature_names']))
    
    # Feature distributions
    st.subheader("Feature Distributions")
    
    # Select features to plot
    feature_cols = ['age', 'sofa_score', 'glucose', 'heart_rate', 'systolic_bp', 'spo2']
    selected_features = st.multiselect("Select features to visualize", feature_cols, default=feature_cols[:3])
    
    if selected_features:
        fig = make_subplots(
            rows=len(selected_features), cols=1,
            subplot_titles=selected_features,
            vertical_spacing=0.1
        )
        
        for i, feature in enumerate(selected_features):
            fig.add_trace(
                go.Histogram(
                    x=df[feature],
                    name=feature,
                    showlegend=False,
                    opacity=0.7
                ),
                row=i+1, col=1
            )
        
        fig.update_layout(height=200*len(selected_features), title_text="Feature Distributions")
        st.plotly_chart(fig, use_container_width=True)
    
    # Data sample
    st.subheader("Data Sample")
    st.dataframe(df.head(10))

with tab2:
    st.header("Make Predictions")
    
    st.markdown("Enter patient features to get mortality prediction:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 18, 90, 65)
        gender = st.selectbox("Gender", ["Male", "Female"])
        sofa_score = st.slider("SOFA Score", 0, 20, 8)
        glucose = st.slider("Glucose (mg/dL)", 50, 300, 110)
    
    with col2:
        heart_rate = st.slider("Heart Rate (bpm)", 40, 150, 85)
        systolic_bp = st.slider("Systolic BP (mmHg)", 70, 200, 120)
        spo2 = st.slider("SpO2 (%)", 70, 100, 96)
    
    # Create input data
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'sofa_score': [sofa_score],
        'glucose': [glucose],
        'heart_rate': [heart_rate],
        'systolic_bp': [systolic_bp],
        'spo2': [spo2]
    })
    
    # Preprocess input
    X_input, _, _ = data_processor.preprocess_data(input_data)
    
    # Make prediction
    if st.button("Predict Mortality Risk"):
        proba = model.predict_proba(X_input)[0]
        mortality_prob = proba[1]  # Probability of mortality
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mortality Risk", f"{mortality_prob:.1%}")
        
        with col2:
            risk_level = "High" if mortality_prob > 0.5 else "Low"
            st.metric("Risk Level", risk_level)
        
        with col3:
            prediction = "Deceased" if mortality_prob > 0.5 else "Survived"
            st.metric("Prediction", prediction)
        
        # Risk visualization
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = mortality_prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Mortality Risk (%)"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Model Performance")
    
    # Get test predictions
    y_pred_proba = model.predict_proba(splits['X_test'])
    y_pred = model.predict(splits['X_test'])
    
    # Initialize evaluator
    evaluator = ICUEvaluator(config)
    
    # Compute metrics
    metrics = evaluator.compute_metrics(splits['y_test'], y_pred_proba)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("AUROC", f"{metrics['auroc']:.3f}")
    with col2:
        st.metric("AUPRC", f"{metrics['auprc']:.3f}")
    with col3:
        st.metric("Sensitivity", f"{metrics['sensitivity']:.3f}")
    with col4:
        st.metric("Specificity", f"{metrics['specificity']:.3f}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("PPV", f"{metrics['ppv']:.3f}")
    with col2:
        st.metric("NPV", f"{metrics['npv']:.3f}")
    with col3:
        st.metric("F1 Score", f"{metrics['f1']:.3f}")
    with col4:
        st.metric("Calibration Error", f"{metrics['calibration_error']:.4f}")
    
    # ROC Curve
    st.subheader("ROC Curve")
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(splits['y_test'], y_pred_proba[:, 1])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUROC = {metrics["auroc"]:.3f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
    
    fig.update_layout(
        title="Receiver Operating Characteristic (ROC) Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=600,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Precision-Recall Curve
    st.subheader("Precision-Recall Curve")
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(splits['y_test'], y_pred_proba[:, 1])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'PR Curve (AUPRC = {metrics["auprc"]:.3f})'))
    
    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        width=600,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Model Explainability")
    
    # Feature importance
    if hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
        st.subheader("Feature Importance")
        
        importance_scores = model.model.feature_importances_
        feature_names = metadata['feature_names']
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        }).sort_values('Importance', ascending=True)
        
        # Plot feature importance
        fig = px.bar(
            importance_df, 
            x='Importance', 
            y='Feature',
            orientation='h',
            title="Feature Importance"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display importance table
        st.dataframe(importance_df)
    
    # SHAP-like explanation for a sample
    st.subheader("Sample Explanation")
    
    # Select a test sample
    sample_idx = st.selectbox("Select test sample", range(len(splits['X_test'])), index=0)
    
    sample_features = splits['X_test'][sample_idx]
    sample_pred = model.predict_proba(splits['X_test'][sample_idx:sample_idx+1])[0, 1]
    sample_true = splits['y_test'][sample_idx]
    
    st.write(f"**Sample {sample_idx}**: Predicted mortality risk = {sample_pred:.1%}, Actual = {'Deceased' if sample_true else 'Survived'}")
    
    # Feature values
    feature_df = pd.DataFrame({
        'Feature': metadata['feature_names'],
        'Value': sample_features
    })
    
    st.dataframe(feature_df)

# Footer
st.markdown("---")
st.markdown("""
**Disclaimer**: This is a research demonstration project. The models and predictions shown here are for educational purposes only and should not be used for clinical decision-making. Always consult with qualified healthcare professionals for medical decisions.
""")
