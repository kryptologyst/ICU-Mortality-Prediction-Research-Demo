#!/usr/bin/env bash

# ICU Mortality Prediction - Quick Start Script

echo "🏥 ICU Mortality Prediction - Research Demo Setup"
echo "================================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python version $python_version is compatible"
else
    echo "❌ Python version $python_version is not compatible. Required: $required_version+"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p data/{raw,processed}
mkdir -p models checkpoints logs results
mkdir -p assets/{plots,explanations}
mkdir -p tests

# Run tests
echo "🧪 Running tests..."
python -m pytest tests/ -v

# Train a quick model
echo "🚀 Training sample model..."
python scripts/train.py --config configs/default.yaml --model random_forest

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Launch the demo: streamlit run demo/app.py"
echo "2. Train different models: python scripts/train.py --model xgboost"
echo "3. Evaluate models: python scripts/evaluate.py --model_path models/random_forest_model.pkl --results_path results/random_forest_results.yaml"
echo ""
echo "⚠️  Remember: This is a research demo only, not for clinical use!"
