#!/bin/bash
set -e

echo "� Starting Build Process..."
echo "================================"

echo "�🐍 Checking Python version..."
python --version

echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip

echo ""
echo "📚 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "🧠 Training ML models..."
echo "This may take a few minutes..."
python train_model.py

# Verify models were created
if [ -f "models/encoders.pkl" ] && [ -f "models/scaler.pkl" ]; then
    echo "✅ Models trained successfully!"
else
    echo "❌ Model training failed - required files not found"
    exit 1
fi

echo ""
echo "✅ Build complete!"
echo "================================"
