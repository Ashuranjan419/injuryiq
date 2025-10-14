#!/bin/bash
set -e

echo "🐍 Checking Python version..."
python --version

echo "📦 Upgrading pip..."
pip install --upgrade pip

echo "📚 Installing dependencies..."
pip install -r requirements.txt

echo "✅ Build complete!"
