#!/bin/bash
# Complete Evaluation Pipeline Runner
# Usage: ./run_evaluation_pipeline.sh

echo "=========================================="
echo "ASD/ADHD Detection - Evaluation Pipeline"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ -d ".venv" ]; then
    echo "📦 Activating virtual environment..."
    source .venv/bin/activate
else
    echo "⚠️  Warning: Virtual environment not found. Make sure dependencies are installed."
fi

echo ""
echo "Step 1: Checking prerequisites..."
echo "-----------------------------------"

# Check if features exist
if [ ! -d "data/processed/features" ] || [ -z "$(ls -A data/processed/features 2>/dev/null)" ]; then
    echo "❌ Features not found. Running feature extraction..."
    python3 01_extract_features.py
    if [ $? -ne 0 ]; then
        echo "❌ Feature extraction failed. Please check your audio files."
        exit 1
    fi
else
    echo "✅ Features found"
fi

# Check if models exist
if [ ! -d "models/saved" ] || [ -z "$(ls -A models/saved/*.pkl 2>/dev/null)" ]; then
    echo "❌ Models not found. Training models..."
    python3 02_train_models.py
    if [ $? -ne 0 ]; then
        echo "❌ Model training failed."
        exit 1
    fi
else
    echo "✅ Models found"
fi

echo ""
echo "Step 2: Running comprehensive evaluation..."
echo "-------------------------------------------"
python3 evaluate_models_comprehensive.py

if [ $? -ne 0 ]; then
    echo "❌ Evaluation failed."
    exit 1
fi

echo ""
echo "Step 3: Generating evaluation reports..."
echo "----------------------------------------"
python3 generate_evaluation_report.py

if [ $? -ne 0 ]; then
    echo "⚠️  Report generation failed (non-critical)."
fi

echo ""
echo "=========================================="
echo ""
echo "✅ Evaluation pipeline complete!"
echo ""
echo "📁 Results saved to: results/evaluation/"
echo ""
echo "💡 Generated files:"
echo "   - evaluation_results.json (metrics)"
echo "   - confusion_matrix_*.png (visualizations)"
echo "   - roc_curve_*.png (ROC curves)"
echo "   - metrics_comparison.png (model comparison)"
echo "   - terminal_training_logs.txt (training summary)"
echo "   - terminal_evaluation_summary.txt (evaluation summary)"
echo ""
echo "💡 Next steps:"
echo "   1. Review results in results/evaluation/"
echo "   2. Use terminal_*.txt files for presentations"
echo "   3. Include PNG visualizations in your report"
echo ""

