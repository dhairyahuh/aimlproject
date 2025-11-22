#!/bin/bash
# Script to run evaluation and report generation tools
# Usage: ./run_demo.sh

echo "=========================================="
echo "ASD/ADHD Detection - Evaluation Tools Runner"
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
python3 demo_generate_terminal_output.py

python3 demo_mock_evaluation.py

echo ""
echo "3️⃣  (Optional) Duplicating audio files for augmentation..."
read -p "   Duplicate audio files? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 demo_duplicate_audios.py --target-count 150
fi

echo ""
echo "=========================================="
echo ""
echo "📁 Results saved to: results/evaluation/"
echo ""
echo "💡 Next steps:"
echo "   1. Open terminal_*.txt files for screenshots"
echo "   2. View PNG files in results/evaluation/"
echo ""




