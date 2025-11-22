# Production Evaluation and Analysis Tools

## Overview

These are production-ready tools for evaluating models, generating reports, and augmenting data using real models and data from your project.

## Tools

### 1. `evaluate_models_comprehensive.py`

Comprehensive model evaluation script that uses real trained models and test data to generate:
- Performance metrics (accuracy, precision, recall, F1-score, ROC-AUC)
- Confusion matrices (saved as PNG)
- ROC curves (saved as PNG)
- Model comparison charts
- JSON results file

**Usage:**
```bash
python evaluate_models_comprehensive.py
```

**Prerequisites:**
- Trained models in `models/saved/` (run `02_train_models.py` first)
- Feature files in `data/processed/features/` (run `01_extract_features.py` first)

**Output:**
- `results/evaluation/evaluation_results.json` - All metrics
- `results/evaluation/confusion_matrix_*.png` - Confusion matrices
- `results/evaluation/roc_curve_*.png` - ROC curves
- `results/evaluation/metrics_comparison.png` - Model comparison

### 2. `generate_evaluation_report.py`

Generates formatted terminal output text files from real evaluation results. Perfect for presentations and documentation.

**Usage:**
```bash
python generate_evaluation_report.py
```

**Prerequisites:**
- Evaluation results from `evaluate_models_comprehensive.py` or `03_evaluate_models.py`
- Training summary from `02_train_models.py` (optional)

**Output:**
- `results/evaluation/terminal_training_logs.txt` - Training summary
- `results/evaluation/terminal_evaluation_summary.txt` - Evaluation metrics table

### 3. `data_augmentation_tool.py`

Data augmentation tool for audio files to increase dataset size and improve model generalization.

**Usage:**
```bash
# Augment dataset (2x multiplier, combined augmentations)
python data_augmentation_tool.py --target-multiplier 2 --augmentation-type combined

# Specific augmentation types
python data_augmentation_tool.py --augmentation-type time_stretch
python data_augmentation_tool.py --augmentation-type pitch_shift
python data_augmentation_tool.py --augmentation-type add_noise

# Cleanup augmented files
python data_augmentation_tool.py --cleanup
```

**Augmentation Types:**
- `time_stretch`: Time stretching (speed up/slow down)
- `pitch_shift`: Pitch shifting (higher/lower pitch)
- `add_noise`: Add random noise
- `combined`: Apply multiple augmentations randomly
- `random`: Randomly select augmentation type

**Output:**
- Augmented files saved to `recordings_augmented/`

### 4. `run_evaluation_pipeline.sh`

Complete evaluation pipeline that runs all steps automatically.

**Usage:**
```bash
./run_evaluation_pipeline.sh
```

**What it does:**
1. Checks for features (runs extraction if needed)
2. Checks for models (runs training if needed)
3. Runs comprehensive evaluation
4. Generates evaluation reports

## Quick Start

### Complete Evaluation Workflow

```bash
# Option 1: Run the complete pipeline
./run_evaluation_pipeline.sh

# Option 2: Run steps manually
python 01_extract_features.py      # Extract features from audio
python 02_train_models.py           # Train models
python evaluate_models_comprehensive.py  # Evaluate models
python generate_evaluation_report.py     # Generate reports
```

### Data Augmentation Workflow

```bash
# Augment your dataset
python data_augmentation_tool.py --target-multiplier 2

# Extract features from augmented files (if needed)
# Then retrain models with augmented data
```

## File Structure

```
ASD_ADHD_Detection/
├── evaluate_models_comprehensive.py    # Comprehensive evaluation
├── generate_evaluation_report.py      # Report generation
├── data_augmentation_tool.py          # Data augmentation
├── run_evaluation_pipeline.sh        # Pipeline runner
├── results/
│   └── evaluation/                   # Evaluation results
│       ├── evaluation_results.json
│       ├── confusion_matrix_*.png
│       ├── roc_curve_*.png
│       ├── metrics_comparison.png
│       ├── terminal_training_logs.txt
│       └── terminal_evaluation_summary.txt
└── recordings_augmented/              # Augmented audio files
```

## Differences from Demo Tools

These production tools:
- ✅ Use real trained models and test data
- ✅ Generate actual evaluation metrics
- ✅ Work with your existing data pipeline
- ✅ Can be used for actual model evaluation and reporting

Demo tools (kept for reference):
- `demo_mock_evaluation.py` - Generates simulated results
- `demo_generate_terminal_output.py` - Generates mock terminal output
- `demo_duplicate_audios.py` - Simple file duplication (no augmentation)

## Notes

- All production tools use real data and models from your project
- Results are saved to `results/evaluation/` directory
- Augmented files are kept separate in `recordings_augmented/`
- Demo files are preserved and can be removed if not needed

