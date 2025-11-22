# ASD/ADHD Detection - Python Scripts

This directory contains Python scripts for the complete autism detection pipeline. All scripts use real audio recordings from the `recordings/` folder.

## 📋 Scripts Overview

### 1. `01_extract_features.py`
**Purpose**: Extract MFCC features from audio recordings

**What it does**:
- Scans `recordings/` folder for audio files (.m4a, .wav, .mp3)
- Extracts 40 MFCC coefficients from each audio file
- Saves features as .npy files in `data/processed/features/`

**Usage**:
```bash
cd ASD_ADHD_Detection
.venv/bin/python 01_extract_features.py
```

**Output**: Feature files in `data/processed/features/`

---

### 2. `02_train_models.py`
**Purpose**: Train classical ML models on extracted features

**What it does**:
- Loads MFCC features from `data/processed/features/`
- Trains 4 models: Random Forest, SVM, Naive Bayes, MLP
- Evaluates each model and displays metrics
- Saves trained models to `models/saved/`

**Usage**:
```bash
.venv/bin/python 02_train_models.py
```

**Output**: 
- Trained models: `models/saved/rf.pkl`, `svm.pkl`, `nb.pkl`, `ann.pkl`
- Training summary: `results/training_summary.json`

---

### 3. `03_evaluate_models.py`
**Purpose**: Comprehensive model evaluation with visualizations

**What it does**:
- Loads trained models and test data
- Evaluates each model with detailed metrics
- Generates confusion matrices and ROC curves
- Creates visualization plots

**Usage**:
```bash
.venv/bin/python 03_evaluate_models.py
```

**Output**:
- Confusion matrices: `results/confusion_matrix_*.png`
- ROC curves: `results/roc_curve_*.png`
- Evaluation results: `results/evaluation_results.json`

---

### 4. `04_kfold_validation.py`
**Purpose**: K-Fold cross-validation for robust evaluation

**What it does**:
- Performs 5-fold stratified cross-validation
- Trains and evaluates models on each fold
- Computes mean and standard deviation of metrics
- Visualizes cross-validation results

**Usage**:
```bash
.venv/bin/python 04_kfold_validation.py
```

**Output**:
- CV results plot: `results/kfold_cv_results.png`
- CV results JSON: `results/kfold_cv_results.json`

---

### 5. `05_hyperparameter_tuning.py`
**Purpose**: Hyperparameter optimization using Grid Search

**What it does**:
- Tunes Random Forest, SVM, and MLP hyperparameters
- Uses Grid Search and Random Search
- Evaluates on validation set
- Saves best models

**Usage**:
```bash
.venv/bin/python 05_hyperparameter_tuning.py
```

**Output**:
- Tuned models: `models/saved/rf_tuned.pkl`, `svm_tuned.pkl`, `mlp_tuned.pkl`
- Tuning results: `results/hyperparameter_tuning_results.json`

**Note**: This script takes longer to run (10-30 minutes depending on data size)

---

### 6. `06_feature_analysis.py`
**Purpose**: Feature importance analysis and selection

**What it does**:
- Computes feature importance using Random Forest
- Analyzes F-scores and Mutual Information
- Evaluates model performance with different feature subsets
- Creates visualization plots

**Usage**:
```bash
.venv/bin/python 06_feature_analysis.py
```

**Output**:
- Feature importance plot: `results/feature_importance.png`
- Feature selection scores: `results/feature_selection_scores.png`
- Performance vs features: `results/feature_selection_performance.png`
- Analysis results: `results/feature_analysis_results.json`

---

### 7. `07_streamlit_ui.py`
**Purpose**: Interactive web UI for audio prediction

**What it does**:
- Provides web interface for uploading audio files
- Extracts features and makes predictions
- Displays results with confidence scores
- Supports all trained models

**Usage**:
```bash
.venv/bin/streamlit run 07_streamlit_ui.py
```

**Access**: Open browser to the URL shown (usually http://localhost:8501)

---

## 🚀 Quick Start Pipeline

Run scripts in this order:

```bash
# 1. Extract features from recordings
.venv/bin/python 01_extract_features.py

# 2. Train models
.venv/bin/python 02_train_models.py

# 3. Evaluate models (see metrics)
.venv/bin/python 03_evaluate_models.py

# 4. Launch UI
.venv/bin/streamlit run 07_streamlit_ui.py
```

---

## 📊 Complete Workflow

For comprehensive analysis:

```bash
# 1. Extract features
.venv/bin/python 01_extract_features.py

# 2. Train models
.venv/bin/python 02_train_models.py

# 3. Evaluate models
.venv/bin/python 03_evaluate_models.py

# 4. Cross-validation (optional, takes longer)
.venv/bin/python 04_kfold_validation.py

# 5. Hyperparameter tuning (optional, takes much longer)
.venv/bin/python 05_hyperparameter_tuning.py

# 6. Feature analysis (optional)
.venv/bin/python 06_feature_analysis.py

# 7. Launch UI
.venv/bin/streamlit run 07_streamlit_ui.py
```

---

## 📁 Directory Structure

```
ASD_ADHD_Detection/
├── recordings/              # Input audio files
├── data/
│   └── processed/
│       └── features/      # Extracted MFCC features (.npy files)
├── models/
│   └── saved/             # Trained models (.pkl files)
├── results/               # Evaluation results, plots, JSON files
├── 01_extract_features.py
├── 02_train_models.py
├── 03_evaluate_models.py
├── 04_kfold_validation.py
├── 05_hyperparameter_tuning.py
├── 06_feature_analysis.py
└── 07_streamlit_ui.py
```

---

## ⚙️ Configuration

All scripts use paths relative to the `ASD_ADHD_Detection` folder:

- **Recordings**: `recordings/` - Input audio files
- **Features**: `data/processed/features/` - Extracted features
- **Models**: `models/saved/` - Trained models
- **Results**: `results/` - Output files and visualizations

Key parameters (can be modified in scripts):
- `N_MFCC = 40` - Number of MFCC coefficients
- `SR = 22050` - Sample rate for audio processing
- `N_FOLDS = 5` - Number of folds for cross-validation

---

## 🔧 Requirements

All dependencies are in `requirements.txt`. Install with:

```bash
cd ASD_ADHD_Detection
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📝 Notes

1. **First time setup**: Run `01_extract_features.py` before training
2. **Model training**: `02_train_models.py` must be run before evaluation or UI
3. **UI access**: Streamlit UI runs on localhost (default port 8501)
4. **Performance**: Hyperparameter tuning and K-fold CV take longer (10-30 min)
5. **Data**: All scripts use real recordings from `recordings/` folder

---

## 🐛 Troubleshooting

**Error: "Features directory not found"**
- Run `01_extract_features.py` first

**Error: "Model not found"**
- Run `02_train_models.py` first

**Error: "No audio files found"**
- Check that `recordings/` folder contains .m4a, .wav, or .mp3 files

**Streamlit not found**
- Install: `pip install streamlit`

---

## 📞 Support

For issues or questions, check:
1. All paths are relative to `ASD_ADHD_Detection` folder
2. Virtual environment is activated
3. All dependencies are installed
4. Audio files are in `recordings/` folder

---

**Last Updated**: November 2025
**Version**: 1.0

