# Quick Start Guide

## 🚀 Run the Complete Pipeline

All scripts are now Python files (no notebooks needed!) and use real recordings from `recordings/` folder.

### Option 1: Run Individual Scripts

```bash
cd ASD_ADHD_Detection

# 1. Extract features from recordings
.venv/bin/python 01_extract_features.py

# 2. Train models
.venv/bin/python 02_train_models.py

# 3. Evaluate and see metrics
.venv/bin/python 03_evaluate_models.py

# 4. Launch UI
.venv/bin/streamlit run 07_streamlit_ui.py
```

### Option 2: Run Complete Pipeline

```bash
cd ASD_ADHD_Detection
.venv/bin/python run_pipeline.py
```

This will run steps 1-3 automatically.

---

## 📁 What Each Script Does

| Script | Purpose | Output |
|--------|---------|--------|
| `01_extract_features.py` | Extract MFCC from recordings | `data/processed/features/*.npy` |
| `02_train_models.py` | Train RF, SVM, NB, MLP models | `models/saved/*.pkl` |
| `03_evaluate_models.py` | Show metrics & visualizations | `results/*.png`, `results/*.json` |
| `04_kfold_validation.py` | Cross-validation analysis | `results/kfold_cv_results.*` |
| `05_hyperparameter_tuning.py` | Optimize hyperparameters | `models/saved/*_tuned.pkl` |
| `06_feature_analysis.py` | Feature importance analysis | `results/feature_*.png` |
| `07_streamlit_ui.py` | Web UI for predictions | Web interface |

---

## ✅ Verify Setup

```bash
cd ASD_ADHD_Detection

# Check virtual environment
.venv/bin/python -c "import numpy, librosa, sklearn, joblib; print('✅ All packages available')"

# Check recordings
ls recordings/*.m4a | wc -l  # Should show number of audio files
```

---

## 📊 Expected Results

After running `02_train_models.py`, you should see:
- Model accuracies printed to console
- Models saved in `models/saved/`
- Training summary in `results/training_summary.json`

After running `03_evaluate_models.py`, you should see:
- Detailed metrics for each model
- Confusion matrices saved as PNG files
- ROC curves (if model supports probabilities)
- Evaluation results JSON

---

## 🎯 Next Steps

1. **Train models**: Run `02_train_models.py` to train on your recordings
2. **View metrics**: Run `03_evaluate_models.py` to see performance
3. **Use UI**: Run `07_streamlit_ui.py` to test predictions interactively

---

## 📝 Notes

- All paths are relative to `ASD_ADHD_Detection/` folder
- Features are saved in `data/processed/features/`
- Models are saved in `models/saved/`
- Results are saved in `results/`
- All scripts use real recordings from `recordings/` folder (no synthetic data!)

---

**Ready to go!** 🎉

