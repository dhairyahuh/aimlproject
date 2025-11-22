# Training Notebooks - Step-by-Step Model Development

This directory contains **interactive Jupyter notebooks** that walk you through the entire ASD/ADHD voice detection pipeline, from feature extraction to model training and refinement.

## 📚 Notebook Roadmap

### Phase 1: Understanding Features
**File:** `01_feature_extraction_tutorial.ipynb`

**Duration:** ~30 minutes (read + run all cells)

**What you'll learn:**
- How audio preprocessing standardizes voice samples
- MFCC features (52) - speech spectrum with dynamics
- Spectral features (24) - frequency content & energy
- Prosodic features (19+) - pitch, formants, voice quality
- How to aggregate into 106-dimensional feature vector
- Feature importance and variance analysis

**Key Sections:**
1. Synthetic audio generation (test data)
2. Audio preprocessing pipeline
3. MFCC extraction with delta/delta-delta
4. Spectral feature extraction
5. Prosodic feature extraction (ASD/ADHD markers)
6. Complete feature aggregation (106 features)
7. Feature statistics and importance
8. Customization options for refinement

**You should run this if:**
- You want to understand what features the model uses
- You're debugging why certain features matter
- You want to customize feature extraction

---

### Phase 2: Building Your First Model
**File:** `02_data_preparation_and_training.ipynb`

**Duration:** ~30-45 minutes (depends on dataset size and number of epochs)

**What you'll learn:**
- How to create synthetic training data
- Feature normalization for neural networks
- Building a 3-layer MLP classifier (106 → 128 → 64 → 32 → 3)
- Training with detailed monitoring (loss, accuracy, precision, recall)
- Model evaluation on test data
- Confusion matrix interpretation
- Identifying refinement opportunities

**Key Sections:**
1. Synthetic data generation (ASD vs ADHD vs Healthy)
2. Data train/test split with class balance
3. Feature standardization (StandardScaler)
4. MLP architecture design
5. Training with callbacks (early stopping, learning rate reduction)
6. Training history visualization
7. Test set evaluation with metrics
8. Refinement recommendations based on performance

**Model Architecture:**
```
Input (106 features)
  ↓
Dense(128) + BatchNorm + ReLU + Dropout(0.3)
  ↓
Dense(64) + BatchNorm + ReLU + Dropout(0.3)
  ↓
Dense(32) + BatchNorm + ReLU + Dropout(0.2)
  ↓
Dense(3, softmax)  → Output (Healthy, ASD, ADHD)
```

**You should run this if:**
- You want to see your first trained model
- You want to understand training dynamics
- You need baseline results to compare against

---

## 🎯 How to Use These Notebooks

### Workflow 1: Learning Journey
1. Start with Notebook 01 to understand features
2. Run Notebook 02 to see end-to-end training
3. Read the analysis sections carefully
4. Modify and re-run to test hypotheses

### Workflow 2: Refinement Cycle
1. Run Notebook 02 and note the accuracy
2. Identify weak areas from confusion matrix
3. Try modifications:
   - Different feature combinations
   - Different hyperparameters
   - Different architectures
4. Re-run and compare results
5. Iterate until satisfied

### Workflow 3: Production Pipeline
1. Extract features using Notebook 01 functions
2. Train model using Notebook 02 code
3. Save model for deployment
4. Use additional notebooks (03-05) for refinement

---

## 🔧 Configuration & Customization

### Modifying Features
In Notebook 01, adjust these parameters in `config/config.py`:

```python
# MFCC settings
config.audio.N_MFCC = 13        # Number of MFCC coefficients
config.audio.N_MEL = 128        # Number of mel bands
config.audio.FMIN = 80          # Minimum frequency (Hz)
config.audio.FMAX = 7600        # Maximum frequency (Hz)

# Spectral settings
config.spectral.HOP_LENGTH = 512
config.spectral.FRAME_LENGTH = 2048

# Prosodic settings
config.prosodic.USE_PARSELMOUTH = False  # Use Praat (True) or librosa (False)
```

### Modifying Model Architecture
In Notebook 02, edit the model definition:

```python
# Current: 128-64-32
# Try: 256-128-64-32 (deeper)
# Try: 64-32 (shallower)
# Try: 128-128-64 (wider)

layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))  # Changed 128 → 256
```

### Modifying Training Parameters
```python
# Learning rate (lower = slower but more stable)
optimizer=Adam(learning_rate=0.0005)

# Batch size (smaller = noisier updates, faster)
batch_size=16  # was 32

# Early stopping patience (how many epochs to wait for improvement)
EarlyStopping(patience=20)  # was 15

# Dropout rates (higher = more regularization, less overfitting)
layers.Dropout(0.5)  # was 0.3
```

---

## 📊 Expected Results

### With Synthetic Data:
- **Accuracy:** 80-95% (synthetic data is easier)
- **Training time:** 1-2 minutes
- **Classes well-separated:** MFCC and prosodic features work well

### With Real Patient Data:
- **Accuracy:** 60-85% (realistic difficulty)
- **Training time:** Depends on dataset size
- **Class imbalance:** May affect results (use class weights)

---

## 🐛 Troubleshooting

### Problem: ImportError - modules not found
**Solution:** Make sure you're in the correct directory and paths are correct
```python
sys.path.insert(0, os.path.abspath('../..'))
```

### Problem: CUDA/GPU warnings
**Solution:** Harmless - TensorFlow falls back to CPU. To use GPU:
```bash
pip install tensorflow[and-cuda]
```

### Problem: Model accuracy is 33% (random guessing)
**Solution:** Check:
1. Data is normalized (`X_train_normalized`)
2. Labels are correct (0, 1, 2)
3. Model is actually training (check loss decreasing)
4. Batch size not too small relative to dataset

### Problem: Model overfitting (training 99%, test 50%)
**Solution:**
1. Increase dropout rates (0.3 → 0.5)
2. Add L2 regularization (1e-4 → 1e-3)
3. Reduce model size (128-64-32 → 64-32-16)
4. More training data

---

## 📈 Next Steps

After mastering these notebooks:

1. **Notebook 03:** K-Fold cross-validation for robust evaluation
2. **Notebook 04:** Hyperparameter tuning (grid search, random search)
3. **Notebook 05:** Feature importance analysis and selection
4. **Real data:** Collect or use public datasets (MIT BIGhand, TIMIT, etc.)
5. **Production:** Deploy as web service or mobile app

---

## 📝 Recommended Reading Order

**For beginners:**
1. Read this README
2. Run Notebook 01 (Feature Extraction)
3. Read markdown cells in Notebook 02
4. Run Notebook 02 (Training)
5. Modify hyperparameters and retrain

**For developers:**
1. Read config.py to understand all parameters
2. Run both notebooks quickly to validate setup
3. Focus on modifying model architecture and data
4. Proceed to advanced notebooks (03-05)

**For researchers:**
1. Read Section 6 of Notebook 02 (Refinement Recommendations)
2. Identify your specific challenge
3. Design experiments to test hypotheses
4. Use advanced notebooks for systematic exploration

---

## 💾 Saving Your Progress

### Save trained model:
```python
model.save('my_model.keras')
```

### Save training history:
```python
import json
with open('training_history.json', 'w') as f:
    json.dump(history.history, f)
```

### Save feature scaler:
```python
import pickle
pickle.dump(scaler, open('feature_scaler.pkl', 'wb'))
```

---

## 📞 Getting Help

If stuck:
1. Check error message carefully
2. Read notebook markdown cells (they explain)
3. Examine similar cells that work
4. Try with smaller dataset first
5. Add `print()` statements to debug

---

**Last Updated:** November 2025
**Notebook Version:** 1.0
**Status:** Ready for step-by-step training! 🚀
