# ASD/ADHD Detection - Project Structure & Implementation Guide

## Overview

This document provides a comprehensive guide to the project structure, module responsibilities, and implementation patterns adapted from reference materials in the AIML folder.

---

## Complete Project Directory Tree

```
f:/AIML/ASD_ADHD_Detection/
│
├── README.md                           # Main project documentation
├── requirements.txt                    # Python dependencies (45+ packages)
├── LICENSE                             # MIT License
├── .gitignore                          # Git ignore patterns
│
├── config/                             # Configuration management
│   ├── __init__.py
│   ├── config.py                       # MAIN: 50+ configuration parameters
│   ├── default_config.yaml             # YAML configuration file
│   └── README.md                       # Config documentation
│
├── src/                                # Source code modules
│   ├── __init__.py
│   │
│   ├── feature_extraction/             # Audio feature extraction
│   │   ├── __init__.py
│   │   ├── mfcc_extractor.py          # MFCC features (13 coeffs + delta)
│   │   ├── spectral_extractor.py      # Spectral features (centroid, rolloff, etc)
│   │   ├── prosodic_extractor.py      # Prosodic features (F0, formants, jitter)
│   │   ├── feature_aggregator.py      # Combine all features
│   │   └── README.md                  # Feature extraction docs
│   │
│   ├── models/                         # MLP classifier
│   │   ├── __init__.py
│   │   ├── mlp_classifier.py          # Main MLP model (128-64-32 architecture)
│   │   ├── model_utils.py             # Model utilities (save/load/evaluate)
│   │   ├── layers.py                  # Custom layers if needed
│   │   └── README.md                  # Model architecture docs
│   │
│   ├── preprocessing/                  # Data preprocessing pipeline
│   │   ├── __init__.py
│   │   ├── audio_preprocessor.py      # Audio loading, trimming, normalization
│   │   ├── feature_normalizer.py      # Feature standardization (z-norm, minmax)
│   │   ├── data_augmentation.py       # Audio augmentation (pitch shift, time shift)
│   │   ├── train_test_split.py        # Data splitting with stratification
│   │   └── README.md                  # Preprocessing docs
│   │
│   ├── evaluation/                     # Model evaluation & metrics
│   │   ├── __init__.py
│   │   ├── metrics.py                 # Accuracy, precision, recall, F1, confusion matrix
│   │   ├── cross_validation.py        # K-fold CV implementation
│   │   ├── visualization.py           # Plots: confusion matrix, ROC, PR curves
│   │   ├── majority_voting.py         # Frame-level aggregation (from DecisionLevelFusion/)
│   │   └── README.md                  # Evaluation docs
│   │
│   └── utils/                          # Utility functions
│       ├── __init__.py
│       ├── logger.py                  # Logging setup
│       ├── file_handler.py            # File I/O operations
│       ├── constants.py               # Global constants
│       ├── validators.py              # Input validation
│       └── README.md                  # Utils docs
│
├── data/                               # Data management
│   ├── README.md
│   ├── raw/                            # Original audio files (not in repo)
│   │   └── .gitkeep
│   ├── processed/                      # Extracted feature files
│   │   └── .gitkeep
│   └── splits/                         # Train/Val/Test NPY files
│       ├── X_train.npy
│       ├── X_val.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       ├── y_val.npy
│       ├── y_test.npy
│       └── .gitkeep
│
├── models/                             # Trained models storage
│   ├── README.md
│   ├── saved/                          # Production models
│   │   ├── asd_adhd_mlp_v1.0.json     # Model architecture
│   │   ├── asd_adhd_mlp_v1.0_weights.h5
│   │   ├── asd_adhd_mlp_v1.0.h5       # Complete model
│   │   ├── scaler.pkl                 # Feature scaler
│   │   ├── label_encoder.pkl          # Class label encoder
│   │   └── metadata.json              # Model metadata
│   │
│   └── checkpoints/                    # Training checkpoints
│       ├── fold_1/
│       │   ├── epoch_10.h5
│       │   ├── epoch_20.h5
│       │   └── best_model.h5
│       └── fold_2/
│           └── ...
│
├── notebooks/                          # Jupyter notebooks for exploration
│   ├── 00_environment_setup.ipynb     # Dependencies & configuration test
│   ├── 01_feature_extraction.ipynb    # Extract & visualize features
│   ├── 02_data_exploration.ipynb      # EDA & class distribution
│   ├── 03_model_training.ipynb        # Train MLP with K-fold CV
│   ├── 04_model_evaluation.ipynb      # Evaluate metrics & plots
│   ├── 05_realtime_demo.ipynb         # Real-time recording & inference
│   ├── 06_comparison_baseline.ipynb   # Compare with RF, SVM baselines
│   └── README.md
│
├── streamlit_app/                      # Web dashboard
│   ├── app.py                         # Main Streamlit app
│   ├── pages/
│   │   ├── 1_Upload_Audio.py          # File upload & processing
│   │   ├── 2_Realtime_Recording.py    # Microphone input
│   │   ├── 3_Model_Info.py            # Model details & performance
│   │   ├── 4_Batch_Processing.py      # Process multiple files
│   │   └── 5_Results_History.py       # Historical predictions
│   ├── assets/
│   │   ├── style.css
│   │   ├── logo.png
│   │   └── icons/
│   ├── utils.py                       # Streamlit utilities
│   └── README.md
│
├── results/                            # Experiment results & outputs
│   ├── plots/
│   │   ├── confusion_matrix.png
│   │   ├── roc_curves.png
│   │   ├── precision_recall.png
│   │   ├── feature_importance.png
│   │   └── training_history.png
│   ├── metrics/
│   │   ├── fold_1_metrics.json
│   │   ├── fold_2_metrics.json
│   │   ├── fold_3_metrics.json
│   │   ├── fold_4_metrics.json
│   │   ├── fold_5_metrics.json
│   │   └── average_metrics.json
│   ├── predictions/
│   │   ├── test_predictions.csv       # Format: file_id, prediction, confidence
│   │   └── realtime_predictions.csv
│   └── README.md
│
├── logs/                               # Training & inference logs
│   ├── training.log
│   ├── inference.log
│   ├── errors.log
│   ├── tensorboard/                   # TensorBoard event files
│   │   ├── fold_1/
│   │   ├── fold_2/
│   │   └── ...
│   └── .gitkeep
│
├── tests/                              # Unit tests
│   ├── __init__.py
│   ├── test_config.py                 # Test configuration
│   ├── test_audio_preprocessor.py     # Test audio preprocessing
│   ├── test_feature_extractors.py     # Test feature extraction
│   ├── test_mlp_model.py              # Test model architecture
│   ├── test_inference.py              # Test inference pipeline
│   └── README.md
│
└── .github/                            # GitHub workflows
    ├── workflows/
    │   ├── tests.yml                  # Automated testing
    │   ├── deploy.yml                 # Deployment workflow
    │   └── code_quality.yml           # Linting & formatting
    └── README.md
```

---

## Module Responsibilities & Implementation

### 1. Configuration Module (`config/config.py`)

**Purpose**: Centralized configuration management  
**Inspired by**: `code/estimate_recs_trained_mdl.py` (YAML loading pattern)

**Key Classes**:
- `AudioConfig` - Audio processing parameters
- `MFCCConfig` - MFCC extraction settings
- `SpectralConfig` - Spectral features
- `ProsodicConfig` - Prosodic analysis settings
- `MLPConfig` - Neural network architecture
- `TrainingConfig` - Training hyperparameters
- `Config` - Master configuration class

**Key Methods**:
```python
config.to_dict()              # Convert to dict
config.to_yaml(filepath)      # Save to YAML
Config.from_yaml(filepath)    # Load from YAML
config.print_config()         # Display all settings
```

**Example Usage**:
```python
from config.config import config
print(config.audio.SAMPLE_RATE)           # 16000
print(config.mlp.HIDDEN_LAYERS)           # [128, 64, 32]
print(config.training.EPOCHS)             # 100
```

---

### 2. Feature Extraction Module (`src/feature_extraction/`)

#### 2.1 MFCC Extractor (`mfcc_extractor.py`)

**Purpose**: Extract MFCC features  
**Inspired by**: `python_speech_features/` and `librosa`  
**Base Pattern from**: `ser_preprocessing.py`

**Expected Features**: 52 (13 base + delta + delta-delta + statistics)

```python
class MFCCExtractor:
    def extract(self, audio: np.ndarray, sr: int) -> np.ndarray:
        # Compute MFCC
        # Add delta and delta-delta
        # Compute statistics (mean, std, min, max, median, q25, q75)
        # Return (52,) feature vector
```

#### 2.2 Spectral Extractor (`spectral_extractor.py`)

**Purpose**: Extract spectral features  
**Inspired by**: `pyAudioAnalysis/`

**Features Extracted**:
- Spectral centroid, rolloff, bandwidth
- Zero-crossing rate (ZCR)
- RMS energy, log energy
- Chroma features (12-dimensional)
- Statistics for each

**Expected Features**: 24

```python
class SpectralExtractor:
    def extract(self, audio: np.ndarray, sr: int) -> np.ndarray:
        # Compute spectral features
        # Return (24,) feature vector
```

#### 2.3 Prosodic Extractor (`prosodic_extractor.py`)

**Purpose**: Extract prosodic features (F0, formants, jitter, shimmer)  
**Inspired by**: `Parselmouth/` and `Dinstein-Lab/ASDSpeech/`

**Key Features** (30+):
- **F0 Analysis**: Mean, std, min, max, median, range, CV
- **Formants**: F1, F2, F3 with bandwidth
- **Jitter**: Pitch perturbation (ASD marker)
- **Shimmer**: Amplitude perturbation (ASD marker)
- **HNR**: Harmonic-to-Noise Ratio
- **Voice Quality**: Voice breaks, voiced rate

```python
class ProsodicExtractor:
    def extract_f0(self, audio: np.ndarray, sr: int) -> Dict:
        # Extract fundamental frequency features
    
    def extract_formants(self, audio: np.ndarray, sr: int) -> Dict:
        # Extract formant features
    
    def extract_jitter_shimmer(self, audio: np.ndarray, sr: int) -> Dict:
        # Extract perturbation features
    
    def extract(self, audio: np.ndarray, sr: int) -> np.ndarray:
        # Combine all prosodic features
        # Return (30+,) feature vector
```

#### 2.4 Feature Aggregator (`feature_aggregator.py`)

**Purpose**: Combine MFCC, spectral, and prosodic features  
**Expected Output**: (106,) feature vector

```python
class FeatureAggregator:
    def aggregate(self, 
                  mfcc_features: np.ndarray,
                  spectral_features: np.ndarray,
                  prosodic_features: np.ndarray) -> np.ndarray:
        # Concatenate all features
        # Optional: Apply PCA dimensionality reduction
        # Return (106,) or (80,) if PCA enabled
```

---

### 3. Model Module (`src/models/`)

#### 3.1 MLP Classifier (`mlp_classifier.py`)

**Purpose**: Main MLP neural network  
**Inspired by**: `mondtorsha/Speech-Emotion-Recognition/` and `x4nth055/emotion-recognition-using-speech/`

**Architecture**:
```
Input (106) → Dense(128, relu) → BatchNorm → Dropout(0.3) → 
              Dense(64, relu)   → BatchNorm → Dropout(0.3) → 
              Dense(32, relu)   → BatchNorm → Dropout(0.2) → 
              Dense(3, softmax)
```

```python
class MLPClassifier:
    def __init__(self, config: Config):
        # Build architecture based on config
    
    def build(self) -> None:
        # Create model
    
    def train(self, X_train, y_train, X_val, y_val) -> Dict:
        # Train model
        # Return training history
    
    def train_with_kfold(self, X, y, n_splits=5) -> Dict:
        # K-fold cross-validation training
        # Return fold metrics and average metrics
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Prediction
        # Return (predictions, confidence scores)
    
    def predict_realtime(self, duration=5.0, sr=16000) -> Dict:
        # Real-time microphone prediction
        # Return {class, confidence, features, markers}
    
    def save(self, filepath: str) -> None:
        # Save model
    
    def load(self, filepath: str) -> None:
        # Load model
```

#### 3.2 Model Utilities (`model_utils.py`)

```python
def save_model_architecture(model, filepath: str):
    # Save model.to_json()

def save_model_weights(model, filepath: str):
    # Save model.save_weights()

def save_complete_model(model, filepath: str):
    # Save model.save() (entire model)

def load_model_architecture_weights(arch_path: str, weights_path: str):
    # Load from JSON + weights

def load_complete_model(filepath: str):
    # Load complete model from H5
```

---

### 4. Preprocessing Module (`src/preprocessing/`)

#### 4.1 Audio Preprocessor (`audio_preprocessor.py`)

**Purpose**: Load and preprocess raw audio  
**Inspired by**: `ser_preprocessing.py`

```python
class AudioPreprocessor:
    def load_audio(self, filepath: str) -> Tuple[np.ndarray, int]:
        # Load audio file
        # Resample to 16000 Hz if needed
        # Return (audio, sr)
    
    def preprocess(self, audio: np.ndarray, sr: int) -> np.ndarray:
        # Normalize audio
        # Trim silence (optional)
        # Pad/truncate to 5 seconds
        # Return processed audio
    
    def trim_silence(self, audio: np.ndarray, top_db=30) -> np.ndarray:
        # Trim leading/trailing silence
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        # Normalize to [-1, 1] range
```

#### 4.2 Feature Normalizer (`feature_normalizer.py`)

**Purpose**: Normalize features for neural network input  
**Inspired by**: `ser_preprocessing.py` (data normalization pattern)

```python
class FeatureNormalizer:
    def __init__(self, method='standard'):  # 'standard' or 'minmax'
        pass
    
    def fit(self, X: np.ndarray):
        # Compute mean/std or min/max
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        # Apply normalization
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        # Fit and transform
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        # Reverse normalization
    
    def save(self, filepath: str):
        # Save scaler (pickle)
    
    def load(self, filepath: str):
        # Load scaler
```

#### 4.3 Data Augmentation (`data_augmentation.py`)

**Purpose**: Augment training data  
**Inspired by**: `pyAudioAnalysis/` (data augmentation patterns)

```python
class AudioAugmentor:
    def pitch_shift(self, audio: np.ndarray, sr: int, n_steps: int) -> np.ndarray:
        # Shift pitch up/down
    
    def time_stretch(self, audio: np.ndarray, rate: float) -> np.ndarray:
        # Stretch/compress time
    
    def add_noise(self, audio: np.ndarray, noise_factor: float) -> np.ndarray:
        # Add Gaussian noise
    
    def random_augmentation(self, audio: np.ndarray, sr: int) -> np.ndarray:
        # Randomly apply augmentations
```

#### 4.4 Train-Test Split (`train_test_split.py`)

```python
def stratified_train_test_split(X, y, train_ratio=0.7, val_ratio=0.15):
    # Stratified split maintaining class distribution
    # Return (X_train, X_val, X_test, y_train, y_val, y_test)

def kfold_split(X, y, n_splits=5):
    # K-fold split generator
    # Yield (X_train, X_test, y_train, y_test) for each fold
```

---

### 5. Evaluation Module (`src/evaluation/`)

#### 5.1 Metrics (`metrics.py`)

**Purpose**: Compute classification metrics  
**Inspired by**: `DecisionLevelFusion/Mean.py` (metrics pattern)

```python
class ClassificationMetrics:
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, classes: List[str]):
        pass
    
    def accuracy(self) -> float:
        pass
    
    def precision(self, average='weighted') -> float:
        pass
    
    def recall(self, average='weighted') -> float:
        pass
    
    def f1_score(self, average='weighted') -> float:
        pass
    
    def confusion_matrix(self) -> np.ndarray:
        pass
    
    def classification_report(self) -> str:
        pass
    
    def roc_auc(self, y_proba) -> float:
        pass
```

#### 5.2 Cross-Validation (`cross_validation.py`)

**Purpose**: K-fold cross-validation evaluation  
**Inspired by**: `AudioModels/DenseNetCNN.py` (K-fold pattern)

```python
def evaluate_kfold(model, X, y, n_splits=5, metric='accuracy'):
    # Run K-fold CV
    # Return fold scores and average score
```

#### 5.3 Visualization (`visualization.py`)

```python
def plot_confusion_matrix(y_true, y_pred, classes, normalize=True):
    # Plot confusion matrix

def plot_roc_curves(y_true, y_proba, classes):
    # Plot ROC curves per class

def plot_precision_recall(y_true, y_proba, classes):
    # Plot precision-recall curves

def plot_training_history(history):
    # Plot loss and accuracy curves
```

#### 5.4 Majority Voting (`majority_voting.py`)

**Purpose**: Aggregate frame-level predictions  
**Inspired by**: `AudioModels/BiLSTM.py` (majority_voting pattern)

```python
def majority_voting(predictions: Dict[str, List[int]]) -> Dict[str, int]:
    # Aggregate predictions by recording ID
    # Return final predictions
```

---

### 6. Utils Module (`src/utils/`)

#### 6.1 Logger (`logger.py`)

```python
def setup_logger(name: str, log_file: str, level='INFO'):
    # Setup logging with file and console handlers
```

#### 6.2 File Handler (`file_handler.py`)

```python
def save_npy(data: np.ndarray, filepath: str):
    # Save NumPy array

def load_npy(filepath: str) -> np.ndarray:
    # Load NumPy array

def save_csv(data: pd.DataFrame, filepath: str):
    # Save CSV

def save_json(data: Dict, filepath: str):
    # Save JSON

def load_json(filepath: str) -> Dict:
    # Load JSON
```

#### 6.3 Constants (`constants.py`)

```python
CLASS_NAMES = ['Healthy', 'ASD', 'ADHD']
CLASS_COLORS = {'Healthy': 'green', 'ASD': 'red', 'ADHD': 'orange'}
EMOTION_LABELS = {0: 'Healthy', 1: 'ASD', 2: 'ADHD'}
```

---

## Data Flow Pipeline

```
Raw Audio File (WAV)
    ↓
[AudioPreprocessor]
    • Load (16000 Hz, mono)
    • Trim silence
    • Normalize
    ↓
Preprocessed Audio (5 sec @ 16000 Hz)
    ↓
┌───────────────────────────────────────────────┐
│  [FeatureExtractor] (Parallel)                │
├───────────────────────────────────────────────┤
│ MFCCExtractor      → 52 features              │
│ SpectralExtractor  → 24 features              │
│ ProsodicExtractor  → 30+ features             │
└───────────────────────────────────────────────┘
    ↓
[FeatureAggregator]
    • Concatenate features
    • Optional: PCA reduction (106 → 80)
    ↓
Feature Vector (106,) or (80,)
    ↓
[FeatureNormalizer]
    • Standardization (z-norm)
    • MinMax scaling
    ↓
Normalized Features (106,)
    ↓
[MLPClassifier]
    • Input: 106
    • Dense(128) → Dense(64) → Dense(32) → Dense(3)
    • Softmax output
    ↓
Class Prediction + Confidence Scores
    ├─ ASD: 0.85
    ├─ ADHD: 0.12
    └─ Healthy: 0.03
    ↓
[VocalMarkerExplainer]
    • Identify important features
    • Map to vocal markers (jitter, F0, etc)
    ↓
Final Report
    ├─ Prediction: ASD
    ├─ Confidence: 85%
    ├─ Vocal Markers: High jitter (1.2%), Low F0 (95 Hz)
    └─ Recommendation: Consult specialist
```

---

## Implementation Roadmap

### Phase 1: Core Infrastructure ✅
- [x] Project structure creation
- [x] Configuration module (50+ parameters)
- [x] Requirements.txt
- [x] README documentation

### Phase 2: Feature Extraction (Next)
- [ ] MFCC extractor (librosa)
- [ ] Spectral features (pyAudioAnalysis patterns)
- [ ] Prosodic features (Parselmouth)
- [ ] Feature aggregation & normalization

### Phase 3: Model Development
- [ ] MLP architecture (TensorFlow/Keras)
- [ ] Training loop with early stopping
- [ ] K-fold cross-validation
- [ ] Model saving/loading utilities

### Phase 4: Evaluation & Visualization
- [ ] Metrics computation
- [ ] Confusion matrix plots
- [ ] ROC/PR curves
- [ ] Training history visualization

### Phase 5: Real-time & Dashboard
- [ ] Microphone input handler
- [ ] Real-time inference pipeline
- [ ] Streamlit web dashboard
- [ ] Audio visualization (waveform, spectrogram)

### Phase 6: Testing & Deployment
- [ ] Unit tests for all modules
- [ ] Integration tests
- [ ] Docker containerization
- [ ] API server (FastAPI)

---

## Adapter Patterns from Reference Repositories

| Component | From | Adaptation |
|-----------|------|-----------|
| MFCC Features | `ser_preprocessing.py` | Feature statistics (mean, std, min, max) |
| Spectral Features | `pyAudioAnalysis/` | Centroid, rolloff, bandwidth, ZCR, chroma |
| Prosodic Features | `Parselmouth/` | F0, formants, jitter, shimmer, HNR |
| MLP Architecture | `mondtorsha/` | 3-layer network (128-64-32) with batch norm |
| Training Loop | `x4nth055/` | Training with validation and callbacks |
| K-fold CV | `AudioModels/DenseNetCNN.py` | Cross-validation with majority voting |
| Model Saving | `final_results_gender_test.ipynb` | JSON architecture + separate weights |
| Evaluation Metrics | `DecisionLevelFusion/Mean.py` | Accuracy, precision, recall, F1, CCC |
| Real-time Recording | `3_realtime_ser.ipynb` | Microphone input with buffer |
| Configuration | `code/estimate_recs_trained_mdl.py` | YAML loading with parameter management |

---

## Example Integration Script

```python
# main.py - End-to-end pipeline example

from config.config import config
from src.preprocessing.audio_preprocessor import AudioPreprocessor
from src.feature_extraction.mfcc_extractor import MFCCExtractor
from src.feature_extraction.spectral_extractor import SpectralExtractor
from src.feature_extraction.prosodic_extractor import ProsodicExtractor
from src.feature_extraction.feature_aggregator import FeatureAggregator
from src.preprocessing.feature_normalizer import FeatureNormalizer
from src.models.mlp_classifier import MLPClassifier

# Load and preprocess audio
preprocessor = AudioPreprocessor(config)
audio, sr = preprocessor.load_audio('sample_audio.wav')
audio_clean = preprocessor.preprocess(audio, sr)

# Extract features
mfcc_ext = MFCCExtractor(config)
spectral_ext = SpectralExtractor(config)
prosodic_ext = ProsodicExtractor(config)

mfcc_feat = mfcc_ext.extract(audio_clean, sr)
spectral_feat = spectral_ext.extract(audio_clean, sr)
prosodic_feat = prosodic_ext.extract(audio_clean, sr)

# Aggregate features
aggregator = FeatureAggregator(config)
features = aggregator.aggregate(mfcc_feat, spectral_feat, prosodic_feat)

# Normalize features
normalizer = FeatureNormalizer(method='standard')
normalizer.fit(X_train_features)
features_norm = normalizer.transform(features[np.newaxis, :])

# Predict
model = MLPClassifier(config)
model.load('models/saved/asd_adhd_mlp_v1.0.h5')
prediction, confidence = model.predict(features_norm)

print(f"Prediction: {config.dataset.CLASSES[prediction[0]]}")
print(f"Confidence: {confidence[0, prediction[0]]:.2%}")
```

---

## Next Steps

1. **Implement Phase 2 modules** - Feature extractors
2. **Create example notebooks** - Demonstrate feature extraction
3. **Build MLP model** - Implement training loop
4. **Add evaluation** - Metrics and visualization
5. **Develop dashboard** - Streamlit web interface
6. **Deploy** - Docker + API server

---

**Last Updated**: November 2025  
**Version**: 1.0.0
