# ASD/ADHD Voice Detection System

## Project Overview

Real-time differential diagnosis system for **Autism Spectrum Disorder (ASD)** vs **Attention-Deficit/Hyperactivity Disorder (ADHD)** using acoustic and prosodic feature analysis with **Multi-Layer Perceptron (MLP)** neural networks.

### Key Features

✅ **Automated Feature Extraction** - 100+ acoustic and prosodic features  
✅ **Real-time Recording & Inference** - 5-second voice sample analysis  
✅ **MLP Classifier** - 3-layer neural network (128-64-32 neurons)  
✅ **3-Class Classification** - ASD, ADHD, or Healthy  
✅ **Confidence Scores** - Probability estimates for each class  
✅ **Vocal Marker Explanation** - Interpretable feedback on key indicators  
✅ **Streamlit Dashboard** - Interactive web-based UI  
✅ **Cross-Validation** - 5-fold CV with comprehensive metrics  

---

## System Architecture

```
ASD_ADHD_Detection/
│
├── config/                      # Configuration management
│   └── config.py               # Centralized configuration (50+ parameters)
│
├── src/                         # Source code modules
│   ├── feature_extraction/     # Acoustic & prosodic feature extraction
│   │   ├── mfcc_extractor.py
│   │   ├── spectral_extractor.py
│   │   └── prosodic_extractor.py
│   │
│   ├── models/                 # MLP model architecture
│   │   ├── mlp_classifier.py
│   │   └── model_utils.py
│   │
│   ├── preprocessing/          # Data preprocessing pipeline
│   │   ├── audio_preprocessor.py
│   │   ├── feature_normalizer.py
│   │   └── data_augmentation.py
│   │
│   ├── evaluation/             # Evaluation & metrics
│   │   ├── metrics.py
│   │   ├── cross_validation.py
│   │   └── visualization.py
│   │
│   └── utils/                  # Utility functions
│       ├── logger.py
│       ├── file_handler.py
│       └── constants.py
│
├── data/                        # Data management
│   ├── raw/                    # Original audio files
│   ├── processed/              # Extracted features
│   └── splits/                 # Train/Val/Test splits
│
├── models/                      # Model storage
│   ├── saved/                  # Trained model checkpoints
│   └── checkpoints/            # Training checkpoints
│
├── streamlit_app/              # Web dashboard
│   ├── app.py
│   ├── pages/
│   └── assets/
│
├── notebooks/                   # Jupyter notebooks for exploration
│   ├── 01_feature_extraction.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_evaluation.ipynb
│   └── 04_realtime_demo.ipynb
│
├── results/                     # Experiment results & plots
├── logs/                        # Training & inference logs
├── tests/                       # Unit tests
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## Feature Engineering

### 1. **MFCC Features** (52 features)
- **13** Mel-Frequency Cepstral Coefficients
- **13** Delta features (velocity)
- **13** Delta-Delta features (acceleration)
- **13** Statistics per feature type (mean, std, min, max)

*Based on: `python_speech_features/` and `librosa`*

### 2. **Spectral Features** (24 features)
- Spectral Centroid, Rolloff, Bandwidth
- Zero-Crossing Rate (ZCR)
- RMS Energy, Log Energy
- Chroma features (12-dimensional)
- Statistics: mean, std, min, max

*Based on: `pyAudioAnalysis/`*

### 3. **Prosodic Features** (30+ features)
- **F0/Pitch** - Mean, Std, Range, Median, Coefficient of Variation
- **Formants** - F1, F2, F3 with statistics
- **Jitter** - Fundamental frequency perturbation (ASD marker)
- **Shimmer** - Amplitude perturbation (ASD marker)
- **HNR** - Harmonic-to-Noise Ratio
- **Voice Quality** - Voice breaks, voiced rate

*Based on: `Parselmouth/` and `Dinstein-Lab/ASDSpeech/` (49 autism features)*

### 4. **Total Feature Dimension**
- **MFCC**: 52
- **Spectral**: 24  
- **Prosodic**: 30+
- **Total**: ~106 features → Dimensionality reduction optional (PCA to 80)

---

## Model Architecture

### MLP Classifier (3 Hidden Layers)

```
Input Layer (106 features)
    ↓
Dense Layer 1 (128 units, ReLU)
├─ Batch Normalization
├─ Dropout (30%)
└─ L2 Regularization (1e-4)
    ↓
Dense Layer 2 (64 units, ReLU)
├─ Batch Normalization
├─ Dropout (30%)
└─ L2 Regularization (1e-4)
    ↓
Dense Layer 3 (32 units, ReLU)
├─ Batch Normalization
├─ Dropout (20%)
└─ L2 Regularization (1e-4)
    ↓
Output Layer (3 units, Softmax)
    ↓
Predictions: [ASD, ADHD, Healthy]
```

**Total Parameters**: ~24,000 trainable parameters

*Based on: `mondtorsha/Speech-Emotion-Recognition/` (MLP pattern) and `x4nth055/emotion-recognition-using-speech/` (architecture)*

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration, optional)
- 4GB RAM minimum

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Test Configuration

```python
from config.config import config
config.print_config()
```

This will display all 50+ configuration parameters.

---

## Usage

### 1. **Feature Extraction from Audio File**

```python
from src.feature_extraction.prosodic_extractor import ProsodicExtractor
from src.preprocessing.audio_preprocessor import AudioPreprocessor
from config.config import config

# Load and preprocess audio
preprocessor = AudioPreprocessor(config)
audio, sr = preprocessor.load_audio('path/to/audio.wav')
audio_clean = preprocessor.preprocess(audio, sr)

# Extract features
prosodic_extractor = ProsodicExtractor(config)
prosodic_features = prosodic_extractor.extract(audio_clean, sr)
```

### 2. **Training MLP Model**

```python
from src.models.mlp_classifier import MLPClassifier
from src.preprocessing.feature_normalizer import FeatureNormalizer
from config.config import config

# Initialize model
model = MLPClassifier(config)
model.build()

# Train with cross-validation
model.train_with_kfold(
    X_features,
    y_labels,
    n_splits=config.dataset.K_FOLDS
)
```

### 3. **Real-time Prediction**

```python
from src.models.mlp_classifier import MLPClassifier

# Load trained model
model = MLPClassifier(config)
model.load('models/saved/asd_adhd_mlp_v1.0.h5')

# Predict from microphone
prediction = model.predict_realtime(
    duration=5.0,  # seconds
    sample_rate=16000
)

print(f"Prediction: {prediction['class']}")
print(f"Confidence: {prediction['confidence']:.2%}")
```

### 4. **Web Dashboard (Streamlit)**

```bash
streamlit run streamlit_app/app.py
```

Open browser → `http://localhost:8501`

---

## Reference Repositories & Adaptations

### Core Implementations

| Repository | Purpose | Adaptation |
|---|---|---|
| **x4nth055/emotion-recognition-using-speech** | MLP architecture, real-time recording | → MLP classifier for ASD/ADHD |
| **mondtorsha/Speech-Emotion-Recognition** | MLP vs LSTM on RAVDESS | → MLP architecture & training loop |
| **pyAudioAnalysis** | Audio feature extraction patterns | → Spectral features extraction |
| **python_speech_features** | MFCC computation | → MFCC extraction pipeline |
| **Parselmouth** | Prosodic analysis interface | → Pitch, formants, jitter, shimmer |
| **Dinstein-Lab/ASDSpeech** | 49 autism acoustic features | → Extended to 100+ features |
| **ronit1706/Autism-Detection** | Multi-class ML models (RF, SVM, ANN) | → Multi-class (ASD/ADHD/Healthy) MLP |
| **56kd/MulitmodalDepressionDetection** | Hybrid CNN-LSTM, audio extraction | → Feature extraction patterns |
| **MITESHPUTHRANNEU/Speech-Emotion-Analyzer** | Web dashboard with visualization | → Streamlit app structure |

### Pattern Mappings

```
Emotion Recognition → ASD/ADHD Detection
EmotionRecognizer   → ASDDetector
emotion_label       → diagnosis_label
8 emotion classes   → 3 diagnostic classes (ASD/ADHD/Healthy)
RAVDESS dataset     → Clinical voice recordings
```

---

## Training Pipeline

### 1. **Data Preparation**
- Load raw audio files (5-second samples)
- Split: 70% train, 15% val, 15% test
- Apply stratified K-fold CV (5 folds)

### 2. **Feature Extraction**
- Extract 106 acoustic + prosodic features per sample
- Normalize features (standardization)
- Optional: Apply PCA (reduce to 80 dimensions)

### 3. **Model Training**
- Initialize MLP with 128-64-32 architecture
- Apply batch norm + dropout regularization
- Train for max 100 epochs with early stopping
- Monitor validation accuracy (patience=15)

### 4. **Evaluation**
- Compute confusion matrix & metrics (accuracy, precision, recall, F1)
- Generate ROC curves per class
- Perform majority voting for frame-level aggregation
- Report class-wise performance

### 5. **Model Saving**
- Save model architecture (JSON)
- Save trained weights (H5)
- Save feature scaler & label encoder (pickle)
- Create model metadata (config, version, timestamp)

---

## Configuration Parameters

Key configuration in `config/config.py`:

| Parameter | Value | Purpose |
|---|---|---|
| `SAMPLE_RATE` | 16000 Hz | Audio sampling rate |
| `DURATION` | 5 sec | Voice sample length |
| `FRAME_LENGTH` | 400 samples | 25ms window |
| `HOP_LENGTH` | 160 samples | 10ms step (60% overlap) |
| `N_MFCC` | 13 | MFCC coefficients |
| `N_MEL` | 128 | Mel frequency bins |
| `F0_MIN` | 80 Hz | Minimum pitch frequency |
| `F0_MAX` | 400 Hz | Maximum pitch frequency |
| `HIDDEN_LAYERS` | [128, 64, 32] | MLP architecture |
| `DROPOUT_RATE` | 0.3 | Regularization |
| `LEARNING_RATE` | 0.001 | Adam optimizer LR |
| `EPOCHS` | 100 | Max training epochs |
| `BATCH_SIZE` | 32 | Training batch size |
| `K_FOLDS` | 5 | Cross-validation folds |

---

## Performance Targets

Based on reference implementations:

- **Accuracy**: ≥ 85% (binary classification baseline)
- **Precision**: ≥ 0.83 per class
- **Recall**: ≥ 0.82 per class
- **F1 Score**: ≥ 0.82

*Reference: ronit1706/Autism-Detection achieved 90% on binary ASD/non-ASD*

---

## API Endpoints (Optional)

```
POST /predict
├─ Input: Audio file (WAV, MP3, OGG)
└─ Output: {class, confidence, features, markers}

POST /predict-realtime
├─ Input: Audio stream
└─ Output: Real-time predictions

GET /model/info
└─ Output: Model architecture, version, performance metrics
```

---

## Troubleshooting

### Issue: Low accuracy
- **Solution**: Check feature normalization, verify class balance, increase training epochs

### Issue: Memory error
- **Solution**: Reduce batch size, use data generators, enable gradient checkpointing

### Issue: Slow feature extraction
- **Solution**: Enable GPU acceleration, use vectorized operations (NumPy/SciPy)

### Issue: Noisy predictions
- **Solution**: Increase confidence threshold, use ensemble predictions

---

## Future Enhancements

- [ ] **Transfer Learning**: Pre-trained models (Speech2Vec, WavLM)
- [ ] **Ensemble Methods**: Combine MLP with Random Forest, SVM
- [ ] **Attention Mechanisms**: Self-attention over time dimension
- [ ] **Temporal Modeling**: LSTM/GRU layers for sequence modeling
- [ ] **Multimodal**: Combine voice with text analysis (BERT)
- [ ] **Explainability**: SHAP/LIME for feature importance
- [ ] **Mobile Deployment**: TensorFlow Lite for mobile apps
- [ ] **A/B Testing**: Clinical validation framework

---

## Citation & References

This project adapts implementations and patterns from multiple open-source repositories:

```bibtex
@misc{emotion_recognition_speech,
  title={Speech Emotion Recognition using MLP},
  author={x4nth055},
  url={https://github.com/x4nth055/emotion-recognition-using-speech}
}

@misc{asd_speech,
  title={Autism Spectrum Disorder Detection from Speech},
  author={Dinstein Lab},
  url={https://github.com/Dinstein-Lab/ASDSpeech}
}

@misc{autism_detection,
  title={Autism Detection using Machine Learning},
  author={Ronit Khurana},
  url={https://github.com/ronit1706/Autism-Detection}
}
```

---

## License

MIT License - See LICENSE file for details

---

## Contact & Support

For questions or issues:
- Create an issue on GitHub
- Check documentation in `/notebooks/`
- Review configuration in `/config/config.py`

---

**Last Updated**: November 2025  
**Version**: 1.0.0  
**Status**: Active Development 🚀
