"""
ASD/ADHD Detection Configuration Module
========================================
Comprehensive configuration file for real-time ASD vs ADHD differential diagnosis system
using MLP neural networks with acoustic and prosodic feature extraction.

Inspired by and adapting patterns from:
- x4nth055/emotion-recognition-using-speech/ (MLP architecture)
- pyAudioAnalysis/ (feature extraction methods)
- Dinstein-Lab/ASDSpeech/ (49 acoustic features for autism detection)
- ronit1706/Autism-Detection/ (multi-class ML approach)
- mondtorsha/Speech-Emotion-Recognition/ (MLP implementation)
- Parselmouth/ (prosodic analysis patterns)
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
import yaml

# ============================================================================
# PROJECT STRUCTURE
# ============================================================================

# Get the root directory of the project
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATA_SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
STREAMLIT_DIR = PROJECT_ROOT / "streamlit_app"
TESTS_DIR = PROJECT_ROOT / "tests"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, DATA_SPLITS_DIR, SAVED_MODELS_DIR,
                   CHECKPOINTS_DIR, RESULTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================================
# AUDIO PROCESSING CONFIGURATION
# Based on pyAudioAnalysis/ and python_speech_features/ patterns
# ============================================================================

class AudioConfig:
    """Audio processing and feature extraction parameters."""
    
    # Audio file specifications
    SAMPLE_RATE = 16000              # Hz, standard for speech processing
    DURATION = 5                     # seconds, 5-second voice sample
    CHUNK_DURATION = 0.025           # seconds (25ms window)
    STEP_DURATION = 0.010            # seconds (10ms step) - 60% overlap
    
    # Audio preprocessing
    FRAME_LENGTH = int(SAMPLE_RATE * CHUNK_DURATION)       # samples
    HOP_LENGTH = int(SAMPLE_RATE * STEP_DURATION)          # samples
    N_FFT = 2048                     # FFT window size
    
    # Audio normalization
    NORMALIZE_AUDIO = True
    TRIM_SILENCE = True
    TRIM_THRESHOLD_DB = 30           # dB threshold for silence trimming
    
    # Voice Activity Detection (VAD)
    ENABLE_VAD = True
    VAD_THRESHOLD = 0.5              # VAD confidence threshold
    
    # Supported audio formats
    AUDIO_FORMATS = ['.wav', '.mp3', '.ogg', '.flac']
    MAX_AUDIO_LENGTH = 10            # seconds, max allowed audio duration


# ============================================================================
# MFCC FEATURE EXTRACTION
# Based on librosa and python_speech_features patterns
# ============================================================================

class MFCCConfig:
    """MFCC (Mel-Frequency Cepstral Coefficient) extraction parameters."""
    
    N_MFCC = 13                      # number of MFCC coefficients
    N_MEL = 128                      # number of mel frequency bins
    FMIN = 80                        # Hz, minimum frequency
    FMAX = 7600                      # Hz, maximum frequency
    
    # Delta features (velocity and acceleration)
    COMPUTE_DELTA = True
    COMPUTE_DELTA_DELTA = True       # acceleration
    
    # Statistics computed from MFCCs over time
    STATISTICS = ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75']
    
    # Total MFCC features: 13 base + 13 delta + 13 delta-delta = 39
    # With statistics: 39 * 7 = 273 features
    # We'll use reduced set for efficiency: mean, std, min, max = 52 features


# ============================================================================
# SPECTRAL FEATURES CONFIGURATION
# Based on pyAudioAnalysis patterns
# ============================================================================

class SpectralConfig:
    """Spectral feature extraction parameters."""
    
    # Spectral envelope features
    COMPUTE_SPECTRAL_CENTROID = True
    COMPUTE_SPECTRAL_ROLLOFF = True
    COMPUTE_SPECTRAL_BANDWIDTH = True
    COMPUTE_ZERO_CROSSING_RATE = True
    
    # Energy features
    COMPUTE_RMS_ENERGY = True
    COMPUTE_LOG_ENERGY = True
    
    # Chroma features (12-dimensional, one for each pitch class)
    COMPUTE_CHROMA = True
    N_CHROMA = 12
    
    # Tempogram (rhythmic structure)
    COMPUTE_TEMPOGRAM = False       # Optional, computationally expensive
    
    # Spectral statistics
    STATISTICS = ['mean', 'std', 'min', 'max']


# ============================================================================
# PROSODIC FEATURES CONFIGURATION
# Based on Parselmouth patterns and Dinstein-Lab/ASDSpeech features
# Inspired by: 49 acoustic features for autism detection
# ============================================================================

class ProsodicConfig:
    """Prosodic feature extraction parameters (pitch, formants, duration)."""
    
    # Fundamental Frequency (F0) / Pitch Analysis
    COMPUTE_F0 = True
    F0_MIN = 80                      # Hz, minimum F0 for male/female
    F0_MAX = 400                     # Hz, maximum F0
    F0_VOICING_THRESHOLD = 0.45      # voicing threshold
    
    # F0 Statistics to extract
    F0_FEATURES = [
        'mean_f0',
        'std_f0',
        'min_f0',
        'max_f0',
        'median_f0',
        'range_f0',                  # max - min
        'cv_f0',                     # coefficient of variation
    ]
    
    # Formants (F1, F2, F3) - resonances in vocal tract
    COMPUTE_FORMANTS = True
    NUM_FORMANTS = 3                 # F1, F2, F3
    MAX_FORMANT = 5000               # Hz, maximum formant frequency
    
    # Formant tracking stability (important for ASD detection)
    COMPUTE_FORMANT_BANDWIDTH = True
    
    # Duration features
    COMPUTE_DURATION = True
    
    # Jitter (pitch perturbation) - ASD marker
    COMPUTE_JITTER = True
    JITTER_WINDOW_MS = 40            # ms window for jitter calculation
    
    # Shimmer (amplitude perturbation) - ASD marker
    COMPUTE_SHIMMER = True
    SHIMMER_WINDOW_MS = 40           # ms window for shimmer calculation
    
    # Harmonic-to-Noise Ratio (HNR) - voice quality
    COMPUTE_HNR = True
    
    # Voice Quality Measures
    COMPUTE_VOICE_BREAKS = True      # number of voice breaks
    COMPUTE_VOICED_RATE = True       # percentage of voiced frames


# ============================================================================
# FEATURE AGGREGATION & DIMENSIONALITY
# Adapted from Dinstein-Lab/ASDSpeech (49 features) and extended
# ============================================================================

class FeatureConfig:
    """Complete feature extraction and aggregation configuration."""
    
    # Feature selection
    COMPUTE_MFCC = True
    COMPUTE_SPECTRAL = True
    COMPUTE_PROSODIC = True
    
    # Expected feature dimensions (approximate)
    # MFCC: ~52 features (13 base + delta + delta-delta with stats)
    # Spectral: ~24 features (6 types * 4 stats)
    # Prosodic: ~30 features (F0, formants, jitter, shimmer, HNR, voice quality)
    # Total: ~106 features (extended from original 49)
    
    EXPECTED_NUM_FEATURES = 106
    
    # Feature normalization
    NORMALIZE_FEATURES = True
    NORMALIZATION_METHOD = 'standard'  # 'standard' (z-norm) or 'minmax'
    
    # Feature aggregation method for variable-length utterances
    AGGREGATION_METHOD = 'statistics'  # 'statistics', 'mean', or 'concatenate'
    
    # Feature dimensionality reduction (optional)
    USE_PCA = False                  # If True, reduce to PCA_COMPONENTS
    PCA_COMPONENTS = 80              # Number of PCA components if enabled
    USE_FEATURE_SELECTION = False    # If True, use SelectKBest
    SELECTED_FEATURES = 50           # Number of top features to select


# ============================================================================
# DATASET CONFIGURATION
# Based on ser_preprocessing.py pattern
# ============================================================================

class DatasetConfig:
    """Dataset and data split configuration."""
    
    # Class labels for ASD/ADHD classification
    CLASSES = {
        0: 'Healthy',
        1: 'ASD',
        2: 'ADHD'
    }
    NUM_CLASSES = 3
    
    # Data split ratios
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Cross-validation
    USE_KFOLD = True
    K_FOLDS = 5
    RANDOM_STATE = 42
    
    # Data augmentation
    ENABLE_DATA_AUGMENTATION = True
    AUGMENTATION_FACTOR = 1.5        # multiply training data by this factor
    
    # Imbalanced class handling
    USE_CLASS_WEIGHTS = True
    HANDLE_IMBALANCE = True          # Use SMOTE or oversampling


# ============================================================================
# MLP NEURAL NETWORK ARCHITECTURE
# Based on mondtorsha/Speech-Emotion-Recognition MLP pattern
# Inspired by: x4nth055/emotion-recognition-using-speech
# ============================================================================

class MLPConfig:
    """MLP (Multi-Layer Perceptron) neural network configuration."""
    
    # Architecture - 3-layer MLP as specified
    INPUT_DIM = 106                  # Number of features (EXPECTED_NUM_FEATURES)
    HIDDEN_LAYERS = [128, 64, 32]    # 3 hidden layers with decreasing units
    OUTPUT_DIM = 3                   # 3 classes: ASD, ADHD, Healthy
    
    # Activation functions
    HIDDEN_ACTIVATION = 'relu'       # ReLU for hidden layers
    OUTPUT_ACTIVATION = 'softmax'    # Softmax for multi-class classification
    
    # Regularization
    USE_BATCH_NORM = True            # Batch normalization after each layer
    DROPOUT_RATE = 0.3               # Dropout probability
    L2_REGULARIZATION = 1e-4         # L2 (Ridge) regularization factor
    
    # Architecture Details
    LAYER_CONFIG = [
        {
            'type': 'dense',
            'units': 128,
            'activation': 'relu',
            'batch_norm': True,
            'dropout': 0.3,
            'l2_reg': 1e-4
        },
        {
            'type': 'dense',
            'units': 64,
            'activation': 'relu',
            'batch_norm': True,
            'dropout': 0.3,
            'l2_reg': 1e-4
        },
        {
            'type': 'dense',
            'units': 32,
            'activation': 'relu',
            'batch_norm': True,
            'dropout': 0.2,
            'l2_reg': 1e-4
        },
        {
            'type': 'dense',
            'units': 3,
            'activation': 'softmax',
            'batch_norm': False,
            'dropout': 0.0,
            'l2_reg': 0
        }
    ]


# ============================================================================
# TRAINING CONFIGURATION
# Based on DeepBeliefModel.py and HybridCNNLSTM.py patterns
# ============================================================================

class TrainingConfig:
    """Model training hyperparameters."""
    
    # Training basics
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    OPTIMIZER = 'adam'               # 'adam', 'rmsprop', 'sgd'
    
    # Loss function and metrics
    LOSS_FUNCTION = 'categorical_crossentropy'
    METRICS = ['accuracy', 'precision', 'recall']
    
    # Learning rate scheduling
    USE_LR_SCHEDULER = True
    LR_SCHEDULER_TYPE = 'step'       # 'step', 'exponential', 'cosine'
    LR_STEP_SIZE = 20                # Reduce LR every N epochs
    LR_GAMMA = 0.5                   # Multiply LR by this factor
    
    # Early stopping (based on code/estimate_recs_trained_mdl.py pattern)
    ENABLE_EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 15     # Stop if no improvement for N epochs
    EARLY_STOPPING_METRIC = 'val_loss'
    EARLY_STOPPING_MODE = 'min'      # 'min' for loss, 'max' for accuracy
    
    # Model checkpointing
    SAVE_BEST_ONLY = True
    CHECKPOINT_METRIC = 'val_accuracy'
    
    # Validation
    VALIDATION_SPLIT = 0.15
    SHUFFLE_TRAINING_DATA = True


# ============================================================================
# REAL-TIME PREDICTION CONFIGURATION
# Based on 3_realtime_ser.ipynb and LSTM.py patterns
# ============================================================================

class RealtimeConfig:
    """Real-time recording and prediction configuration."""
    
    # Microphone recording
    ENABLE_REALTIME = True
    RECORDING_DEVICE = None          # None = default device
    CHANNELS = 1                     # Mono audio
    CHUNK_SIZE = 1024                # samples per chunk
    
    # Recording parameters
    RECORD_DURATION = 5              # seconds
    BUFFER_SIZE = 10                 # number of chunks to buffer
    
    # Real-time processing
    PROCESS_CHUNK_BY_CHUNK = True
    OVERLAP_RATIO = 0.5              # overlap between consecutive frames
    
    # Inference
    BATCH_INFERENCE = False
    CONFIDENCE_THRESHOLD = 0.7       # minimum confidence for prediction
    
    # Display/Logging
    SHOW_WAVEFORM = True
    SHOW_SPECTROGRAM = True
    LOG_PREDICTIONS = True
    LOG_FILE = str(LOGS_DIR / 'realtime_predictions.log')


# ============================================================================
# EVALUATION & METRICS CONFIGURATION
# Based on DecisionLevelFusion/ metrics patterns
# ============================================================================

class EvaluationConfig:
    """Evaluation metrics and reporting configuration."""
    
    # Classification metrics
    COMPUTE_CONFUSION_MATRIX = True
    COMPUTE_ROC_CURVE = True
    COMPUTE_PRECISION_RECALL = True
    COMPUTE_F1_SCORE = True
    
    # CCC (Concordance Correlation Coefficient) - from config_file.yaml
    COMPUTE_CCC = False              # For regression tasks
    
    # Cross-validation evaluation
    USE_CROSS_VALIDATION = True
    CV_FOLDS = 5
    
    # Class-wise metrics
    REPORT_CLASS_WISE = True
    AVERAGE_METHOD = 'weighted'      # 'weighted', 'macro', 'micro'
    
    # Confusion matrix normalization
    NORMALIZE_CONFUSION_MATRIX = True
    
    # Results aggregation (from DecisionLevelFusion/Mean.py)
    AGGREGATION_METHOD = 'majority_voting'  # 'majority_voting', 'mean', 'weighted'
    
    # Output directory
    RESULTS_SAVE_DIR = str(RESULTS_DIR)
    SAVE_PLOTS = True
    PLOT_FORMAT = 'png'
    DPI = 300


# ============================================================================
# MODEL PERSISTENCE CONFIGURATION
# Based on final_results_gender_test.ipynb save/load patterns
# ============================================================================

class PersistenceConfig:
    """Model saving, loading, and versioning configuration."""
    
    # Model saving
    SAVE_MODEL_JSON = True           # Save model architecture as JSON
    SAVE_MODEL_WEIGHTS = True        # Save weights separately
    SAVE_MODEL_H5 = True             # Save complete model as H5
    SAVE_MODEL_ONNX = False          # Optional: cross-platform format
    
    # Checkpoint management
    KEEP_BEST_CHECKPOINTS = 3        # Keep top N checkpoints
    CHECKPOINT_FORMAT = 'h5'         # 'h5' or 'pt' for PyTorch
    
    # Model naming
    MODEL_NAME_PREFIX = 'asd_adhd_mlp'
    MODEL_VERSION = 'v1.0'
    INCLUDE_TIMESTAMP = True
    
    # Model directories
    MODEL_SAVE_DIR = str(SAVED_MODELS_DIR)
    CHECKPOINT_DIR = str(CHECKPOINTS_DIR)
    
    # Metadata saving
    SAVE_SCALER = True              # Save feature scaler for preprocessing
    SAVE_LABEL_ENCODER = True       # Save label encoder
    METADATA_FORMAT = 'json'        # 'json' or 'pickle'


# ============================================================================
# LOGGING & DEBUGGING CONFIGURATION
# ============================================================================

class LoggingConfig:
    """Logging and debugging configuration."""
    
    # Logging levels
    LOG_LEVEL = 'INFO'               # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    
    # Log files
    LOG_DIR = str(LOGS_DIR)
    TRAINING_LOG = 'training.log'
    INFERENCE_LOG = 'inference.log'
    ERROR_LOG = 'errors.log'
    
    # Log format
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    
    # Tensorboard logging
    USE_TENSORBOARD = True
    TENSORBOARD_DIR = str(LOGS_DIR / 'tensorboard')
    
    # Debug mode
    DEBUG_MODE = False
    VERBOSE = True
    SAVE_INTERMEDIATE_FEATURES = False


# ============================================================================
# STREAMLIT DASHBOARD CONFIGURATION
# Inspired by MITESHPUTHRANNEU/Speech-Emotion-Analyzer pattern
# ============================================================================

class StreamlitConfig:
    """Streamlit web application configuration."""
    
    # App settings
    PAGE_TITLE = "ASD/ADHD Voice Detection System"
    PAGE_ICON = "🎙️"
    LAYOUT = "wide"
    
    # UI Components
    SHOW_AUDIO_PLAYER = True
    SHOW_WAVEFORM_PLOT = True
    SHOW_SPECTROGRAM = True
    SHOW_FEATURE_HEATMAP = True
    SHOW_CONFIDENCE_BARS = True
    
    # Upload settings
    MAX_UPLOAD_SIZE_MB = 25
    ALLOWED_FORMATS = ['wav', 'mp3', 'ogg', 'flac']
    
    # Real-time recording
    ENABLE_REALTIME_RECORDING = True
    REALTIME_TIMEOUT_SECONDS = 30
    
    # Visualization
    PLOT_QUALITY = 'high'
    PLOT_WIDTH = 800
    PLOT_HEIGHT = 600
    
    # Export options
    ENABLE_EXPORT_RESULTS = True
    EXPORT_FORMATS = ['csv', 'json', 'pdf']


# ============================================================================
# HYPERPARAMETER TUNING CONFIGURATION
# Based on code/hyper_tune.py pattern
# ============================================================================

class HypertuneConfig:
    """Hyperparameter tuning and optimization configuration."""
    
    # Tuning mode
    ENABLE_HYPERTUNING = False
    TUNING_METRIC = 'val_accuracy'
    TUNING_MODE = 'max'              # 'min' or 'max'
    
    # Search method
    SEARCH_METHOD = 'grid'            # 'grid', 'random', 'bayesian'
    N_ITERATIONS = 100
    
    # Parameter ranges
    LEARNING_RATE_RANGE = [1e-5, 1e-3]
    BATCH_SIZE_RANGE = [8, 64]
    DROPOUT_RANGE = [0.1, 0.5]
    HIDDEN_UNITS_RANGE = [64, 256]
    
    # CV settings for tuning
    TUNING_CV_FOLDS = 3
    RANDOM_STATE = 42


# ============================================================================
# GPU/DEVICE CONFIGURATION
# ============================================================================

class DeviceConfig:
    """GPU and device configuration."""
    
    # GPU usage
    USE_GPU = True
    GPU_ID = 0                       # Which GPU to use
    GPU_MEMORY_FRACTION = 0.8        # Use 80% of GPU memory
    ALLOW_GROWTH = True              # Allow GPU memory to grow dynamically
    
    # Mixed precision training (for faster training on newer GPUs)
    USE_MIXED_PRECISION = False
    
    # Parallel processing
    NUM_WORKERS = 4                  # Number of data loading workers
    PIN_MEMORY = True                # Pin memory for data loading


# ============================================================================
# INFERENCE & PREDICTION OUTPUT CONFIGURATION
# ============================================================================

class InferenceConfig:
    """Inference and prediction configuration."""
    
    # Prediction format
    RETURN_CLASS_LABEL = True
    RETURN_CONFIDENCE_SCORES = True
    RETURN_TOP_K_PREDICTIONS = 3     # Return top 3 predictions
    
    # Batch prediction
    BATCH_SIZE = 32
    PREFETCH_BUFFER = 10
    
    # Output interpretation
    PROVIDE_EXPLANATION = True
    HIGHLIGHT_VOCAL_MARKERS = True   # Show which features contributed to prediction
    
    # Caching
    CACHE_PREDICTIONS = False
    CACHE_DIR = str(PROJECT_ROOT / '.cache')


# ============================================================================
# INTEGRATION & AUXILIARY CONFIGURATION
# ============================================================================

class AuxiliaryConfig:
    """Auxiliary and integration configuration."""
    
    # External APIs
    ENABLE_API_SERVER = True
    API_PORT = 8000
    API_HOST = '0.0.0.0'
    
    # Database (if needed)
    USE_DATABASE = False
    DATABASE_TYPE = 'sqlite'         # 'sqlite', 'postgres', 'mysql'
    DATABASE_URL = 'sqlite:///./asd_adhd.db'
    
    # Notifications
    SEND_EMAIL_REPORTS = False
    EMAIL_RECIPIENTS = []
    
    # Experiment tracking
    USE_MLFLOW = False
    MLFLOW_TRACKING_URI = 'http://localhost:5000'
    
    # Version control
    GIT_SYNC = False
    
    # Random seeds for reproducibility
    RANDOM_SEED = 42


# ============================================================================
# CONFIGURATION AGGREGATION CLASS
# ============================================================================

class Config:
    """
    Master configuration class that aggregates all sub-configurations.
    Provides centralized access to all configuration parameters.
    """
    
    def __init__(self):
        """Initialize all configuration sub-classes."""
        self.audio = AudioConfig()
        self.mfcc = MFCCConfig()
        self.spectral = SpectralConfig()
        self.prosodic = ProsodicConfig()
        self.features = FeatureConfig()
        self.dataset = DatasetConfig()
        self.mlp = MLPConfig()
        self.training = TrainingConfig()
        self.realtime = RealtimeConfig()
        self.evaluation = EvaluationConfig()
        self.persistence = PersistenceConfig()
        self.logging = LoggingConfig()
        self.streamlit = StreamlitConfig()
        self.hypertuning = HypertuneConfig()
        self.device = DeviceConfig()
        self.inference = InferenceConfig()
        self.auxiliary = AuxiliaryConfig()
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary format."""
        config_dict = {}
        for attr_name in dir(self):
            if not attr_name.startswith('_'):
                attr = getattr(self, attr_name)
                if isinstance(attr, object) and hasattr(attr, '__dict__'):
                    config_dict[attr_name] = vars(attr)
        return config_dict
    
    def to_yaml(self, filepath: str) -> None:
        """Save configuration to YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @staticmethod
    def from_yaml(filepath: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = Config()
        # Update config with loaded values
        for section_name, section_values in config_dict.items():
            if hasattr(config, section_name):
                section = getattr(config, section_name)
                for key, value in section_values.items():
                    if hasattr(section, key.upper()):
                        setattr(section, key.upper(), value)
        return config
    
    def print_config(self) -> None:
        """Pretty print configuration."""
        print("\n" + "="*70)
        print("ASD/ADHD DETECTION SYSTEM - CONFIGURATION")
        print("="*70)
        for attr_name in sorted(dir(self)):
            if not attr_name.startswith('_'):
                attr = getattr(self, attr_name)
                if isinstance(attr, object) and hasattr(attr, '__dict__'):
                    print(f"\n[{attr_name.upper()}]")
                    for key, value in sorted(vars(attr).items()):
                        print(f"  {key}: {value}")
        print("\n" + "="*70 + "\n")


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

# Create a singleton instance for easy access throughout the project
config = Config()

# Optional: Uncomment to save default config to YAML
# config.to_yaml(str(PROJECT_ROOT / 'config' / 'default_config.yaml'))


if __name__ == "__main__":
    """Test configuration loading and display."""
    config.print_config()
    
    # Example: Access specific configuration
    print("\n" + "="*70)
    print("EXAMPLE ACCESS PATTERNS:")
    print("="*70)
    print(f"Sample Rate: {config.audio.SAMPLE_RATE} Hz")
    print(f"Recording Duration: {config.audio.DURATION} seconds")
    print(f"MLP Input Dimension: {config.mlp.INPUT_DIM}")
    print(f"MLP Hidden Layers: {config.mlp.HIDDEN_LAYERS}")
    print(f"Training Epochs: {config.training.EPOCHS}")
    print(f"Learning Rate: {config.training.LEARNING_RATE}")
    print(f"Dataset Classes: {config.dataset.CLASSES}")
    print(f"K-Folds: {config.dataset.K_FOLDS}")
    print(f"Project Root: {PROJECT_ROOT}")
    print("="*70 + "\n")
