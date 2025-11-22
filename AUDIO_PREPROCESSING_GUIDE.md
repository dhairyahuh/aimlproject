# Audio Preprocessing Utility - Complete Guide

## Overview

`utils/audio_preprocessing.py` provides production-ready audio preprocessing following patterns from:
- **x4nth055/emotion-recognition-using-speech** - Class structure and loading approach
- **pyAudioAnalysis** - STFT-based analysis patterns  
- **sukesh167** - Energy-based silence detection

## Class Hierarchy

```
AudioLoader
├── Purpose: Load audio files with format detection
├── Methods:
│   ├── load() - Main loading method
│   └── is_supported_format() - Format validation
└── Supports: .wav, .mp3, .ogg, .flac, .m4a

SilenceTrimmer
├── Purpose: Remove silence using energy detection
├── Methods:
│   ├── compute_energy() - STFT-based energy
│   ├── find_speech_frames() - Detect speech regions
│   ├── trim() - Custom trimming
│   └── trim_librosa() - Fast librosa trimming
└── Algorithms: STFT energy, percentile thresholding

AmplitudeNormalizer
├── Purpose: Standardize audio amplitude
├── Methods:
│   ├── normalize_peak() - Peak normalization
│   ├── normalize_rms() - RMS energy normalization
│   ├── normalize_z_score() - Statistical standardization
│   └── normalize_minmax() - Min-max scaling
└── Use Cases: Consistent volume levels, model input preparation

NoiseReducer
├── Purpose: Reduce background noise
├── Methods:
│   ├── spectral_subtraction() - Frequency domain subtraction
│   └── wiener_filter() - Frequency domain filtering
└── Assumption: First ~1 second contains noise profile

AudioPreprocessor
├── Purpose: Complete preprocessing pipeline
├── Methods:
│   ├── process_file() - Single file processing
│   ├── process_batch() - Multiple files
│   ├── process_directory() - Directory processing
│   └── __repr__() - Configuration display
└── Pipeline: Load → Trim → Denoise → Normalize
```

## Quick Start Examples

### 1. Basic File Loading

```python
from utils.audio_preprocessing import AudioLoader

loader = AudioLoader(sr=16000, mono=True)
audio, sr = loader.load('speech.wav', verbose=True)

# Output:
# ✓ Loaded: speech.wav
#   Shape: (48000,)
#   Sample rate: 16000 Hz
#   Duration: 3.00s
#   Range: [-0.5234, 0.4892]
```

### 2. Process Single File with Full Pipeline

```python
from utils.audio_preprocessing import AudioPreprocessor

preprocessor = AudioPreprocessor(
    sr=16000,
    trim_silence=True,
    normalize='peak',
    apply_noise_reduction=True,
    noise_reduction_method='spectral_subtraction',
    verbose=True
)

audio, sr = preprocessor.process_file('noisy_speech.wav')

# Output:
# [1/4] Loading: noisy_speech.wav
# ✓ Loaded: noisy_speech.wav
#   Shape: (48000,)
#   Sample rate: 16000 Hz
#   Duration: 3.00s
#   Range: [-0.2934, 0.3109]
# [2/4] Trimming silence... (original: 48000 samples)
#       After trim: 44320 samples (2.77s)
# [3/4] Applying noise reduction (spectral_subtraction)...
# [4/4] Normalizing (peak method)...
# ✓ Processing complete
#   Final shape: (44320,)
#   Final range: [-1.0000, 1.0000]
#   Final RMS: 0.3421
```

### 3. Silence Trimming Comparison

```python
from utils.audio_preprocessing import SilenceTrimmer
import numpy as np

trimmer = SilenceTrimmer()

# Method 1: Energy-based (more control)
trimmed_energy = trimmer.trim(
    audio, sr=16000,
    threshold_percentile=30,  # Lower = trim more
    margin_frames=5
)

# Method 2: librosa (faster, standard)
trimmed_librosa = trimmer.trim_librosa(
    audio,
    top_db=20  # 20dB threshold (standard for speech)
)

print(f"Original:  {len(audio)} samples")
print(f"Energy:    {len(trimmed_energy)} samples (margin=5 frames)")
print(f"Librosa:   {len(trimmed_librosa)} samples (20dB threshold)")
```

### 4. Noise Reduction Methods

```python
from utils.audio_preprocessing import NoiseReducer

reducer = NoiseReducer(fft_size=2048, hop_length=512)

# Method 1: Spectral Subtraction
# Assumes first 1 second is noise
clean_audio = reducer.spectral_subtraction(
    noisy_audio,
    sr=16000,
    noise_duration=1.0,
    alpha=2.0,              # Over-subtraction factor
    floor_factor=0.002      # Floor to prevent artifacts
)

# Method 2: Wiener Filter
# Frequency domain filtering using noise variance
clean_audio = reducer.wiener_filter(
    noisy_audio,
    sr=16000,
    noise_duration=1.0,
    frame_length=2048
)
```

### 5. Normalization Methods

```python
from utils.audio_preprocessing import AmplitudeNormalizer

normalizer = AmplitudeNormalizer()
test_audio = np.random.randn(16000)

# Method 1: Peak normalization (loudest sample = 1.0)
audio_peak = normalizer.normalize_peak(test_audio, target_level=1.0)
# Result: max(|audio_peak|) == 1.0

# Method 2: RMS normalization (energy standardization)
audio_rms = normalizer.normalize_rms(test_audio, target_rms=0.1)
# Result: RMS energy = 0.1

# Method 3: Z-score normalization (statistical)
audio_zscore = normalizer.normalize_z_score(test_audio)
# Result: mean=0, std=1

# Method 4: Min-max scaling
audio_minmax = normalizer.normalize_minmax(test_audio, min_val=-1.0, max_val=1.0)
# Result: min=−1, max=1
```

### 6. Batch Processing

```python
from utils.audio_preprocessing import AudioPreprocessor
from pathlib import Path

preprocessor = AudioPreprocessor(
    sr=16000,
    trim_silence=True,
    normalize='peak',
    apply_noise_reduction=False
)

# Process multiple files
files = ['audio1.wav', 'audio2.wav', 'audio3.wav']
results = preprocessor.process_batch(files, show_progress=True)

for i, (audio, sr) in enumerate(results):
    if audio is not None:
        print(f"File {i+1}: {len(audio)} samples, {len(audio)/sr:.2f}s")
    else:
        print(f"File {i+1}: Error loading")

# Output:
# [1/3] Processing audio1.wav...
# File 1: 44320 samples, 2.77s
# [2/3] Processing audio2.wav...
# File 2: 41600 samples, 2.60s
# [3/3] Processing audio3.wav...
# File 3: 39680 samples, 2.48s
```

### 7. Directory Processing

```python
preprocessor = AudioPreprocessor(sr=16000, trim_silence=True)

# Process all .wav files in directory
results = preprocessor.process_directory(
    'audio_data/',
    pattern='*.wav',
    recursive=False
)

for filepath, audio, sr in results:
    print(f"{Path(filepath).name}: {len(audio)/sr:.2f}s")
```

### 8. Convenience Functions

```python
from utils.audio_preprocessing import load_audio, preprocess_audio

# Quick load
audio, sr = load_audio('speech.wav', sr=16000)

# Quick preprocess
cleaned_audio = preprocess_audio(
    audio,
    sr=16000,
    trim=True,
    normalize='peak',
    reduce_noise=True
)
```

## Detailed API Reference

### AudioLoader

```python
class AudioLoader:
    def __init__(self, sr: int = 16000, mono: bool = True, 
                 offset: float = 0.0, duration: Optional[float] = None)
    
    def load(filepath: str, verbose: bool = False) -> Tuple[np.ndarray, int]
        """Load audio file with format detection."""
    
    def is_supported_format(filepath: str) -> bool
        """Check if format is supported."""
```

**Parameters:**
- `sr`: Target sampling rate (Hz) - 16000 for speech
- `mono`: Convert to mono (True for speech processing)
- `offset`: Start reading after this time (seconds)
- `duration`: Only load this duration (seconds)

**Supported Formats:** .wav, .mp3, .ogg, .flac, .m4a

**Raises:**
- `FileNotFoundError`: File doesn't exist
- `ValueError`: Unsupported format
- `RuntimeError`: Loading error

---

### SilenceTrimmer

```python
class SilenceTrimmer:
    def __init__(self, fft_window: int = 2048, hop_length: int = 512)
    
    def trim(audio: np.ndarray, sr: int, 
             threshold_percentile: float = 30,
             margin_frames: int = 5) -> np.ndarray
        """Trim silence using energy detection."""
    
    def trim_librosa(audio: np.ndarray, top_db: float = 20) -> np.ndarray
        """Trim silence using librosa (faster)."""
```

**Energy-based Trimming:**
- `threshold_percentile`: Lower = trim more (20-40 typical)
- `margin_frames`: Frames to keep at edges for context

**Librosa Trimming:**
- `top_db`: Threshold in dB (20dB standard for speech)

**Speed:** librosa ~2ms, energy-based ~5ms per file

---

### AmplitudeNormalizer

```python
class AmplitudeNormalizer:
    @staticmethod
    def normalize_peak(audio: np.ndarray, target_level: float = 1.0) -> np.ndarray
        """Normalize by peak value."""
    
    @staticmethod
    def normalize_rms(audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray
        """Normalize by RMS energy."""
    
    @staticmethod
    def normalize_z_score(audio: np.ndarray) -> np.ndarray
        """Standardize to mean=0, std=1."""
    
    @staticmethod
    def normalize_minmax(audio: np.ndarray, min_val: float = -1.0,
                        max_val: float = 1.0) -> np.ndarray
        """Scale to [min_val, max_val] range."""
```

**Method Comparison:**

| Method | Use Case | Output Range |
|--------|----------|--------------|
| Peak | Ensure no clipping | [-1, 1] |
| RMS | Consistent energy | ~[-0.3, 0.3] |
| Z-score | Statistical standardization | ~[-3, 3] |
| MinMax | Bounded scaling | [min, max] |

---

### NoiseReducer

```python
class NoiseReducer:
    def __init__(self, fft_size: int = 2048, hop_length: int = 512)
    
    def spectral_subtraction(audio: np.ndarray, sr: int,
                             noise_duration: float = 1.0,
                             alpha: float = 2.0,
                             floor_factor: float = 0.002) -> np.ndarray
        """Reduce noise using spectral subtraction."""
    
    def wiener_filter(audio: np.ndarray, sr: int,
                      noise_duration: float = 1.0,
                      frame_length: int = 2048) -> np.ndarray
        """Reduce noise using Wiener filter."""
```

**Spectral Subtraction:**
- `noise_duration`: How long is noise profile at start (seconds)
- `alpha`: Over-subtraction factor (1.0-3.0)
  - Lower (1.0) = less aggressive
  - Higher (3.0) = more aggressive, more artifacts
- `floor_factor`: Prevent over-subtraction artifacts (0.001-0.01)

**Wiener Filter:**
- More sophisticated frequency domain filtering
- Better for varying noise levels
- Slower but higher quality

---

### AudioPreprocessor

```python
class AudioPreprocessor:
    def __init__(self, sr: int = 16000, trim_silence: bool = True,
                 normalize: str = 'peak', apply_noise_reduction: bool = False,
                 noise_reduction_method: str = 'spectral_subtraction',
                 verbose: bool = False)
    
    def process_file(filepath: str) -> Tuple[np.ndarray, int]
        """Load and preprocess single file."""
    
    def process_batch(filepaths: List[str], show_progress: bool = True) -> List[Tuple]
        """Process multiple files."""
    
    def process_directory(directory: str, pattern: str = '*.wav',
                          recursive: bool = False) -> List[Tuple]
        """Process all files in directory."""
```

**Pipeline Order:**
1. Load
2. Trim silence (if enabled)
3. Noise reduction (if enabled)
4. Normalization (if enabled)

**Return Values:**
- `process_file()`: (audio, sr)
- `process_batch()`: [(audio, sr), ...]
- `process_directory()`: [(filepath, audio, sr), ...]

---

## Integration Example

```python
from utils.audio_preprocessing import AudioPreprocessor
import numpy as np

# Initialize preprocessor
preprocessor = AudioPreprocessor(
    sr=16000,
    trim_silence=True,
    normalize='peak',
    apply_noise_reduction=True,
    noise_reduction_method='spectral_subtraction',
    verbose=True
)

# Process audio files
audio_files = ['speech1.wav', 'speech2.wav', 'speech3.wav']
processed_audio_list = []

for filepath, audio, sr in preprocessor.process_directory('data/audio/', pattern='*.wav'):
    # Use for feature extraction
    from src.feature_extraction.feature_aggregator import FeatureAggregator
    
    aggregator = FeatureAggregator(sr=sr)
    features = aggregator.extract_all_features(audio)  # 106-d features
    
    processed_audio_list.append((filepath, audio, features))

print(f"Processed {len(processed_audio_list)} files")
```

## Performance Notes

- **Loading:** ~10-20ms per file (format dependent)
- **Silence Trimming:** ~2-5ms per file
- **Noise Reduction:**
  - Spectral Subtraction: ~50-100ms
  - Wiener Filter: ~100-200ms
- **Normalization:** < 1ms per file

## Common Configurations

### Real-time Speech Processing
```python
preprocessor = AudioPreprocessor(
    sr=16000,
    trim_silence=True,
    normalize='rms',
    apply_noise_reduction=False  # Speed over quality
)
```

### High-Quality Speech Analysis
```python
preprocessor = AudioPreprocessor(
    sr=16000,
    trim_silence=True,
    normalize='peak',
    apply_noise_reduction=True,
    noise_reduction_method='wiener_filter'
)
```

### Batch Noise Cleaning
```python
preprocessor = AudioPreprocessor(
    sr=16000,
    trim_silence=True,
    normalize='z_score',
    apply_noise_reduction=True,
    noise_reduction_method='spectral_subtraction'
)
```

## Testing

```bash
python utils/audio_preprocessing.py
```

This runs built-in examples showing:
- Synthetic audio creation with noise
- Full preprocessing pipeline
- Normalization method comparison

## Dependencies

- numpy
- librosa
- scipy

All included in `requirements.txt`
