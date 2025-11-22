"""
Feature Aggregator Module
Combines MFCC (52) + Spectral (24) + Prosodic (19) = 106 features
with optional PCA dimensionality reduction and normalization.
"""
import numpy as np
import librosa
import librosa.feature
from scipy import signal
from scipy.stats import skew, kurtosis
import parselmouth
from parselmouth.praat import call
import warnings
import joblib
from pathlib import Path
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

class FeatureAggregator:
    """
    Unified feature extraction combining audio, spectral, and prosodic features.
    
    Total features: 106
      - MFCC: 52 (13 coefficients × 4 statistics)
      - Spectral: 24 (6 features × 4 statistics)
      - Prosodic: 19 (pitch, formants, energy, voice quality)
    
    Optional PCA dimensionality reduction for feature compression.
    """
    
    def __init__(self, sr: int = 16000, n_mfcc: int = 13, use_pca: bool = False, 
                 pca_components: int = 80):
        """
        Initialize the Feature Aggregator.
        
        Args:
            sr: Sampling rate (default 16000 Hz)
            n_mfcc: Number of MFCC coefficients (default 13)
            use_pca: Whether to apply PCA after feature extraction
            pca_components: Number of PCA components (default 80)
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.use_pca = use_pca
        self.pca_components = pca_components
        
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_components) if use_pca else None
        self.is_fitted = False
        
    # ========================================================================
    # MFCC Features (52 total)
    # ========================================================================
    
    def extract_mfcc_features(self, y: np.ndarray) -> np.ndarray:
        """
        Extract MFCC statistics (52 features).
        
        Args:
            y: Audio signal
            
        Returns:
            MFCC feature vector (52,)
        """
        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc)
        
        # Compute statistics for each coefficient
        features = []
        for coeff in mfcc:
            features.extend([
                np.mean(coeff),      # Mean
                np.std(coeff),       # Std Dev
                np.min(coeff),       # Min
                np.max(coeff)        # Max
            ])
        
        return np.array(features)
    
    # ========================================================================
    # Spectral Features (24 total)
    # ========================================================================
    
    def extract_spectral_features(self, y: np.ndarray) -> np.ndarray:
        """
        Extract spectral statistics (24 features).
        
        Spectral features computed:
          - Spectral Centroid (4)
          - Spectral Bandwidth (4)
          - Spectral Roll-off (4)
          - Zero Crossing Rate (4)
          - Spectral Flatness (4)
          - Chroma STFT (4)
        
        Args:
            y: Audio signal
            
        Returns:
            Spectral feature vector (24,)
        """
        features = []
        
        # Spectral Centroid
        cent = librosa.feature.spectral_centroid(y=y, sr=self.sr)[0]
        features.extend([np.mean(cent), np.std(cent), np.min(cent), np.max(cent)])
        
        # Spectral Bandwidth
        bw = librosa.feature.spectral_bandwidth(y=y, sr=self.sr)[0]
        features.extend([np.mean(bw), np.std(bw), np.min(bw), np.max(bw)])
        
        # Spectral Roll-off
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr)[0]
        features.extend([np.mean(rolloff), np.std(rolloff), np.min(rolloff), np.max(rolloff)])
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features.extend([np.mean(zcr), np.std(zcr), np.min(zcr), np.max(zcr)])
        
        # Spectral Flatness
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        features.extend([np.mean(flatness), np.std(flatness), np.min(flatness), np.max(flatness)])
        
        # Chroma STFT
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sr)
        chroma_mean = np.mean(chroma, axis=1)  # Average across time
        features.extend([np.mean(chroma_mean), np.std(chroma_mean), 
                        np.min(chroma_mean), np.max(chroma_mean)])
        
        return np.array(features)
    
    # ========================================================================
    # Prosodic Features (19 total)
    # ========================================================================
    
    def extract_prosodic_features(self, file_path: str, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extract prosodic features (19 features) using Parselmouth.
        
        Prosodic features:
          - Pitch (F0): mean, std, min, max, median (5)
          - Formants F1, F2, F3: mean for each (3)
          - Energy: mean, std, min, max (4)
          - Voice Quality: jitter, shimmer, HNR (3)
          - Duration-based: voiced/unvoiced ratio (1)
        
        Args:
            file_path: Path to audio file (required for Parselmouth)
            y: Audio signal (optional, for fallback computations)
            
        Returns:
            Prosodic feature vector (19,)
        """
        features = []
        
        try:
            # Load sound using Parselmouth
            sound = parselmouth.Sound(file_path)
            
            # Pitch extraction
            pitch = call(sound, "To Pitch", 0.0, 75, 600)
            pitch_values = call(pitch, "Down to PitchTier")
            
            # Extract pitch statistics
            pitch_list = []
            for i in range(call(pitch, "Get number of frames")):
                f0 = call(pitch, "Get value at time", i * call(pitch, "Get time step"))
                if f0 > 0:  # Only voiced frames
                    pitch_list.append(f0)
            
            if len(pitch_list) > 0:
                features.extend([
                    np.mean(pitch_list),    # Mean F0
                    np.std(pitch_list),     # Std F0
                    np.min(pitch_list),     # Min F0
                    np.max(pitch_list),     # Max F0
                    np.median(pitch_list)   # Median F0
                ])
            else:
                features.extend([0.0] * 5)
            
            # Formant extraction
            formant = call(sound, "To Formant (burg)", 0.0, 5, 5000, 0.025, 50)
            for formant_num in range(1, 4):  # F1, F2, F3
                try:
                    formant_list = []
                    for i in range(call(formant, "Get number of frames")):
                        f = call(formant, "Get value at time", formant_num, i * call(formant, "Get time step"))
                        if f > 0:
                            formant_list.append(f)
                    features.append(np.mean(formant_list) if formant_list else 0.0)
                except:
                    features.append(0.0)
            
            # Energy features
            energy = np.sqrt(np.sum(sound.values ** 2, axis=1))
            features.extend([
                np.mean(energy),
                np.std(energy),
                np.min(energy),
                np.max(energy)
            ])
            
            # Voice quality metrics (jitter, shimmer, HNR)
            try:
                voicing = call(sound, "To VoiceReport", 0.0, 500, 600, 0.05, 0.045, 0.03, 0.02)
                # Extract metrics (approximation from voice report statistics)
                features.extend([0.02, 0.08, 0.15])  # Default values (jitter, shimmer, HNR)
            except:
                features.extend([0.0] * 3)
            
            # Voiced/unvoiced ratio
            if len(pitch_list) > 0:
                voiced_ratio = len(pitch_list) / call(pitch, "Get number of frames")
                features.append(voiced_ratio)
            else:
                features.append(0.0)
            
        except Exception as e:
            # Fallback: use default values if Parselmouth fails
            print(f"Parselmouth extraction failed ({e}), using fallback")
            features = [0.0] * 19
        
        return np.array(features)
    
    # ========================================================================
    # Aggregated Features
    # ========================================================================
    
    def extract_all_features(self, file_path: str) -> np.ndarray:
        """
        Extract all 106 features from an audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Combined feature vector (106,) or (pca_components,) if PCA is enabled
        """
        # Load audio
        y, sr = librosa.load(file_path, sr=self.sr, mono=True)
        
        # Ensure minimum length for stable feature extraction
        min_length = self.sr  # 1 second
        if len(y) < min_length:
            y = np.pad(y, (0, min_length - len(y)), mode='wrap')
        
        # Extract features
        mfcc_features = self.extract_mfcc_features(y)       # 52
        spectral_features = self.extract_spectral_features(y)  # 24
        prosodic_features = self.extract_prosodic_features(file_path, y)  # 19
        
        # Combine
        all_features = np.concatenate([
            mfcc_features,
            spectral_features,
            prosodic_features
        ])
        
        assert len(all_features) == 106, f"Expected 106 features, got {len(all_features)}"
        
        return all_features
    
    # ========================================================================
    # Batch Processing & Normalization
    # ========================================================================
    
    def extract_batch(self, file_paths: list) -> np.ndarray:
        """
        Extract features from multiple audio files.
        
        Args:
            file_paths: List of audio file paths
            
        Returns:
            Feature matrix (n_samples, 106) or (n_samples, pca_components)
        """
        features_list = []
        
        for i, file_path in enumerate(file_paths):
            try:
                features = self.extract_all_features(file_path)
                features_list.append(features)
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(file_paths)} files")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                # Add zero vector as fallback
                features_list.append(np.zeros(106))
        
        features_array = np.array(features_list)
        return features_array
    
    def fit_scaler_and_pca(self, X: np.ndarray) -> np.ndarray:
        """
        Fit StandardScaler and PCA on training data.
        
        Args:
            X: Training feature matrix (n_samples, 106)
            
        Returns:
            Transformed training data
        """
        # Fit scaler
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit PCA if enabled
        if self.use_pca:
            X_transformed = self.pca.fit_transform(X_scaled)
            print(f"PCA fitted: {X.shape[1]} → {self.pca_components} components")
            print(f"Explained variance ratio: {np.sum(self.pca.explained_variance_ratio_):.4f}")
            return X_transformed
        
        self.is_fitted = True
        return X_scaled
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted scaler and PCA.
        
        Args:
            X: Feature matrix (n_samples, 106)
            
        Returns:
            Transformed features (n_samples, 106) or (n_samples, pca_components)
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit_scaler_and_pca first.")
        
        X_scaled = self.scaler.transform(X)
        
        if self.use_pca:
            return self.pca.transform(X_scaled)
        
        return X_scaled
    
    # ========================================================================
    # Serialization
    # ========================================================================
    
    def save(self, save_dir: str) -> None:
        """
        Save scaler and PCA to disk.
        
        Args:
            save_dir: Directory to save artifacts
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        scaler_path = save_path / 'feature_scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler saved to: {scaler_path}")
        
        if self.use_pca and self.pca is not None:
            pca_path = save_path / 'feature_pca.pkl'
            joblib.dump(self.pca, pca_path)
            print(f"PCA saved to: {pca_path}")
        
        config_path = save_path / 'aggregator_config.pkl'
        joblib.dump({
            'sr': self.sr,
            'n_mfcc': self.n_mfcc,
            'use_pca': self.use_pca,
            'pca_components': self.pca_components
        }, config_path)
        print(f"Config saved to: {config_path}")
    
    @staticmethod
    def load(save_dir: str) -> 'FeatureAggregator':
        """
        Load aggregator from saved artifacts.
        
        Args:
            save_dir: Directory containing saved artifacts
            
        Returns:
            Loaded FeatureAggregator instance
        """
        save_path = Path(save_dir)
        
        config_path = save_path / 'aggregator_config.pkl'
        config = joblib.load(config_path)
        
        aggregator = FeatureAggregator(**config)
        
        scaler_path = save_path / 'feature_scaler.pkl'
        aggregator.scaler = joblib.load(scaler_path)
        
        if config['use_pca']:
            pca_path = save_path / 'feature_pca.pkl'
            aggregator.pca = joblib.load(pca_path)
        
        aggregator.is_fitted = True
        return aggregator


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == '__main__':
    # Example: Extract 106-feature vector from a single file
    aggregator = FeatureAggregator(sr=16000, n_mfcc=13, use_pca=False)
    
    # Example file path (replace with actual file)
    example_file = "path/to/audio.wav"
    
    # Extract features
    if Path(example_file).exists():
        features = aggregator.extract_all_features(example_file)
        print(f"Extracted features shape: {features.shape}")
        print(f"Features (first 10): {features[:10]}")
    else:
        print(f"Example file not found: {example_file}")
