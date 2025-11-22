"""
MFCC Feature Extractor Module
Extracts Mel-Frequency Cepstral Coefficients and derivatives

Based on: python_speech_features/ and ser_preprocessing.py patterns
Author: ASD/ADHD Detection Team
"""

import numpy as np
import librosa
from typing import Tuple, List, Dict
import warnings

warnings.filterwarnings('ignore')


class MFCCExtractor:
    """Extract MFCC features from audio signals."""
    
    def __init__(self, config):
        """
        Initialize MFCC extractor with configuration.
        
        Args:
            config: Configuration object with audio and MFCC parameters
        """
        self.config = config
        self.sr = config.audio.SAMPLE_RATE
        self.n_mfcc = config.mfcc.N_MFCC
        self.n_mel = config.mfcc.N_MEL
        self.fmin = config.mfcc.FMIN
        self.fmax = config.mfcc.FMAX
        self.compute_delta = config.mfcc.COMPUTE_DELTA
        self.compute_delta_delta = config.mfcc.COMPUTE_DELTA_DELTA
        self.statistics = config.mfcc.STATISTICS
        
    def extract_mfcc(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract MFCC coefficients.
        
        Args:
            audio: Audio time series
            sr: Sampling rate
            
        Returns:
            MFCC features shape (n_mfcc, time_steps)
        """
        try:
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=self.n_mfcc,
                n_mels=self.n_mel,
                fmin=self.fmin,
                fmax=self.fmax
            )
            return mfcc
        except Exception as e:
            print(f"Error extracting MFCC: {e}")
            return np.zeros((self.n_mfcc, 1))
    
    def extract_delta(self, mfcc: np.ndarray, order: int = 1) -> np.ndarray:
        """
        Extract delta (velocity) features.
        
        Args:
            mfcc: MFCC features
            order: Derivative order (1=delta, 2=delta-delta)
            
        Returns:
            Delta features same shape as MFCC
        """
        try:
            delta = librosa.feature.delta(mfcc, order=order)
            return delta
        except Exception as e:
            print(f"Error extracting delta: {e}")
            return np.zeros_like(mfcc)
    
    def compute_statistics(self, features: np.ndarray) -> np.ndarray:
        """
        Compute statistics over time dimension.
        
        Args:
            features: Time series features (n_features, time_steps)
            
        Returns:
            Statistics vector (n_features * n_statistics,)
        """
        stats = []
        
        for stat_name in self.statistics:
            if stat_name == 'mean':
                stats.append(np.mean(features, axis=1))
            elif stat_name == 'std':
                stats.append(np.std(features, axis=1))
            elif stat_name == 'min':
                stats.append(np.min(features, axis=1))
            elif stat_name == 'max':
                stats.append(np.max(features, axis=1))
            elif stat_name == 'median':
                stats.append(np.median(features, axis=1))
            elif stat_name == 'q25':
                stats.append(np.percentile(features, 25, axis=1))
            elif stat_name == 'q75':
                stats.append(np.percentile(features, 75, axis=1))
        
        # Stack statistics: (n_statistics, n_features) -> (n_features * n_statistics,)
        stats_vector = np.concatenate(stats)
        return stats_vector
    
    def extract(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract complete MFCC features including delta and delta-delta.
        
        Args:
            audio: Audio time series
            sr: Sampling rate
            
        Returns:
            Feature vector containing:
            - 13 base MFCC coefficients with statistics
            - 13 delta features with statistics
            - 13 delta-delta features with statistics
            Total: 13 * 7 statistics types = 91 features
            Simplified to 52 features (mean, std, min, max only)
        """
        all_features = []
        
        # Extract base MFCC
        mfcc = self.extract_mfcc(audio, sr)
        mfcc_stats = self.compute_statistics(mfcc)
        all_features.append(mfcc_stats)
        
        # Extract delta (first derivative)
        if self.compute_delta:
            delta = self.extract_delta(mfcc, order=1)
            delta_stats = self.compute_statistics(delta)
            all_features.append(delta_stats)
        
        # Extract delta-delta (second derivative)
        if self.compute_delta_delta:
            delta_delta = self.extract_delta(mfcc, order=2)
            delta_delta_stats = self.compute_statistics(delta_delta)
            all_features.append(delta_delta_stats)
        
        # Concatenate all features
        features_vector = np.concatenate(all_features)
        
        return features_vector
    
    def get_feature_names(self) -> List[str]:
        """Get names of all extracted features for interpretability."""
        names = []
        feature_types = ['MFCC']
        if self.compute_delta:
            feature_types.append('Delta')
        if self.compute_delta_delta:
            feature_types.append('Delta-Delta')
        
        for ftype in feature_types:
            for coeff in range(self.n_mfcc):
                for stat in self.statistics:
                    names.append(f"{ftype}_C{coeff}_{stat}")
        
        return names


# Example usage
if __name__ == "__main__":
    from config.config import config
    import librosa
    
    # Load sample audio
    audio_path = "sample_audio.wav"
    try:
        audio, sr = librosa.load(audio_path, sr=config.audio.SAMPLE_RATE)
        
        # Extract features
        extractor = MFCCExtractor(config)
        features = extractor.extract(audio, sr)
        
        print(f"Audio shape: {audio.shape}")
        print(f"Features shape: {features.shape}")
        print(f"Expected shape: ({52 * len(extractor.statistics)},)")
        print(f"Feature names: {len(extractor.get_feature_names())} features")
    except FileNotFoundError:
        print("Sample audio file not found. Create one first.")
