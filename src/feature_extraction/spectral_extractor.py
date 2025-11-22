"""
Spectral Features Extractor Module
Extracts spectral features (centroid, rolloff, bandwidth, chroma, energy, ZCR)

Based on: pyAudioAnalysis/ patterns
Author: ASD/ADHD Detection Team
"""

import numpy as np
import librosa
from typing import List
import warnings

warnings.filterwarnings('ignore')


class SpectralExtractor:
    """Extract spectral features from audio signals."""
    
    def __init__(self, config):
        """
        Initialize spectral extractor with configuration.
        
        Args:
            config: Configuration object with audio parameters
        """
        self.config = config
        self.sr = config.audio.SAMPLE_RATE
        self.n_fft = config.audio.N_FFT
        self.hop_length = config.audio.HOP_LENGTH
        self.statistics = config.spectral.STATISTICS
        
    def extract_spectral_centroid(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract spectral centroid (center of mass of spectrum).
        
        Args:
            audio: Audio time series
            
        Returns:
            Spectral centroid values (time_steps,)
        """
        try:
            centroid = librosa.feature.spectral_centroid(
                y=audio,
                sr=self.sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )[0]
            return centroid
        except Exception as e:
            print(f"Error extracting spectral centroid: {e}")
            return np.array([0])
    
    def extract_spectral_rolloff(self, audio: np.ndarray, roll_percent: float = 0.95) -> np.ndarray:
        """
        Extract spectral rolloff (frequency below which specified % of energy is contained).
        
        Args:
            audio: Audio time series
            roll_percent: Percentage of energy (default 0.95 = 95%)
            
        Returns:
            Spectral rolloff values (time_steps,)
        """
        try:
            rolloff = librosa.feature.spectral_rolloff(
                y=audio,
                sr=self.sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                roll_percent=roll_percent
            )[0]
            return rolloff
        except Exception as e:
            print(f"Error extracting spectral rolloff: {e}")
            return np.array([0])
    
    def extract_spectral_bandwidth(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract spectral bandwidth (width of spectrum).
        
        Args:
            audio: Audio time series
            
        Returns:
            Spectral bandwidth values (time_steps,)
        """
        try:
            bandwidth = librosa.feature.spectral_bandwidth(
                y=audio,
                sr=self.sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )[0]
            return bandwidth
        except Exception as e:
            print(f"Error extracting spectral bandwidth: {e}")
            return np.array([0])
    
    def extract_zcr(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract Zero-Crossing Rate (number of times signal crosses zero).
        
        Args:
            audio: Audio time series
            
        Returns:
            Zero-crossing rate (time_steps,)
        """
        try:
            zcr = librosa.feature.zero_crossing_rate(
                audio,
                frame_length=self.n_fft,
                hop_length=self.hop_length
            )[0]
            return zcr
        except Exception as e:
            print(f"Error extracting ZCR: {e}")
            return np.array([0])
    
    def extract_rms_energy(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract RMS (Root Mean Square) Energy.
        
        Args:
            audio: Audio time series
            
        Returns:
            RMS energy (time_steps,)
        """
        try:
            rms = librosa.feature.rms(
                y=audio,
                frame_length=self.n_fft,
                hop_length=self.hop_length
            )[0]
            return rms
        except Exception as e:
            print(f"Error extracting RMS energy: {e}")
            return np.array([0])
    
    def extract_log_energy(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract log energy (log of power).
        
        Args:
            audio: Audio time series
            
        Returns:
            Log energy (time_steps,)
        """
        try:
            S = librosa.magphase(librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length))[0]
            power = np.abs(S) ** 2
            log_energy = np.log1p(np.mean(power, axis=0))
            return log_energy
        except Exception as e:
            print(f"Error extracting log energy: {e}")
            return np.array([0])
    
    def extract_chroma(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract Chroma features (12 pitch classes).
        
        Args:
            audio: Audio time series
            
        Returns:
            Chroma features (n_chroma=12, time_steps)
        """
        try:
            chroma = librosa.feature.chroma_stft(
                y=audio,
                sr=self.sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_chroma=12
            )
            return chroma
        except Exception as e:
            print(f"Error extracting chroma: {e}")
            return np.zeros((12, 1))
    
    def compute_statistics(self, features: np.ndarray) -> np.ndarray:
        """
        Compute statistics over time dimension.
        
        Args:
            features: Time series or single feature (1, time_steps) or (n_features, time_steps)
            
        Returns:
            Statistics vector
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
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
        
        return np.concatenate(stats)
    
    def extract(self, audio: np.ndarray, sr: int = None) -> np.ndarray:
        """
        Extract complete spectral features.
        
        Args:
            audio: Audio time series
            sr: Sampling rate (uses config if None)
            
        Returns:
            Feature vector (24,) for simplified version:
            - Spectral centroid stats (4)
            - Spectral rolloff stats (4)
            - Spectral bandwidth stats (4)
            - Zero-crossing rate stats (4)
            - RMS energy stats (4)
            - Log energy stats (4)
            Total: 24 features (with 4 statistics: mean, std, min, max)
        """
        if sr is None:
            sr = self.sr
        
        all_features = []
        
        # Extract individual features
        centroid = self.extract_spectral_centroid(audio)
        centroid_stats = self.compute_statistics(centroid)
        all_features.append(centroid_stats)
        
        rolloff = self.extract_spectral_rolloff(audio)
        rolloff_stats = self.compute_statistics(rolloff)
        all_features.append(rolloff_stats)
        
        bandwidth = self.extract_spectral_bandwidth(audio)
        bandwidth_stats = self.compute_statistics(bandwidth)
        all_features.append(bandwidth_stats)
        
        zcr = self.extract_zcr(audio)
        zcr_stats = self.compute_statistics(zcr)
        all_features.append(zcr_stats)
        
        rms = self.extract_rms_energy(audio)
        rms_stats = self.compute_statistics(rms)
        all_features.append(rms_stats)
        
        log_energy = self.extract_log_energy(audio)
        log_energy_stats = self.compute_statistics(log_energy)
        all_features.append(log_energy_stats)
        
        # Chroma features (already multi-dimensional)
        chroma = self.extract_chroma(audio)
        chroma_stats = self.compute_statistics(chroma)
        all_features.append(chroma_stats)
        
        # Concatenate all features
        features_vector = np.concatenate(all_features)
        
        return features_vector
    
    def get_feature_names(self) -> List[str]:
        """Get names of all extracted features for interpretability."""
        feature_names = []
        feature_types = [
            'Centroid', 'Rolloff', 'Bandwidth', 'ZCR', 
            'RMS_Energy', 'Log_Energy'
        ]
        
        for ftype in feature_types:
            for stat in self.statistics:
                feature_names.append(f"{ftype}_{stat}")
        
        # Chroma features (12 pitch classes)
        pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        for pitch in pitches:
            for stat in self.statistics:
                feature_names.append(f"Chroma_{pitch}_{stat}")
        
        return feature_names


# Example usage
if __name__ == "__main__":
    from config.config import config
    import librosa
    
    try:
        # Create simple test audio
        sr = config.audio.SAMPLE_RATE
        duration = 1  # 1 second
        freq = 440  # A4 note
        t = np.linspace(0, duration, sr * duration)
        audio = 0.5 * np.sin(2 * np.pi * freq * t)
        
        # Extract features
        extractor = SpectralExtractor(config)
        features = extractor.extract(audio, sr)
        
        print(f"Audio shape: {audio.shape}")
        print(f"Features shape: {features.shape}")
        print(f"Feature names count: {len(extractor.get_feature_names())}")
    except Exception as e:
        print(f"Error in example: {e}")
