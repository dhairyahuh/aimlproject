"""
Audio Preprocessing Module
Loads, cleans, and prepares audio for feature extraction

Based on: ser_preprocessing.py patterns
Author: ASD/ADHD Detection Team
"""

import numpy as np
import librosa
from typing import Tuple
import warnings

warnings.filterwarnings('ignore')


class AudioPreprocessor:
    """Load and preprocess audio files."""
    
    def __init__(self, config):
        """
        Initialize audio preprocessor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.sr = config.audio.SAMPLE_RATE
        self.target_duration = config.audio.DURATION
        self.trim_silence = config.audio.TRIM_SILENCE
        self.trim_threshold = config.audio.TRIM_THRESHOLD_DB
        self.normalize = config.audio.NORMALIZE_AUDIO
        
    def load_audio(self, filepath: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and resample to target sampling rate.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Tuple of (audio, sampling_rate)
        """
        try:
            audio, sr = librosa.load(filepath, sr=self.sr, mono=True)
            return audio, sr
        except Exception as e:
            print(f"Error loading audio: {e}")
            return np.array([0]), self.sr
    
    def trim_silence_from_audio(self, audio: np.ndarray, top_db: float = 30) -> np.ndarray:
        """
        Trim leading and trailing silence from audio.
        
        Args:
            audio: Audio time series
            top_db: Threshold in dB below reference
            
        Returns:
            Trimmed audio
        """
        try:
            trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
            return trimmed_audio
        except Exception as e:
            print(f"Error trimming silence: {e}")
            return audio
    
    def normalize_audio_signal(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range.
        
        Args:
            audio: Audio time series
            
        Returns:
            Normalized audio
        """
        try:
            max_abs = np.max(np.abs(audio))
            if max_abs > 0:
                normalized = audio / max_abs
            else:
                normalized = audio
            return normalized
        except Exception as e:
            print(f"Error normalizing audio: {e}")
            return audio
    
    def pad_or_truncate(self, audio: np.ndarray, sr: int, target_duration: float) -> np.ndarray:
        """
        Pad or truncate audio to target duration.
        
        Args:
            audio: Audio time series
            sr: Sampling rate
            target_duration: Target duration in seconds
            
        Returns:
            Resized audio
        """
        target_samples = int(sr * target_duration)
        
        if len(audio) > target_samples:
            # Truncate
            audio = audio[:target_samples]
        elif len(audio) < target_samples:
            # Pad with zeros
            pad_width = target_samples - len(audio)
            audio = np.pad(audio, (0, pad_width), mode='constant', constant_values=0)
        
        return audio
    
    def preprocess(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply complete preprocessing pipeline.
        
        Args:
            audio: Audio time series
            sr: Sampling rate
            
        Returns:
            Preprocessed audio
        """
        # 1. Trim silence
        if self.trim_silence:
            audio = self.trim_silence_from_audio(audio, top_db=self.trim_threshold)
        
        # 2. Normalize
        if self.normalize:
            audio = self.normalize_audio_signal(audio)
        
        # 3. Pad or truncate to target duration
        audio = self.pad_or_truncate(audio, sr, self.target_duration)
        
        return audio
    
    def load_and_preprocess(self, filepath: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and apply complete preprocessing.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Tuple of (preprocessed_audio, sampling_rate)
        """
        audio, sr = self.load_audio(filepath)
        audio = self.preprocess(audio, sr)
        return audio, sr


# Example usage
if __name__ == "__main__":
    from config.config import config
    
    # Create test audio
    sr = config.audio.SAMPLE_RATE
    duration = 3
    t = np.linspace(0, duration, sr * duration)
    
    # Multi-frequency signal
    audio = (
        0.3 * np.sin(2 * np.pi * 220 * t) +  # A3
        0.2 * np.sin(2 * np.pi * 440 * t) +  # A4
        0.1 * np.sin(2 * np.pi * 880 * t)    # A5
    )
    
    # Add noise
    audio = audio + 0.05 * np.random.randn(len(audio))
    
    preprocessor = AudioPreprocessor(config)
    
    print(f"Original audio shape: {audio.shape}")
    print(f"Original audio range: [{np.min(audio):.4f}, {np.max(audio):.4f}]")
    print(f"Original audio duration: {len(audio) / sr:.2f} seconds")
    
    # Preprocess
    processed = preprocessor.preprocess(audio, sr)
    
    print(f"\nAfter preprocessing:")
    print(f"Processed audio shape: {processed.shape}")
    print(f"Processed audio range: [{np.min(processed):.4f}, {np.max(processed):.4f}]")
    print(f"Processed audio duration: {len(processed) / sr:.2f} seconds")
