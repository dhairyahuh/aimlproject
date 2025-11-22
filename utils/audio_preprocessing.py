"""
Audio Preprocessing Utilities
Based on: x4nth055/emotion-recognition-using-speech and pyAudioAnalysis patterns

Provides:
- Audio loading with format auto-detection
- Silence trimming (sukesh167 approach)
- Amplitude normalization
- Noise reduction (spectral subtraction)
- Batch processing capabilities
"""

import os
import numpy as np
import librosa
import librosa.effects
from typing import Tuple, Optional, Union, List
from pathlib import Path
import warnings
from scipy import signal as scipy_signal
from scipy.fftpack import fft, ifft

warnings.filterwarnings('ignore')


class AudioLoader:
    """
    Audio file loading with format detection and resampling.
    
    Based on x4nth055's loading approach but adapted for flexibility.
    """
    
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.ogg', '.flac', '.m4a'}
    
    def __init__(self, sr: int = 16000, mono: bool = True, offset: float = 0.0,
                 duration: Optional[float] = None):
        """
        Initialize AudioLoader.
        
        Args:
            sr: Target sampling rate (default: 16000 Hz for speech)
            mono: Convert to mono (default: True)
            offset: Start reading after this time (seconds)
            duration: Only load this much audio (seconds)
        """
        self.sr = sr
        self.mono = mono
        self.offset = offset
        self.duration = duration
        
    def is_supported_format(self, filepath: str) -> bool:
        """Check if file format is supported."""
        ext = Path(filepath).suffix.lower()
        return ext in self.SUPPORTED_FORMATS
    
    def load(self, filepath: str, verbose: bool = False) -> Tuple[np.ndarray, int]:
        """
        Load audio file with error handling.
        
        Args:
            filepath: Path to audio file
            verbose: Print loading information
            
        Returns:
            Tuple of (audio_array, sampling_rate)
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If format not supported
        """
        filepath = str(filepath)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Audio file not found: {filepath}")
        
        if not self.is_supported_format(filepath):
            raise ValueError(
                f"Unsupported format: {Path(filepath).suffix}. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )
        
        try:
            audio, sr = librosa.load(
                filepath,
                sr=self.sr,
                mono=self.mono,
                offset=self.offset,
                duration=self.duration
            )
            
            if verbose:
                print(f"✓ Loaded: {filepath}")
                print(f"  Shape: {audio.shape}")
                print(f"  Sample rate: {sr} Hz")
                print(f"  Duration: {len(audio) / sr:.2f}s")
                print(f"  Range: [{audio.min():.4f}, {audio.max():.4f}]")
            
            return audio, sr
            
        except Exception as e:
            raise RuntimeError(f"Error loading audio file {filepath}: {str(e)}")


class SilenceTrimmer:
    """
    Trim silence from audio using energy-based detection.
    
    Inspired by sukesh167's implementation but generalized.
    """
    
    def __init__(self, fft_window: int = 2048, hop_length: int = 512):
        """
        Initialize SilenceTrimmer.
        
        Args:
            fft_window: FFT window size for STFT
            hop_length: Number of samples between frames
        """
        self.fft_window = fft_window
        self.hop_length = hop_length
    
    def compute_energy(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Compute frame-wise energy using STFT.
        
        Args:
            audio: Audio time series
            sr: Sampling rate
            
        Returns:
            Energy per frame
        """
        # Compute STFT
        D = librosa.stft(audio, n_fft=self.fft_window, hop_length=self.hop_length)
        # Magnitude spectrogram
        S = np.abs(D)
        # Energy per frame (sum of squared magnitudes)
        energy = np.sum(S ** 2, axis=0)
        return energy
    
    def find_speech_frames(self, energy: np.ndarray, threshold_percentile: float = 30) -> Tuple[int, int]:
        """
        Find start and end frames of speech activity.
        
        Args:
            energy: Energy per frame
            threshold_percentile: Percentile for silence threshold
            
        Returns:
            Tuple of (start_frame, end_frame)
        """
        threshold = np.percentile(energy, threshold_percentile)
        speech_frames = np.where(energy > threshold)[0]
        
        if len(speech_frames) == 0:
            # No speech detected, return full range
            return 0, len(energy)
        
        start_frame = speech_frames[0]
        end_frame = speech_frames[-1]
        
        return start_frame, end_frame
    
    def trim(self, audio: np.ndarray, sr: int, threshold_percentile: float = 30,
             margin_frames: int = 5) -> np.ndarray:
        """
        Trim silence from audio using energy-based detection.
        
        Args:
            audio: Audio time series
            sr: Sampling rate
            threshold_percentile: Percentile for silence threshold (lower = trim more)
            margin_frames: Frames to keep at start/end for context
            
        Returns:
            Trimmed audio
            
        Example:
            >>> trimmer = SilenceTrimmer()
            >>> trimmed = trimmer.trim(audio, sr=16000)
        """
        energy = self.compute_energy(audio, sr)
        start_frame, end_frame = self.find_speech_frames(energy, threshold_percentile)
        
        # Add margin (convert frames to samples)
        start_sample = max(0, (start_frame - margin_frames) * self.hop_length)
        end_sample = min(len(audio), (end_frame + margin_frames) * self.hop_length)
        
        return audio[start_sample:end_sample]
    
    def trim_librosa(self, audio: np.ndarray, top_db: float = 20) -> np.ndarray:
        """
        Trim silence using librosa's built-in method (faster, less control).
        
        Args:
            audio: Audio time series
            top_db: Threshold in dB relative to reference power
            
        Returns:
            Trimmed audio
        """
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed


class AmplitudeNormalizer:
    """
    Normalize audio amplitude to standard range.
    """
    
    @staticmethod
    def normalize_peak(audio: np.ndarray, target_level: float = 1.0) -> np.ndarray:
        """
        Normalize by peak value (loudest sample).
        
        Args:
            audio: Audio time series
            target_level: Target peak level (default 1.0 for [-1, 1] range)
            
        Returns:
            Normalized audio
        """
        peak = np.max(np.abs(audio))
        if peak > 0:
            return audio * (target_level / peak)
        return audio
    
    @staticmethod
    def normalize_rms(audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
        """
        Normalize by RMS (root mean square) energy.
        
        Args:
            audio: Audio time series
            target_rms: Target RMS level (default 0.1)
            
        Returns:
            Normalized audio
        """
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            return audio * (target_rms / rms)
        return audio
    
    @staticmethod
    def normalize_z_score(audio: np.ndarray) -> np.ndarray:
        """
        Standardize audio using z-score (zero mean, unit variance).
        
        Args:
            audio: Audio time series
            
        Returns:
            Normalized audio with mean=0, std=1
        """
        mean = np.mean(audio)
        std = np.std(audio)
        if std > 0:
            return (audio - mean) / std
        return audio - mean
    
    @staticmethod
    def normalize_minmax(audio: np.ndarray, min_val: float = -1.0, 
                        max_val: float = 1.0) -> np.ndarray:
        """
        Scale audio to [min_val, max_val] range.
        
        Args:
            audio: Audio time series
            min_val: Minimum value in output
            max_val: Maximum value in output
            
        Returns:
            Scaled audio in [min_val, max_val] range
        """
        audio_min = np.min(audio)
        audio_max = np.max(audio)
        audio_range = audio_max - audio_min
        
        if audio_range > 0:
            normalized = (audio - audio_min) / audio_range
            return normalized * (max_val - min_val) + min_val
        return audio


class NoiseReducer:
    """
    Reduce background noise using spectral subtraction and related methods.
    """
    
    def __init__(self, fft_size: int = 2048, hop_length: int = 512):
        """
        Initialize NoiseReducer.
        
        Args:
            fft_size: FFT size for frequency analysis
            hop_length: Hop length for STFT
        """
        self.fft_size = fft_size
        self.hop_length = hop_length
    
    def spectral_subtraction(self, audio: np.ndarray, sr: int,
                            noise_duration: float = 1.0,
                            alpha: float = 2.0,
                            floor_factor: float = 0.002) -> np.ndarray:
        """
        Reduce noise using spectral subtraction.
        
        Assumes first noise_duration seconds contain mostly noise.
        
        Args:
            audio: Audio time series
            sr: Sampling rate
            noise_duration: Duration of noise profile at start (seconds)
            alpha: Over-subtraction factor (higher = more aggressive)
            floor_factor: Floor to prevent over-subtraction (fraction of noise spectrum)
            
        Returns:
            Noise-reduced audio
            
        Example:
            >>> reducer = NoiseReducer()
            >>> clean_audio = reducer.spectral_subtraction(audio, sr=16000)
        """
        # Compute STFT
        D = librosa.stft(audio, n_fft=self.fft_size, hop_length=self.hop_length)
        magnitude = np.abs(D)
        phase = np.angle(D)
        
        # Estimate noise spectrum from first noise_duration
        noise_frames = int(sr * noise_duration / self.hop_length)
        noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        
        # Spectral subtraction
        floor = floor_factor * noise_spectrum
        magnitude_cleaned = magnitude - alpha * noise_spectrum
        magnitude_cleaned = np.maximum(magnitude_cleaned, floor)
        
        # Reconstruct complex spectrogram
        D_cleaned = magnitude_cleaned * np.exp(1j * phase)
        
        # Inverse STFT
        audio_cleaned = librosa.istft(D_cleaned, hop_length=self.hop_length)
        
        return audio_cleaned
    
    def wiener_filter(self, audio: np.ndarray, sr: int,
                     noise_duration: float = 1.0,
                     frame_length: int = 2048) -> np.ndarray:
        """
        Reduce noise using Wiener filtering.
        
        Args:
            audio: Audio time series
            sr: Sampling rate
            noise_duration: Duration of noise profile (seconds)
            frame_length: Frame length for noise estimation
            
        Returns:
            Filtered audio
        """
        # Estimate noise variance from silence
        noise_samples = int(sr * noise_duration)
        noise_variance = np.var(audio[:noise_samples])
        
        # Compute signal variance using Welch's method
        f, Pxx = scipy_signal.welch(audio, sr, nperseg=frame_length)
        signal_variance = np.mean(Pxx)
        
        # Apply Wiener filter in frequency domain
        D = librosa.stft(audio, n_fft=self.fft_size, hop_length=self.hop_length)
        magnitude = np.abs(D)
        phase = np.angle(D)
        
        # Wiener filter gain
        snr = np.maximum(signal_variance - noise_variance, 0) / (signal_variance + 1e-10)
        H = snr / (1 + snr)
        
        # Apply to each frame
        magnitude_filtered = magnitude * H
        
        # Reconstruct
        D_filtered = magnitude_filtered * np.exp(1j * phase)
        audio_filtered = librosa.istft(D_filtered, hop_length=self.hop_length)
        
        return audio_filtered


class AudioPreprocessor:
    """
    Complete audio preprocessing pipeline.
    
    Combines loading, trimming, normalization, and optional noise reduction.
    
    Based on x4nth055 class structure pattern.
    """
    
    def __init__(self, sr: int = 16000, trim_silence: bool = True,
                 normalize: str = 'peak', apply_noise_reduction: bool = False,
                 noise_reduction_method: str = 'spectral_subtraction',
                 verbose: bool = False):
        """
        Initialize AudioPreprocessor.
        
        Args:
            sr: Target sampling rate (Hz)
            trim_silence: Whether to trim silence
            normalize: Normalization method: 'peak', 'rms', 'z_score', 'minmax'
            apply_noise_reduction: Whether to apply noise reduction
            noise_reduction_method: 'spectral_subtraction' or 'wiener_filter'
            verbose: Print debug information
        """
        self.sr = sr
        self.trim_silence = trim_silence
        self.normalize = normalize
        self.apply_noise_reduction = apply_noise_reduction
        self.noise_reduction_method = noise_reduction_method
        self.verbose = verbose
        
        # Initialize components
        self.loader = AudioLoader(sr=sr)
        self.trimmer = SilenceTrimmer()
        self.normalizer = AmplitudeNormalizer()
        self.reducer = NoiseReducer()
        
    def __repr__(self) -> str:
        """String representation of preprocessor configuration."""
        return (
            f"AudioPreprocessor(sr={self.sr}, "
            f"trim_silence={self.trim_silence}, "
            f"normalize={self.normalize}, "
            f"noise_reduction={self.apply_noise_reduction})"
        )
    
    def process_file(self, filepath: str) -> Tuple[np.ndarray, int]:
        """
        Load and preprocess a single audio file.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Tuple of (processed_audio, sampling_rate)
        """
        # Step 1: Load
        if self.verbose:
            print(f"\n[1/4] Loading: {filepath}")
        audio, sr = self.loader.load(filepath, verbose=self.verbose)
        
        # Step 2: Trim silence
        if self.trim_silence:
            if self.verbose:
                print(f"[2/4] Trimming silence... (original: {len(audio)} samples)")
            audio = self.trimmer.trim_librosa(audio, top_db=20)
            if self.verbose:
                print(f"      After trim: {len(audio)} samples ({len(audio)/sr:.2f}s)")
        else:
            if self.verbose:
                print("[2/4] Skipping silence trim")
        
        # Step 3: Noise reduction
        if self.apply_noise_reduction:
            if self.verbose:
                print(f"[3/4] Applying noise reduction ({self.noise_reduction_method})...")
            if self.noise_reduction_method == 'spectral_subtraction':
                audio = self.reducer.spectral_subtraction(audio, sr)
            elif self.noise_reduction_method == 'wiener_filter':
                audio = self.reducer.wiener_filter(audio, sr)
        else:
            if self.verbose:
                print("[3/4] Skipping noise reduction")
        
        # Step 4: Normalize
        if self.normalize:
            if self.verbose:
                print(f"[4/4] Normalizing ({self.normalize} method)...")
            if self.normalize == 'peak':
                audio = self.normalizer.normalize_peak(audio)
            elif self.normalize == 'rms':
                audio = self.normalizer.normalize_rms(audio)
            elif self.normalize == 'z_score':
                audio = self.normalizer.normalize_z_score(audio)
            elif self.normalize == 'minmax':
                audio = self.normalizer.normalize_minmax(audio)
        else:
            if self.verbose:
                print("[4/4] Skipping normalization")
        
        if self.verbose:
            print(f"✓ Processing complete")
            print(f"  Final shape: {audio.shape}")
            print(f"  Final range: [{audio.min():.4f}, {audio.max():.4f}]")
            print(f"  Final RMS: {np.sqrt(np.mean(audio**2)):.4f}")
        
        return audio, sr
    
    def process_batch(self, filepaths: List[str], show_progress: bool = True) -> List[Tuple[np.ndarray, int]]:
        """
        Process multiple audio files.
        
        Args:
            filepaths: List of file paths
            show_progress: Show progress bar
            
        Returns:
            List of (audio, sr) tuples
        """
        results = []
        
        for i, filepath in enumerate(filepaths):
            if show_progress:
                print(f"[{i+1}/{len(filepaths)}] Processing {Path(filepath).name}...")
            
            try:
                audio, sr = self.process_file(filepath)
                results.append((audio, sr))
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
                results.append((None, None))
        
        return results
    
    def process_directory(self, directory: str, pattern: str = '*.wav',
                         recursive: bool = False) -> List[Tuple[str, np.ndarray, int]]:
        """
        Process all audio files in a directory.
        
        Args:
            directory: Directory path
            pattern: File pattern to match (e.g., '*.wav')
            recursive: Search recursively
            
        Returns:
            List of (filepath, audio, sr) tuples
        """
        dir_path = Path(directory)
        
        # Find files
        if recursive:
            files = sorted(dir_path.rglob(pattern))
        else:
            files = sorted(dir_path.glob(pattern))
        
        results = []
        for filepath in files:
            try:
                audio, sr = self.process_file(str(filepath))
                results.append((str(filepath), audio, sr))
            except Exception as e:
                print(f"Error processing {filepath}: {str(e)}")
        
        return results


# Convenience functions for quick use
def load_audio(filepath: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Quick load audio file."""
    loader = AudioLoader(sr=sr)
    return loader.load(filepath)


def preprocess_audio(audio: np.ndarray, sr: int = 16000, trim: bool = True,
                    normalize: str = 'peak', reduce_noise: bool = False) -> np.ndarray:
    """
    Quick preprocess audio array.
    
    Args:
        audio: Audio time series
        sr: Sampling rate
        trim: Trim silence
        normalize: Normalization method
        reduce_noise: Apply noise reduction
        
    Returns:
        Preprocessed audio
    """
    processor = AudioPreprocessor(
        sr=sr, trim_silence=trim, normalize=normalize,
        apply_noise_reduction=reduce_noise
    )
    
    # Simulate processing by running components
    if trim:
        audio = processor.trimmer.trim_librosa(audio)
    
    if reduce_noise:
        audio = processor.reducer.spectral_subtraction(audio, sr)
    
    if normalize == 'peak':
        audio = processor.normalizer.normalize_peak(audio)
    elif normalize == 'rms':
        audio = processor.normalizer.normalize_rms(audio)
    elif normalize == 'z_score':
        audio = processor.normalizer.normalize_z_score(audio)
    
    return audio


# Example usage and testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("Audio Preprocessing Utility - Usage Examples\n")
    print("=" * 60)
    
    # Example 1: Create synthetic audio
    print("\n[Example 1] Creating synthetic audio with noise")
    print("-" * 60)
    
    sr = 16000
    duration = 3
    t = np.linspace(0, duration, sr * duration)
    
    # Multi-frequency signal (simulating speech)
    audio = (
        0.3 * np.sin(2 * np.pi * 220 * t) +  # A3
        0.2 * np.sin(2 * np.pi * 440 * t) +  # A4
        0.1 * np.sin(2 * np.pi * 880 * t)    # A5
    )
    
    # Add silence at start and end
    silence = np.zeros(int(sr * 0.5))
    audio_with_silence = np.concatenate([silence, audio, silence])
    
    # Add background noise
    audio_noisy = audio_with_silence + 0.05 * np.random.randn(len(audio_with_silence))
    
    print(f"Original audio shape: {audio_noisy.shape}")
    print(f"Original duration: {len(audio_noisy) / sr:.2f}s")
    print(f"Original range: [{audio_noisy.min():.4f}, {audio_noisy.max():.4f}]")
    print(f"Original RMS: {np.sqrt(np.mean(audio_noisy**2)):.4f}")
    
    # Example 2: Full preprocessing pipeline
    print("\n[Example 2] Full preprocessing pipeline")
    print("-" * 60)
    
    preprocessor = AudioPreprocessor(
        sr=sr,
        trim_silence=True,
        normalize='peak',
        apply_noise_reduction=True,
        noise_reduction_method='spectral_subtraction',
        verbose=True
    )
    
    print(repr(preprocessor))
    
    # Create temporary audio array for processing
    processor = AudioPreprocessor(sr=sr, trim_silence=True, normalize='peak',
                                  apply_noise_reduction=True)
    
    # Manual processing pipeline
    processed = audio_noisy.copy()
    
    # Trim silence
    processed = processor.trimmer.trim_librosa(processed, top_db=20)
    print(f"\nAfter trim: {len(processed)} samples ({len(processed)/sr:.2f}s)")
    
    # Noise reduction
    processed = processor.reducer.spectral_subtraction(processed, sr, noise_duration=0.5)
    print(f"After noise reduction: {processed.shape}")
    
    # Normalize
    processed = processor.normalizer.normalize_peak(processed)
    print(f"After normalization: range [{processed.min():.4f}, {processed.max():.4f}]")
    print(f"After normalization: RMS = {np.sqrt(np.mean(processed**2)):.4f}")
    
    # Example 3: Normalization methods comparison
    print("\n[Example 3] Comparing normalization methods")
    print("-" * 60)
    
    test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr))
    test_audio = test_audio * 0.3 + 0.1  # Offset and scale
    
    methods = ['peak', 'rms', 'z_score', 'minmax']
    normalizer = AmplitudeNormalizer()
    
    for method in methods:
        if method == 'peak':
            result = normalizer.normalize_peak(test_audio)
        elif method == 'rms':
            result = normalizer.normalize_rms(test_audio)
        elif method == 'z_score':
            result = normalizer.normalize_z_score(test_audio)
        elif method == 'minmax':
            result = normalizer.normalize_minmax(test_audio)
        
        print(f"{method:15} | range: [{result.min():7.4f}, {result.max():7.4f}] | RMS: {np.sqrt(np.mean(result**2)):.4f}")
    
    print("\n" + "=" * 60)
    print("Examples complete!")
