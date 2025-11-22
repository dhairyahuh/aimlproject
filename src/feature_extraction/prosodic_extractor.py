"""
Prosodic Features Extractor Module
Extracts prosodic features (F0, formants, jitter, shimmer, HNR, voice quality)

Based on: Parselmouth/ and Dinstein-Lab/ASDSpeech patterns
Author: ASD/ADHD Detection Team
"""

import numpy as np
import librosa
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore')

try:
    import parselmouth
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False
    print("Warning: Parselmouth not available. Install with: pip install parselmouth")


class ProsodicExtractor:
    """Extract prosodic features from audio signals."""
    
    def __init__(self, config):
        """
        Initialize prosodic extractor with configuration.
        
        Args:
            config: Configuration object with prosodic parameters
        """
        self.config = config
        self.sr = config.audio.SAMPLE_RATE
        self.f0_min = config.prosodic.F0_MIN
        self.f0_max = config.prosodic.F0_MAX
        self.n_fft = config.audio.N_FFT
        
    def extract_f0_librosa(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract F0 using librosa's pyin algorithm (more robust fallback).
        
        Args:
            audio: Audio time series
            sr: Sampling rate
            
        Returns:
            F0 values (time_steps,) in Hz
        """
        try:
            # Use pYIN algorithm for robust pitch tracking
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=self.f0_min,
                fmax=self.f0_max,
                sr=sr,
                frame_length=self.n_fft
            )
            
            # Replace NaN with 0 (unvoiced frames)
            f0 = np.nan_to_num(f0, nan=0.0)
            return f0
        except Exception as e:
            print(f"Error extracting F0: {e}")
            return np.array([0])
    
    def extract_f0_parselmouth(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract F0 using Parselmouth (Praat interface) - more accurate.
        
        Args:
            audio: Audio time series
            sr: Sampling rate
            
        Returns:
            F0 values (time_steps,) in Hz
        """
        if not PARSELMOUTH_AVAILABLE:
            print("Parselmouth not available, using librosa fallback")
            return self.extract_f0_librosa(audio, sr)
        
        try:
            # Create Praat Sound object
            sound = parselmouth.Sound(audio, sr)
            
            # Extract pitch with Praat's algorithm
            pitch = sound.to_pitch(
                time_step=0.01,  # 10ms step
                pitch_floor=self.f0_min,
                pitch_ceiling=self.f0_max
            )
            
            # Get F0 values
            f0_values = pitch.selected_array['frequency']
            f0_values = np.nan_to_num(f0_values, nan=0.0)
            
            return f0_values
        except Exception as e:
            print(f"Error with Parselmouth: {e}, using librosa fallback")
            return self.extract_f0_librosa(audio, sr)
    
    def compute_f0_statistics(self, f0: np.ndarray) -> Dict[str, float]:
        """
        Compute F0 statistics (important for ASD detection).
        
        Args:
            f0: F0 contour (time_steps,)
            
        Returns:
            Dictionary with F0 statistics
        """
        # Filter out unvoiced (f0=0) frames
        f0_voiced = f0[f0 > 0]
        
        if len(f0_voiced) == 0:
            # All unvoiced
            return {
                'f0_mean': 0, 'f0_std': 0, 'f0_min': 0, 'f0_max': 0,
                'f0_median': 0, 'f0_range': 0, 'f0_cv': 0, 'voiced_rate': 0
            }
        
        stats = {
            'f0_mean': np.mean(f0_voiced),
            'f0_std': np.std(f0_voiced),
            'f0_min': np.min(f0_voiced),
            'f0_max': np.max(f0_voiced),
            'f0_median': np.median(f0_voiced),
            'f0_range': np.max(f0_voiced) - np.min(f0_voiced),
            'f0_cv': np.std(f0_voiced) / (np.mean(f0_voiced) + 1e-8),  # Coefficient of variation
            'voiced_rate': len(f0_voiced) / len(f0)
        }
        
        return stats
    
    def extract_formants_librosa(self, audio: np.ndarray, sr: int, num_formants: int = 3) -> Dict[str, float]:
        """
        Estimate formants using spectral peaks (simplified method).
        
        Args:
            audio: Audio time series
            sr: Sampling rate
            num_formants: Number of formants to extract (default 3: F1, F2, F3)
            
        Returns:
            Dictionary with formant frequencies
        """
        try:
            # Compute STFT
            S = librosa.stft(audio, n_fft=self.n_fft)
            mag = np.abs(S)
            freq_bins = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
            
            # Average spectrum
            avg_spectrum = np.mean(mag, axis=1)
            
            # Find spectral peaks (formants)
            peaks = []
            for i in range(1, len(avg_spectrum) - 1):
                if avg_spectrum[i] > avg_spectrum[i-1] and avg_spectrum[i] > avg_spectrum[i+1]:
                    peaks.append((freq_bins[i], avg_spectrum[i]))
            
            # Sort by magnitude and get top N
            peaks.sort(key=lambda x: x[1], reverse=True)
            top_peaks = sorted(peaks[:num_formants], key=lambda x: x[0])  # Sort by frequency
            
            formant_stats = {}
            for idx, (freq, mag) in enumerate(top_peaks):
                formant_stats[f'F{idx+1}_freq'] = freq
                formant_stats[f'F{idx+1}_mag'] = mag
            
            # Pad with zeros if fewer formants found
            for idx in range(len(top_peaks), num_formants):
                formant_stats[f'F{idx+1}_freq'] = 0
                formant_stats[f'F{idx+1}_mag'] = 0
            
            return formant_stats
        except Exception as e:
            print(f"Error extracting formants: {e}")
            return {f'F{i}_freq': 0 for i in range(1, num_formants+1)} | {f'F{i}_mag': 0 for i in range(1, num_formants+1)}
    
    def extract_jitter(self, f0: np.ndarray) -> float:
        """
        Calculate Jitter (pitch perturbation) - ASD marker.
        Measures cycle-to-cycle variation in pitch.
        
        Args:
            f0: F0 contour (time_steps,)
            
        Returns:
            Jitter value (0-1 range, lower is better)
        """
        try:
            # Filter voiced frames
            f0_voiced = f0[f0 > 0]
            
            if len(f0_voiced) < 2:
                return 0
            
            # Calculate period differences
            periods = 1.0 / (f0_voiced + 1e-8)
            period_diffs = np.abs(np.diff(periods))
            
            # Jitter = mean absolute difference / mean period
            jitter = np.mean(period_diffs) / (np.mean(periods) + 1e-8)
            
            return float(np.clip(jitter, 0, 1))
        except Exception as e:
            print(f"Error calculating jitter: {e}")
            return 0
    
    def extract_shimmer(self, audio: np.ndarray, sr: int) -> float:
        """
        Calculate Shimmer (amplitude perturbation) - ASD marker.
        Measures cycle-to-cycle variation in amplitude.
        
        Args:
            audio: Audio time series
            sr: Sampling rate
            
        Returns:
            Shimmer value (0-1 range, lower is better)
        """
        try:
            # Extract envelope using Hilbert transform
            analytic_signal = librosa.stft(audio)
            amplitude_envelope = np.abs(analytic_signal)
            
            # Calculate frame amplitudes
            frame_amps = np.mean(amplitude_envelope, axis=0)
            
            if len(frame_amps) < 2:
                return 0
            
            # Calculate amplitude differences
            amp_diffs = np.abs(np.diff(frame_amps))
            
            # Shimmer = mean absolute difference / mean amplitude
            shimmer = np.mean(amp_diffs) / (np.mean(frame_amps) + 1e-8)
            
            return float(np.clip(shimmer, 0, 1))
        except Exception as e:
            print(f"Error calculating shimmer: {e}")
            return 0
    
    def extract_hnr(self, audio: np.ndarray, sr: int) -> float:
        """
        Extract HNR (Harmonic-to-Noise Ratio) - voice quality measure.
        
        Args:
            audio: Audio time series
            sr: Sampling rate
            
        Returns:
            HNR in dB
        """
        try:
            if PARSELMOUTH_AVAILABLE:
                sound = parselmouth.Sound(audio, sr)
                hnr = sound.to_harmonicity()
                return float(np.nanmean(hnr.values))
            else:
                # Fallback: simplified HNR calculation
                S = librosa.stft(audio, n_fft=self.n_fft)
                mag = np.abs(S)
                
                # Harmonic: sum of peaks
                # Noise: sum of valleys
                harmonic = np.mean(np.max(mag, axis=1))
                noise = np.mean(np.min(mag, axis=1))
                
                hnr = 10 * np.log10((harmonic + 1e-8) / (noise + 1e-8))
                return float(hnr)
        except Exception as e:
            print(f"Error extracting HNR: {e}")
            return 0
    
    def extract_voice_quality(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract voice quality measures.
        
        Args:
            audio: Audio time series
            sr: Sampling rate
            
        Returns:
            Dictionary with voice quality metrics
        """
        try:
            # Voice Activity Detection using energy
            S = librosa.stft(audio, n_fft=self.n_fft)
            mag = np.abs(S)
            power = mag ** 2
            energy = np.mean(power, axis=0)
            
            # Threshold for voiced frames
            energy_threshold = np.mean(energy) * 0.3
            voiced_frames = energy > energy_threshold
            
            stats = {
                'voice_activity_rate': float(np.mean(voiced_frames)),
                'voice_breaks': int(np.sum(np.diff(voiced_frames.astype(int)) < 0))
            }
            
            return stats
        except Exception as e:
            print(f"Error extracting voice quality: {e}")
            return {'voice_activity_rate': 0, 'voice_breaks': 0}
    
    def extract(self, audio: np.ndarray, sr: int = None) -> np.ndarray:
        """
        Extract complete prosodic features.
        
        Args:
            audio: Audio time series
            sr: Sampling rate (uses config if None)
            
        Returns:
            Feature vector (30+) containing:
            - F0 statistics (8): mean, std, min, max, median, range, CV, voiced_rate
            - Formants (6): F1_freq, F1_mag, F2_freq, F2_mag, F3_freq, F3_mag
            - Jitter (1)
            - Shimmer (1)
            - HNR (1)
            - Voice quality (2): activity_rate, voice_breaks
            Total: ~19-20 features (simplified)
        """
        if sr is None:
            sr = self.sr
        
        all_features = []
        
        # Extract F0
        f0 = self.extract_f0_parselmouth(audio, sr)
        f0_stats = self.compute_f0_statistics(f0)
        all_features.extend(list(f0_stats.values()))
        
        # Extract formants
        formant_stats = self.extract_formants_librosa(audio, sr, num_formants=3)
        all_features.extend(list(formant_stats.values()))
        
        # Extract jitter (pitch perturbation) - ASD MARKER
        jitter = self.extract_jitter(f0)
        all_features.append(jitter)
        
        # Extract shimmer (amplitude perturbation) - ASD MARKER
        shimmer = self.extract_shimmer(audio, sr)
        all_features.append(shimmer)
        
        # Extract HNR
        hnr = self.extract_hnr(audio, sr)
        all_features.append(hnr)
        
        # Extract voice quality
        voice_quality = self.extract_voice_quality(audio, sr)
        all_features.extend(list(voice_quality.values()))
        
        features_vector = np.array(all_features, dtype=np.float32)
        
        return features_vector
    
    def get_feature_names(self) -> List[str]:
        """Get names of all extracted features for interpretability."""
        names = [
            'F0_Mean', 'F0_Std', 'F0_Min', 'F0_Max', 'F0_Median', 'F0_Range', 'F0_CV', 'Voiced_Rate',
            'F1_Freq', 'F1_Mag', 'F2_Freq', 'F2_Mag', 'F3_Freq', 'F3_Mag',
            'Jitter', 'Shimmer', 'HNR',
            'Voice_Activity_Rate', 'Voice_Breaks'
        ]
        return names


# Example usage
if __name__ == "__main__":
    from config.config import config
    
    try:
        # Create simple test audio (440 Hz sine wave)
        sr = config.audio.SAMPLE_RATE
        duration = 2  # 2 seconds
        freq = 220  # A3 note
        t = np.linspace(0, duration, sr * duration)
        audio = 0.5 * np.sin(2 * np.pi * freq * t)
        
        # Extract features
        extractor = ProsodicExtractor(config)
        features = extractor.extract(audio, sr)
        
        print(f"Audio shape: {audio.shape}")
        print(f"Features shape: {features.shape}")
        print(f"Feature names: {extractor.get_feature_names()}")
        print(f"Feature values:\n{dict(zip(extractor.get_feature_names(), features))}")
    except Exception as e:
        print(f"Error in example: {e}")
