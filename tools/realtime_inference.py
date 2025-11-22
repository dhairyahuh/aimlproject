"""
Real-time ASD/ADHD Detection
=============================
Records audio from microphone and performs real-time prediction.
"""

import os
import sys
import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
import warnings
import time
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.mlp_classifier import MLPClassifier
from src.feature_extraction.feature_aggregator import FeatureAggregator
import pickle

# Paths
MODELS_DIR = project_root / 'models' / 'saved'
ROOT_DATA_DIR = project_root.parent / 'data'

# Configuration
SAMPLE_RATE = 16000
DURATION = 5  # seconds
CHANNELS = 1


class RealtimeDetector:
    """Real-time ASD/ADHD voice detector."""
    
    def __init__(self, model_path: str = None, scaler_path: str = None):
        """
        Initialize real-time detector.
        
        Args:
            model_path: Path to trained model
            scaler_path: Path to feature scaler
        """
        # Default paths
        if model_path is None:
            model_path = MODELS_DIR / 'asd_adhd_mlp_model.keras'
        if scaler_path is None:
            scaler_path = ROOT_DATA_DIR / 'data_scaler.pkl'
        
        print("="*70)
        print("INITIALIZING REAL-TIME DETECTOR")
        print("="*70)
        
        # Load model
        print(f"Loading model from: {model_path}")
        self.model = MLPClassifier()
        self.scaler = self.model.load(str(model_path), load_scaler=True)
        
        if self.scaler is None and Path(scaler_path).exists():
            print(f"Loading scaler from: {scaler_path}")
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        # Initialize feature extractor
        self.feature_extractor = FeatureAggregator(sr=SAMPLE_RATE, n_mfcc=13, use_pca=False)
        
        print("✓ Detector initialized successfully!")
        print("-"*70)
    
    def record_audio(self, duration: float = DURATION, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        """
        Record audio from microphone.
        
        Args:
            duration: Recording duration in seconds
            sample_rate: Sample rate in Hz
        
        Returns:
            Audio signal array
        """
        print(f"\n🎤 Recording audio for {duration} seconds...")
        print("   Speak now...")
        
        try:
            # Record audio
            audio = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=CHANNELS,
                dtype='float32'
            )
            sd.wait()  # Wait until recording is finished
            
            # Convert to 1D array
            audio = audio.flatten()
            
            print("✓ Recording complete!")
            return audio
            
        except Exception as e:
            print(f"✗ Error recording audio: {e}")
            raise
    
    def save_audio(self, audio: np.ndarray, filepath: str, sample_rate: int = SAMPLE_RATE):
        """Save recorded audio to file."""
        sf.write(filepath, audio, sample_rate)
        print(f"✓ Audio saved to: {filepath}")
    
    def predict_from_audio(self, audio: np.ndarray, save_temp: bool = True) -> dict:
        """
        Predict from audio array.
        
        Args:
            audio: Audio signal array
            save_temp: Whether to save temporary audio file for feature extraction
        
        Returns:
            Prediction dictionary
        """
        # Save temporary audio file for feature extraction
        # (Prosodic features require file path)
        if save_temp:
            temp_file = project_root / 'temp_audio.wav'
            self.save_audio(audio, str(temp_file), SAMPLE_RATE)
            audio_file = str(temp_file)
        else:
            # For MFCC and spectral features, we can use array directly
            # But prosodic features need file path, so we'll save anyway
            temp_file = project_root / 'temp_audio.wav'
            self.save_audio(audio, str(temp_file), SAMPLE_RATE)
            audio_file = str(temp_file)
        
        try:
            # Extract features
            print("\n🔍 Extracting features...")
            features = self.feature_extractor.extract_all_features(audio_file)
            
            # Normalize features
            if self.scaler is not None:
                features = self.scaler.transform(features.reshape(1, -1))
            else:
                features = features.reshape(1, -1)
            
            # Predict
            print("🤖 Making prediction...")
            prediction = self.model.predict_realtime(
                audio_file=audio_file,
                feature_extractor=self.feature_extractor
            )
            
            # Clean up temp file
            if save_temp and temp_file.exists():
                temp_file.unlink()
            
            return prediction
            
        except Exception as e:
            print(f"✗ Error during prediction: {e}")
            # Clean up temp file
            if save_temp and temp_file.exists():
                temp_file.unlink()
            raise
    
    def predict_from_file(self, audio_file: str) -> dict:
        """
        Predict from audio file.
        
        Args:
            audio_file: Path to audio file
        
        Returns:
            Prediction dictionary
        """
        print(f"\n📁 Processing audio file: {audio_file}")
        
        # Extract features
        print("🔍 Extracting features...")
        features = self.feature_extractor.extract_all_features(audio_file)
        
        # Normalize features
        if self.scaler is not None:
            features = self.scaler.transform(features.reshape(1, -1))
        else:
            features = features.reshape(1, -1)
        
        # Predict
        print("🤖 Making prediction...")
        prediction = self.model.predict_realtime(
            audio_file=audio_file,
            feature_extractor=self.feature_extractor
        )
        
        return prediction
    
    def display_prediction(self, prediction: dict):
        """Display prediction results."""
        print("\n" + "="*70)
        print("PREDICTION RESULTS")
        print("="*70)
        print(f"\n🎯 Predicted Class: {prediction['class_label']}")
        print(f"📊 Confidence: {prediction['confidence']:.2%}")
        print("\n📈 Class Probabilities:")
        for class_name, prob in prediction['probabilities'].items():
            bar_length = int(prob * 50)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"   {class_name:8s}: {prob:.2%} {bar}")
        print("="*70)


def interactive_mode():
    """Run interactive real-time detection."""
    detector = RealtimeDetector()
    
    print("\n" + "="*70)
    print("REAL-TIME ASD/ADHD DETECTION - INTERACTIVE MODE")
    print("="*70)
    print("\nInstructions:")
    print("  - Press Enter to start recording")
    print("  - Speak for 5 seconds")
    print("  - Wait for prediction")
    print("  - Type 'quit' to exit")
    print("-"*70)
    
    while True:
        try:
            user_input = input("\nPress Enter to record (or 'quit' to exit): ").strip().lower()
            
            if user_input == 'quit':
                print("\n👋 Exiting...")
                break
            
            # Record audio
            audio = detector.record_audio(duration=DURATION, sample_rate=SAMPLE_RATE)
            
            # Predict
            prediction = detector.predict_from_audio(audio)
            
            # Display results
            detector.display_prediction(prediction)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")
            print("Please try again.")


def file_mode(audio_file: str):
    """Run detection on audio file."""
    detector = RealtimeDetector()
    
    if not Path(audio_file).exists():
        print(f"✗ Audio file not found: {audio_file}")
        return
    
    prediction = detector.predict_from_file(audio_file)
    detector.display_prediction(prediction)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time ASD/ADHD Detection')
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='Path to audio file for prediction'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Path to trained model (default: models/saved/asd_adhd_mlp_model.keras)'
    )
    parser.add_argument(
        '--scaler', '-s',
        type=str,
        help='Path to feature scaler (default: data/data_scaler.pkl)'
    )
    
    args = parser.parse_args()
    
    if args.file:
        # File mode
        file_mode(args.file)
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    # Check if sounddevice is available
    try:
        import sounddevice as sd
    except ImportError:
        print("✗ sounddevice not installed. Install with: pip install sounddevice")
        sys.exit(1)
    
    main()

