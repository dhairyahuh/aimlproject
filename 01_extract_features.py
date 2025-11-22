#!/usr/bin/env python3
"""
Extract MFCC features from audio recordings in recordings/ folder.
Saves features to data/processed/features/ directory.
"""

import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import librosa
import numpy as np

# ---------------------------------------------------------------------------
# Configuration - All paths relative to ASD_ADHD_Detection folder
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
RECORDINGS_DIR = PROJECT_ROOT / "recordings"
FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features"
SR = 22050  # Sample rate
N_MFCC = 40  # Number of MFCC coefficients


def extract_mfcc_features(audio_path: Path) -> np.ndarray:
    """Load audio file and compute MFCC feature matrix."""
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    return mfcc


def main():
    """Extract MFCCs for every recording and store them as .npy files."""
    if not RECORDINGS_DIR.exists():
        raise FileNotFoundError(
            f"Recordings directory not found: {RECORDINGS_DIR}\n"
            f"Please ensure audio files are in: {RECORDINGS_DIR}"
        )

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # Find all audio files
    audio_files = sorted(
        p for p in RECORDINGS_DIR.iterdir() 
        if p.suffix.lower() in {".m4a", ".wav", ".mp3"}
    )
    
    if not audio_files:
        print(f"No audio files found in {RECORDINGS_DIR}")
        return

    print(f"Extracting MFCC features for {len(audio_files)} audio files...")
    print(f"Output directory: {FEATURES_DIR}\n")
    
    success_count = 0
    for audio_path in audio_files:
        try:
            mfcc = extract_mfcc_features(audio_path)
            feature_name = audio_path.name + ".npy"
            feature_path = FEATURES_DIR / feature_name
            np.save(feature_path, mfcc)
            print(f"  ✓ {audio_path.name} -> {feature_name} (shape: {mfcc.shape})")
            success_count += 1
        except Exception as exc:
            print(f"  ✗ Failed to process {audio_path.name}: {exc}")

    print(f"\n✅ Successfully processed {success_count}/{len(audio_files)} files")
    print(f"Features saved to: {FEATURES_DIR}")


if __name__ == "__main__":
    main()

