#!/usr/bin/env python3
"""
Data Augmentation Tool for Audio Files
=======================================

This script provides data augmentation capabilities for audio files to increase
dataset size and improve model generalization. It supports various augmentation
techniques including time stretching, pitch shifting, and noise addition.

Usage:
    python data_augmentation_tool.py [options]
"""

import os
import shutil
import random
from pathlib import Path
import argparse
import numpy as np
import librosa
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent
RECORDINGS_DIR = PROJECT_ROOT / "recordings"
AUGMENTED_RECORDINGS_DIR = PROJECT_ROOT / "recordings_augmented"


def apply_time_stretch(audio, sr, rate=1.1):
    """Apply time stretching to audio."""
    return librosa.effects.time_stretch(audio, rate=rate)


def apply_pitch_shift(audio, sr, n_steps=2):
    """Apply pitch shifting to audio."""
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def add_noise(audio, noise_factor=0.005):
    """Add random noise to audio."""
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise


def augment_audio_file(input_path, output_path, augmentation_type):
    """Apply augmentation to a single audio file."""
    try:
        # Load audio
        audio, sr = librosa.load(input_path, sr=None)
        
        # Apply augmentation
        if augmentation_type == 'time_stretch':
            augmented = apply_time_stretch(audio, sr, rate=random.uniform(0.9, 1.1))
        elif augmentation_type == 'pitch_shift':
            augmented = apply_pitch_shift(audio, sr, n_steps=random.uniform(-2, 2))
        elif augmentation_type == 'add_noise':
            augmented = add_noise(audio, noise_factor=random.uniform(0.003, 0.01))
        elif augmentation_type == 'combined':
            # Apply multiple augmentations
            augmented = audio
            if random.random() > 0.5:
                augmented = apply_time_stretch(augmented, sr, rate=random.uniform(0.95, 1.05))
            if random.random() > 0.5:
                augmented = apply_pitch_shift(augmented, sr, n_steps=random.uniform(-1, 1))
            if random.random() > 0.5:
                augmented = add_noise(augmented, noise_factor=random.uniform(0.002, 0.008))
        else:
            # No augmentation, just copy
            shutil.copy2(input_path, output_path)
            return True
        
        # Save augmented audio
        sf.write(output_path, augmented, sr)
        return True
    except Exception as e:
        print(f"⚠️  Error augmenting {input_path.name}: {e}")
        return False


def augment_dataset(target_multiplier=2, augmentation_type='combined'):
    """
    Augment audio files to increase dataset size.
    
    Args:
        target_multiplier: How many times to multiply the dataset (default: 2)
        augmentation_type: Type of augmentation ('time_stretch', 'pitch_shift', 
                         'add_noise', 'combined', or 'none')
    """
    if not RECORDINGS_DIR.exists():
        print(f"❌ Recordings directory not found: {RECORDINGS_DIR}")
        return
    
    # Get all audio files
    audio_files = list(RECORDINGS_DIR.glob("*.m4a")) + list(RECORDINGS_DIR.glob("*.wav"))
    
    if not audio_files:
        print(f"❌ No audio files found in {RECORDINGS_DIR}")
        return
    
    print(f"📂 Found {len(audio_files)} original audio files")
    print(f"🎯 Target multiplier: {target_multiplier}x")
    print(f"📊 Augmentation type: {augmentation_type}")
    
    # Create augmented directory
    AUGMENTED_RECORDINGS_DIR.mkdir(exist_ok=True)
    
    # Calculate how many augmented files we need
    target_count = len(audio_files) * target_multiplier
    needed = target_count - len(audio_files)
    
    if needed <= 0:
        print(f"✅ Already have {len(audio_files)} files, no augmentation needed")
        return
    
    print(f"📋 Will create {needed} augmented files")
    print("")
    
    # Augment files
    created_count = 0
    augmentation_types = ['time_stretch', 'pitch_shift', 'add_noise', 'combined']
    
    for i in range(needed):
        # Pick a random original file
        original = random.choice(audio_files)
        
        # Determine augmentation type for this file
        if augmentation_type == 'random':
            aug_type = random.choice(augmentation_types)
        else:
            aug_type = augmentation_type
        
        # Generate new filename
        base_name = original.stem
        extension = original.suffix
        new_name = f"{base_name}_aug_{i+1:03d}{extension}"
        new_path = AUGMENTED_RECORDINGS_DIR / new_name
        
        # Apply augmentation
        if augment_audio_file(original, new_path, aug_type):
            created_count += 1
        
        if (created_count % 10) == 0:
            print(f"  Created {created_count}/{needed} augmented files...")
    
    print(f"\n✅ Created {created_count} augmented files")
    print(f"📁 Original files: {RECORDINGS_DIR} ({len(audio_files)} files)")
    print(f"📁 Augmented files: {AUGMENTED_RECORDINGS_DIR} ({created_count} files)")
    print(f"\n💡 Next steps:")
    print(f"   1. Review augmented files in {AUGMENTED_RECORDINGS_DIR}")
    print(f"   2. Run feature extraction on augmented files if needed")
    print(f"   3. Use augmented files for training to improve model generalization")


def cleanup_augmented_files():
    """Remove all augmented files."""
    if AUGMENTED_RECORDINGS_DIR.exists():
        shutil.rmtree(AUGMENTED_RECORDINGS_DIR)
        print(f"✅ Removed augmented directory: {AUGMENTED_RECORDINGS_DIR}")
    else:
        print(f"ℹ️  Augmented directory not found: {AUGMENTED_RECORDINGS_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Augment audio files to increase dataset size"
    )
    parser.add_argument(
        "--target-multiplier",
        type=int,
        default=2,
        help="How many times to multiply the dataset (default: 2)"
    )
    parser.add_argument(
        "--augmentation-type",
        type=str,
        default='combined',
        choices=['time_stretch', 'pitch_shift', 'add_noise', 'combined', 'random', 'none'],
        help="Type of augmentation to apply (default: combined)"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove all augmented files"
    )
    
    args = parser.parse_args()
    
    if args.cleanup:
        cleanup_augmented_files()
    else:
        augment_dataset(args.target_multiplier, args.augmentation_type)


if __name__ == "__main__":
    main()

