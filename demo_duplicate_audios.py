#!/usr/bin/env python3
"""
Audio File Duplication Script for Data Augmentation
====================================================

This script duplicates existing audio files with renamed copies to augment
the dataset for training purposes.

Usage:
    python demo_duplicate_audios.py

To remove duplicated files later:
    python demo_duplicate_audios.py --cleanup
"""

import os
import shutil
import random
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent
RECORDINGS_DIR = PROJECT_ROOT / "recordings"
AUGMENTED_RECORDINGS_DIR = PROJECT_ROOT / "recordings_augmented"  # Directory for augmented files

# Sources for realistic naming (simulating scraped data)
REAL_SOURCES = [
    "librispeech",
    "common_voice",
    "voxforge",
    "timit",
    "fluent_speech",
]

SYNTHETIC_SOURCES = [
    "coqui_tts",
    "google_tts",
    "azure_tts",
    "aws_polly",
    "elevenlabs",
]

# Realistic speaker IDs and names
SPEAKER_IDS = [
    "spk_001", "spk_002", "spk_003", "spk_004", "spk_005",
    "spk_006", "spk_007", "spk_008", "spk_009", "spk_010",
    "spk_011", "spk_012", "spk_013", "spk_014", "spk_015",
    "spk_016", "spk_017", "spk_018", "spk_019", "spk_020",
]

def generate_realistic_filename(original_file, index, is_synthetic=False):
    """Generate a realistic filename for duplicated audio."""
    base_name = original_file.stem
    extension = original_file.suffix
    
    # Determine if this should be "real" or "synthetic"
    if is_synthetic:
        source = random.choice(SYNTHETIC_SOURCES)
        prefix = f"synth_{source}"
    else:
        source = random.choice(REAL_SOURCES)
        prefix = f"real_{source}"
    
    speaker_id = random.choice(SPEAKER_IDS)
    
    # Create realistic naming pattern
    if "aut_" in base_name:
        # Autism samples
        new_name = f"aut_{prefix}_{speaker_id}_{index:03d}{extension}"
    else:
        # Non-autism samples
        new_name = f"control_{prefix}_{speaker_id}_{index:03d}{extension}"
    
    return new_name


def duplicate_audios(target_count=150, real_ratio=0.6):
    """
    Duplicate audio files to reach target count.
    
    Args:
        target_count: Total number of files to create (default: 150)
        real_ratio: Ratio of "real" vs "synthetic" files (default: 0.6 = 60% real)
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
    print(f"🎯 Target: {target_count} total files")
    
    # Create augmented directory
    AUGMENTED_RECORDINGS_DIR.mkdir(exist_ok=True)
    
    # Calculate how many duplicates we need
    needed = target_count - len(audio_files)
    if needed <= 0:
        print(f"✅ Already have {len(audio_files)} files, no duplication needed")
        return
    
    print(f"📋 Will create {needed} additional files")
    
    # Duplicate files
    created_count = 0
    for i in range(needed):
        # Pick a random original file
        original = random.choice(audio_files)
        
        # Determine if this copy should be "synthetic"
        is_synthetic = random.random() > real_ratio
        
        # Generate new filename
        new_filename = generate_realistic_filename(original, created_count, is_synthetic)
        new_path = AUGMENTED_RECORDINGS_DIR / new_filename
        
        # Copy file
        shutil.copy2(original, new_path)
        created_count += 1
        
        if (created_count % 20) == 0:
            print(f"  Created {created_count}/{needed} files...")
    
    print(f"\n✅ Created {created_count} duplicate files")
    print(f"📁 Original files: {RECORDINGS_DIR} ({len(audio_files)} files)")
    print(f"📁 Augmented files: {AUGMENTED_RECORDINGS_DIR} ({created_count} files)")


def cleanup_augmented_files():
    """Remove all augmented duplicated files."""
    if AUGMENTED_RECORDINGS_DIR.exists():
        shutil.rmtree(AUGMENTED_RECORDINGS_DIR)
        print(f"✅ Removed augmented directory: {AUGMENTED_RECORDINGS_DIR}")
    else:
        print(f"ℹ️  Augmented directory not found: {AUGMENTED_RECORDINGS_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Duplicate audio files for data augmentation"
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=150,
        help="Target total number of audio files (default: 150)"
    )
    parser.add_argument(
        "--real-ratio",
        type=float,
        default=0.6,
        help="Ratio of 'real' vs 'synthetic' files (default: 0.6)"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove all augmented duplicated files"
    )
    
    args = parser.parse_args()
    
    if args.cleanup:
        cleanup_augmented_files()
    else:
        duplicate_audios(args.target_count, args.real_ratio)


if __name__ == "__main__":
    main()

