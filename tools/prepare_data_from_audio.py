"""
Data Preparation Script
========================
Loads audio files from the root data folder, extracts features,
and creates train/val/test splits for ASD/ADHD detection.

This script processes audio files and creates the necessary data files
for training the MLP classifier.
"""

import os
import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.feature_extraction.feature_aggregator import FeatureAggregator

# Paths
ROOT_DATA_DIR = project_root.parent / 'data'
OUTPUT_DIR = ROOT_DATA_DIR
PROJECT_DATA_DIR = project_root / 'data' / 'processed'

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROJECT_DATA_DIR.mkdir(parents=True, exist_ok=True)


def find_audio_files(data_dir: Path) -> list:
    """
    Find all audio files in the data directory.
    
    Args:
        data_dir: Root data directory
        
    Returns:
        List of audio file paths
    """
    audio_files = []
    audio_extensions = ['.wav', '.mp3', '.ogg', '.flac']
    
    # Search in subdirectories
    for ext in audio_extensions:
        audio_files.extend(list(data_dir.rglob(f'*{ext}')))
    
    print(f"Found {len(audio_files)} audio files")
    return sorted(audio_files)


def get_label_from_filename(filename: str) -> str:
    """
    Extract label from filename or directory structure.
    
    This is a placeholder function. You'll need to adapt this based on
    how your audio files are organized/labeled.
    
    Possible strategies:
    1. Check parent directory name (e.g., 'ASD/', 'ADHD/', 'Healthy/')
    2. Check filename prefix (e.g., 'ASD_001.wav')
    3. Use a separate label file/mapping
    
    Args:
        filename: Audio file path
        
    Returns:
        Label string ('ASD', 'ADHD', or 'Healthy')
    """
    filename_lower = filename.lower()
    path_parts = Path(filename).parts
    
    # Strategy 1: Check directory names
    for part in path_parts:
        part_lower = part.lower()
        if 'asd' in part_lower or 'autism' in part_lower:
            return 'ASD'
        elif 'adhd' in part_lower:
            return 'ADHD'
        elif 'healthy' in part_lower or 'normal' in part_lower or 'control' in part_lower:
            return 'Healthy'
    
    # Strategy 2: Check filename
    if 'asd' in filename_lower or 'autism' in filename_lower:
        return 'ASD'
    elif 'adhd' in filename_lower:
        return 'ADHD'
    elif 'healthy' in filename_lower or 'normal' in filename_lower:
        return 'Healthy'
    
    # Default: If you have pre-labeled data, you might want to raise an error
    # For now, we'll return None and filter these out
    return None


def extract_features_from_audio(audio_files: list, labels: list = None,
                               sample_size: int = None) -> tuple:
    """
    Extract features from audio files.
    
    Args:
        audio_files: List of audio file paths
        labels: List of labels (optional, will try to infer if None)
        sample_size: Maximum number of files to process (for testing)
        
    Returns:
        Tuple of (features_array, labels_array)
    """
    print("\n" + "="*70)
    print("EXTRACTING FEATURES FROM AUDIO FILES")
    print("="*70)
    
    # Initialize feature extractor
    feature_extractor = FeatureAggregator(sr=16000, n_mfcc=13, use_pca=False)
    
    # Limit sample size if specified
    if sample_size and sample_size < len(audio_files):
        print(f"Processing {sample_size} files (sample mode)")
        audio_files = audio_files[:sample_size]
    
    features_list = []
    labels_list = []
    failed_files = []
    
    for i, audio_file in enumerate(audio_files):
        try:
            # Extract label
            if labels is None:
                label = get_label_from_filename(str(audio_file))
                if label is None:
                    print(f"  ⚠ Skipping {audio_file.name}: No label found")
                    continue
            else:
                label = labels[i]
            
            # Extract features
            features = feature_extractor.extract_all_features(str(audio_file))
            features_list.append(features)
            labels_list.append(label)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(audio_files)} files...")
                
        except Exception as e:
            print(f"  ✗ Error processing {audio_file.name}: {e}")
            failed_files.append(audio_file)
            continue
    
    if failed_files:
        print(f"\n⚠ Failed to process {len(failed_files)} files")
    
    features_array = np.array(features_list)
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels_array = label_encoder.fit_transform(labels_list)
    
    print(f"\n✓ Feature extraction complete!")
    print(f"  Total samples: {len(features_list)}")
    print(f"  Feature dimension: {features_array.shape[1]}")
    print(f"  Label distribution: {dict(zip(*np.unique(labels_array, return_counts=True)))}")
    
    # Save label encoder
    le_path = OUTPUT_DIR / 'le_classes.npy'
    np.save(le_path, label_encoder.classes_)
    print(f"  Label encoder saved to: {le_path}")
    
    return features_array, labels_array, label_encoder


def create_train_val_test_split(X: np.ndarray, y: np.ndarray,
                                train_ratio: float = 0.7,
                                val_ratio: float = 0.15,
                                test_ratio: float = 0.15,
                                random_state: int = 42) -> tuple:
    """
    Create train/validation/test splits.
    
    Args:
        X: Feature array
        y: Label array
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print("\n" + "="*70)
    print("CREATING TRAIN/VAL/TEST SPLITS")
    print("="*70)
    
    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=y
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_size),
        random_state=random_state,
        stratify=y_temp
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Val set:   {X_val.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def normalize_features(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> tuple:
    """
    Normalize features using StandardScaler.
    
    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        
    Returns:
        Tuple of normalized arrays and scaler
    """
    print("\n" + "="*70)
    print("NORMALIZING FEATURES")
    print("="*70)
    
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test)
    
    print("✓ Features normalized using StandardScaler")
    
    return X_train_norm, X_val_norm, X_test_norm, scaler


def save_data_splits(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                    y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                    scaler: StandardScaler, output_dir: Path) -> None:
    """
    Save data splits to disk.
    
    Args:
        X_train, X_val, X_test: Feature arrays
        y_train, y_val, y_test: Label arrays
        scaler: Fitted scaler
        output_dir: Output directory
    """
    print("\n" + "="*70)
    print("SAVING DATA SPLITS")
    print("="*70)
    
    # Save feature arrays
    np.save(output_dir / 'X_train.npy', X_train)
    np.save(output_dir / 'X_val.npy', X_val)
    np.save(output_dir / 'X_test.npy', X_test)
    
    # Save label arrays
    np.save(output_dir / 'y_train.npy', y_train)
    np.save(output_dir / 'y_val.npy', y_val)
    np.save(output_dir / 'y_test.npy', y_test)
    
    # Save scaler
    scaler_path = output_dir / 'data_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"✓ Data splits saved to: {output_dir}")
    print(f"  - X_train.npy: {X_train.shape}")
    print(f"  - X_val.npy: {X_val.shape}")
    print(f"  - X_test.npy: {X_test.shape}")
    print(f"  - y_train.npy: {y_train.shape}")
    print(f"  - y_val.npy: {y_val.shape}")
    print(f"  - y_test.npy: {y_test.shape}")
    print(f"  - data_scaler.pkl: Scaler saved")


def main():
    """Main data preparation pipeline."""
    print("="*70)
    print("ASD/ADHD DETECTION - DATA PREPARATION")
    print("="*70)
    
    # Step 1: Find audio files
    print(f"\nSearching for audio files in: {ROOT_DATA_DIR}")
    audio_files = find_audio_files(ROOT_DATA_DIR)
    
    if len(audio_files) == 0:
        print("⚠ No audio files found!")
        print("Please ensure audio files are in the data directory.")
        return
    
    # Step 2: Extract features
    # Note: You may want to use sample_size for initial testing
    # Remove sample_size parameter for full processing
    X, y, label_encoder = extract_features_from_audio(
        audio_files,
        labels=None,  # Will infer from filenames
        sample_size=None  # Set to a number (e.g., 100) for testing
    )
    
    if len(X) == 0:
        print("⚠ No valid samples extracted!")
        return
    
    # Step 3: Create splits
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_split(
        X, y,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42
    )
    
    # Step 4: Normalize features
    X_train_norm, X_val_norm, X_test_norm, scaler = normalize_features(
        X_train, X_val, X_test
    )
    
    # Step 5: Save data splits
    save_data_splits(
        X_train_norm, X_val_norm, X_test_norm,
        y_train, y_val, y_test,
        scaler,
        OUTPUT_DIR
    )
    
    print("\n" + "="*70)
    print("✓ DATA PREPARATION COMPLETE!")
    print("="*70)
    print(f"\nYou can now train the model using:")
    print(f"  python {project_root}/tools/train_model.py")


if __name__ == "__main__":
    main()

