#!/usr/bin/env python3
"""
Train classical ML models (Random Forest, SVM, Naive Bayes, MLP) on MFCC features.
Uses real autism recordings from recordings/ folder.
"""

import os
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
import joblib

# ---------------------------------------------------------------------------
# Configuration - All paths relative to ASD_ADHD_Detection folder
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"
RESULTS_DIR = PROJECT_ROOT / "results"

# Model configurations
MODEL_CONFIGS = {
    "rf.pkl": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "svm.pkl": SVC(kernel="rbf", C=1, gamma="scale", random_state=42, probability=True),
    "nb.pkl": GaussianNB(),
    "ann.pkl": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
}


def load_and_average_features(file_list):
    """Load MFCC features and average across time dimension."""
    data_list = []
    for file_name in sorted(file_list):
        mfcc = np.load(FEATURES_DIR / file_name)
        # Average across time frames to get a single feature vector per audio file
        avg_features = np.mean(mfcc, axis=1)
        data_list.append(avg_features)
    return np.vstack(data_list)


def build_training_set():
    """Build training set from feature files."""
    if not FEATURES_DIR.exists():
        raise FileNotFoundError(
            f"Features directory not found: {FEATURES_DIR}\n"
            f"Please run 01_extract_features.py first to generate features."
        )
    
    # Find feature files
    all_files = os.listdir(FEATURES_DIR)
    autistic_files = [f for f in all_files if f.startswith("aut_") and f.endswith(".npy")]
    non_autistic_files = [f for f in all_files if f.startswith("split-") and f.endswith(".npy")]
    
    if not autistic_files:
        raise ValueError(f"No autism feature files found in {FEATURES_DIR}")
    if not non_autistic_files:
        raise ValueError(f"No non-autism feature files found in {FEATURES_DIR}")
    
    print(f"Found {len(autistic_files)} autism samples and {len(non_autistic_files)} non-autism samples")
    
    # Load and average features
    autistic_data = load_and_average_features(autistic_files)
    non_autistic_data = load_and_average_features(non_autistic_files)
    
    # Combine data
    X = np.vstack((autistic_data, non_autistic_data))
    y = np.hstack((np.ones(autistic_data.shape[0]), np.zeros(non_autistic_data.shape[0])))
    
    print(f"Total samples: {len(X)}")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"Class distribution: Autism={int(y.sum())}, Non-autism={len(y) - int(y.sum())}")
    
    # Split into train/test
    # WARNING: With small datasets (<50 samples), test set will be very small
    # and 100% accuracy is likely overfitting. Use K-fold CV for reliable metrics.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    if len(X_test) < 15:
        print(f"\n⚠️  WARNING: Test set is very small ({len(X_test)} samples)!")
        print(f"   100% accuracy on small test sets often indicates overfitting.")
        print(f"   For reliable metrics, run: python 04_kfold_validation.py")
    
    return X_train, X_test, y_train, y_test


def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """Train a model and evaluate it."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}...")
    print(f"{'='*60}")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
    
    print(f"\nMetrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Autism', 'Autism']))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                Predicted")
    print(f"              Non-Aut  Autism")
    print(f"Actual Non-Aut   {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"       Autism    {cm[1,0]:4d}   {cm[1,1]:4d}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist()
    }


def main():
    """Main training pipeline."""
    print("="*60)
    print("ASD/ADHD Detection - Model Training")
    print("="*60)
    
    # Create directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Build training set
    X_train, X_test, y_train, y_test = build_training_set()
    
    # Train all models
    results = {}
    for model_filename, model in MODEL_CONFIGS.items():
        model_path = MODELS_DIR / model_filename
        metrics = train_and_evaluate_model(model, model_filename, X_train, X_test, y_train, y_test)
        
        # Save model
        joblib.dump(model, model_path)
        print(f"\n✅ Model saved to: {model_path}")
        
        results[model_filename] = metrics
    
    # Save summary
    import json
    summary_path = RESULTS_DIR / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"\nModels saved to: {MODELS_DIR}")
    print(f"Results saved to: {summary_path}")
    
    # Print best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nBest Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")
    
    # Overfitting warning
    if best_model[1]['accuracy'] >= 0.95 and len(X_test) < 15:
        print(f"\n⚠️  CAUTION: Very high accuracy ({best_model[1]['accuracy']:.2%}) on small test set!")
        print(f"   This may indicate overfitting. For more reliable evaluation:")
        print(f"   - Run K-fold cross-validation: python 04_kfold_validation.py")
        print(f"   - Run Leave-One-Out CV: python 08_leave_one_out_cv.py")
        print(f"   - Collect more data if possible")
        print(f"   - Consider simpler models or regularization")


if __name__ == "__main__":
    main()

