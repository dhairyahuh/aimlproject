"""
Model Training Script
=====================
Trains the MLP classifier on prepared data for ASD/ADHD detection.
"""

import os
import sys
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

from src.models.mlp_classifier import MLPClassifier

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Paths
ROOT_DATA_DIR = project_root.parent / 'data'
MODELS_DIR = project_root / 'models' / 'saved'
RESULTS_DIR = project_root / 'results' / 'training'

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load preprocessed data splits."""
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    X_train = np.load(ROOT_DATA_DIR / 'X_train.npy')
    X_val = np.load(ROOT_DATA_DIR / 'X_val.npy')
    X_test = np.load(ROOT_DATA_DIR / 'X_test.npy')
    y_train = np.load(ROOT_DATA_DIR / 'y_train.npy')
    y_val = np.load(ROOT_DATA_DIR / 'y_val.npy')
    y_test = np.load(ROOT_DATA_DIR / 'y_test.npy')
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape:   {X_val.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_val shape:   {y_val.shape}")
    print(f"y_test shape:  {y_test.shape}")
    
    n_classes = len(np.unique(y_train))
    input_dim = X_train.shape[1]
    
    print(f"\nInput dimension: {input_dim}")
    print(f"Number of classes: {n_classes}")
    print(f"Class distribution (train): {np.bincount(y_train)}")
    print(f"Class distribution (val):   {np.bincount(y_val)}")
    print(f"Class distribution (test):  {np.bincount(y_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, input_dim, n_classes


def train_model(X_train, y_train, X_val, y_val, input_dim, n_classes):
    """Train the MLP model."""
    print("\n" + "="*70)
    print("TRAINING MODEL")
    print("="*70)
    
    # Initialize model
    model = MLPClassifier(input_dim=input_dim, n_classes=n_classes)
    
    # Build model
    model.build(
        hidden_layers=[128, 64, 32],
        dropout_rate=0.3,
        l2_reg=1e-4,
        learning_rate=0.001
    )
    
    # Setup callbacks
    checkpoint_path = MODELS_DIR / 'best_model.keras'
    callbacks_list = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            str(checkpoint_path),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train model
    history = model.train(
        X_train, y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=100,
        batch_size=32,
        callbacks_list=callbacks_list,
        verbose=1
    )
    
    return model, history


def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Evaluate model on all splits."""
    print("\n" + "="*70)
    print("EVALUATING MODEL")
    print("="*70)
    
    # Evaluate on all splits
    train_metrics = model.evaluate(X_train, y_train)
    val_metrics = model.evaluate(X_val, y_val)
    test_metrics = model.evaluate(X_test, y_test)
    
    print("\n" + "-"*70)
    print("TRAIN SET METRICS")
    print("-"*70)
    print(f"Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"Precision: {train_metrics['precision']:.4f}")
    print(f"Recall:    {train_metrics['recall']:.4f}")
    print(f"F1-Score:  {train_metrics['f1_score']:.4f}")
    
    print("\n" + "-"*70)
    print("VALIDATION SET METRICS")
    print("-"*70)
    print(f"Accuracy:  {val_metrics['accuracy']:.4f}")
    print(f"Precision: {val_metrics['precision']:.4f}")
    print(f"Recall:    {val_metrics['recall']:.4f}")
    print(f"F1-Score:  {val_metrics['f1_score']:.4f}")
    
    print("\n" + "-"*70)
    print("TEST SET METRICS")
    print("-"*70)
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1-Score:  {test_metrics['f1_score']:.4f}")
    
    print("\n" + "-"*70)
    print("TEST SET CLASSIFICATION REPORT")
    print("-"*70)
    print(classification_report(y_test, model.predict(X_test)))
    
    return train_metrics, val_metrics, test_metrics


def save_results(model, history, train_metrics, val_metrics, test_metrics):
    """Save model and training results."""
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    # Save model
    model_path = MODELS_DIR / 'asd_adhd_mlp_model.keras'
    model.save(str(model_path))
    
    # Load scaler and save with model
    scaler_path = ROOT_DATA_DIR / 'data_scaler.pkl'
    if scaler_path.exists():
        import pickle
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        model.save(str(model_path), save_scaler=True, scaler=scaler)
    
    # Save metrics
    metrics = {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    metrics_path = RESULTS_DIR / 'training_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Model saved to: {model_path}")
    print(f"✓ Metrics saved to: {metrics_path}")
    
    # Plot training curves
    plot_training_curves(history, RESULTS_DIR)
    
    # Plot confusion matrix
    plot_confusion_matrix(model, RESULTS_DIR)


def plot_training_curves(history, output_dir):
    """Plot training and validation curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    axes[0].plot(history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / 'training_curves.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Training curves saved to: {plot_path}")


def plot_confusion_matrix(model, output_dir):
    """Plot confusion matrix on test set."""
    # Load test data
    X_test = np.load(ROOT_DATA_DIR / 'X_test.npy')
    y_test = np.load(ROOT_DATA_DIR / 'y_test.npy')
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Healthy', 'ASD', 'ADHD'],
                yticklabels=['Healthy', 'ASD', 'ADHD'])
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    cm_path = output_dir / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved to: {cm_path}")


def main():
    """Main training pipeline."""
    print("="*70)
    print("ASD/ADHD DETECTION - MODEL TRAINING")
    print("="*70)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, input_dim, n_classes = load_data()
    
    # Train model
    model, history = train_model(X_train, y_train, X_val, y_val, input_dim, n_classes)
    
    # Evaluate model
    train_metrics, val_metrics, test_metrics = evaluate_model(
        model, X_train, y_train, X_val, y_val, X_test, y_test
    )
    
    # Save results
    save_results(model, history, train_metrics, val_metrics, test_metrics)
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModel saved to: {MODELS_DIR / 'asd_adhd_mlp_model.keras'}")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

