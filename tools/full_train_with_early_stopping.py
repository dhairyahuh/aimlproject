"""
Full training script with early stopping, learning rate scheduler, and comprehensive evaluation.
Saves trained model, training plots, metrics, and evaluation results.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Add parent dirs to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve, auc
)
import json
import pickle

# Configuration
DATA_DIR = Path(__file__).parent.parent.parent / 'data'
OUTPUT_DIR = Path(__file__).parent.parent / 'results' / 'full_training'
EXTERNAL_HELPERS = Path(__file__).parent.parent / 'external_helpers'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EXTERNAL_HELPERS, exist_ok=True)

# ============================================================================
# Load Data
# ============================================================================
print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

X_train = np.load(DATA_DIR / 'X_train.npy')
X_val = np.load(DATA_DIR / 'X_val.npy')
X_test = np.load(DATA_DIR / 'X_test.npy')
y_train = np.load(DATA_DIR / 'y_train.npy')
y_val = np.load(DATA_DIR / 'y_val.npy')
y_test = np.load(DATA_DIR / 'y_test.npy')

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

# ============================================================================
# Normalize Data
# ============================================================================
print("\n" + "="*70)
print("NORMALIZING DATA")
print("="*70)

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)

print("Data normalized using StandardScaler")

# Save scaler
scaler_path = EXTERNAL_HELPERS / 'data_scaler.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Scaler saved to: {scaler_path}")

# ============================================================================
# Build Model
# ============================================================================
print("\n" + "="*70)
print("BUILDING MODEL")
print("="*70)

model = models.Sequential([
    layers.Dense(256, input_dim=input_dim, kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),
    
    layers.Dense(128, kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),
    
    layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
    
    layers.Dense(32, kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
    
    layers.Dense(n_classes, activation='softmax')
])

model.summary()

# ============================================================================
# Compile Model
# ============================================================================
print("\n" + "="*70)
print("COMPILING MODEL")
print("="*70)

optimizer = optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print("Model compiled with Adam optimizer (lr=0.001)")

# ============================================================================
# Define Callbacks
# ============================================================================
print("\n" + "="*70)
print("SETTING UP CALLBACKS")
print("="*70)

checkpoint_path = OUTPUT_DIR / 'best_model.keras'
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = callbacks.ModelCheckpoint(
    str(checkpoint_path),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

lr_schedule = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

print("Callbacks configured:")
print("  - EarlyStopping (patience=15, monitor=val_loss)")
print("  - ModelCheckpoint (save best by val_accuracy)")
print("  - ReduceLROnPlateau (factor=0.5, patience=5)")

# ============================================================================
# Train Model
# ============================================================================
print("\n" + "="*70)
print("TRAINING MODEL")
print("="*70)

history = model.fit(
    X_train_norm, y_train,
    validation_data=(X_val_norm, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, model_checkpoint, lr_schedule],
    verbose=1
)

print(f"\nTraining completed. Total epochs: {len(history.history['loss'])}")

# ============================================================================
# Evaluate Model
# ============================================================================
print("\n" + "="*70)
print("EVALUATING MODEL")
print("="*70)

# Load best model
best_model = keras.models.load_model(str(checkpoint_path))

# Train evaluation
train_loss, train_acc = best_model.evaluate(X_train_norm, y_train, verbose=0)
print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

# Validation evaluation
val_loss, val_acc = best_model.evaluate(X_val_norm, y_val, verbose=0)
print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

# Test evaluation
test_loss, test_acc = best_model.evaluate(X_test_norm, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Predictions
y_train_pred = best_model.predict(X_train_norm, verbose=0).argmax(axis=1)
y_val_pred = best_model.predict(X_val_norm, verbose=0).argmax(axis=1)
y_test_pred = best_model.predict(X_test_norm, verbose=0).argmax(axis=1)

y_train_proba = best_model.predict(X_train_norm, verbose=0)
y_val_proba = best_model.predict(X_val_norm, verbose=0)
y_test_proba = best_model.predict(X_test_norm, verbose=0)

# Classification reports
print("\n" + "-"*70)
print("TRAIN CLASSIFICATION REPORT")
print("-"*70)
print(classification_report(y_train, y_train_pred))

print("\n" + "-"*70)
print("VALIDATION CLASSIFICATION REPORT")
print("-"*70)
print(classification_report(y_val, y_val_pred))

print("\n" + "-"*70)
print("TEST CLASSIFICATION REPORT")
print("-"*70)
test_report = classification_report(y_test, y_test_pred, output_dict=True)
print(classification_report(y_test, y_test_pred))

# Confusion matrix
cm_test = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix (Test):")
print(cm_test)

# ============================================================================
# Save Results
# ============================================================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Save model
model_path = EXTERNAL_HELPERS / 'full_trained_model.keras'
best_model.save(str(model_path))
print(f"Model saved to: {model_path}")

# Save history
history_path = OUTPUT_DIR / 'training_history.pkl'
with open(history_path, 'wb') as f:
    pickle.dump(history.history, f)
print(f"Training history saved to: {history_path}")

# Save metrics
metrics = {
    'train_loss': float(train_loss),
    'train_accuracy': float(train_acc),
    'val_loss': float(val_loss),
    'val_accuracy': float(val_acc),
    'test_loss': float(test_loss),
    'test_accuracy': float(test_acc),
    'epochs_trained': len(history.history['loss']),
    'test_classification_report': test_report,
}

metrics_path = OUTPUT_DIR / 'metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved to: {metrics_path}")

# ============================================================================
# Generate Plots
# ============================================================================
print("\n" + "="*70)
print("GENERATING PLOTS")
print("="*70)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 5)

# Plot 1: Loss and Accuracy curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = OUTPUT_DIR / 'training_curves.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Training curves saved to: {plot_path}")
plt.close()

# Plot 2: Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', cbar=True, 
            xticklabels=range(n_classes), yticklabels=range(n_classes))
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
plt.title('Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
plt.tight_layout()
cm_path = OUTPUT_DIR / 'confusion_matrix_test.png'
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
print(f"Confusion matrix saved to: {cm_path}")
plt.close()

# Plot 3: Per-class metrics
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred)

axes[0].bar(range(n_classes), precision, color='skyblue', alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Class', fontsize=11)
axes[0].set_ylabel('Precision', fontsize=11)
axes[0].set_title('Per-Class Precision', fontsize=12, fontweight='bold')
axes[0].set_ylim([0, 1.0])
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].bar(range(n_classes), recall, color='lightgreen', alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Class', fontsize=11)
axes[1].set_ylabel('Recall', fontsize=11)
axes[1].set_title('Per-Class Recall', fontsize=12, fontweight='bold')
axes[1].set_ylim([0, 1.0])
axes[1].grid(True, alpha=0.3, axis='y')

axes[2].bar(range(n_classes), f1, color='lightsalmon', alpha=0.7, edgecolor='black')
axes[2].set_xlabel('Class', fontsize=11)
axes[2].set_ylabel('F1-Score', fontsize=11)
axes[2].set_title('Per-Class F1-Score', fontsize=12, fontweight='bold')
axes[2].set_ylim([0, 1.0])
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
metrics_plot_path = OUTPUT_DIR / 'per_class_metrics_test.png'
plt.savefig(metrics_plot_path, dpi=150, bbox_inches='tight')
print(f"Per-class metrics saved to: {metrics_plot_path}")
plt.close()

# ============================================================================
# Summary Report
# ============================================================================
print("\n" + "="*70)
print("SUMMARY REPORT")
print("="*70)

summary_text = f"""
{'='*70}
ASD/ADHD DETECTION - FULL TRAINING SUMMARY
{'='*70}

TRAINING CONFIGURATION:
  - Input Dimension: {input_dim}
  - Number of Classes: {n_classes}
  - Total Epochs Trained: {len(history.history['loss'])}
  - Batch Size: 32
  - Optimizer: Adam (initial lr=0.001)
  - Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

FINAL RESULTS:
  - Train Accuracy: {train_acc:.4f} | Loss: {train_loss:.4f}
  - Val Accuracy:   {val_acc:.4f} | Loss: {val_loss:.4f}
  - Test Accuracy:  {test_acc:.4f} | Loss: {test_loss:.4f}

TEST SET PERFORMANCE:
  - Weighted Precision: {test_report['weighted avg']['precision']:.4f}
  - Weighted Recall:    {test_report['weighted avg']['recall']:.4f}
  - Weighted F1-Score:  {test_report['weighted avg']['f1-score']:.4f}

CLASS DISTRIBUTION:
  - Train: {dict(zip(*np.unique(y_train, return_counts=True)))}
  - Val:   {dict(zip(*np.unique(y_val, return_counts=True)))}
  - Test:  {dict(zip(*np.unique(y_test, return_counts=True)))}

SAVED ARTIFACTS:
  - Model: {model_path}
  - Scaler: {scaler_path}
  - Metrics: {metrics_path}
  - History: {history_path}
  - Training Curves: {plot_path}
  - Confusion Matrix: {cm_path}
  - Per-Class Metrics: {metrics_plot_path}

TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}
"""

print(summary_text)

# Save summary to file
summary_path = OUTPUT_DIR / 'training_summary.txt'
with open(summary_path, 'w') as f:
    f.write(summary_text)
print(f"Summary saved to: {summary_path}")

print("\n✓ Full training completed successfully!")
