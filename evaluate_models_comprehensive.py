#!/usr/bin/env python3
"""
Comprehensive Model Evaluation with Training Curves and Visualizations
======================================================================

This script evaluates trained models using real data and generates comprehensive
evaluation metrics, training curves, confusion matrices, and ROC curves.

Usage:
    python evaluate_models_comprehensive.py
"""

import os
import json
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
)
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"
RESULTS_DIR = PROJECT_ROOT / "results"
EVAL_RESULTS_DIR = RESULTS_DIR / "evaluation"

# Set style for professional-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Model name mapping
MODEL_NAME_MAP = {
    "rf.pkl": "Random Forest",
    "svm.pkl": "SVM",
    "nb.pkl": "Naive Bayes",
    "ann.pkl": "MLP (Neural Network)"
}

MODEL_SHORT_MAP = {
    "rf.pkl": "rf",
    "svm.pkl": "svm",
    "nb.pkl": "nb",
    "ann.pkl": "ann"
}


def load_and_average_features(file_list):
    """Load MFCC features and average across time dimension."""
    data_list = []
    for file_list_item in sorted(file_list):
        mfcc = np.load(FEATURES_DIR / file_list_item)
        avg_features = np.mean(mfcc, axis=1)
        data_list.append(avg_features)
    return np.vstack(data_list)


def load_data():
    """Load all data for evaluation."""
    if not FEATURES_DIR.exists():
        raise FileNotFoundError(
            f"Features directory not found: {FEATURES_DIR}\n"
            f"Please run 01_extract_features.py first to generate features."
        )
    
    all_files = os.listdir(FEATURES_DIR)
    autistic_files = [f for f in all_files if f.startswith("aut_") and f.endswith(".npy")]
    non_autistic_files = [f for f in all_files if f.startswith("split-") and f.endswith(".npy")]
    
    if not autistic_files:
        raise ValueError(f"No autism feature files found in {FEATURES_DIR}")
    if not non_autistic_files:
        raise ValueError(f"No non-autism feature files found in {FEATURES_DIR}")
    
    autistic_data = load_and_average_features(autistic_files)
    non_autistic_data = load_and_average_features(non_autistic_files)
    
    X = np.vstack((autistic_data, non_autistic_data))
    y = np.hstack((np.ones(autistic_data.shape[0]), np.zeros(non_autistic_data.shape[0])))
    
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


def plot_confusion_matrix(cm, model_name, save_path):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Control', 'ASD'],
                yticklabels=['Control', 'ASD'],
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 Confusion matrix saved to: {save_path}")


def plot_roc_curve(y_test, y_proba, roc_auc, model_name, save_path):
    """Generate ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 ROC curve saved to: {save_path}")


def plot_metrics_comparison(all_metrics, save_path):
    """Plot comparison of all models."""
    models = list(all_metrics.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, metric in enumerate(metrics_names):
        values = [all_metrics[model][metric] for model in models]
        ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(fontsize=10)
    ax.set_ylim([0.0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 Metrics comparison saved to: {save_path}")


def evaluate_model(model, model_name, model_short, X_test, y_test):
    """Evaluate a single model."""
    print(f"\n{'='*70}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*70}")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Probabilities (if available)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        has_proba = True
    except AttributeError:
        y_proba = None
        has_proba = False
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
    
    print(f"\n📊 Performance Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    if has_proba:
        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"  ROC-AUC:   {roc_auc:.4f}")
    else:
        roc_auc = None
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Autism', 'Autism']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n🔢 Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Non-Aut  Autism")
    print(f"Actual Non-Aut   {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"       Autism    {cm[1,0]:4d}   {cm[1,1]:4d}")
    
    # Save confusion matrix plot
    cm_path = EVAL_RESULTS_DIR / f"confusion_matrix_{model_short}.png"
    plot_confusion_matrix(cm, model_name, cm_path)
    
    # Save ROC curve (if probabilities available)
    if has_proba:
        roc_path = EVAL_RESULTS_DIR / f"roc_curve_{model_short}.png"
        plot_roc_curve(y_test, y_proba, roc_auc, model_name, roc_path)
    
    return {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'roc_auc': round(roc_auc, 4) if roc_auc else None,
        'confusion_matrix': cm.tolist()
    }


def print_evaluation_summary(all_metrics):
    """Print final evaluation summary table."""
    print(f"\n{'='*70}")
    print("📊 Final Evaluation Summary")
    print(f"{'='*70}")
    print(f"\n{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
    print("-" * 85)
    
    for model_name, metrics in all_metrics.items():
        roc_auc_str = f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "N/A"
        print(f"{model_name:<25} "
              f"{metrics['accuracy']:<12.4f} "
              f"{metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} "
              f"{metrics['f1_score']:<12.4f} "
              f"{roc_auc_str:<12}")
    
    # Find best model
    best_model = max(all_metrics.items(), key=lambda x: x[1]['accuracy'])
    print("-" * 85)
    print(f"\n🏆 Best Model: {best_model[0]}")
    print(f"   Accuracy: {best_model[1]['accuracy']:.4f} ({best_model[1]['accuracy']*100:.2f}%)")
    print(f"   F1-Score: {best_model[1]['f1_score']:.4f}")
    if best_model[1]['roc_auc']:
        print(f"   ROC-AUC:  {best_model[1]['roc_auc']:.4f}")


def main():
    """Main evaluation pipeline."""
    # Create results directory
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("ASD/ADHD Detection - Comprehensive Model Evaluation")
    print("="*70)
    
    # Load data
    print("\n📂 Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    
    # Find all models
    if not MODELS_DIR.exists():
        raise FileNotFoundError(f"Models directory not found: {MODELS_DIR}\n"
                              f"Please run 02_train_models.py first to train models.")
    
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
    if not model_files:
        raise FileNotFoundError(f"No models found in {MODELS_DIR}\n"
                              f"Please run 02_train_models.py first to train models.")
    
    print(f"\n📦 Found {len(model_files)} models to evaluate")
    
    # Evaluate each model
    all_metrics = {}
    for model_file in sorted(model_files):
        model_path = MODELS_DIR / model_file
        model = joblib.load(model_path)
        
        model_name = MODEL_NAME_MAP.get(model_file, model_file.replace('.pkl', ''))
        model_short = MODEL_SHORT_MAP.get(model_file, model_file.replace('.pkl', ''))
        
        metrics = evaluate_model(model, model_name, model_short, X_test, y_test)
        all_metrics[model_name] = metrics
    
    # Generate metrics comparison
    print(f"\n{'='*70}")
    print("Generating Comparison Visualizations...")
    print("="*70)
    comparison_path = EVAL_RESULTS_DIR / "metrics_comparison.png"
    plot_metrics_comparison(all_metrics, comparison_path)
    
    # Print summary
    print_evaluation_summary(all_metrics)
    
    # Save JSON results
    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset_info': {
            'total_samples': len(X_train) + len(X_test),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_dimension': X_test.shape[1]
        },
        'models': {}
    }
    
    for model_name, model_short in zip(MODEL_NAME_MAP.values(), MODEL_SHORT_MAP.values()):
        if model_name in all_metrics:
            results['models'][model_short] = all_metrics[model_name]
    
    json_path = EVAL_RESULTS_DIR / "evaluation_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("Evaluation Complete!")
    print(f"{'='*70}")
    print(f"\n📁 All results saved to: {EVAL_RESULTS_DIR}")
    print(f"   - Evaluation metrics: {json_path}")
    print(f"   - Confusion matrices: {EVAL_RESULTS_DIR}/confusion_matrix_*.png")
    print(f"   - ROC curves: {EVAL_RESULTS_DIR}/roc_curve_*.png")
    print(f"   - Metrics comparison: {comparison_path}")


if __name__ == "__main__":
    main()

