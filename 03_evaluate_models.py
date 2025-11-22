#!/usr/bin/env python3
"""
Evaluate trained models and display comprehensive metrics.
"""

import os
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"
RESULTS_DIR = PROJECT_ROOT / "results"


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
    all_files = os.listdir(FEATURES_DIR)
    autistic_files = [f for f in all_files if f.startswith("aut_") and f.endswith(".npy")]
    non_autistic_files = [f for f in all_files if f.startswith("split-") and f.endswith(".npy")]
    
    autistic_data = load_and_average_features(autistic_files)
    non_autistic_data = load_and_average_features(non_autistic_files)
    
    X = np.vstack((autistic_data, non_autistic_data))
    y = np.hstack((np.ones(autistic_data.shape[0]), np.zeros(non_autistic_data.shape[0])))
    
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


def evaluate_model(model, model_name, X_test, y_test):
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
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Autism', 'Autism']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n🔢 Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Non-Aut  Autism")
    print(f"Actual Non-Aut   {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"       Autism    {cm[1,0]:4d}   {cm[1,1]:4d}")
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Autism', 'Autism'],
                yticklabels=['Non-Autism', 'Autism'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    cm_path = RESULTS_DIR / f"confusion_matrix_{model_name.replace('.pkl', '')}.png"
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Confusion matrix saved to: {cm_path}")
    plt.close()
    
    # ROC Curve (if probabilities available)
    if has_proba:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        
        roc_path = RESULTS_DIR / f"roc_curve_{model_name.replace('.pkl', '')}.png"
        plt.savefig(roc_path, dpi=150, bbox_inches='tight')
        print(f"💾 ROC curve saved to: {roc_path}")
        plt.close()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist(),
        'roc_auc': roc_auc if has_proba else None
    }


def main():
    """Main evaluation pipeline."""
    print("="*70)
    print("ASD/ADHD Detection - Model Evaluation")
    print("="*70)
    
    # Load data
    print("\n📂 Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"  Test set: {len(X_test)} samples")
    
    # Find all models
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
    if not model_files:
        raise FileNotFoundError(f"No models found in {MODELS_DIR}")
    
    print(f"\n📦 Found {len(model_files)} models to evaluate")
    
    # Evaluate each model
    results = {}
    for model_file in sorted(model_files):
        model_path = MODELS_DIR / model_file
        model = joblib.load(model_path)
        results[model_file] = evaluate_model(model, model_file, X_test, y_test)
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 Evaluation Summary")
    print(f"{'='*70}")
    
    print(f"\n{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 70)
    for model_name, metrics in results.items():
        print(f"{model_name:<20} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")
    
    # Best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🏆 Best Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")
    
    # Save results
    import json
    results_path = RESULTS_DIR / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {results_path}")


if __name__ == "__main__":
    main()

