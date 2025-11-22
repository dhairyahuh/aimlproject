#!/usr/bin/env python3
"""
K-Fold Cross-Validation for robust model evaluation.
"""

import os
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features"
RESULTS_DIR = PROJECT_ROOT / "results"
N_FOLDS = 5


def load_and_average_features(file_list):
    """Load MFCC features and average across time dimension."""
    data_list = []
    for file_list_item in sorted(file_list):
        mfcc = np.load(FEATURES_DIR / file_list_item)
        avg_features = np.mean(mfcc, axis=1)
        data_list.append(avg_features)
    return np.vstack(data_list)


def load_all_data():
    """Load all data for cross-validation."""
    all_files = os.listdir(FEATURES_DIR)
    autistic_files = [f for f in all_files if f.startswith("aut_") and f.endswith(".npy")]
    non_autistic_files = [f for f in all_files if f.startswith("split-") and f.endswith(".npy")]
    
    autistic_data = load_and_average_features(autistic_files)
    non_autistic_data = load_and_average_features(non_autistic_files)
    
    X = np.vstack((autistic_data, non_autistic_data))
    y = np.hstack((np.ones(autistic_data.shape[0]), np.zeros(non_autistic_data.shape[0])))
    
    return X, y


def evaluate_fold(model, X_test, y_test, fold_num):
    """Evaluate a single fold."""
    y_pred = model.predict(X_test)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='binary', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }


def kfold_cross_validation(model_class, model_name, X, y, n_folds=N_FOLDS):
    """Perform K-Fold cross-validation."""
    print(f"\n{'='*70}")
    print(f"K-Fold Cross-Validation: {model_name}")
    print(f"{'='*70}")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n📊 Fold {fold}/{n_folds}")
        
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        # Normalize features
        scaler = StandardScaler()
        X_train_fold = scaler.fit_transform(X_train_fold)
        X_test_fold = scaler.transform(X_test_fold)
        
        # Create and train model
        if model_name == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_name == "SVM":
            model = SVC(kernel="rbf", C=1, gamma="scale", random_state=42)
        elif model_name == "Naive Bayes":
            model = GaussianNB()
        elif model_name == "MLP":
            model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
        else:
            model = model_class()
        
        model.fit(X_train_fold, y_train_fold)
        
        # Evaluate
        fold_metrics = evaluate_fold(model, X_test_fold, y_test_fold, fold)
        fold_results.append(fold_metrics)
        
        print(f"  Accuracy:  {fold_metrics['accuracy']:.4f}")
        print(f"  Precision: {fold_metrics['precision']:.4f}")
        print(f"  Recall:    {fold_metrics['recall']:.4f}")
        print(f"  F1-Score:  {fold_metrics['f1']:.4f}")
    
    # Aggregate results
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    aggregated = {}
    for metric in metrics:
        values = [r[metric] for r in fold_results]
        aggregated[f'{metric}_mean'] = np.mean(values)
        aggregated[f'{metric}_std'] = np.std(values)
        aggregated[f'{metric}_min'] = np.min(values)
        aggregated[f'{metric}_max'] = np.max(values)
    
    print(f"\n{'='*70}")
    print(f"📊 Aggregated Results ({n_folds}-Fold CV)")
    print(f"{'='*70}")
    for metric in metrics:
        mean_val = aggregated[f'{metric}_mean']
        std_val = aggregated[f'{metric}_std']
        print(f"  {metric.capitalize():<12}: {mean_val:.4f} ± {std_val:.4f}")
    
    return {
        'fold_results': fold_results,
        'aggregated': aggregated
    }


def visualize_results(all_results):
    """Visualize cross-validation results."""
    model_names = list(all_results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        means = [all_results[m]['aggregated'][f'{metric}_mean'] for m in model_names]
        stds = [all_results[m]['aggregated'][f'{metric}_std'] for m in model_names]
        
        axes[idx].bar(model_names, means, yerr=stds, capsize=5, alpha=0.7)
        axes[idx].set_title(f'{metric.capitalize()} (Mean ± Std)')
        axes[idx].set_ylabel('Score')
        axes[idx].set_ylim([0, 1])
        axes[idx].grid(True, alpha=0.3)
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plot_path = RESULTS_DIR / "kfold_cv_results.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Visualization saved to: {plot_path}")
    plt.close()


def main():
    """Main cross-validation pipeline."""
    print("="*70)
    print("ASD/ADHD Detection - K-Fold Cross-Validation")
    print("="*70)
    
    # Load data
    print("\n📂 Loading data...")
    X, y = load_all_data()
    print(f"  Total samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes: Autism={int(y.sum())}, Non-autism={len(y) - int(y.sum())}")
    
    # Models to evaluate
    models = {
        "Random Forest": RandomForestClassifier,
        "SVM": SVC,
        "Naive Bayes": GaussianNB,
        "MLP": MLPClassifier
    }
    
    # Run cross-validation for each model
    all_results = {}
    for model_name, model_class in models.items():
        results = kfold_cross_validation(model_class, model_name, X, y, N_FOLDS)
        all_results[model_name] = results
    
    # Visualize
    visualize_results(all_results)
    
    # Save results
    results_path = RESULTS_DIR / "kfold_cv_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 Results saved to: {results_path}")
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 Cross-Validation Summary")
    print(f"{'='*70}")
    print(f"\n{'Model':<20} {'Accuracy (Mean±Std)':<25} {'F1-Score (Mean±Std)':<25}")
    print("-" * 70)
    for model_name, results in all_results.items():
        acc_mean = results['aggregated']['accuracy_mean']
        acc_std = results['aggregated']['accuracy_std']
        f1_mean = results['aggregated']['f1_mean']
        f1_std = results['aggregated']['f1_std']
        print(f"{model_name:<20} {acc_mean:.4f}±{acc_std:.4f}     {f1_mean:.4f}±{f1_std:.4f}")
    
    # Best model
    best_model = max(all_results.items(), 
                    key=lambda x: x[1]['aggregated']['accuracy_mean'])
    print(f"\n🏆 Best Model: {best_model[0]} "
          f"(Accuracy: {best_model[1]['aggregated']['accuracy_mean']:.4f} ± "
          f"{best_model[1]['aggregated']['accuracy_std']:.4f})")


if __name__ == "__main__":
    main()

