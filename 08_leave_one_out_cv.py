#!/usr/bin/env python3
"""
Leave-One-Out Cross-Validation for small datasets.
This is the most rigorous evaluation method for small datasets.
"""

import os
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import json
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_and_average_features(file_list):
    """Load MFCC features and average across time dimension."""
    data_list = []
    for file_list_item in sorted(file_list):
        mfcc = np.load(FEATURES_DIR / file_list_item)
        avg_features = np.mean(mfcc, axis=1)
        data_list.append(avg_features)
    return np.vstack(data_list)


def load_all_data():
    """Load all data."""
    all_files = os.listdir(FEATURES_DIR)
    autistic_files = [f for f in all_files if f.startswith("aut_") and f.endswith(".npy")]
    non_autistic_files = [f for f in all_files if f.startswith("split-") and f.endswith(".npy")]
    
    autistic_data = load_and_average_features(autistic_files)
    non_autistic_data = load_and_average_features(non_autistic_files)
    
    X = np.vstack((autistic_data, non_autistic_data))
    y = np.hstack((np.ones(autistic_data.shape[0]), np.zeros(non_autistic_data.shape[0])))
    
    return X, y


def leave_one_out_cv(model_class, model_name, X, y):
    """Perform Leave-One-Out Cross-Validation."""
    print(f"\n{'='*70}")
    print(f"Leave-One-Out CV: {model_name}")
    print(f"{'='*70}")
    
    loo = LeaveOneOut()
    fold_results = []
    all_predictions = []
    all_true_labels = []
    
    for fold, (train_idx, test_idx) in enumerate(loo.split(X), 1):
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        # Normalize features
        scaler = StandardScaler()
        X_train_fold = scaler.fit_transform(X_train_fold)
        X_test_fold = scaler.transform(X_test_fold)
        
        # Create and train model
        if model_name == "Random Forest":
            model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
        elif model_name == "SVM":
            model = SVC(kernel="rbf", C=1, gamma="scale", random_state=42)
        elif model_name == "Naive Bayes":
            model = GaussianNB()
        elif model_name == "MLP":
            model = MLPClassifier(hidden_layer_sizes=(20,), max_iter=500, random_state=42)
        else:
            model = model_class()
        
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_test_fold)
        
        # Store results
        all_predictions.append(y_pred[0])
        all_true_labels.append(y_test_fold[0])
        
        fold_results.append({
            'fold': fold,
            'correct': int(y_pred[0] == y_test_fold[0])
        })
        
        if fold % 10 == 0:
            print(f"  Processed {fold}/{len(X)} folds...")
    
    # Calculate overall metrics
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    
    accuracy = accuracy_score(all_true_labels, all_predictions)
    precision = precision_score(all_true_labels, all_predictions, average='binary', zero_division=0)
    recall = recall_score(all_true_labels, all_predictions, average='binary', zero_division=0)
    f1 = f1_score(all_true_labels, all_predictions, average='binary', zero_division=0)
    cm = confusion_matrix(all_true_labels, all_predictions)
    
    print(f"\n📊 Leave-One-Out CV Results:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
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
    plt.title(f'Confusion Matrix - {model_name} (Leave-One-Out CV)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    cm_path = RESULTS_DIR / f"loocv_confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Confusion matrix saved to: {cm_path}")
    plt.close()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist(),
        'fold_results': fold_results
    }


def main():
    """Main Leave-One-Out CV pipeline."""
    print("="*70)
    print("ASD/ADHD Detection - Leave-One-Out Cross-Validation")
    print("="*70)
    print("\n⚠️  This is the most rigorous evaluation for small datasets.")
    print("   Each sample is tested using all other samples for training.")
    
    # Load data
    print("\n📂 Loading data...")
    X, y = load_all_data()
    print(f"  Total samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes: Autism={int(y.sum())}, Non-autism={len(y) - int(y.sum())}")
    print(f"\n  This will run {len(X)} separate train/test splits...")
    
    # Models to evaluate (using simpler configurations to reduce overfitting)
    models = {
        "Random Forest": RandomForestClassifier,
        "SVM": SVC,
        "Naive Bayes": GaussianNB,
        "MLP": MLPClassifier
    }
    
    # Run Leave-One-Out CV for each model
    all_results = {}
    for model_name, model_class in models.items():
        results = leave_one_out_cv(model_class, model_name, X, y)
        all_results[model_name] = results
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 Leave-One-Out CV Summary")
    print(f"{'='*70}")
    
    print(f"\n{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 70)
    for model_name, results in all_results.items():
        print(f"{model_name:<20} {results['accuracy']:<12.4f} {results['precision']:<12.4f} "
              f"{results['recall']:<12.4f} {results['f1']:<12.4f}")
    
    # Best model
    best_model = max(all_results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🏆 Best Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")
    
    # Overfitting analysis
    print(f"\n{'='*70}")
    print("🔍 Overfitting Analysis")
    print(f"{'='*70}")
    
    high_accuracy_models = [m for m, r in all_results.items() if r['accuracy'] >= 0.95]
    if high_accuracy_models:
        print(f"\n⚠️  Models with ≥95% accuracy: {', '.join(high_accuracy_models)}")
        print(f"   With {len(X)} samples, this could indicate:")
        print(f"   1. Classes are genuinely separable (good!)")
        print(f"   2. Dataset is too small/homogeneous to detect overfitting")
        print(f"   3. Features are too discriminative (check for data leakage)")
        print(f"\n   Recommendation: Collect more diverse data to validate.")
    else:
        print(f"\n✅ Accuracy values are more realistic for this dataset size.")
    
    # Save results
    results_path = RESULTS_DIR / "leave_one_out_cv_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 Results saved to: {results_path}")


if __name__ == "__main__":
    main()

