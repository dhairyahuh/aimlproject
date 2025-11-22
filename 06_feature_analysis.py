#!/usr/bin/env python3
"""
Feature importance analysis and feature selection.
"""

import os
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

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


def load_data():
    """Load all data."""
    all_files = os.listdir(FEATURES_DIR)
    autistic_files = [f for f in all_files if f.startswith("aut_") and f.endswith(".npy")]
    non_autistic_files = [f for f in all_files if f.startswith("split-") and f.endswith(".npy")]
    
    autistic_data = load_and_average_features(autistic_files)
    non_autistic_data = load_and_average_features(non_autistic_files)
    
    X = np.vstack((autistic_data, non_autistic_data))
    y = np.hstack((np.ones(autistic_data.shape[0]), np.zeros(non_autistic_data.shape[0])))
    
    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, scaler


def compute_feature_importance(X, y):
    """Compute feature importance using Random Forest."""
    print("\n" + "="*70)
    print("Computing Feature Importance (Random Forest)")
    print("="*70)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print(f"\nTop 10 Most Important Features:")
    print(f"{'Rank':<6} {'Feature Index':<15} {'Importance':<15}")
    print("-" * 40)
    for i in range(min(10, len(importances))):
        idx = indices[i]
        print(f"{i+1:<6} {idx:<15} {importances[idx]:<15.6f}")
    
    # Visualize
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importances)), importances[indices])
    plt.title('Feature Importance (Random Forest)')
    plt.xlabel('Feature Rank')
    plt.ylabel('Importance')
    plt.tight_layout()
    
    plot_path = RESULTS_DIR / "feature_importance.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Feature importance plot saved to: {plot_path}")
    plt.close()
    
    return importances, indices


def feature_selection_analysis(X, y):
    """Analyze feature selection methods."""
    print("\n" + "="*70)
    print("Feature Selection Analysis")
    print("="*70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # F-test
    f_selector = SelectKBest(score_func=f_classif, k='all')
    f_selector.fit(X_train, y_train)
    f_scores = f_selector.scores_
    
    # Mutual Information
    mi_selector = SelectKBest(score_func=mutual_info_classif, k='all')
    mi_selector.fit(X_train, y_train)
    mi_scores = mi_selector.scores_
    
    # Create DataFrame
    feature_df = pd.DataFrame({
        'feature_idx': range(len(f_scores)),
        'f_score': f_scores,
        'mi_score': mi_scores
    })
    feature_df = feature_df.sort_values('f_score', ascending=False)
    
    print(f"\nTop 10 Features by F-Score:")
    print(feature_df.head(10).to_string(index=False))
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].bar(range(len(f_scores)), np.sort(f_scores)[::-1])
    axes[0].set_title('F-Score (ANOVA F-value)')
    axes[0].set_xlabel('Feature Rank')
    axes[0].set_ylabel('F-Score')
    
    axes[1].bar(range(len(mi_scores)), np.sort(mi_scores)[::-1])
    axes[1].set_title('Mutual Information Score')
    axes[1].set_xlabel('Feature Rank')
    axes[1].set_ylabel('MI Score')
    
    plt.tight_layout()
    plot_path = RESULTS_DIR / "feature_selection_scores.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Feature selection scores plot saved to: {plot_path}")
    plt.close()
    
    return feature_df


def evaluate_with_selected_features(X, y, k_values=[10, 20, 30, 40]):
    """Evaluate model performance with different numbers of selected features."""
    print("\n" + "="*70)
    print("Evaluating Model with Selected Features")
    print("="*70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    results = []
    
    for k in k_values:
        if k > X.shape[1]:
            continue
        
        # Select K best features
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train_selected, y_train)
        
        # Evaluate
        y_pred = rf.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        
        results.append({
            'k': k,
            'accuracy': accuracy,
            'f1': f1
        })
        
        print(f"\nK={k:2d} features: Accuracy={accuracy:.4f}, F1={f1:.4f}")
    
    # Baseline (all features)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, y_pred)
    baseline_f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
    
    print(f"\nBaseline (all {X.shape[1]} features): Accuracy={baseline_accuracy:.4f}, F1={baseline_f1:.4f}")
    
    # Visualize
    k_vals = [r['k'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    f1_scores = [r['f1'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_vals, accuracies, 'o-', label='Accuracy')
    plt.plot(k_vals, f1_scores, 's-', label='F1-Score')
    plt.axhline(y=baseline_accuracy, color='r', linestyle='--', label='Baseline Accuracy')
    plt.axhline(y=baseline_f1, color='g', linestyle='--', label='Baseline F1')
    plt.xlabel('Number of Selected Features (K)')
    plt.ylabel('Score')
    plt.title('Model Performance vs Number of Selected Features')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = RESULTS_DIR / "feature_selection_performance.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Performance plot saved to: {plot_path}")
    plt.close()
    
    return results


def main():
    """Main feature analysis pipeline."""
    print("="*70)
    print("ASD/ADHD Detection - Feature Analysis & Selection")
    print("="*70)
    
    # Load data
    print("\n📂 Loading data...")
    X, y, scaler = load_data()
    print(f"  Total samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    
    # Feature importance
    importances, indices = compute_feature_importance(X, y)
    
    # Feature selection analysis
    feature_df = feature_selection_analysis(X, y)
    
    # Evaluate with selected features
    selection_results = evaluate_with_selected_features(X, y)
    
    # Save results
    results = {
        'feature_importance': {
            'top_10_indices': indices[:10].tolist(),
            'top_10_importances': importances[indices[:10]].tolist()
        },
        'feature_selection': {
            'top_10_f_score': feature_df.head(10).to_dict('records')
        },
        'selection_performance': selection_results
    }
    
    results_path = RESULTS_DIR / "feature_analysis_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_path}")


if __name__ == "__main__":
    main()

