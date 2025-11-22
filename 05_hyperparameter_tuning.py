#!/usr/bin/env python3
"""
Hyperparameter tuning using Grid Search and Random Search.
"""

import os
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import json
import time

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
    """Load all data."""
    all_files = os.listdir(FEATURES_DIR)
    autistic_files = [f for f in all_files if f.startswith("aut_") and f.endswith(".npy")]
    non_autistic_files = [f for f in all_files if f.startswith("split-") and f.endswith(".npy")]
    
    autistic_data = load_and_average_features(autistic_files)
    non_autistic_data = load_and_average_features(non_autistic_files)
    
    X = np.vstack((autistic_data, non_autistic_data))
    y = np.hstack((np.ones(autistic_data.shape[0]), np.zeros(non_autistic_data.shape[0])))
    
    # Split into train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def tune_random_forest(X_train, y_train, X_val, y_val):
    """Tune Random Forest hyperparameters."""
    print("\n" + "="*70)
    print("Tuning Random Forest")
    print("="*70)
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        rf, param_grid, cv=3, scoring='accuracy', 
        n_jobs=-1, verbose=1
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - start_time
    
    print(f"\n⏱️  Time taken: {elapsed:.2f} seconds")
    print(f"✅ Best parameters: {grid_search.best_params_}")
    print(f"✅ Best CV score: {grid_search.best_score_:.4f}")
    
    # Evaluate on validation set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred)
    val_f1 = f1_score(y_val, y_pred, average='binary', zero_division=0)
    
    print(f"✅ Validation Accuracy: {val_accuracy:.4f}")
    print(f"✅ Validation F1-Score: {val_f1:.4f}")
    
    return {
        'best_params': grid_search.best_params_,
        'best_cv_score': grid_search.best_score_,
        'val_accuracy': val_accuracy,
        'val_f1': val_f1,
        'model': best_model
    }


def tune_svm(X_train, y_train, X_val, y_val):
    """Tune SVM hyperparameters."""
    print("\n" + "="*70)
    print("Tuning SVM")
    print("="*70)
    
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'kernel': ['rbf', 'linear', 'poly']
    }
    
    svm = SVC(random_state=42, probability=True)
    grid_search = GridSearchCV(
        svm, param_grid, cv=3, scoring='accuracy',
        n_jobs=-1, verbose=1
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - start_time
    
    print(f"\n⏱️  Time taken: {elapsed:.2f} seconds")
    print(f"✅ Best parameters: {grid_search.best_params_}")
    print(f"✅ Best CV score: {grid_search.best_score_:.4f}")
    
    # Evaluate on validation set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred)
    val_f1 = f1_score(y_val, y_pred, average='binary', zero_division=0)
    
    print(f"✅ Validation Accuracy: {val_accuracy:.4f}")
    print(f"✅ Validation F1-Score: {val_f1:.4f}")
    
    return {
        'best_params': grid_search.best_params_,
        'best_cv_score': grid_search.best_score_,
        'val_accuracy': val_accuracy,
        'val_f1': val_f1,
        'model': best_model
    }


def tune_mlp(X_train, y_train, X_val, y_val):
    """Tune MLP hyperparameters."""
    print("\n" + "="*70)
    print("Tuning MLP (Neural Network)")
    print("="*70)
    
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (100, 50), (128, 64)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01, 0.1],
        'max_iter': [500, 1000]
    }
    
    mlp = MLPClassifier(random_state=42, early_stopping=True, validation_fraction=0.1)
    grid_search = RandomizedSearchCV(
        mlp, param_grid, n_iter=20, cv=3, scoring='accuracy',
        n_jobs=1, verbose=1, random_state=42
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - start_time
    
    print(f"\n⏱️  Time taken: {elapsed:.2f} seconds")
    print(f"✅ Best parameters: {grid_search.best_params_}")
    print(f"✅ Best CV score: {grid_search.best_score_:.4f}")
    
    # Evaluate on validation set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred)
    val_f1 = f1_score(y_val, y_pred, average='binary', zero_division=0)
    
    print(f"✅ Validation Accuracy: {val_accuracy:.4f}")
    print(f"✅ Validation F1-Score: {val_f1:.4f}")
    
    return {
        'best_params': grid_search.best_params_,
        'best_cv_score': grid_search.best_score_,
        'val_accuracy': val_accuracy,
        'val_f1': val_f1,
        'model': best_model
    }


def main():
    """Main hyperparameter tuning pipeline."""
    print("="*70)
    print("ASD/ADHD Detection - Hyperparameter Tuning")
    print("="*70)
    
    # Load data
    print("\n📂 Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_data()
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    
    # Tune models
    results = {}
    
    # Random Forest
    rf_results = tune_random_forest(X_train, y_train, X_val, y_val)
    results['random_forest'] = {
        'best_params': rf_results['best_params'],
        'best_cv_score': rf_results['best_cv_score'],
        'val_accuracy': rf_results['val_accuracy'],
        'val_f1': rf_results['val_f1']
    }
    joblib.dump(rf_results['model'], MODELS_DIR / "rf_tuned.pkl")
    
    # SVM
    svm_results = tune_svm(X_train, y_train, X_val, y_val)
    results['svm'] = {
        'best_params': svm_results['best_params'],
        'best_cv_score': svm_results['best_cv_score'],
        'val_accuracy': svm_results['val_accuracy'],
        'val_f1': svm_results['val_f1']
    }
    joblib.dump(svm_results['model'], MODELS_DIR / "svm_tuned.pkl")
    
    # MLP
    mlp_results = tune_mlp(X_train, y_train, X_val, y_val)
    results['mlp'] = {
        'best_params': mlp_results['best_params'],
        'best_cv_score': mlp_results['best_cv_score'],
        'val_accuracy': mlp_results['val_accuracy'],
        'val_f1': mlp_results['val_f1']
    }
    joblib.dump(mlp_results['model'], MODELS_DIR / "mlp_tuned.pkl")
    
    # Save results
    results_path = RESULTS_DIR / "hyperparameter_tuning_results.json"
    with open(results_path, 'w') as f:
        # Convert numpy types to native Python types for JSON
        json_results = json.loads(json.dumps(results, default=str))
        json.dump(json_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("📊 Hyperparameter Tuning Summary")
    print(f"{'='*70}")
    print(f"\n{'Model':<20} {'Val Accuracy':<15} {'Val F1-Score':<15}")
    print("-" * 50)
    for model_name, result in results.items():
        print(f"{model_name:<20} {result['val_accuracy']:<15.4f} {result['val_f1']:<15.4f}")
    
    print(f"\n💾 Results saved to: {results_path}")
    print(f"💾 Tuned models saved to: {MODELS_DIR}")


if __name__ == "__main__":
    main()

