"""
Evaluation and Manual Verification Script
==========================================
Evaluates the model on test data and provides detailed results for manual verification.
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

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.mlp_classifier import MLPClassifier

# Paths
ROOT_DATA_DIR = project_root.parent / 'data'
MODELS_DIR = project_root / 'models' / 'saved'
RESULTS_DIR = project_root / 'results' / 'evaluation'

# Create directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_model_and_data():
    """Load trained model and test data."""
    print("="*70)
    print("LOADING MODEL AND DATA")
    print("="*70)
    
    # Load model
    model_path = MODELS_DIR / 'asd_adhd_mlp_model.keras'
    if not model_path.exists():
        print(f"✗ Model not found: {model_path}")
        print("Please train the model first using: python tools/train_model.py")
        return None, None, None, None
    
    model = MLPClassifier()
    model.load(str(model_path))
    print(f"✓ Model loaded from: {model_path}")
    
    # Load test data
    X_test = np.load(ROOT_DATA_DIR / 'X_test.npy')
    y_test = np.load(ROOT_DATA_DIR / 'y_test.npy')
    print(f"✓ Test data loaded: {X_test.shape[0]} samples")
    
    return model, X_test, y_test, model_path


def evaluate_model(model, X_test, y_test):
    """Evaluate model and generate detailed report."""
    print("\n" + "="*70)
    print("EVALUATING MODEL")
    print("="*70)
    
    # Get predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)
    y_proba = model.predict(X_test, return_proba=True)
    
    # Calculate metrics
    metrics = model.evaluate(X_test, y_test)
    
    # Print summary
    print("\n" + "-"*70)
    print("OVERALL METRICS")
    print("-"*70)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    
    # Classification report
    print("\n" + "-"*70)
    print("DETAILED CLASSIFICATION REPORT")
    print("-"*70)
    print(classification_report(y_test, y_pred, 
                                target_names=['Healthy', 'ASD', 'ADHD']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n" + "-"*70)
    print("CONFUSION MATRIX")
    print("-"*70)
    print("Rows = True labels, Columns = Predicted labels")
    print("\n" + " " * 15 + "Healthy    ASD    ADHD")
    for i, label in enumerate(['Healthy', 'ASD', 'ADHD']):
        print(f"{label:12s} {cm[i, 0]:6d} {cm[i, 1]:6d} {cm[i, 2]:6d}")
    
    return y_pred, y_proba, metrics, cm


def create_detailed_results(y_test, y_pred, y_proba, output_dir):
    """Create detailed results DataFrame for manual verification."""
    print("\n" + "="*70)
    print("CREATING DETAILED RESULTS")
    print("="*70)
    
    # Create DataFrame
    results_df = pd.DataFrame({
        'sample_id': range(len(y_test)),
        'true_label': y_test,
        'predicted_label': y_pred,
        'true_label_name': [['Healthy', 'ASD', 'ADHD'][int(label)] for label in y_test],
        'predicted_label_name': [['Healthy', 'ASD', 'ADHD'][int(label)] for label in y_pred],
        'confidence': [y_proba[i, pred] for i, pred in enumerate(y_pred)],
        'prob_healthy': y_proba[:, 0],
        'prob_asd': y_proba[:, 1],
        'prob_adhd': y_proba[:, 2],
        'correct': (y_test == y_pred)
    })
    
    # Save to CSV
    csv_path = output_dir / 'detailed_predictions.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"✓ Detailed results saved to: {csv_path}")
    
    # Print statistics
    print("\n" + "-"*70)
    print("PREDICTION STATISTICS")
    print("-"*70)
    print(f"Total samples: {len(results_df)}")
    print(f"Correct predictions: {results_df['correct'].sum()} ({results_df['correct'].mean():.2%})")
    print(f"Incorrect predictions: {(~results_df['correct']).sum()} ({(~results_df['correct']).mean():.2%})")
    
    print("\n" + "-"*70)
    print("CONFIDENCE STATISTICS")
    print("-"*70)
    print(f"Mean confidence: {results_df['confidence'].mean():.4f}")
    print(f"Median confidence: {results_df['confidence'].median():.4f}")
    print(f"Min confidence: {results_df['confidence'].min():.4f}")
    print(f"Max confidence: {results_df['confidence'].max():.4f}")
    
    print("\n" + "-"*70)
    print("MISCLASSIFICATIONS")
    print("-"*70)
    misclassified = results_df[~results_df['correct']]
    if len(misclassified) > 0:
        print(f"\nTotal misclassifications: {len(misclassified)}")
        print("\nMisclassification breakdown:")
        for true_label in ['Healthy', 'ASD', 'ADHD']:
            true_mask = misclassified['true_label_name'] == true_label
            if true_mask.sum() > 0:
                print(f"\n  True label: {true_label}")
                pred_counts = misclassified[true_mask]['predicted_label_name'].value_counts()
                for pred_label, count in pred_counts.items():
                    print(f"    → Predicted as {pred_label}: {count}")
    else:
        print("No misclassifications!")
    
    return results_df


def plot_results(y_test, y_pred, y_proba, cm, output_dir):
    """Create visualization plots."""
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Healthy', 'ASD', 'ADHD'],
                yticklabels=['Healthy', 'ASD', 'ADHD'],
                ax=ax1)
    ax1.set_xlabel('Predicted', fontsize=12)
    ax1.set_ylabel('True', fontsize=12)
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Plot 2: Confidence Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    confidences = [y_proba[i, pred] for i, pred in enumerate(y_pred)]
    ax2.hist(confidences, bins=30, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Confidence', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Per-class Accuracy
    ax3 = fig.add_subplot(gs[1, 0])
    class_names = ['Healthy', 'ASD', 'ADHD']
    class_accuracies = []
    for i, name in enumerate(class_names):
        mask = y_test == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == i).mean()
            class_accuracies.append(acc)
        else:
            class_accuracies.append(0.0)
    
    bars = ax3.bar(class_names, class_accuracies, color=['skyblue', 'lightcoral', 'lightgreen'],
                   alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax3.set_ylim([0, 1.0])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, acc in zip(bars, class_accuracies):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=11)
    
    # Plot 4: Probability Distributions
    ax4 = fig.add_subplot(gs[1, 1])
    for i, name in enumerate(class_names):
        ax4.hist(y_proba[:, i], bins=30, alpha=0.5, label=name, edgecolor='black')
    ax4.set_xlabel('Probability', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Probability Distributions by Class', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Error Analysis
    ax5 = fig.add_subplot(gs[2, :])
    errors = y_test != y_pred
    error_confidences = [y_proba[i, pred] for i, pred in enumerate(y_pred) if errors[i]]
    correct_confidences = [y_proba[i, pred] for i, pred in enumerate(y_pred) if not errors[i]]
    
    ax5.hist(correct_confidences, bins=30, alpha=0.6, label='Correct', 
             color='green', edgecolor='black')
    ax5.hist(error_confidences, bins=30, alpha=0.6, label='Incorrect',
             color='red', edgecolor='black')
    ax5.set_xlabel('Confidence', fontsize=12)
    ax5.set_ylabel('Frequency', fontsize=12)
    ax5.set_title('Confidence Distribution: Correct vs Incorrect Predictions', 
                  fontsize=14, fontweight='bold')
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3)
    
    plt.suptitle('Model Evaluation Results', fontsize=16, fontweight='bold', y=0.995)
    
    # Save plot
    plot_path = output_dir / 'evaluation_plots.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Evaluation plots saved to: {plot_path}")


def main():
    """Main evaluation pipeline."""
    print("="*70)
    print("ASD/ADHD DETECTION - MODEL EVALUATION & VERIFICATION")
    print("="*70)
    
    # Load model and data
    model, X_test, y_test, model_path = load_model_and_data()
    if model is None:
        return
    
    # Evaluate model
    y_pred, y_proba, metrics, cm = evaluate_model(model, X_test, y_test)
    
    # Create detailed results
    results_df = create_detailed_results(y_test, y_pred, y_proba, RESULTS_DIR)
    
    # Generate plots
    plot_results(y_test, y_pred, y_proba, cm, RESULTS_DIR)
    
    print("\n" + "="*70)
    print("✓ EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"  - detailed_predictions.csv: Detailed predictions for manual verification")
    print(f"  - evaluation_plots.png: Visualization plots")
    print("\nYou can now review the detailed_predictions.csv file to manually verify results.")


if __name__ == "__main__":
    main()

