#!/usr/bin/env python3
"""
Generate Terminal Output Reports from Evaluation Results
========================================================

This script reads real evaluation results and generates formatted terminal
output text files that can be used for presentations and documentation.

Usage:
    python generate_evaluation_report.py
"""

import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
EVAL_RESULTS_DIR = RESULTS_DIR / "evaluation"
TRAINING_SUMMARY_PATH = RESULTS_DIR / "training_summary.json"
EVAL_RESULTS_PATH = EVAL_RESULTS_DIR / "evaluation_results.json"

# Model name mapping
MODEL_NAME_MAP = {
    "rf.pkl": "Random Forest",
    "svm.pkl": "SVM",
    "nb.pkl": "Naive Bayes",
    "ann.pkl": "MLP (Neural Network)"
}


def load_training_summary():
    """Load training summary if available."""
    if TRAINING_SUMMARY_PATH.exists():
        with open(TRAINING_SUMMARY_PATH, 'r') as f:
            return json.load(f)
    return None


def load_evaluation_results():
    """Load evaluation results."""
    if EVAL_RESULTS_PATH.exists():
        with open(EVAL_RESULTS_PATH, 'r') as f:
            return json.load(f)
    return None


def generate_training_log(training_summary):
    """Generate training log output from real training summary."""
    output = []
    
    output.append("="*70)
    output.append("ASD/ADHD Detection - Model Training")
    output.append("="*70)
    output.append("")
    
    if training_summary:
        # Get dataset info from evaluation results if available
        eval_results = load_evaluation_results()
        if eval_results and 'dataset_info' in eval_results:
            info = eval_results['dataset_info']
            output.append(f"Dataset Information:")
            output.append(f"  - Total samples: {info.get('total_samples', 'N/A')}")
            output.append(f"  - Training samples: {info.get('training_samples', 'N/A')}")
            output.append(f"  - Test samples: {info.get('test_samples', 'N/A')}")
            output.append(f"  - Feature dimension: {info.get('feature_dimension', 'N/A')}")
        else:
            output.append("Dataset: Real audio recordings from recordings/ folder")
        
        output.append("Train/Test split: 70/30")
        output.append("")
        
        for model_file, metrics in training_summary.items():
            model_name = MODEL_NAME_MAP.get(model_file, model_file.replace('.pkl', ''))
            output.append("")
            output.append("="*70)
            output.append(f"Training {model_name}...")
            output.append("="*70)
            output.append("")
            output.append(f"✅ Training completed")
            output.append(f"   Test Accuracy: {metrics.get('accuracy', 0):.4f}")
            output.append(f"   Precision:     {metrics.get('precision', 0):.4f}")
            output.append(f"   Recall:        {metrics.get('recall', 0):.4f}")
            output.append(f"   F1-Score:      {metrics.get('f1', 0):.4f}")
    else:
        output.append("Training summary not found. Please run 02_train_models.py first.")
    
    return "\n".join(output)


def generate_evaluation_summary(eval_results):
    """Generate evaluation summary output from real results."""
    output = []
    
    output.append("="*70)
    output.append("ASD/ADHD Detection - Model Evaluation")
    output.append("="*70)
    output.append("")
    
    if eval_results:
        dataset_info = eval_results.get('dataset_info', {})
        output.append("📂 Loading data...")
        output.append(f"  Test set: {dataset_info.get('test_samples', 'N/A')} samples")
        output.append("")
        
        models = eval_results.get('models', {})
        output.append(f"📦 Found {len(models)} models to evaluate")
        output.append("")
        
        # Generate output for each model
        for model_short, metrics in models.items():
            # Map short name to full name
            model_name = None
            for key, value in MODEL_NAME_MAP.items():
                if MODEL_NAME_MAP.get(key, '').replace(' ', '_').lower() == model_short or \
                   key.replace('.pkl', '') == model_short:
                    model_name = value
                    break
            
            if not model_name:
                model_name = model_short.upper()
            
            output.append("")
            output.append("="*70)
            output.append(f"Evaluating: {model_name}")
            output.append("="*70)
            output.append("")
            output.append("📊 Performance Metrics:")
            output.append(f"  Accuracy:  {metrics.get('accuracy', 0):.4f} ({metrics.get('accuracy', 0)*100:.2f}%)")
            output.append(f"  Precision: {metrics.get('precision', 0):.4f}")
            output.append(f"  Recall:    {metrics.get('recall', 0):.4f}")
            output.append(f"  F1-Score:  {metrics.get('f1_score', 0):.4f}")
            
            if metrics.get('roc_auc'):
                output.append(f"  ROC-AUC:   {metrics.get('roc_auc', 0):.4f}")
            
            output.append("")
            output.append("📋 Classification Report:")
            output.append("              precision    recall  f1-score   support")
            output.append("")
            
            # Extract confusion matrix info
            cm = metrics.get('confusion_matrix', [[0, 0], [0, 0]])
            if isinstance(cm, list) and len(cm) == 2:
                tn, fp = cm[0]
                fn, tp = cm[1]
                support_0 = tn + fp
                support_1 = fn + tp
                
                prec_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
                prec_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
                rec_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
                rec_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_0 = 2 * (prec_0 * rec_0) / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0
                f1_1 = 2 * (prec_1 * rec_1) / (prec_1 + rec_1) if (prec_1 + rec_1) > 0 else 0
                
                output.append(f"   Non-Autism      {prec_0:.2f}       {rec_0:.2f}       {f1_0:.2f}        {support_0}")
                output.append(f"        Autism      {prec_1:.2f}       {rec_1:.2f}       {f1_1:.2f}        {support_1}")
                output.append("")
                output.append(f"     accuracy                          {metrics.get('accuracy', 0):.2f}        {support_0 + support_1}")
                output.append(f"    macro avg      {(prec_0 + prec_1)/2:.2f}       {(rec_0 + rec_1)/2:.2f}       {(f1_0 + f1_1)/2:.2f}        {support_0 + support_1}")
                output.append(f" weighted avg      {(prec_0 * support_0 + prec_1 * support_1)/(support_0 + support_1):.2f}       {(rec_0 * support_0 + rec_1 * support_1)/(support_0 + support_1):.2f}       {(f1_0 * support_0 + f1_1 * support_1)/(support_0 + support_1):.2f}        {support_0 + support_1}")
            
            output.append("")
            output.append("🔢 Confusion Matrix:")
            output.append("                Predicted")
            output.append("              Non-Aut  Autism")
            if isinstance(cm, list) and len(cm) == 2:
                output.append(f"Actual Non-Aut   {cm[0][0]:4d}   {cm[0][1]:4d}")
                output.append(f"       Autism    {cm[1][0]:4d}   {cm[1][1]:4d}")
            
            output.append("")
            output.append(f"💾 Confusion matrix saved to: results/evaluation/confusion_matrix_{model_short}.png")
            if metrics.get('roc_auc'):
                output.append(f"💾 ROC curve saved to: results/evaluation/roc_curve_{model_short}.png")
        
        # Summary table
        output.append("")
        output.append("="*70)
        output.append("📊 Evaluation Summary")
        output.append("="*70)
        output.append("")
        output.append(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
        output.append("-" * 85)
        
        metrics_list = []
        for model_short, metrics in models.items():
            model_name = None
            for key, value in MODEL_NAME_MAP.items():
                if key.replace('.pkl', '') == model_short:
                    model_name = value
                    break
            if not model_name:
                model_name = model_short.upper()
            
            roc_auc_str = f"{metrics.get('roc_auc', 0):.4f}" if metrics.get('roc_auc') else "N/A"
            output.append(f"{model_name:<25} "
                         f"{metrics.get('accuracy', 0):<12.4f} "
                         f"{metrics.get('precision', 0):<12.4f} "
                         f"{metrics.get('recall', 0):<12.4f} "
                         f"{metrics.get('f1_score', 0):<12.4f} "
                         f"{roc_auc_str:<12}")
            
            metrics_list.append((model_name, metrics))
        
        output.append("-" * 85)
        
        if metrics_list:
            best = max(metrics_list, key=lambda x: x[1].get('accuracy', 0))
            output.append("")
            output.append(f"🏆 Best Model: {best[0]}")
            output.append(f"   Accuracy: {best[1].get('accuracy', 0):.4f} ({best[1].get('accuracy', 0)*100:.2f}%)")
            output.append(f"   F1-Score: {best[1].get('f1_score', 0):.4f}")
            if best[1].get('roc_auc'):
                output.append(f"   ROC-AUC:  {best[1].get('roc_auc', 0):.4f}")
        
        output.append("")
        output.append(f"💾 Results saved to: results/evaluation/evaluation_results.json")
    else:
        output.append("Evaluation results not found. Please run evaluate_models_comprehensive.py first.")
    
    return "\n".join(output)


def main():
    """Generate terminal output files from real results."""
    print("Generating evaluation reports from real results...")
    
    # Create output directory
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate training log
    training_summary = load_training_summary()
    training_log = generate_training_log(training_summary)
    training_path = EVAL_RESULTS_DIR / "terminal_training_logs.txt"
    with open(training_path, 'w') as f:
        f.write(training_log)
    print(f"✅ Training logs saved: {training_path}")
    
    # Generate evaluation summary
    eval_results = load_evaluation_results()
    eval_summary = generate_evaluation_summary(eval_results)
    eval_path = EVAL_RESULTS_DIR / "terminal_evaluation_summary.txt"
    with open(eval_path, 'w') as f:
        f.write(eval_summary)
    print(f"✅ Evaluation summary saved: {eval_path}")
    
    print(f"\n💡 You can now:")
    print(f"   1. Open these files in a terminal/text editor")
    print(f"   2. Take screenshots for your presentation")
    print(f"   3. Or copy/paste the content into your report")


if __name__ == "__main__":
    main()

