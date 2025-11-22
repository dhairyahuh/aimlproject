#!/usr/bin/env python3
"""
Main pipeline script to run the complete autism detection workflow.
Run: python run_pipeline.py
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


def run_script(script_name, description):
    """Run a Python script and handle errors."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    
    script_path = PROJECT_ROOT / script_name
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run(
            [str(VENV_PYTHON), str(script_path)],
            cwd=str(PROJECT_ROOT),
            check=True,
            capture_output=False
        )
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script_name}: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  {description} interrupted by user")
        return False


def main():
    """Run the complete pipeline."""
    print("="*70)
    print("ASD/ADHD Detection - Complete Pipeline")
    print("="*70)
    print("\nThis will run the following steps:")
    print("  1. Extract MFCC features from recordings")
    print("  2. Train ML models (RF, SVM, NB, MLP)")
    print("  3. Evaluate models and show metrics")
    print("\nOptional steps (commented out):")
    print("  4. K-Fold cross-validation")
    print("  5. Hyperparameter tuning")
    print("  6. Feature analysis")
    
    response = input("\nProceed with steps 1-3? (y/n): ").strip().lower()
    if response != 'y':
        print("Pipeline cancelled.")
        return
    
    # Step 1: Extract features
    if not run_script("01_extract_features.py", "Step 1: Extracting MFCC Features"):
        print("\n❌ Pipeline failed at feature extraction")
        return
    
    # Step 2: Train models
    if not run_script("02_train_models.py", "Step 2: Training Models"):
        print("\n❌ Pipeline failed at model training")
        return
    
    # Step 3: Evaluate models
    if not run_script("03_evaluate_models.py", "Step 3: Evaluating Models"):
        print("\n❌ Pipeline failed at model evaluation")
        return
    
    print("\n" + "="*70)
    print("✅ Pipeline completed successfully!")
    print("="*70)
    print("\nNext steps:")
    print("  - View results in: results/")
    print("  - Launch UI: streamlit run 07_streamlit_ui.py")
    print("  - Run optional analyses: 04_kfold_validation.py, 05_hyperparameter_tuning.py, 06_feature_analysis.py")


if __name__ == "__main__":
    main()

