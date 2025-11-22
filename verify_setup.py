#!/usr/bin/env python3
"""
Quick verification script to check if the environment is set up correctly.
Run this to diagnose any interpreter or dependency issues.
"""

import sys
from pathlib import Path

def check_python():
    print("="*70)
    print("Python Environment Check")
    print("="*70)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Virtual env active: {'venv' in sys.executable or 'ASD_ADHD_Detection' in sys.executable}")
    print()

def check_imports():
    print("="*70)
    print("Checking Critical Dependencies")
    print("="*70)
    
    required = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scikit-learn': 'sklearn',
        'librosa': 'librosa',
        'tensorflow': 'tensorflow',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'joblib': 'joblib',
    }
    
    missing = []
    for module, import_name in required.items():
        try:
            __import__(import_name)
            print(f"  ✓ {module}")
        except ImportError:
            print(f"  ✗ {module} - MISSING")
            missing.append(module)
    
    print()
    if missing:
        print(f"⚠️  Missing packages: {', '.join(missing)}")
        print("   Install with: pip install " + " ".join(missing))
        return False
    else:
        print("✅ All dependencies installed!")
        return True

def check_paths():
    print("="*70)
    print("Checking Project Paths")
    print("="*70)
    
    project_root = Path(__file__).parent
    paths_to_check = {
        'Project root': project_root,
        'Notebooks': project_root / 'notebooks',
        'Features': project_root.parent / 'features',
        'Recordings': project_root / 'recordings',
        'Models': project_root / 'models' / 'saved',
    }
    
    for name, path in paths_to_check.items():
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {path}")
        if not exists and name in ['Recordings', 'Features']:
            print(f"      (Optional - may not exist yet)")
    
    print()

def check_data():
    print("="*70)
    print("Checking Data Availability")
    print("="*70)
    
    project_root = Path(__file__).parent
    features_dir = project_root.parent / 'features'
    
    if features_dir.exists():
        import os
        files = os.listdir(features_dir)
        aut_files = [f for f in files if f.startswith('aut_')]
        non_files = [f for f in files if f.startswith('split-')]
        print(f"  Features directory: {features_dir}")
        print(f"  Autism samples: {len(aut_files)}")
        print(f"  Non-autism samples: {len(non_files)}")
        print(f"  Total: {len(aut_files) + len(non_files)}")
    else:
        print(f"  ⚠️  Features directory not found: {features_dir}")
        print(f"     Run mfcc_extract.py first to generate features")
    
    print()

def main():
    print("\n" + "="*70)
    print("ASD_ADHD_Detection - Environment Verification")
    print("="*70 + "\n")
    
    check_python()
    deps_ok = check_imports()
    check_paths()
    check_data()
    
    print("="*70)
    if deps_ok:
        print("✅ Environment looks good! You should be able to run notebooks.")
    else:
        print("⚠️  Some dependencies are missing. Install them first.")
    print("="*70)
    print("\nTo use notebooks:")
    print("1. Open Jupyter: jupyter notebook")
    print("2. Select kernel: 'Python (ASD_ADHD_Detection)'")
    print("3. Run the helper cells at the top of each notebook")
    print()

if __name__ == "__main__":
    main()

