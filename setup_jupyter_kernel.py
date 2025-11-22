#!/usr/bin/env python3
"""
Setup Jupyter Kernel for ASD_ADHD_Detection Project
====================================================
This script registers the project's virtual environment as a Jupyter kernel
so notebooks can use the correct Python interpreter with all dependencies.
"""

import sys
import subprocess
from pathlib import Path

def main():
    project_root = Path(__file__).parent
    venv_python = project_root / ".venv" / "bin" / "python"
    
    if not venv_python.exists():
        print("❌ Virtual environment not found!")
        print(f"   Expected: {venv_python}")
        print("\nPlease create the virtual environment first:")
        print(f"   cd {project_root}")
        print("   python3 -m venv venv")
        print("   source venv/bin/activate")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    print("="*70)
    print("Setting up Jupyter Kernel for ASD_ADHD_Detection")
    print("="*70)
    
    # Install ipykernel if not already installed
    print("\n1. Installing ipykernel in virtual environment...")
    try:
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "-q", "ipykernel", "jupyter"],
            check=True
        )
        print("   ✓ ipykernel installed")
    except subprocess.CalledProcessError as e:
        print(f"   ✗ Failed to install ipykernel: {e}")
        sys.exit(1)
    
    # Register kernel
    print("\n2. Registering kernel with Jupyter...")
    try:
        subprocess.run(
            [
                str(venv_python), "-m", "ipykernel", "install",
                "--user",
                "--name=asd_adhd_detection",
                "--display-name=Python (ASD_ADHD_Detection)"
            ],
            check=True
        )
        print("   ✓ Kernel registered successfully!")
    except subprocess.CalledProcessError as e:
        print(f"   ✗ Failed to register kernel: {e}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("✅ Setup Complete!")
    print("="*70)
    print("\nNext steps:")
    print("1. Open Jupyter Notebook or JupyterLab")
    print("2. When creating/opening a notebook, select kernel:")
    print("   'Python (ASD_ADHD_Detection)' from the kernel menu")
    print("3. All notebooks in the project should now use the correct interpreter")
    print("\nTo verify, run:")
    print("   jupyter kernelspec list")
    print("\nYou should see 'asd_adhd_detection' in the list.")

if __name__ == "__main__":
    main()

