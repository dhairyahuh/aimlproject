#!/usr/bin/env python3
"""
Fix Notebook Interpreter Configuration
======================================
Updates all notebooks in the notebooks/ directory to use the correct kernel.
"""

import json
from pathlib import Path

def fix_notebook_kernel(notebook_path: Path):
    """Update notebook metadata to use the correct kernel."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Update kernel specification
    if 'metadata' not in nb:
        nb['metadata'] = {}
    
    if 'kernelspec' not in nb['metadata']:
        nb['metadata']['kernelspec'] = {}
    
    nb['metadata']['kernelspec'].update({
        'name': 'asd_adhd_detection',
        'display_name': 'Python (ASD_ADHD_Detection)',
        'language': 'python'
    })
    
    # Also update language_info
    if 'language_info' not in nb['metadata']:
        nb['metadata']['language_info'] = {}
    
    nb['metadata']['language_info'].update({
        'name': 'python',
        'version': '3.13'
    })
    
    # Save updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"  ✓ Fixed: {notebook_path.name}")

def main():
    project_root = Path(__file__).parent
    notebooks_dir = project_root / 'notebooks'
    
    print("="*70)
    print("Fixing Notebook Interpreter Configuration")
    print("="*70)
    print(f"\nScanning: {notebooks_dir}")
    
    notebook_files = list(notebooks_dir.glob('*.ipynb'))
    
    if not notebook_files:
        print("  ⚠️  No notebooks found!")
        return
    
    print(f"\nFound {len(notebook_files)} notebooks\n")
    
    for nb_path in sorted(notebook_files):
        try:
            fix_notebook_kernel(nb_path)
        except Exception as e:
            print(f"  ✗ Error fixing {nb_path.name}: {e}")
    
    print("\n" + "="*70)
    print("✅ All notebooks updated!")
    print("="*70)
    print("\nNext steps:")
    print("1. Open Jupyter: jupyter notebook")
    print("2. Open any notebook")
    print("3. Select kernel: 'Python (ASD_ADHD_Detection)'")
    print("4. Run the helper cells at the top of each notebook")
    print()

if __name__ == "__main__":
    main()

