"""
Data Structure Checker
======================
Helps you understand how your audio files are organized and suggests
how to modify the label detection function.
"""

import os
from pathlib import Path
from collections import Counter

# Paths
project_root = Path(__file__).parent.parent
ROOT_DATA_DIR = project_root.parent / 'data'


def analyze_data_structure():
    """Analyze the structure of audio files in the data directory."""
    print("="*70)
    print("DATA STRUCTURE ANALYSIS")
    print("="*70)
    
    if not ROOT_DATA_DIR.exists():
        print(f"✗ Data directory not found: {ROOT_DATA_DIR}")
        return
    
    # Find all audio files
    audio_extensions = ['.wav', '.mp3', '.ogg', '.flac', '.m4a']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(list(ROOT_DATA_DIR.rglob(f'*{ext}')))
    
    print(f"\nFound {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        print("\n⚠ No audio files found!")
        print("Please ensure audio files are in the data directory.")
        return
    
    # Analyze directory structure
    print("\n" + "-"*70)
    print("DIRECTORY STRUCTURE")
    print("-"*70)
    
    directories = set()
    for audio_file in audio_files:
        parent_dir = audio_file.parent.name
        directories.add(parent_dir)
    
    print(f"Found {len(directories)} unique parent directories:")
    for dir_name in sorted(directories):
        count = sum(1 for f in audio_files if f.parent.name == dir_name)
        print(f"  - {dir_name}: {count} files")
    
    # Analyze filenames
    print("\n" + "-"*70)
    print("FILENAME PATTERNS")
    print("-"*70)
    
    filename_keywords = {
        'ASD': [],
        'ADHD': [],
        'Healthy': [],
        'Normal': [],
        'Control': [],
        'Autism': []
    }
    
    for audio_file in audio_files[:20]:  # Sample first 20
        filename_lower = audio_file.name.lower()
        for keyword in filename_keywords:
            if keyword.lower() in filename_lower:
                filename_keywords[keyword].append(audio_file.name)
    
    for keyword, matches in filename_keywords.items():
        if matches:
            print(f"\n  '{keyword}' found in {len(matches)} filenames:")
            for match in matches[:5]:  # Show first 5
                print(f"    - {match}")
            if len(matches) > 5:
                print(f"    ... and {len(matches) - 5} more")
    
    # Sample file paths
    print("\n" + "-"*70)
    print("SAMPLE FILE PATHS")
    print("-"*70)
    for i, audio_file in enumerate(audio_files[:10], 1):
        rel_path = audio_file.relative_to(ROOT_DATA_DIR)
        print(f"{i}. {rel_path}")
    
    # Suggestions
    print("\n" + "="*70)
    print("SUGGESTIONS FOR LABEL DETECTION")
    print("="*70)
    
    has_asd_dir = any('asd' in d.lower() or 'autism' in d.lower() for d in directories)
    has_adhd_dir = any('adhd' in d.lower() for d in directories)
    has_healthy_dir = any('healthy' in d.lower() or 'normal' in d.lower() or 'control' in d.lower() for d in directories)
    
    if has_asd_dir or has_adhd_dir or has_healthy_dir:
        print("\n✓ Your data appears to be organized by directories.")
        print("  The label detection function should check parent directory names.")
        print("\n  Suggested code:")
        print("  ```python")
        print("  def get_label_from_filename(filename: str) -> str:")
        print("      path_parts = Path(filename).parts")
        print("      for part in path_parts:")
        print("          part_lower = part.lower()")
        print("          if 'asd' in part_lower or 'autism' in part_lower:")
        print("              return 'ASD'")
        print("          elif 'adhd' in part_lower:")
        print("              return 'ADHD'")
        print("          elif 'healthy' in part_lower or 'normal' in part_lower:")
        print("              return 'Healthy'")
        print("      return None")
        print("  ```")
    else:
        print("\n⚠ Directory-based labeling not detected.")
        print("  Checking filenames for keywords...")
        
        asd_in_filename = any('asd' in f.name.lower() or 'autism' in f.name.lower() 
                             for f in audio_files[:50])
        adhd_in_filename = any('adhd' in f.name.lower() for f in audio_files[:50])
        healthy_in_filename = any('healthy' in f.name.lower() or 'normal' in f.name.lower() 
                                 for f in audio_files[:50])
        
        if asd_in_filename or adhd_in_filename or healthy_in_filename:
            print("  ✓ Keywords found in filenames.")
            print("  The label detection function should check filename patterns.")
        else:
            print("  ⚠ No clear labeling pattern detected in filenames or directories.")
            print("  You may need to:")
            print("    1. Organize files into labeled directories (ASD/, ADHD/, Healthy/)")
            print("    2. Rename files with labels (ASD_001.wav, ADHD_001.wav, etc.)")
            print("    3. Create a separate label mapping file")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    analyze_data_structure()

