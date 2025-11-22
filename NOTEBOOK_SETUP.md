# Jupyter Notebook Setup Guide

## ✅ Quick Setup (Already Done!)

The Jupyter kernel has been registered. Your notebooks should now work!

## How to Use the Correct Interpreter

### Option 1: Select Kernel in Jupyter (Recommended)

1. **Open Jupyter Notebook/Lab:**
   ```bash
   cd /Users/tusharchopra/Downloads/AIML/ASD_ADHD_Detection
   jupyter notebook
   # or
   jupyter lab
   ```

2. **When opening a notebook:**
   - Click on "Kernel" → "Change Kernel"
   - Select **"Python (ASD_ADHD_Detection)"**
   - The notebook will now use the virtual environment with all dependencies

### Option 2: Set Kernel in Notebook Metadata

If you want to set the kernel permanently for all notebooks, the kernel name is:
- **Kernel name:** `asd_adhd_detection`
- **Display name:** `Python (ASD_ADHD_Detection)`

### Option 3: Verify Setup

Run this to check if the kernel is available:
```bash
jupyter kernelspec list
```

You should see `asd_adhd_detection` in the list.

## Troubleshooting

### Issue: "Kernel not found" or "No module named X"

**Solution:** Make sure you're using the correct kernel:
1. In Jupyter, go to Kernel → Change Kernel
2. Select "Python (ASD_ADHD_Detection)"
3. If it's not listed, run the setup script again:
   ```bash
   python3 setup_jupyter_kernel.py
   ```

### Issue: "Interpreter not found"

**Solution:** The virtual environment might not be set up correctly:
```bash
cd /Users/tusharchopra/Downloads/AIML/ASD_ADHD_Detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 setup_jupyter_kernel.py
```

### Issue: Dependencies missing

**Solution:** Install all dependencies:
```bash
cd /Users/tusharchopra/Downloads/AIML/ASD_ADHD_Detection
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Test

Open any notebook and run this cell to verify:
```python
import sys
print(f"Python: {sys.version}")
print(f"Python path: {sys.executable}")
print(f"Virtual env: {'venv' in sys.executable}")

# Test imports
import numpy as np
import librosa
import tensorflow as tf
print("✅ All imports successful!")
```

You should see the Python path pointing to `venv/bin/python`.

## Alternative: Run Notebooks Directly

If Jupyter is still having issues, you can also run notebooks programmatically:

```bash
cd /Users/tusharchopra/Downloads/AIML/ASD_ADHD_Detection
source venv/bin/activate
jupyter nbconvert --to notebook --execute notebooks/02_data_preparation_and_training.ipynb --inplace
```

Or use `papermill` for parameterized execution.

---

**The kernel is now set up!** Just select "Python (ASD_ADHD_Detection)" from the kernel menu in Jupyter.

