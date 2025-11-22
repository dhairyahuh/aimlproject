"""
Integration helper - copies helper scripts from repository root into
ASD_ADHD_Detection/external_helpers and runs a quick check using available
precomputed data and saved models.

Usage: python integrate_helpers.py
"""
import os
import shutil
import traceback
import joblib
import numpy as np

root_dir = r"f:/AIML"
dest_dir = os.path.join(root_dir, 'ASD_ADHD_Detection', 'external_helpers')
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

helpers = [
    'mfcc_extract.py',
    'extract_audio.py',
    'ser_preprocessing.py',
    'spectrogram_conversion.py',
    'extractBERT.py',
    'predictor.py',
    'model.py'
]

print('Scanning root for helper scripts...')
found = []
for h in helpers:
    p = os.path.join(root_dir, h)
    if os.path.exists(p):
        found.append((h, p))

if not found:
    print('No helper scripts found in root. Exiting.')
else:
    print(f'Found {len(found)} helper scripts. Copying to {dest_dir}')
    for name, p in found:
        try:
            shutil.copy(p, os.path.join(dest_dir, name))
            print(f' - Copied {name}')
        except Exception as e:
            print(f' - Failed to copy {name}: {e}')

# Quick data split check
data_dir = os.path.join(root_dir, 'data')
expected = ['X_train.npy','X_val.npy','X_test.npy','y_train.npy','y_val.npy','y_test.npy']
print('\nChecking for precomputed Data splits in:', data_dir)
available_splits = {f: os.path.exists(os.path.join(data_dir, f)) for f in expected}
for k,v in available_splits.items():
    print(f' - {k}: {v}')

if all(available_splits.values()):
    try:
        X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
        X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
        X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
        y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        print('\nLoaded splits:')
        print(f' X_train: {X_train.shape}')
        print(f' X_val:   {X_val.shape}')
        print(f' X_test:  {X_test.shape}')
        print(f' y_train: {y_train.shape}')
        print(f' y_test:  {y_test.shape}')
    except Exception as e:
        print('Error loading .npy splits:', e)
        traceback.print_exc()
else:
    print('\nNot all splits available - skipping data load.')

# Quick model artifact check
model_files = ['rf.pkl','svm.pkl','ann.pkl','model.pkl','model.json']
print('\nScanning for saved models in root:')
for m in model_files:
    mpath = os.path.join(root_dir, m)
    print(f' - {m}:', os.path.exists(mpath))

# Try to load any joblib model and run a quick predict if X_test exists
if os.path.exists(os.path.join(root_dir, 'rf.pkl')) or os.path.exists(os.path.join(root_dir, 'svm.pkl')) or os.path.exists(os.path.join(root_dir, 'ann.pkl')):
    for m in ['rf.pkl','svm.pkl','ann.pkl']:
        mpath = os.path.join(root_dir, m)
        if os.path.exists(mpath):
            try:
                mdl = joblib.load(mpath)
                print(f'\nLoaded model {m} -> type: {type(mdl)}')
                sample = None
                if all(available_splits.values()):
                    # Use first test sample
                    sample = X_test[:1]
                else:
                    # Try features folder
                    feats_dir = os.path.join(root_dir, 'features')
                    if os.path.exists(feats_dir):
                        files = [f for f in os.listdir(feats_dir) if f.endswith('.npy')]
                        if files:
                            arr = np.load(os.path.join(feats_dir, files[0]))
                            # Flatten or average to match model input shape
                            sample = arr.reshape(1, -1)
                if sample is not None:
                    try:
                        pred = mdl.predict(sample)
                        print(' Quick prediction:', pred)
                    except Exception as e:
                        print(' Model loaded but prediction failed (shape mismatch?):', e)
                else:
                    print(' No sample available to run prediction')
            except Exception as e:
                print(' Failed loading model', m, ':', e)
                traceback.print_exc()

print('\nIntegration helper finished.')
