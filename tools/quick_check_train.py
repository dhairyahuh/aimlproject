"""
Quick check trainer: loads f:/AIML/data splits and runs a short training of the MLP
architecture adapted to the loaded feature dimension. Prints evaluation metrics.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

root_dir = r"f:/AIML"
data_dir = os.path.join(root_dir, 'data')

# Load splits
X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

print('Loaded data shapes:')
print(' X_train', X_train.shape)
print(' X_val  ', X_val.shape)
print(' X_test ', X_test.shape)
print(' y_train', y_train.shape)

# Combine train+val for scaler fit or fit on train only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

input_dim = X_train.shape[1]
num_classes = len(np.unique(y_train))
print(f'Input dim: {input_dim}, classes: {num_classes}')

# Build MLP matching project architecture but adapt input size
model = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),

    layers.Dense(64, kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),

    layers.Dense(32, kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),

    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# Quick training
history = model.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val), epochs=5, batch_size=32, verbose=2)

# Evaluate on test
pred_prob = model.predict(X_test_scaled)
pred = np.argmax(pred_prob, axis=1)
acc = accuracy_score(y_test, pred)
print('\nTest accuracy:', acc)
print('\nClassification report:\n', classification_report(y_test, pred))

# Save quick model to external_helpers for later use
outpath = os.path.join(root_dir, 'ASD_ADHD_Detection', 'external_helpers', 'quick_mlp.h5')
model.save(outpath)
print('\nSaved quick model to', outpath)
