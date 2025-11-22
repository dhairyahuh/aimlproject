"""
MLP Classifier for ASD/ADHD Detection
======================================
Multi-Layer Perceptron neural network for 3-class classification:
- Class 0: Healthy
- Class 1: ASD (Autism Spectrum Disorder)
- Class 2: ADHD (Attention-Deficit/Hyperactivity Disorder)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from pathlib import Path
import pickle
import json
from datetime import datetime
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class MLPClassifier:
    """
    Multi-Layer Perceptron classifier for ASD/ADHD voice detection.
    
    Architecture:
    - Input: 106 features (MFCC + Spectral + Prosodic)
    - Hidden Layers: 128 → 64 → 32 neurons
    - Output: 3 classes (Softmax)
    """
    
    def __init__(self, config=None, input_dim: int = 106, n_classes: int = 3):
        """
        Initialize MLP Classifier.
        
        Args:
            config: Configuration object (optional)
            input_dim: Input feature dimension (default 106)
            n_classes: Number of output classes (default 3)
        """
        self.config = config
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.model = None
        self.history = None
        self.is_trained = False
        
        # Class labels
        self.class_labels = {0: 'Healthy', 1: 'ASD', 2: 'ADHD'}
        
    def build(self, hidden_layers: List[int] = None, dropout_rate: float = 0.3,
              l2_reg: float = 1e-4, learning_rate: float = 0.001):
        """
        Build the MLP model architecture.
        
        Args:
            hidden_layers: List of hidden layer sizes (default [128, 64, 32])
            dropout_rate: Dropout probability (default 0.3)
            l2_reg: L2 regularization factor (default 1e-4)
            learning_rate: Learning rate for optimizer (default 0.001)
        """
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]
        
        # Build sequential model
        self.model = models.Sequential()
        
        # Input layer
        self.model.add(layers.Dense(
            hidden_layers[0],
            input_dim=self.input_dim,
            kernel_regularizer=regularizers.l2(l2_reg),
            name='dense_1'
        ))
        self.model.add(layers.BatchNormalization(name='bn_1'))
        self.model.add(layers.Activation('relu', name='relu_1'))
        self.model.add(layers.Dropout(dropout_rate, name='dropout_1'))
        
        # Hidden layers
        for i, units in enumerate(hidden_layers[1:], start=2):
            self.model.add(layers.Dense(
                units,
                kernel_regularizer=regularizers.l2(l2_reg),
                name=f'dense_{i}'
            ))
            self.model.add(layers.BatchNormalization(name=f'bn_{i}'))
            self.model.add(layers.Activation('relu', name=f'relu_{i}'))
            self.model.add(layers.Dropout(dropout_rate, name=f'dropout_{i}'))
        
        # Output layer
        self.model.add(layers.Dense(
            self.n_classes,
            activation='softmax',
            name='output'
        ))
        
        # Compile model
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✓ MLP model built successfully")
        print(f"  Architecture: {self.input_dim} → {hidden_layers} → {self.n_classes}")
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 100, batch_size: int = 32,
              validation_split: float = 0.15, verbose: int = 1,
              callbacks_list: List = None) -> Dict:
        """
        Train the MLP model.
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split ratio if X_val not provided
            verbose: Verbosity level
            callbacks_list: List of Keras callbacks
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        # Prepare validation data
        val_data = None
        if X_val is not None and y_val is not None:
            val_data = (X_val, y_val)
            validation_split = None
        elif validation_split > 0:
            val_data = None
        
        # Default callbacks
        if callbacks_list is None:
            callbacks_list = [
                callbacks.EarlyStopping(
                    monitor='val_loss' if val_data else 'loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss' if val_data else 'loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                    verbose=1
                )
            ]
        
        # Train model
        print("\n" + "="*70)
        print("TRAINING MLP MODEL")
        print("="*70)
        print(f"Training samples: {X_train.shape[0]}")
        if val_data:
            print(f"Validation samples: {X_val.shape[0]}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        print("-"*70)
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=val_data,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=verbose,
            shuffle=True
        )
        
        self.is_trained = True
        print("\n✓ Training completed!")
        
        return self.history.history
    
    def predict(self, X: np.ndarray, return_proba: bool = False) -> np.ndarray:
        """
        Make predictions on input features.
        
        Args:
            X: Input features (n_samples, n_features)
            return_proba: If True, return probability distribution
        
        Returns:
            Predicted class labels or probability distribution
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        predictions = self.model.predict(X, verbose=0)
        
        if return_proba:
            return predictions
        else:
            return predictions.argmax(axis=1)
    
    def predict_realtime(self, audio_file: str = None, audio_array: np.ndarray = None,
                        sr: int = 16000, feature_extractor=None) -> Dict:
        """
        Predict from audio file or array for real-time inference.
        
        Args:
            audio_file: Path to audio file
            audio_array: Audio signal array
            sr: Sample rate
            feature_extractor: Feature extractor instance
        
        Returns:
            Dictionary with prediction, confidence, and class label
        """
        if feature_extractor is None:
            raise ValueError("Feature extractor required for real-time prediction")
        
        # Extract features
        if audio_file:
            features = feature_extractor.extract_all_features(audio_file)
        elif audio_array is not None:
            # For real-time, we need to save temporarily or use different method
            # This is a simplified version - you may need to adapt
            raise NotImplementedError("Direct audio array prediction not yet implemented")
        else:
            raise ValueError("Either audio_file or audio_array must be provided")
        
        # Reshape for prediction (add batch dimension)
        features = features.reshape(1, -1)
        
        # Predict
        proba = self.predict(features, return_proba=True)[0]
        predicted_class = int(np.argmax(proba))
        confidence = float(proba[predicted_class])
        
        return {
            'class': predicted_class,
            'class_label': self.class_labels[predicted_class],
            'confidence': confidence,
            'probabilities': {
                'Healthy': float(proba[0]),
                'ASD': float(proba[1]),
                'ADHD': float(proba[2])
            }
        }
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate model on test data.
        
        Args:
            X: Test features
            y: Test labels
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        loss, accuracy = self.model.evaluate(X, y, verbose=0)
        
        predictions = self.predict(X)
        
        from sklearn.metrics import (
            precision_score, recall_score, f1_score,
            confusion_matrix, classification_report
        )
        
        precision = precision_score(y, predictions, average='weighted', zero_division=0)
        recall = recall_score(y, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y, predictions, average='weighted', zero_division=0)
        
        cm = confusion_matrix(y, predictions)
        report = classification_report(y, predictions, output_dict=True, zero_division=0)
        
        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
    
    def save(self, save_path: str, save_scaler: bool = False, scaler=None) -> None:
        """
        Save model to disk.
        
        Args:
            save_path: Path to save model
            save_scaler: Whether to save scaler
            scaler: Scaler object to save
        """
        if self.model is None:
            raise ValueError("No model to save. Train or load a model first.")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save(str(save_path))
        print(f"✓ Model saved to: {save_path}")
        
        # Save scaler if provided
        if save_scaler and scaler is not None:
            scaler_path = save_path.parent / f"{save_path.stem}_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"✓ Scaler saved to: {scaler_path}")
        
        # Save metadata
        metadata = {
            'input_dim': self.input_dim,
            'n_classes': self.n_classes,
            'class_labels': self.class_labels,
            'is_trained': self.is_trained,
            'saved_at': datetime.now().isoformat()
        }
        
        metadata_path = save_path.parent / f"{save_path.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved to: {metadata_path}")
    
    def load(self, load_path: str, load_scaler: bool = False) -> Optional[object]:
        """
        Load model from disk.
        
        Args:
            load_path: Path to saved model
            load_scaler: Whether to load scaler
        
        Returns:
            Scaler object if load_scaler=True, else None
        """
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model not found: {load_path}")
        
        # Load model
        self.model = keras.models.load_model(str(load_path))
        self.is_trained = True
        print(f"✓ Model loaded from: {load_path}")
        
        # Load metadata if available
        metadata_path = load_path.parent / f"{load_path.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.input_dim = metadata.get('input_dim', self.input_dim)
                self.n_classes = metadata.get('n_classes', self.n_classes)
                self.class_labels = metadata.get('class_labels', self.class_labels)
        
        # Load scaler if requested
        scaler = None
        if load_scaler:
            scaler_path = load_path.parent / f"{load_path.stem}_scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                print(f"✓ Scaler loaded from: {scaler_path}")
        
        return scaler
    
    def summary(self) -> None:
        """Print model summary."""
        if self.model is None:
            print("Model not built yet.")
        else:
            self.model.summary()

