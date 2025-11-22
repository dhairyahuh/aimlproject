import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf


# emotion mapping
emotion_mapping = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised"
}

DATA_PATH = "data/"

def extract_features(file_path, max_len=173):
    try:
        signal, sr = librosa.load(file_path, sr=22050)
        # Trim silence
        signal, _ = librosa.effects.trim(signal, top_db=20)
        # Pad or truncate
        if len(signal) > max_len * sr:
            signal = signal[:max_len * sr]
        else:
            pad_width = max_len * sr - len(signal)
            signal = np.pad(signal, (0, pad_width), 'constant')
        # Extract features
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}")
        return None
    
def load_data():
    emotions = []
    features = []

    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                # Extract emotion label
                file_name = os.path.splitext(file)[0]
                parts = file_name.split("-")
                emotion_code = int(parts[2])
                emotion = emotion_mapping.get(emotion_code)
                if emotion:
                    feature = extract_features(file_path)
                    if feature is not None:
                        features.append(feature)
                        emotions.append(emotion)

    # Create DataFrame
    df = pd.DataFrame(features)
    df['emotion'] = emotions
    return df

def main():
    df = load_data()
    print(f"Dataset Shape: {df.shape}")
    print(df.head())

    # Save to CSV for future use
    df.to_csv("data/features.csv", index=False)

    # Label Encoding
    le = LabelEncoder()
    df['emotion_label'] = le.fit_transform(df['emotion'])

    # Plot Emotion Distribution
    plt.figure(figsize=(10,6))
    sns.countplot(x='emotion', data=df)
    plt.title("Emotion Distribution")
    plt.savefig("emotion_distribution.png")
    plt.show()

    # Prepare Features and Labels
    X = df.drop(['emotion', 'emotion_label'], axis=1).values
    y = df['emotion_label'].values

    # Shuffle the data
    X, y = shuffle(X, y, random_state=42)

    # Train, Validation, Test Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")

    # Save the splits
    np.save("Data/X_train.npy", X_train)
    np.save("Data/X_val.npy", X_val)
    np.save("Data/X_test.npy", X_test)
    np.save("Data/y_train.npy", y_train)
    np.save("Data/y_val.npy", y_val)
    np.save("Data/y_test.npy", y_test)
    np.save("Data/le_classes.npy", le.classes_)

if __name__ == "__main__":
    main()



# Q1. Why during features Extract you have chosen max_len = 173




