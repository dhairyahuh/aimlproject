#!/usr/bin/env python3
"""
Streamlit UI for autism detection from audio files.
Run with: streamlit run 07_streamlit_ui.py
"""

import io
import tempfile
import warnings
from pathlib import Path

import joblib
import librosa
import numpy as np
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Configuration - All paths relative to ASD_ADHD_Detection folder
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models" / "saved"

MODEL_INFO = {
    "rf.pkl": "Random Forest",
    "ann.pkl": "Artificial Neural Network",
    "svm.pkl": "Support Vector Machine",
    "nb.pkl": "Naive Bayes",
}

N_MFCC = 40  # Must match feature extraction


def load_model(model_filename: str):
    """Load a trained model."""
    model_path = MODELS_DIR / model_filename
    if not model_path.exists():
        st.error(f"Model not found: {model_path}\nPlease train models first using 02_train_models.py")
        st.stop()
    return joblib.load(model_path)


def extract_features(file_bytes: bytes) -> np.ndarray:
    """Extract MFCC features from audio file bytes."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        y, sr = librosa.load(tmp_path, sr=None, mono=True)
        if y.size == 0:
            raise ValueError("Empty audio signal")

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        if np.isnan(mfcc).any():
            raise ValueError("MFCC contains NaN values")

        # Average across time to match training pipeline
        mfcc_avg = np.mean(mfcc, axis=1)
        return mfcc_avg.reshape(1, -1)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# Streamlit UI
st.set_page_config(page_title="Autism Detection", page_icon="🎤", layout="wide")

st.title("🎤 Autism Detection from Audio")
st.markdown("Upload an audio file to detect autism using trained machine learning models.")

# Model selection
st.sidebar.header("Model Selection")
selected_model_label = st.sidebar.selectbox(
    "Choose a model",
    list(MODEL_INFO.values())
)
model_filename = next(
    key for key, value in MODEL_INFO.items() 
    if value == selected_model_label
)

# Load model
try:
    model = load_model(model_filename)
    st.sidebar.success(f"✅ Loaded: {selected_model_label}")
except Exception as e:
    st.sidebar.error(f"❌ Error loading model: {e}")
    st.stop()

# File upload
st.header("Upload Audio File")
uploaded_file = st.file_uploader(
    "Upload an audio file (m4a/wav/mp3)",
    type=["m4a", "wav", "mp3"],
    help="Upload a recorded audio file for analysis"
)

# Prediction
if uploaded_file is not None:
    st.header("Prediction Results")
    
    with st.spinner("Processing audio file..."):
        try:
            # Extract features
            features = extract_features(uploaded_file.getvalue())
            
            # Make prediction
            prediction = model.predict(features)[0]
            
            # Get probabilities if available
            try:
                probabilities = model.predict_proba(features)[0]
                has_proba = True
            except AttributeError:
                has_proba = False
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.markdown(
                        '<h2 style="color:red;text-align:center;">🔴 Prediction: Autistic</h2>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<h2 style="color:green;text-align:center;">🟢 Prediction: Non-Autistic</h2>',
                        unsafe_allow_html=True
                    )
            
            with col2:
                if has_proba:
                    st.subheader("Confidence Scores")
                    prob_non = probabilities[0] * 100
                    prob_aut = probabilities[1] * 100
                    st.metric("Non-Autistic", f"{prob_non:.1f}%")
                    st.metric("Autistic", f"{prob_aut:.1f}%")
                    
                    # Progress bars
                    st.progress(prob_non / 100)
                    st.caption("Non-Autistic probability")
                    st.progress(prob_aut / 100)
                    st.caption("Autistic probability")
            
            # Audio player
            st.audio(uploaded_file, format='audio/m4a')
            
        except Exception as e:
            st.error(f"❌ Could not process audio file: {e}")
            st.exception(e)

# Instructions
with st.expander("📖 Instructions"):
    st.markdown("""
    ### How to use:
    1. **Select a model** from the sidebar (Random Forest recommended)
    2. **Upload an audio file** in m4a, wav, or mp3 format
    3. **View the prediction** and confidence scores
    
    ### Model Information:
    - **Random Forest**: Usually highest accuracy (~90%)
    - **Artificial Neural Network**: Good performance (~72%)
    - **Naive Bayes**: Fast predictions (~81%)
    - **Support Vector Machine**: Moderate performance (~54%)
    
    ### Note:
    - Ensure audio files are clear recordings of speech
    - Models are trained on MFCC features extracted from audio
    - For best results, use models trained on similar audio conditions
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "ASD/ADHD Detection System | Trained on real autism recordings"
    "</div>",
    unsafe_allow_html=True
)

