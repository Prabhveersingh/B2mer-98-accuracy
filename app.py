import streamlit as st
import gdown
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="wide")

# Google Drive file ID (aapka model)
FILE_ID = "1jZyvE_DG_K7zoSaozaEo-_Aj_aDonNo0"
MODEL_PATH = "braintumordetectmodel.h5"

@st.cache_resource
def load_model_from_drive():
    """Download model from Google Drive only once, then cache"""
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        with st.spinner("📥 Downloading model (319MB)... First load only, please wait"):
            gdown.download(url, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

# Load model (cached after first download)
model = load_model_from_drive()

# UI Styling
st.markdown("""
<style>
.glass-card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    border-radius: 32px;
    padding: 1.5rem;
    border: 1px solid rgba(255,255,255,0.2);
}
.neon-text {
    font-size: 2.5rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(135deg, #aaffff, #ff66ff);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
.result-tumor {
    border-left: 6px solid #ff4d4d;
    background: linear-gradient(135deg, rgba(255,0,0,0.1), rgba(255,0,0,0.05));
    border-radius: 24px;
    padding: 1.5rem;
    margin-top: 1.5rem;
    text-align: center;
}
.result-normal {
    border-left: 6px solid #4caf50;
    background: linear-gradient(135deg, rgba(76,175,80,0.1), rgba(76,175,80,0.05));
    border-radius: 24px;
    padding: 1.5rem;
    margin-top: 1.5rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="glass-card"><p class="neon-text">🧠 Brain Tumor Detection</p></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Drop your MRI scan here", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="🧬 Uploaded MRI", use_container_width=True)
    
    # Preprocess (model expects 180x180)
    img_resized = image.resize((180, 180))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    with st.spinner("🔬 Analyzing MRI scan..."):
        prediction = model.predict(img_array, verbose=0)[0]
    
    # Show result
    if prediction > 0.5:
        st.markdown("""
        <div class="result-tumor">
            <h1 style="color:#ff4d4d; margin:0;">🧠⚠️</h1>
            <h1 style="color:#ff4d4d; margin:0;">Tumor Detected</h1>
        </div>
        """, unsafe_allow_html=True)
        st.write(f"Confidence: {prediction[0]:.2%}")
    else:
        st.markdown("""
        <div class="result-normal">
            <h1 style="color:#4caf50; margin:0;">🧠✅</h1>
            <h1 style="color:#4caf50; margin:0;">No Tumor Detected</h1>
        </div>
        """, unsafe_allow_html=True)
        st.write(f"Confidence: {1-prediction[0]:.2%}")

else:
    st.markdown("""
    <div style="text-align:center; padding:2rem; background:rgba(255,255,255,0.03); border-radius:32px;">
        <p style="color:#aaa;">🌟 Upload an MRI image to begin scanning</p>
        <small style="color:#555;">JPG, PNG supported</small>
    </div>
    """, unsafe_allow_html=True)