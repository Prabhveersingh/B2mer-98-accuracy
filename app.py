import streamlit as st
import gdown
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠")

# Google Drive file ID (aapka model)
FILE_ID = "1jZyvE_DG_K7zoSaozaEo-_Aj_aDonNo0"
MODEL_PATH = "braintumordetectmodel.h5"

@st.cache_resource
def load_model_from_drive():
    """Download model from Google Drive only once"""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model (319MB)... First load only"):
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
    
    return load_model(MODEL_PATH)

model = load_model_from_drive()

# UI (same as your working repo style)
st.title("🧠 Brain Tumor Detection")
st.write("Upload MRI scan - Model will detect Tumor or No Tumor")

uploaded_file = st.file_uploader("Choose MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="MRI Scan", use_column_width=True)
    
    # Preprocess (model expects 180x180 as per original)
    img = image.resize((180, 180))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    with st.spinner("Analyzing..."):
        prediction = model.predict(img_array, verbose=0)[0]
    
    # Show result
    if prediction > 0.5:
        st.error("⚠️ **Tumor Detected**")
        st.write(f"Confidence: {prediction[0]:.2%}")
    else:
        st.success("✅ **No Tumor Detected**")
        st.write(f"Confidence: {1-prediction[0]:.2%}")