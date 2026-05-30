import os
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from tensorflow.keras.models import load_model
from PIL import Image
import gdown

app = FastAPI()

# Google Drive model ID
FILE_ID = "1jZyvE_DG_K7zoSaozaEo-_Aj_aDonNo0"
MODEL_PATH = "/tmp/braintumordetectmodel.h5"

# Global variable for cached model
_model = None

def get_model():
    """Load model once and cache it in memory"""
    global _model
    if _model is None:
        # Download only if file doesn't exist
        if not os.path.exists(MODEL_PATH):
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
        _model = load_model(MODEL_PATH)
    return _model

# HTML form (same as before)
html_form = """
<!DOCTYPE html>
<html>
<head>
    <title>Brain Tumor Detection</title>
    <style>
        body { font-family: Arial; text-align: center; padding: 50px; background: linear-gradient(135deg, #0f0c29, #302b63); color: white; min-height: 100vh; margin: 0; }
        .container { max-width: 600px; margin: auto; background: rgba(255,255,255,0.1); border-radius: 20px; padding: 30px; backdrop-filter: blur(10px); }
        input { margin: 20px; padding: 10px; background: white; border: none; border-radius: 10px; cursor: pointer; }
        button { background: #ff66ff; color: white; border: none; padding: 12px 30px; border-radius: 25px; cursor: pointer; font-size: 16px; margin-top: 10px; }
        button:hover { background: #ff33ff; }
        .result { margin-top: 20px; padding: 20px; border-radius: 15px; }
        .tumor { background: rgba(255,0,0,0.3); border-left: 5px solid #ff4d4d; }
        .normal { background: rgba(0,255,0,0.3); border-left: 5px solid #4caf50; }
        h1 { margin-bottom: 30px; }
        img { margin-top: 20px; max-width: 100%; border-radius: 15px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Brain Tumor Detection</h1>
        <p>Upload an MRI scan to check for tumors</p>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required>
            <br>
            <button type="submit">Analyze MRI</button>
        </form>
        <div id="result"></div>
    </div>
</body>
</html>
"""

@app.get("/")
async def home():
    return HTMLResponse(content=html_form, status_code=200)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Get cached model
    model = get_model()
    
    # Read and preprocess image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Display uploaded image in result
    import base64
    img_str = base64.b64encode(contents).decode()
    
    # Preprocess (180x180 as model expects)
    img_resized = image.resize((180, 180))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    prediction = model.predict(img_array, verbose=0)[0][0]
    is_tumor = prediction > 0.5
    confidence = prediction if is_tumor else 1 - prediction
    
    result_html = f"""
    <div class="result {'tumor' if is_tumor else 'normal'}">
        <img src="data:image/jpeg;base64,{img_str}" style="max-width: 300px; border-radius: 15px; margin-bottom: 15px;">
        <h2>{'⚠️ TUMOR DETECTED' if is_tumor else '✅ NO TUMOR DETECTED'}</h2>
        <p>Confidence: {confidence:.2%}</p>
        <a href="/" style="color: #aaffff;">← Upload Another Scan</a>
    </div>
    """
    return HTMLResponse(content=result_html, status_code=200)