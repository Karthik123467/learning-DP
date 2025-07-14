import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import requests

# Define model path and Google Drive file ID
model_path = "flower_model.keras"
drive_file_id = "1qJfWFr7NoDxEyPP7znGQLGl870mYTa4I"

def download_file_from_google_drive(file_id, destination):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    with requests.get(url, stream=True) as r:
        if r.status_code != 200:
            raise Exception(f"Download failed with status {r.status_code}")
        with open(destination, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

# Download model if not already present
if not os.path.exists(model_path):
    st.info("📥 Downloading model from Google Drive...")
    try:
        download_file_from_google_drive(drive_file_id, model_path)
        st.success("✅ Model downloaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to download model: {e}")
        st.stop()

# Load the model
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

class_names = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']

# UI Header
st.markdown("""
    <h1 style='text-align: center; color: #9c27b0;'>🌸 Flower Image Classifier</h1>
    <p style='text-align: center; color: yellow;'>
    The model can classify the following flowers: <b>Roses</b>, <b>Daisy</b>, <b>Dandelion</b>, <b>Sunflowers</b>, <b>Tulips</b>.
    </p>
    <hr style="border: 1px solid #e0e0e0;">
""", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📤 Upload a flower image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = image.resize((180, 180))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = prediction[predicted_index] * 100

    # Display result
    st.markdown(f"""
        <div style="text-align: center; margin-top: 20px;">
            <h3>🔍 Prediction: <span style="color:#4caf50;">{predicted_class.capitalize()}</span></h3>
            <p>💡 Confidence: <b>{confidence:.2f}%</b></p>
        </div>
    """, unsafe_allow_html=True)

    # Bar chart
    st.subheader("📊 Prediction Confidence")
    fig, ax = plt.subplots(figsize=(7, 2))
    bars = ax.bar(class_names, prediction, color='#7e57c2')
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    st.pyplot(fig)

# Footer
st.markdown("""
    <hr>
    <div style="text-align:center;">
        <small>Created by <b>Pavankumar</b> | 💻 Powered by TensorFlow & Streamlit</small>
    </div>
""", unsafe_allow_html=True)
