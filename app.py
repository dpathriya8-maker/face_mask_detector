import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os
import requests

# Set up the webpage design
st.set_page_config(page_title="Face Mask Detector", page_icon="😷", layout="centered")

st.title("Face Mask Detector 😷")
st.write("Upload an image or use your webcam to check if people are wearing masks!")

MODEL_PATH = "face_mask_detector.h5"
MODEL_URL = "https://github.com/dpathriya8-maker/face_mask_detector/releases/download/v1.0.0/face_mask_detector.h5"

# Improved download logic that follows GitHub redirects properly
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading the AI Model from GitHub Releases... This might take a minute."):
        try:
            # stream=True allows us to download large files smoothly
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status() # Raise an error if the download failed
            
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download the model. Error: {e}")
            st.stop()

# Load the model normally
# Load the model normally (ignoring version compile differences)
@st.cache_resource
def load_mask_model():
    return load_model(MODEL_PATH, compile=False)

try:
    model = load_mask_model()
except Exception as e:
    st.error("🚨 Error loading the model!")
    st.exception(e)  # This will print the full technical error trace on your screen
    st.stop()

# Load the Face Detection Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the prediction function
def detect_and_predict_mask(image):
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    
    for (x, y, w, h) in faces:
        face_roi = img_array[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (224, 224))
        face_normalized = face_resized.astype('float32') / 255.0
        face_expanded = np.expand_dims(face_normalized, axis=0)
        
        prediction = model.predict(face_expanded, verbose=0)
        
        if prediction[0][0] > 0.5:
            label = "No Mask"
            color = (255, 0, 0) 
        else:
            label = "Mask ON"
            color = (0, 255, 0) 
            
        cv2.rectangle(img_array, (x, y), (x+w, y+h), color, 3)
        cv2.putText(img_array, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        
    return img_array, len(faces)

# --- WEB APP INTERFACE ---
option = st.radio("Choose input method:", ("Take a Picture", "Upload Image"))

image_file = None
if option == "Take a Picture":
    image_file = st.camera_input("Smile for the camera!")
else:
    image_file = st.file_uploader("Upload an image from your computer", type=["jpg", "jpeg", "png"])

if image_file is not None:
    image = Image.open(image_file)
    
    if st.button("Detect Face Mask"):
        with st.spinner("Analyzing..."):
            processed_image, face_count = detect_and_predict_mask(image)
            st.image(processed_image, caption="Detection Result", use_column_width=True)
            
            if face_count == 0:
                st.warning("No faces were detected in this image. Try another one!")
            else:
                st.success(f"Successfully analyzed {face_count} face(s)!")
