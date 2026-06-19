import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Set up the webpage design
st.set_page_config(page_title="Face Mask Detector", page_icon="😷", layout="centered")

st.title("Face Mask Detector 😷")
st.write("Upload an image or use your webcam to check if people are wearing masks!")

# Load the trained model (Using @st.cache_resource so it only loads once)
@st.cache_resource
def load_mask_model():
    # Make sure your trained model file is named EXACTLY 'mask_model.h5' and is in the same folder!
    return load_model('mask_model.h5')

try:
    model = load_mask_model()
except Exception as e:
    st.error("🚨 Could not find 'mask_model.h5'. Please make sure you saved your trained model and uploaded it to GitHub!")
    st.stop()

# Load the Face Detection Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the prediction function
def detect_and_predict_mask(image):
    # Convert PIL Image to NumPy array (Streamlit uses RGB natively)
    img_array = np.array(image)
    
    # Haar Cascade needs a grayscale image to find faces
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Find faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    
    for (x, y, w, h) in faces:
        # Extract just the face area
        face_roi = img_array[y:y+h, x:x+w]
        
        # Resize to 224x224 and normalize (Exactly like our Jupyter Notebook!)
        face_resized = cv2.resize(face_roi, (224, 224))
        face_normalized = face_resized.astype('float32') / 255.0
        face_expanded = np.expand_dims(face_normalized, axis=0)
        
        # Make the prediction
        prediction = model.predict(face_expanded, verbose=0)
        
        # Determine the label and color
        if prediction[0][0] > 0.5:
            label = "No Mask"
            color = (255, 0, 0) # Red in RGB
        else:
            label = "Mask ON"
            color = (0, 255, 0) # Green in RGB
            
        # Draw the rectangle and text on the image
        cv2.rectangle(img_array, (x, y), (x+w, y+h), color, 3)
        cv2.putText(img_array, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        
    return img_array, len(faces)

# --- WEB APP INTERFACE ---

# Let the user choose how to input a photo
option = st.radio("Choose input method:", ("Take a Picture", "Upload Image"))

image_file = None
if option == "Take a Picture":
    image_file = st.camera_input("Smile for the camera!")
else:
    image_file = st.file_uploader("Upload an image from your computer", type=["jpg", "jpeg", "png"])

# Process the image if one was provided
if image_file is not None:
    # Open the image using Pillow
    image = Image.open(image_file)
    
    if st.button("Detect Face Mask"):
        with st.spinner("Analyzing..."):
            # Run our AI pipeline
            processed_image, face_count = detect_and_predict_mask(image)
            
            # Show the final result!
            st.image(processed_image, caption="Detection Result", use_column_width=True)
            
            if face_count == 0:
                st.warning("No faces were detected in this image. Try another one!")
            else:
                st.success(f"Successfully analyzed {face_count} face(s)!")
