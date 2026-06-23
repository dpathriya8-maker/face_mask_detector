# 😷 Face Mask Detector AI

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red?style=for-the-badge&logo=streamlit)

A real-time computer vision web application that detects whether a person is wearing a face mask. Built with a custom Convolutional Neural Network (using a VGG16 base) and deployed directly to the web using Streamlit.

🔗 **[Click here to try the live Web App!](INSERT_YOUR_STREAMLIT_LINK_HERE)**

## 🚀 Features
* **Real-Time Detection:** Uses OpenCV Haar Cascades to instantly map faces via webcam or uploaded images.
* **Deep Learning Pipeline:** Classifies the extracted face regions using a fine-tuned VGG16 architecture.
* **Cloud-Optimized Deployment:** Engineered to run efficiently on limited-RAM cloud servers by dynamically loading weights into a locally reconstructed Keras skeleton, bypassing standard version-conflict serialization issues.

## 🛠️ Tech Stack
* **Computer Vision:** OpenCV (`opencv-python-headless`)
* **Deep Learning:** TensorFlow / Keras (VGG16 Base + Dense Layers)
* **Web Framework:** Streamlit
* **Data Processing:** NumPy, Pillow

## 💻 Running it Locally
If you want to run this project on your own machine:

1. Clone this repository:
   ```bash
   git clone [https://github.com/your-username/face_mask_detector.git](https://github.com/your-username/face_mask_detector.git)
