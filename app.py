import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

st.title("YOLOv8 Waste Detection")
st.write("Upload an image to detect waste categories using your trained model.")

# -----------------------------
# ✅ Load YOLOv8 Model
# -----------------------------
@st.cache_resource
def load_model():
    model = YOLO("best.pt")   # ensure path is correct
    return model

model = load_model()

# -----------------------------
# ✅ Upload Image
# -----------------------------
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image using PIL (always RGB)
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("Detecting objects...")

    # Convert PIL → NumPy
    img_array = np.array(image)

    # -----------------------------
    # ✅ Run YOLOv8 detection
    # -----------------------------
    results = model.predict(img_array, conf=0.5)

    # YOLOv8 plot output (OpenCV BGR image)
    res_plotted = results[0].plot()

    # -----------------------------
    # ✅ Color and dtype correction
    # -----------------------------
    # Sometimes YOLOv8 returns uint8 BGR array; Streamlit expects RGB float32 or uint8 in [0–255]
    if res_plotted.dtype != np.uint8:
        res_plotted = (res_plotted * 255).astype(np.uint8)

    # Convert BGR → RGB safely
    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

    # -----------------------------
    # ✅ Display the final detection image
    # -----------------------------
    st.image(res_rgb, caption='Detection Result', use_container_width=True)
