import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2  # ✅ Added for color conversion

st.title("YOLOv8 Waste Detection")
st.write("Upload an image to detect waste categories using your trained model.")

# ---------------------------------------------------------
# ✅ Load the YOLO model (cached to avoid reloading each time)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    # Make sure 'best.pt' path is correct (same folder or specify full path)
    model = YOLO("best.pt")
    return model

model = load_model()

# ---------------------------------------------------------
# ✅ File upload section
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded image to RGB PIL image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("Detecting objects...")

    # Convert PIL → NumPy array for YOLOv8
    img_array = np.array(image)

    # Run YOLOv8 model on the uploaded image
    results = model.predict(img_array, conf=0.5)

    # ---------------------------------------------------------
    # ✅ FIX: Convert BGR → RGB before displaying in Streamlit
    # ---------------------------------------------------------
    res_plotted = results[0].plot()  # YOLOv8 returns a BGR image
    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

    # Display final detection image
    st.image(res_rgb, caption='Detection Result', use_container_width=True)
