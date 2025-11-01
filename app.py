import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("YOLOv8 Waste Detection")
st.write("Upload an image to detect waste categories using your trained model.")

@st.cache_resource
def load_model():
    # Load your trained model (make sure best.pt is in the same folder or correct path)
    model = YOLO("best.pt")
    return model

model = load_model()

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")   # ✅ Ensures consistent color format
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("Detecting objects...")

    # ✅ Convert PIL → numpy array for consistent prediction
    results = model.predict(np.array(image), conf=0.5)

    # ✅ Plot the first result correctly
    res_plotted = results[0].plot()

    st.image(res_plotted, caption='Detection Result', use_container_width=True)