import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

st.title("YOLOv8 Waste Detection")
st.write("Upload an image to detect waste categories using your trained model.")

@st.cache_resource
def load_model():
    return YOLO("best.pt")  # ensure correct path

model = load_model()

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read as RGB
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.image(img_array, caption="Uploaded Image", use_container_width=True)
    st.write("Detecting objects...")

    results = model.predict(img_array, conf=0.5)

    # Get first result
    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    classes = result.boxes.cls.cpu().numpy().astype(int)
    names = result.names

    # Copy image to draw on
    drawn = img_array.copy()

    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0)  # green boxes
        label = names[cls] if cls in names else str(cls)

        # Draw rectangle and label
        cv2.rectangle(drawn, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            drawn,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            lineType=cv2.LINE_AA,
        )

    # Show the drawn image (no color distortion)
    st.image(drawn, caption="Detection Result (True Colors)", use_container_width=True)
