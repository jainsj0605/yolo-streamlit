import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

st.title("YOLOv8 Waste Detection")
st.write("Upload an image to detect waste categories using your trained model.")

# ----------------------------------------------------------
# ✅ Load YOLOv8 model (cached for performance)
# ----------------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # ensure correct path

model = load_model()

# ----------------------------------------------------------
# ✅ Upload image and run detection
# ----------------------------------------------------------
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image in RGB
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.image(img_array, caption="Uploaded Image", use_container_width=True)
    st.write("Detecting objects...")

    # Run YOLOv8 prediction
    results = model.predict(img_array, conf=0.5)
    result = results[0]

    boxes = result.boxes.xyxy.cpu().numpy()          # bounding box coordinates
    classes = result.boxes.cls.cpu().numpy().astype(int)  # class IDs
    confs = result.boxes.conf.cpu().numpy()          # confidence scores (0–1)
    names = result.names                             # class name dictionary

    # Copy image to draw on
    drawn = img_array.copy()

    for box, cls, conf in zip(boxes, classes, confs):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0)  # Green for all boxes
        label = names.get(cls, str(cls))
        
        # ✅ Confidence value between 0 and 1 (e.g., 0.87)
        conf_text = f"{label} {conf:.2f}"

        # Draw rectangle
        cv2.rectangle(drawn, (x1, y1), (x2, y2), color, 2)

        # Background rectangle for text
        (text_w, text_h), baseline = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(drawn, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)

        # Put text (black text on green background)
        cv2.putText(
            drawn,
            conf_text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),  # Black text
            2,
            lineType=cv2.LINE_AA
        )

    # ✅ Display detection result (true colors + confidence)
    st.image(drawn, caption="Detection Result (with Confidence Scores 0–1)", use_container_width=True)
