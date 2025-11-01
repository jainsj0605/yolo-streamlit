import streamlit as st
from ultralytics import YOLO
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np

st.set_page_config(page_title="YOLOv8 Live Detection", layout="wide")
st.title("📸 YOLOv8 Live Object Detection")
st.caption("Allow webcam access to start real-time detection 🔒")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # your model file
    return model

model = load_model()

confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.25, 0.05)
imgsz = st.sidebar.selectbox("Image size", [320, 480, 640, 800], index=2)
st.sidebar.markdown("---")
st.sidebar.info("Running model: `best.pt`")

class YOLOVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = self.model.predict(img, conf=confidence, imgsz=imgsz, verbose=False)
        annotated = results[0].plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

webrtc_streamer(
    key="yolo",
    mode="SENDRECV",
    rtc_configuration=RTC_CONFIGURATION,
    video_transformer_factory=YOLOVideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
)
