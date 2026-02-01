import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

st.set_page_config(page_title="Face Detection", layout="wide")

st.title("🎥 Live Face Detection (YOLOv8 + WebRTC)")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

conf_threshold = st.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.7,
    step=0.05
)

class FaceProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        results = model.predict(
            source=img,
            conf=conf_threshold,
            imgsz=640,
            verbose=False
        )

        annotated = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

webrtc_streamer(
    key="face-detection",
    video_processor_factory=FaceProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
