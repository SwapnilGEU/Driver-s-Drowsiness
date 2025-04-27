import time
import cv2
from ultralytics import YOLO
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import threading
from playsound import playsound

# Load your YOLO model
model = YOLO("y_new2.pt")

# For sound alert
def play_alert():
    playsound("mixkit-rooster-crowing-in-the-morning-2462.wav")

# UI
st.title("Drowsiness Detection with YOLOv8 + Streamlit")
st.markdown("Detecting 'Closed Eye' with sound alert if detected for 3 seconds")

# Main video transformer
class DrowsinessTransformer(VideoTransformerBase):
    def __init__(self):
        self.drowsy_detected_time = None
        self.alert_played = False
        self.alert_threshold = 3  # seconds

        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        results = model(img)[0]
        class_names = [model.names[int(cls)] for cls in results.boxes.cls]

        # FPS calculation
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 1:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()

        # Drowsiness logic
        if class_names.count('Closed Eye') >= 2:
            if self.drowsy_detected_time is None:
                self.drowsy_detected_time = time.time()
            else:
                elapsed = time.time() - self.drowsy_detected_time
                if elapsed >= self.alert_threshold and not self.alert_played:
                    threading.Thread(target=play_alert).start()
                    self.alert_played = True
        else:
            self.drowsy_detected_time = None
            self.alert_played = False

        annotated_frame = results.plot()
        cv2.putText(annotated_frame, f"FPS: {self.fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return annotated_frame

# Stream video
webrtc_streamer(key="drowsiness", video_processor_factory=DrowsinessTransformer)
