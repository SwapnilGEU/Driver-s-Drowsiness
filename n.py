import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO  


model = YOLO('y_new2.pt')

# Streamlit UI
st.title("YOLO Object Detection App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    img = np.array(image)

    # Prediction
    results = model.predict(img)

    # Get results image
    result_img = results[0].plot()  # plots bounding boxes

    # Show result
    st.image(result_img, caption='Detected Image', use_column_width=True)

video_file = st.file_uploader("Upload a Video...", type=["mp4", "mov", "avi"])

if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame)
        result_frame = results[0].plot()
        st.image(result_frame, channels="BGR")

