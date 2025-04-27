import streamlit as st
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
    result_img = results[0].plot()

    # Show result
    st.image(result_img, caption='Detected Image', use_column_width=True)


