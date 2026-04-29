import streamlit as st
import cv2
import numpy as np
import time

st.set_page_config(page_title="Sketch Drawer", layout="centered")

st.title("✍️ Sketch Drawing Animation")
st.write("Upload an image and watch it being drawn step-by-step.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Resize for performance
    img = cv2.resize(img, (400, 400))

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Extract edge points
    points = np.column_stack(np.where(edges > 0))

    # Create blank white canvas
    canvas = np.ones_like(gray) * 255

    st.subheader("🎬 Drawing Animation")
    frame = st.empty()

    # Speed control slider
    speed = st.slider("Drawing Speed", 1, 50, 10)

    # Animate drawing
    for i in range(0, len(points), speed):
        y, x = points[i]
        canvas[y, x] = 0

        frame.image(canvas, clamp=True)

        time.sleep(0.001)

    # Final output
    st.subheader("✅ Final Sketch")
    st.image(canvas, clamp=True)
