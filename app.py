import streamlit as st
import cv2
import numpy as np
import time

st.set_page_config(page_title="Live Sketch", layout="centered")

st.title("✍️ Live Sketch Drawing")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (400, 400))

    # Show original image
    st.subheader("📷 Original Image")
    st.image(img, channels="BGR")

    # Button to start drawing
    if st.button("▶️ Start Drawing"):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 80, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # Sort contours (biggest first)
        contours = sorted(contours, key=lambda x: len(x), reverse=True)

        canvas = np.ones_like(gray) * 255

        st.subheader("🎬 Drawing Animation")
        frame = st.empty()

        speed = st.slider("Drawing Speed", 1, 20, 5)
        thickness = st.slider("Pencil Thickness", 1, 3, 1)

        # Drawing process
        for contour in contours:
            for i in range(1, len(contour)):
                x1, y1 = contour[i-1][0]
                x2, y2 = contour[i][0]

                cv2.line(canvas, (x1, y1), (x2, y2), 0, thickness)

                if i % speed == 0:
                    frame.image(canvas, clamp=True)
                    time.sleep(0.005)

        frame.image(canvas, clamp=True)
        st.success("✅ Drawing Completed!")
