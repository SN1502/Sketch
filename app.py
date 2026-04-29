import streamlit as st
import cv2
import numpy as np
import time

st.set_page_config(page_title="Pencil Sketch Drawing", layout="centered")

st.title("✍️ Pencil Sketch Drawing Animation")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (400, 400))

    # Show original
    st.subheader("📷 Original Image")
    st.image(img, channels="BGR")

    if st.button("▶️ Start Drawing"):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- Pencil Sketch Effect ---
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        inv_blur = 255 - blur
        sketch = cv2.divide(gray, inv_blur, scale=256.0)

        # --- Edge for tracing ---
        edges = cv2.Canny(gray, 50, 150)

        # Get contours
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # Sort contours (longer first)
        contours = sorted(contours, key=lambda x: len(x), reverse=True)

        # Blank canvas
        canvas = np.ones_like(gray) * 255

        frame = st.empty()

        speed = st.slider("Drawing Speed", 1, 20, 5)
        thickness = st.slider("Pencil Thickness", 1, 2, 1)

        st.subheader("🎬 Drawing...")

        # Draw with pencil intensity
        for contour in contours:
            for i in range(1, len(contour)):
                x1, y1 = contour[i-1][0]
                x2, y2 = contour[i][0]

                # Use sketch intensity instead of pure black
                intensity = int(sketch[y1, x1])

                cv2.line(canvas, (x1, y1), (x2, y2), intensity, thickness)

                if i % speed == 0:
                    frame.image(canvas, clamp=True)
                    time.sleep(0.003)

        frame.image(canvas, clamp=True)

        st.subheader("✅ Final Pencil Sketch")
        st.image(canvas, clamp=True)
