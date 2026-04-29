import streamlit as st
import cv2
import numpy as np
import time

st.set_page_config(page_title="Live Pencil Sketch", layout="centered")

st.title("✍️ Live Pencil Drawing Animation")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (400, 400))

    st.subheader("📷 Original Image")
    st.image(img, channels="BGR")

    if st.button("▶️ Start Drawing"):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Pencil sketch effect
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        inv_blur = 255 - blur
        sketch = cv2.divide(gray, inv_blur, scale=256.0)

        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda x: len(x), reverse=True)

        canvas = np.ones_like(gray) * 255

        # Load pencil image
        pencil = cv2.imread("pencil.png", cv2.IMREAD_UNCHANGED)
        pencil = cv2.resize(pencil, (30, 30))

        frame = st.empty()

        speed = st.slider("Speed", 1, 20, 5)

        for contour in contours:
            for i in range(1, len(contour)):

                x1, y1 = contour[i-1][0]
                x2, y2 = contour[i][0]

                intensity = int(sketch[y1, x1])

                cv2.line(canvas, (x1, y1), (x2, y2), intensity, 1)

                if i % speed == 0:
                    display = cv2.cvtColor(canvas.copy(), cv2.COLOR_GRAY2BGR)

                    # Overlay pencil
                    h, w = pencil.shape[:2]
                    y_offset = max(0, y1 - h//2)
                    x_offset = max(0, x1 - w//2)

                    for c in range(0, 3):
                        display[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
                            display[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1 - pencil[:,:,3]/255.0) + \
                            pencil[:,:,c] * (pencil[:,:,3]/255.0)

                    frame.image(display, channels="BGR")
                    time.sleep(0.003)

        frame.image(canvas, clamp=True)
        st.success("✅ Drawing Complete!")
