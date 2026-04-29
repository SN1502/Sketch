import streamlit as st
import cv2
import numpy as np
import time

st.set_page_config(page_title="Live Pencil Tracing", layout="centered")

st.title("✍️ Live Pencil Tracing (No Image)")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])


# -------- Draw emoji pencil --------
def draw_pencil_icon(img, x, y):
    img = img.copy()
    cv2.putText(
        img,
        "✏️",  # emoji
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return img


# -------- Main --------
if uploaded_file:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (400, 400))

    st.subheader("📷 Original Image")
    st.image(img, channels="BGR")

    if st.button("▶️ Start Drawing"):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Better edge detection
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 120)

        edges = cv2.dilate(edges, None)
        edges = cv2.erode(edges, None)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda x: len(x), reverse=True)

        canvas = np.ones_like(gray) * 255

        frame = st.empty()

        st.subheader("🎬 Tracing...")

        for contour in contours:
            pts = contour.reshape(-1, 2)

            for i in range(1, len(pts)):
                x1, y1 = pts[i - 1]
                x2, y2 = pts[i]

                steps = 8

                for t in range(steps):
                    xi = int(x1 + (x2 - x1) * t / steps)
                    yi = int(y1 + (y2 - y1) * t / steps)

                    # draw line
                    cv2.circle(canvas, (xi, yi), 1, 0, -1)

                    display = cv2.cvtColor(canvas.copy(), cv2.COLOR_GRAY2BGR)

                    # draw moving icon
                    display = draw_pencil_icon(display, xi, yi)

                    frame.image(display, channels="BGR")

                    time.sleep(0.0005)

        frame.image(canvas, clamp=True)
        st.success("✅ Tracing Completed!")
