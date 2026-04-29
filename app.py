import streamlit as st
import cv2
import numpy as np
import time

st.set_page_config(page_title="Live Pencil Sketch", layout="centered")

st.title("✍️ Live Pencil Sketch Animation")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# -------- Load pencil safely --------
@st.cache_resource
def load_pencil():
    pencil = cv2.imread("pencil.png", cv2.IMREAD_UNCHANGED)

    if pencil is None:
        return None

    pencil = cv2.resize(pencil, (40, 40))
    return pencil

# -------- Overlay function --------
def overlay_pencil(background, pencil, x, y):
    if pencil is None:
        return background

    h, w = pencil.shape[:2]

    y1 = max(0, y - h//2)
    y2 = min(background.shape[0], y1 + h)
    x1 = max(0, x - w//2)
    x2 = min(background.shape[1], x1 + w)

    pencil_crop = pencil[0:(y2-y1), 0:(x2-x1)]

    if pencil_crop.shape[2] == 4:
        alpha = pencil_crop[:, :, 3] / 255.0
    else:
        alpha = np.ones((pencil_crop.shape[0], pencil_crop.shape[1]))

    for c in range(3):
        background[y1:y2, x1:x2, c] = (
            background[y1:y2, x1:x2, c] * (1 - alpha) +
            pencil_crop[:, :, c] * alpha
        )

    return background

# -------- Main --------
if uploaded_file:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (400, 400))

    st.subheader("📷 Original Image")
    st.image(img, channels="BGR")

    if st.button("▶️ Start Drawing"):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        inv_blur = 255 - blur
        sketch = cv2.divide(gray, inv_blur, scale=256.0)

        edges = cv2.Canny(gray, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda x: len(x), reverse=True)

        canvas = np.ones_like(gray) * 255

        pencil = load_pencil()

        if pencil is None:
            st.warning("⚠️ pencil.png not found → drawing without pencil animation")

        frame = st.empty()

        speed = st.slider("Drawing Speed", 1, 20, 5)
        thickness = st.slider("Pencil Thickness", 1, 2, 1)

        st.subheader("🎬 Drawing...")

        for contour in contours:
            for i in range(1, len(contour)):

                x1, y1 = contour[i-1][0]
                x2, y2 = contour[i][0]

                intensity = int(sketch[y1, x1])
                cv2.line(canvas, (x1, y1), (x2, y2), intensity, thickness)

                if i % speed == 0:
                    display = cv2.cvtColor(canvas.copy(), cv2.COLOR_GRAY2BGR)
                    display = overlay_pencil(display, pencil, x1, y1)

                    frame.image(display, channels="BGR")
                    time.sleep(0.003)

        frame.image(canvas, clamp=True)
        st.success("✅ Drawing Completed!")
