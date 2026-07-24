import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

st.set_page_config(page_title="Digit Recognizer", page_icon="✏️", layout="centered")


@st.cache_resource
def get_model():
    return load_model("mnist_cnn.keras")


model = get_model()


# ---------- Preprocessing ----------
def preprocess_image(image):
    """
    Convert canvas image to MNIST-like format:
    - Grayscale
    - Crop to digit
    - Resize while preserving aspect ratio
    - Center on 28x28 canvas
    - Normalize
    """

    img = Image.fromarray(image.astype("uint8")).convert("L")

    arr = np.array(img)

    # Find bounding box of the digit
    coords = np.argwhere(arr > 20)

    if len(coords) == 0:
        return None, None

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    img = img.crop((x0, y0, x1, y1))

    # Resize while keeping aspect ratio
    img.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # Create 28x28 black image
    background = Image.new("L", (28, 28), 0)

    # Center the digit
    x = (28 - img.width) // 2
    y = (28 - img.height) // 2

    background.paste(img, (x, y))

    arr = np.array(background).astype("float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)

    return arr, background


st.title("✏️ Handwritten Digit Recognizer")

st.write(
    "Draw a digit (0–9) and click **Predict**."
)

col1, col2 = st.columns([1, 1])

with col1:

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=12,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:

    if st.button("Predict", use_container_width=True):

        if canvas_result.image_data is None:

            st.warning("Please draw a digit.")

        else:

            processed, preview = preprocess_image(canvas_result.image_data)

            if processed is None:

                st.warning("Please draw a digit.")

            else:

                preds = model.predict(processed, verbose=0)[0]

                digit = np.argmax(preds)

                confidence = np.max(preds) * 100

                st.success(f"Prediction: **{digit}**")

                st.write(f"Confidence: **{confidence:.2f}%**")

                st.bar_chart(preds)

                st.subheader("Image sent to CNN")

                st.image(
                    preview.resize((280, 280)),
                    clamp=True
                )

st.markdown("---")

st.caption(
    "CNN trained on MNIST (Conv2D → MaxPool → Conv2D → MaxPool → Dense → Dropout → Softmax)"
)