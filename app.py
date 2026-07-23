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

st.title("✏️ Handwritten Digit Recognizer")
st.write(
    "Draw a single digit (0-9) in the box below, then click **Predict**. "
    "The model is a CNN trained on MNIST (~99% test accuracy)."
)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Draw here")
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=18,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.subheader("Prediction")
    predict_clicked = st.button("Predict", use_container_width=True)
    clear_note = st.caption("Use the canvas trash icon (top right of canvas) to clear and redraw.")

    result_placeholder = st.empty()
    chart_placeholder = st.empty()

    if predict_clicked:
        if canvas_result.image_data is None:
            st.warning("Please draw a digit first.")
        else:
            # Canvas gives RGBA image; convert to grayscale
            img = canvas_result.image_data.astype("uint8")
            pil_img = Image.fromarray(img).convert("L")

            # Check if canvas is essentially blank
            if np.array(pil_img).max() < 10:
                st.warning("Please draw a digit first.")
            else:
                # Resize to 28x28 like MNIST, preserving aspect via simple resize
                pil_img_small = pil_img.resize((28, 28), Image.LANCZOS)

                arr = np.array(pil_img_small).astype("float32") / 255.0
                arr = arr.reshape(1, 28, 28, 1)

                preds = model.predict(arr, verbose=0)[0]
                pred_digit = int(np.argmax(preds))
                confidence = float(np.max(preds)) * 100

                result_placeholder.markdown(
                    f"## Predicted digit: **{pred_digit}**\n"
                    f"Confidence: **{confidence:.1f}%**"
                )

                chart_placeholder.bar_chart(
                    {"probability": preds},
                )

                with st.expander("See the 28x28 image fed to the model"):
                    st.image(pil_img_small.resize((140, 140)), clamp=True)

st.divider()
st.caption(
    "Model architecture: Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → "
    "Dense(128, relu) → Dropout(0.5) → Dense(10, softmax). Trained for 5 epochs on MNIST."
)