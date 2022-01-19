from typing import Tuple

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit.uploaded_file_manager import UploadedFile

# function untuk load model


@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load("model/random_forest.joblib")
    return model

# untuk memproses gambar


def preprocess_image(file: UploadedFile) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess uploaded image file.

    Steps:
    1. convert to array
    2. resize image into 64x64 dimension
    3. convert to grayscale image
    4. flattened image

    Args:
        file (UploadedFile): uploaded file

    Returns:
        Tuple[ndarray, ndarray]: tuple of preprocessed image and flatttened
            image
    """
    image: np.ndarray = np.asarray(Image.open(file))
    image = cv2.resize(image, (64, 64))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    flattened = image.flatten()

    return image, flattened

# function untuk mendapatkan probabilitas dan prediksi


def classify(image: np.ndarray) -> Tuple[str, pd.DataFrame]:
    model = load_model()
    prediction = model.predict([image])[0]
    probs = pd.DataFrame(model.predict_proba([image]),
                         columns=["cat", "dog"],
                         index=["probability"])
    return prediction, probs


def main():
    st.title("Cats or Dogs?")

    uploaded_image = st.sidebar.file_uploader(
        "Upload your dog/cat image here:",
        key="uploaded_image"
    )

    st.write("Uploaded image:")
    if uploaded_image:
        st.image(uploaded_image)

        show_preview = st.checkbox("Preview")
        image, flat_image = preprocess_image(uploaded_image)
        fig, ax = plt.subplots()
        ax.imshow(image, cmap="gray")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Preprocessed Image")
        if show_preview:
            st.pyplot(fig)
        print(type(image), image.shape)

        run_model = st.button("Classify", key="classify")
        if run_model:
            prediction, probs = classify(flat_image)
            st.write(f"Prediction: it's a **{prediction.upper()}**")
            st.bar_chart(probs.T)
    else:
        st.write("_no image uploaded_")


if __name__ == "__main__":
    main()
