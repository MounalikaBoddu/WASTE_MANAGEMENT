import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load trained model
model = load_model("waste_classifier_model.h5")

# Class labels (must match training folder order)
class_names = ['plastic', 'hazardous', 'recyclable', 'organic']

st.set_page_config(page_title="Waste Classification App", layout="centered")

st.title("♻️ Waste Image Classifier")
st.write("Upload an image of waste to classify it into one of the categories:")

uploaded_file = st.file_uploader("Choose a waste image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Predict Button
    if st.button("🔍 Predict Category"):
        try:
            # Load and preprocess image
            img = Image.open(uploaded_file).convert("RGB")
            img = img.resize((128, 128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # Predict
            prediction = model.predict(img_array)
            class_index = np.argmax(prediction)
            class_label = class_names[class_index]
            confidence = prediction[0][class_index] * 100

            st.success(f"🧠 Predicted: **{class_label.capitalize()}**")
            st.info(f"📊 Confidence: **{confidence:.2f}%**")
        except Exception as e:
            st.error(f"Error: {e}")
