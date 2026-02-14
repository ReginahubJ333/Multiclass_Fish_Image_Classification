import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load model
model = tf.keras.models.load_model("best_fish_model.h5")

# Class labels (must match training order)
class_names = [
    'fish sea_food shrimp',
    'fish sea_food red_mullet',
    'fish sea_food hourse_mackerel',
    'fish sea_food trout',
    'fish sea_food sea_bass',
    'fish sea_food red_sea_bream',
    'animal fish',
    'fish sea_food black_sea_sprat',
    'fish sea_food striped_red_mullet',
    'fish sea_food gilt_head_bream',
    'animal fish bass'
]

st.title("🐟 Fish Image Classification")
st.write("Upload a fish image to predict its category.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")
