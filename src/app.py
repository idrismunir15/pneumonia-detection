import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Set paths
MODEL_PATH = 'models/pneumonia_model.h5'

# Load model
@st.cache_resource
def load_model_cached():
    return load_model(MODEL_PATH)

model = load_model_cached()

# Preprocess image
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Generate Grad-CAM heatmap
def make_gradcam_heatmap(img_array, model, last_conv_layer_name='conv5_block3_out'):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Superimpose heatmap
def apply_gradcam(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    return superimposed_img

# Streamlit app
st.title("Pneumonia Detection from Chest X-Rays")
st.write("Upload a chest X-ray image to detect pneumonia.")

uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "png"])

if uploaded_file is not None:
    # Read and preprocess image
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_processed = preprocess_image(img)

    # Make prediction
    prediction = model.predict(img_processed)[0][0]
    label = 'Pneumonia' if prediction > 0.5 else 'Normal'
    confidence = prediction if prediction > 0.5 else 1 - prediction

    # Display prediction
    st.write(f"**Prediction**: {label} (Confidence: {confidence:.2f})")
    st.image(img_display, caption="Uploaded X-ray", use_column_width=True)

    # Generate and display Grad-CAM
    heatmap = make_gradcam_heatmap(img_processed, model)
    gradcam_img = apply_gradcam(img_display, heatmap)
    st.image(gradcam_img, caption="Grad-CAM Heatmap", use_column_width=True)
