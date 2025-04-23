import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Set paths
MODEL_PATH = 'models/pneumonia_model.h5'
IMG_PATH = 'data/sample_xray.jpg'
OUTPUT_PATH = 'figures/gradcam_example.png'

# Load model
model = load_model(MODEL_PATH)

# Load and preprocess image
def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Grad-CAM implementation
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Superimpose heatmap on image
def save_gradcam(img_path, heatmap, output_path, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    cv2.imwrite(output_path, superimposed_img)

# Generate Grad-CAM
img_array = load_image(IMG_PATH)
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name='conv5_block3_out')
save_gradcam(IMG_PATH, heatmap, OUTPUT_PATH)

# Display prediction
prediction = model.predict(img_array)[0][0]
print(f"Prediction: {'Pneumonia' if prediction > 0.5 else 'Normal'} (Confidence: {prediction:.2f})")
