import streamlit as st
import tensorflow as tf
import numpy as np
import os
import cv2

from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input


# 1. Page Configuration
st.set_page_config(
    page_title="NEU Surface Defect Detection",
    page_icon="🔍",
    layout="wide"
)

# 2. Project Path
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# 3. Load Trained Model
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "resnet50_fine_tuned.keras"
)

model = load_model(MODEL_PATH)

# Grad-CAM Model

last_conv_layer = model.get_layer("conv5_block3_out")

grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[
        last_conv_layer.output,
        model.output
    ]
)

print("Grad-CAM model created successfully!")

# Grad-CAM Heatmap Function

def make_gradcam_heatmap(img_array, grad_model):

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        predicted_class = tf.argmax(predictions[0])

        class_output = predictions[:, predicted_class]

    grads = tape.gradient(class_output, conv_outputs)

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    heatmap = heatmap / (
        tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
    )

    return heatmap.numpy()


# 4. Defect Classes
classes = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

# Professional Application Header

st.title("🔍 AI-Based Steel Surface Defect Detection")

st.markdown(
    """
    ### Deep Learning Based Quality Inspection

    This application uses **ResNet50 Transfer Learning**
    and **Grad-CAM Explainable AI** to detect steel
    surface defects from uploaded images.

    Upload a steel surface image to identify its defect category.
    """
)

st.markdown("---")
# Defect Categories

st.subheader("Supported Defect Categories")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("1. Crazing")

with col2:
    st.info("2. Inclusion")

with col3:
    st.info("3. Patches")

col4, col5, col6 = st.columns(3)

with col4:
    st.info("4. Pitted Surface")

with col5:
    st.info("5. Rolled-in Scale")

with col6:
    st.info("6. Scratches")

st.markdown("---")

# Image Upload Section

st.subheader("📤 Upload Steel Surface Image")

st.write(
    "Please upload a steel surface image in JPG, JPEG or PNG format."
)

# 7. Image Upload
uploaded_file = st.file_uploader(
    "Upload Steel Surface Image",
    type=["jpg", "jpeg", "png"]
)

# 8. Prediction
if uploaded_file is not None:

    # Read uploaded image
    img = Image.open(uploaded_file).convert("RGB")

    # Resize image
    img = img.resize((224, 224))

    # Convert image into array
    img_array = np.array(img)

    # Display uploaded image
    st.image(
        img,
        caption="Uploaded Steel Surface Image"
    )

    # Preprocessing
    input_image = np.expand_dims(img_array, axis=0)

    input_image = preprocess_input(input_image)

    # Prediction
    predictions = model.predict(input_image)

    predicted_index = np.argmax(predictions[0])

    predicted_label = classes[predicted_index]

    confidence = predictions[0][predicted_index] * 100

  # Prediction Result Section

st.markdown("---")
st.subheader("🎯 Prediction Result")

col1, col2 = st.columns(2)

with col1:
    st.success(f"Detected Defect: {predicted_label}")

with col2:
    st.metric(
        "Prediction Confidence",
        f"{confidence:.2f}%"
    )
    
  # Generate Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(input_image, grad_model)

  # Resize heatmap
    heatmap_resized = cv2.resize(heatmap, (224, 224))

  # Apply color map
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(
        heatmap_colored,
        cv2.COLORMAP_JET
    )

    heatmap_colored = cv2.cvtColor(
        heatmap_colored,
        cv2.COLOR_BGR2RGB
    )

    # Create overlay
    original_image = np.array(img)

    superimposed_img = cv2.addWeighted(
        original_image, 0.6,
        heatmap_colored, 0.4,
        0
    )

  # Explainable AI Section

st.markdown("---")

st.subheader("🧠 Explainable AI - Grad-CAM")

st.write(
    "Grad-CAM highlights the important image regions "
    "that influenced the model's prediction."
)

col1, col2 = st.columns(2)

with col1:
    st.image(
        original_image,
        caption="Original Steel Surface Image",
        use_container_width=True
    )

with col2:
    st.image(
        superimposed_img,
        caption="Grad-CAM Heatmap Overlay",
        use_container_width=True
    )

    # Application Footer

st.markdown("---")

st.markdown(
    """
    <div style="text-align: center;">
        <p><b>NEU Surface Defect Detection</b></p>
        <p>Developed using Python, TensorFlow, ResNet50 and Streamlit</p>
        <p>Deep Learning Based Industrial Quality Inspection</p>
    </div>
    """,
    unsafe_allow_html=True
)