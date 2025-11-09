%%writefile streamlit_app.py
# ================================================================
# streamlit_app.py — Skin Lesion Two-Stage Classification Demo
# ================================================================

import streamlit as st
import numpy as np
import cv2, os
from PIL import Image
from inference_sdk import InferenceHTTPClient
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization,
    GlobalAveragePooling2D, GlobalMaxPooling2D,
    Concatenate, Conv2D, Activation, Add, Multiply, Reshape, Lambda
)
from tensorflow.keras.applications import EfficientNetV2M
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnet_preprocess


# ================================================================
# --- 1. CBAM & Model Definition ---
# ================================================================

def compute_spatial_output_shape(input_shape):
    return input_shape[:-1] + (1,)

def cbam_block(cbam_feature, ratio=8):
    channel = int(cbam_feature.shape[-1])
    shared_dense_one = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True)
    shared_dense_two = Dense(channel, kernel_initializer='he_normal', use_bias=True)

    # Channel Attention
    avg_pool = GlobalAveragePooling2D()(cbam_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_dense_two(shared_dense_one(avg_pool))

    max_pool = GlobalMaxPooling2D()(cbam_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_dense_two(shared_dense_one(max_pool))

    channel_attention = Activation('sigmoid')(Add()([avg_pool, max_pool]))
    x = Multiply()([cbam_feature, channel_attention])

    # Spatial Attention
    avg_pool = Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True),
                      output_shape=compute_spatial_output_shape)(x)
    max_pool = Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True),
                      output_shape=compute_spatial_output_shape)(x)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    spatial_attention = Conv2D(1, (7, 7), padding='same', activation='sigmoid',
                               kernel_initializer='he_normal')(concat)
    x = Multiply()([x, spatial_attention])
    return x


def build_model_with_concat_pooling(input_shape=(256,256,3), num_classes=4):
    inputs = Input(shape=input_shape)
    base = EfficientNetV2M(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = False
    x = base(inputs, training=False)
    x = cbam_block(x)
    gap = GlobalAveragePooling2D()(x)
    gmp = GlobalMaxPooling2D()(x)
    x = Concatenate()([gap, gmp])
    x = Dense(512, activation='relu', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, output, name="EfficientNetV2M_CBAM_ConcatPool")
    return model


def preprocess_lesion(img):
    """Enhance lesion region using CLAHE and sharpening."""
    if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 1: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    if img.dtype != np.uint8:
        img = (img * 255.0).astype(np.uint8) if img.max() <= 1.0 else np.clip(img, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2)
    img = cv2.addWeighted(img, 1.4, blurred, -0.4, 0)
    return img


def load_finetuned_model(model_path, image_size=(256,256)):
    """Load your Stage 2 fine-tuned model."""
    model = build_model_with_concat_pooling(input_shape=(image_size[0], image_size[1], 3), num_classes=4)
    model.load_weights(model_path)
    return model


# ================================================================
# --- 2. Roboflow Stage 1 Functions ---
# ================================================================

def run_roboflow_workflow(client, image_path):
    """Call Roboflow workflow for Benign/Malignant classification."""
    try:
        result = client.run_workflow(
            workspace_name="qmul-lfuwr",
            workflow_id="custom-workflow-4",
            images={"image": image_path},
            use_cache=True
        )
        return result
    except Exception as e:
        st.error(f"Roboflow API Error: {e}")
        return None


def parse_roboflow_workflow_result(result):
    """Extract class + confidence from Roboflow result JSON."""
    try:
        step_output = result[0]
        prediction_data = step_output.get('predictions')
        if prediction_data and prediction_data.get('prediction_type') == 'classification':
            predicted_class = prediction_data['top']
            confidence = prediction_data['confidence']
            return predicted_class, confidence
        else:
            return None, None
    except Exception as e:
        st.error(f"Parsing Error: {e}")
        return None, None


# ================================================================
# --- 3. Streamlit Frontend ---
# ================================================================

st.set_page_config(page_title="Skin Lesion Two-Stage Classifier", layout="centered")
st.title("Skin Lesion Two-Stage Classification System")
st.write("Stage 1: Roboflow Benign/Malignant Detection → Stage 2: Benign Sub-class Prediction")

# --- Configuration (predefined) ---
MODEL_PATH = "/content/drive/MyDrive/QMUL_SkinLesion/best_specialist_model_stage1_Finetune_Code2.keras"
API_KEY = "BhquO0k4o5JNlJLjVP5s"

# --- Load Stage 2 Model Once ---
@st.cache_resource
def load_model_once():
    if os.path.exists(MODEL_PATH):
        return load_finetuned_model(MODEL_PATH)
    else:
        st.stop()

model = load_model_once()
st.success(" Stage 2 Model Loaded Successfully")

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload a Skin Lesion Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    tmp_path = "temp_uploaded_image.jpg"
    img.save(tmp_path)

    if st.button("Run Two-Stage Inference"):
        # Stage 1
        st.subheader(" Stage 1: Benign vs Malignant Detection")
        client = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=API_KEY)
        result = run_roboflow_workflow(client, tmp_path)
        if result:
            stage1_class, stage1_conf = parse_roboflow_workflow_result(result)
            if stage1_class:
                st.write(f"**Stage 1 Result:** {stage1_class} ({stage1_conf:.2%})")

                # Stage 2 only if Benign
                if stage1_class.lower() == "benign":
                    st.subheader("Stage 2: Benign Sub-type Classification")
                    img_cv = np.array(img)
                    enhanced = preprocess_lesion(img_cv)
                    resized = cv2.resize(enhanced, (256, 256))
                    batch = np.expand_dims(efficientnet_preprocess(resized), axis=0)
                    pred = model.predict(batch)[0]
                    class_names = ['ISIC-BenignOther', 'ISIC-Cherry', 'ISIC-images_Nevus', 'ISIC-images_SeborrheicKeratosis']
                    idx = np.argmax(pred)
                    st.success(f"Predicted Sub-type: **{class_names[idx]}** ({pred[idx]:.2%})")
                    st.bar_chart(pred)
                else:
                    st.warning("Classified as Malignant — Stage 2 skipped.")
            else:
                st.error("Failed to parse Roboflow response.")
        else:
            st.error("No response from Roboflow API.")
