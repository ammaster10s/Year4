import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import io
import tempfile
import base64
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv

# Hair Removal Function -- Demo
def remove_hair(images, cfg=None):
    """Removes hair from lesion images using morphological filtering + inpainting."""
    if cfg is None:
        class CFG:
            def __init__(self):
                self.edge_low_threshold = 100
                self.edge_high_threshold = 220
                self.dark_spot_threshold = 150
                self.linelength_threshold = 10
                self.divergence_threshold = 0.25
                self.patchiness_threshold = 0.15
        cfg = CFG()

    img_filtered_all = []
    for img_orig in images:
        image_size = img_orig.shape[:2]

        if img_orig.ndim == 3:
            img = img_orig.mean(-1)
        else:
            img = img_orig.copy()

        # Detect possible hair regions
        kernel = np.ones((3, 3), np.uint8)
        img_filt = cv2.morphologyEx(np.uint8(img), cv2.MORPH_BLACKHAT, kernel)
        img_filt = np.where(img_filt > 15, img_filt, 0)
        kernel = np.ones((4, 4), np.uint8)
        img_filt = cv2.morphologyEx(img_filt, cv2.MORPH_DILATE, kernel)
        dark_spots = (img < cfg.dark_spot_threshold).astype(np.uint8)
        kernel = np.ones((4, 4), np.uint8)
        dark_spots = cv2.morphologyEx(dark_spots, cv2.MORPH_DILATE, kernel)
        img_filt = img_filt * dark_spots

        # Detect hair lines
        lines = cv2.HoughLinesP(img_filt, cv2.HOUGH_PROBABILISTIC, np.pi / 90, 20, None, 1, 20)
        mask = np.zeros(image_size, dtype=np.uint8)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(mask, (x1, y1), (x2, y2), 255, 1)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

        # Inpaint the image
        img_filtered = cv2.inpaint(img_orig.astype(np.uint8), mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        img_filtered_all.append(img_filtered)

    return img_filtered_all


# Streamlit App UI
st.set_page_config(page_title="Skin Lesion Analyzer", layout="wide")
st.title("🩺 Skin Lesion Analyzer")
st.write("Upload a dermoscopic image. The app will remove hair, then send it to Roboflow for segmentation + classification.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    load_dotenv()
    api_key = os.getenv("RoboflowAPI")

    workspace_name = st.text_input("Workspace Name", value="qmul-lfuwr")
    workflow_id = st.text_input("Workflow ID", value="custom-workflow-2")
    workflow_id_2 = st.text_input("Workflow ID 2", value="detect-and-classify")

    # Read and show original
    image = np.array(Image.open(uploaded_file).convert("RGB"))

    # Hair removal preprocessing
    with st.spinner("Removing hair artifacts..."):
        # kernel = np.array([[0, -1, 0],
        #            [-1, 5, -1],
        #            [0, -1, 0]])

        # image = cv2.filter2D(image, -1, kernel)
        filtered_img = remove_hair([image])[0]

    # Save temporarily for Roboflow API
    temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(temp_file.name, cv2.cvtColor(filtered_img, cv2.COLOR_RGB2BGR))

    with st.spinner("Running Roboflow Workflows..."):
        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key
        )

        # --- Run first workflow (segmentation) ---
        result = client.run_workflow(
            workspace_name=workspace_name,
            workflow_id=workflow_id,
            images={"image": temp_file.name},
            use_cache=True
        )

        # --- Run second workflow (classification or refined analysis) ---
        result_2 = client.run_workflow(
            workspace_name=workspace_name,
            workflow_id=workflow_id_2,
            images={"image": temp_file.name},
            use_cache=True
        )

# First model predictions
    label_1, confidence_1, img_np_1 = "Unknown", 0.0, None
    if isinstance(result, list) and len(result) > 0:
        data = result[0]
        try:
            cls_pred = data["output_classification"][0]["predictions"]["predictions"][0]
            label_1 = cls_pred.get("class", "Unknown").capitalize()
            confidence_1 = cls_pred.get("confidence", 0.0)
        except Exception:
            pass

        # Decode visualization image
        img_field = data.get("Img")
        img_b64 = None
        if isinstance(img_field, list) and len(img_field) > 0:
            first = img_field[0]
            if isinstance(first, dict) and "value" in first:
                img_b64 = first["value"]
            elif isinstance(first, str):
                img_b64 = first
        elif isinstance(img_field, str):
            img_b64 = img_field

        if img_b64:
            img_bytes = base64.b64decode(img_b64)
            img_np_1 = np.array(Image.open(io.BytesIO(img_bytes)))

# Model Predictions Second Model
label_2, confidence_2, img_np_2 = "Unknown", 0.0, None

if isinstance(result_2, list) and len(result_2) > 0:
    data2 = result_2[0]

    try:
        # Handle two possible structures
        if "model_predictions" in data2:
            pred_info = data2["model_predictions"]
            preds = pred_info.get("predictions", [])
            if preds:
                cls_pred_2 = preds[0]
                label_2 = cls_pred_2.get("class", "Unknown").capitalize()
                confidence_2 = cls_pred_2.get("confidence", 0.0)
        elif "output_classification" in data2:
            cls_pred_2 = data2["output_classification"][0]["predictions"]["predictions"][0]
            label_2 = cls_pred_2.get("class", "Unknown").capitalize()
            confidence_2 = cls_pred_2.get("confidence", 0.0)
    except Exception as e:
        st.warning(f"Model 2 parsing error: {e}")

# Layout grid
    top1, top2 = st.columns(2)
    with top1:
        st.image(image, caption="Original Image", width="stretch")
    with top2:
        st.image(filtered_img, caption="Hair Removed", width="stretch")

    bottom1, bottom2 = st.columns(2)
    with bottom1:
        if img_np_1 is not None:
            st.image(img_np_1, caption=f"Model 1: {workflow_id}", width="stretch")
        else:
            st.info("Model 1 returned no visualization.")
        
    with bottom2:
        if img_np_2 is not None:
            st.image(img_np_2, caption=f"Model 2: {workflow_id_2}", width="stretch")
        
            # st.info("Model 2 returned no visualization.")
        color1 = "red" if label_1.lower() == "malignant" else "green"
        st.markdown(f"<h4 style='color:{color1}'>Result 1: {label_1}</h4>", unsafe_allow_html=True)
        st.progress(confidence_1)
        st.caption(f"Confidence: {confidence_1:.2%}")

        color2 = "red" if label_2.lower() == "malignant" else "green"
        st.markdown(f"<h4 style='color:{color2}'>Result 2: {label_2}</h4>", unsafe_allow_html=True)
        st.progress(confidence_2)
        st.caption(f"Confidence: {confidence_2:.2%}")

else:
    st.info("Please upload an image to begin.")