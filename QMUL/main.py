import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import io
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv

# -------------------- Hair Removal -------------------- #
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
        kernel = np.ones((3,3), np.uint8)
        img_filt = cv2.morphologyEx(np.uint8(img), cv2.MORPH_BLACKHAT, kernel)
        img_filt = np.where(img_filt > 15, img_filt, 0)
        kernel = np.ones((4,4), np.uint8)
        img_filt = cv2.morphologyEx(img_filt, cv2.MORPH_DILATE, kernel)
        dark_spots = (img < cfg.dark_spot_threshold).astype(np.uint8)
        kernel = np.ones((4,4), np.uint8)
        dark_spots = cv2.morphologyEx(dark_spots, cv2.MORPH_DILATE, kernel)
        img_filt = img_filt * dark_spots

        # Detect hair lines
        lines = cv2.HoughLinesP(img_filt, cv2.HOUGH_PROBABILISTIC, np.pi/90, 20, None, 1, 20)
        mask = np.zeros(image_size, dtype=np.uint8)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(mask, (x1, y1), (x2, y2), 255, 1)
        mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)

        # Inpaint the image
        img_filtered = cv2.inpaint(img_orig.astype(np.uint8), mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        img_filtered_all.append(img_filtered)

    return img_filtered_all


# -------------------- Streamlit UI -------------------- #
st.set_page_config(page_title="Skin Lesion Analyzer", layout="wide")

st.title("🩺 AI Skin Lesion Analyzer (Roboflow Workflow)")
st.write("Upload a dermoscopic image. The app will remove hair, then send it to Roboflow for segmentation + classification.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

load_dotenv()
api_key = os.getenv("RoboflowAPI")

# User can override API and workflow if needed
custom_api = st.text_input("🔑 (Optional) Enter API Key", value=api_key or "")
workspace_name = st.text_input("🏢 Workspace Name", value="qmul-lfuwr")
workflow_id = st.text_input("🧠 Workflow ID", value="custom-workflow-2")

if uploaded_file and (api_key or custom_api):
    api_key = custom_api or api_key

    # Load and display image
    image = np.array(Image.open(uploaded_file).convert("RGB"))
    st.image(image, caption="Original Image", use_container_width=True)

    # Step 1: Hair Removal
    with st.spinner("🧼 Removing hair artifacts..."):
        filtered_img = remove_hair([image])[0]
    st.image(filtered_img, caption="Hair Removed Image", use_container_width=True)

    # Step 2: Roboflow Workflow
    with st.spinner("🤖 Running Roboflow Workflow..."):
        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key
        )

        # Save temporarily in memory
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(filtered_img, cv2.COLOR_RGB2BGR))
        img_bytes = io.BytesIO(buffer)

        result = client.run_workflow(
            workspace_name=workspace_name,
            workflow_id=workflow_id,
            images={"image": img_bytes},
            use_cache=True
        )

    # Step 3: Display Results
    st.subheader("🧩 Segmentation + Classification Results")
    if "results" in result:
        st.json(result)

    # Try to show visual output if Roboflow returns one
    if "visualization" in result:
        st.image(result["visualization"], caption="Segmentation Overlay", use_container_width=True)

    # Extract classification if available
    if "predictions" in result:
        predictions = result["predictions"]
        if isinstance(predictions, list) and len(predictions) > 0:
            pred = predictions[0]
            label = pred.get("class", "Unknown")
            conf = pred.get("confidence", None)
            st.success(f"**Prediction:** {label}  \n**Confidence:** {conf:.2f}" if conf else f"**Prediction:** {label}")
else:
    st.info("⬆️ Upload an image and set up your API Key and workflow to start.")