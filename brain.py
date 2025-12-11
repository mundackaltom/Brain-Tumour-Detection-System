import streamlit as st
from ultralytics import YOLO
import PIL.Image
from PIL import ImageOps
import pandas as pd
import plotly.express as px

# Load YOLO model (cached so it doesn't reload every time)
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # replace with your trained weights

model = load_model()

# Page config
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="wide")

# Title
st.markdown("<h1 style='text-align:center; color:#2E86C1;'>🧠 Brain Tumor Detection with YOLO</h1>", unsafe_allow_html=True)
st.write("Upload an MRI scan to detect tumors and their respective types.")

# Sidebar
with st.sidebar:
    st.header("🔍 About this App")
    st.info(
        "This app uses a **YOLO deep learning model** to detect brain tumors "
        "from MRI scans. Upload an image to see predictions with bounding boxes."
    )
    st.write("Model: **YOLOv8**")
    st.write("Author: *Your Name*")

# File uploader
uploaded_file = st.file_uploader("📤 Upload an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = PIL.Image.open(uploaded_file).convert("RGB")
    image = ImageOps.exif_transpose(image)  # Fix orientation if needed

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Uploaded MRI Image")
        st.image(image, caption="Original MRI", use_container_width=True)

    # Run inference
    results = model(image)

    # Collect predictions
    preds = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            preds.append((label, conf))

    # Show predictions
    with col2:
        st.subheader("🧾 Detection Results")
        if preds:
            for label, conf in preds:
                st.success(f"**{label}**  —  {conf:.2%} confidence")

            # Confidence bar chart
            df = pd.DataFrame(preds, columns=["Tumor Type", "Confidence"])
            fig = px.bar(
                df,
                x="Tumor Type",
                y="Confidence",
                text="Confidence",
                color="Tumor Type",
                title="Confidence Levels for Detected Tumors",
                range_y=[0, 1]
            )
            fig.update_traces(texttemplate='%{text:.2%}', textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No tumor detected.")

    # Annotated image
    st.markdown("---")
    st.subheader("🩻 Annotated Detection Result")
    results[0].save(filename="result.jpg")
    st.image("result.jpg", caption="Detected Tumor Regions", use_container_width=True)
else:
    st.info("👆 Upload an MRI image to get started.")
