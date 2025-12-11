import streamlit as st
from ultralytics import YOLO
import PIL.Image
import matplotlib.pyplot as plt

# Load YOLO model once
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # change to your trained model path

model = load_model()

# Custom page style
st.markdown(
    """
    <style>
        .main {background-color: white;}
        .stFileUploader {background-color: #e6f2ff; padding: 20px; border-radius: 10px;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧠 Brain Tumor Detection with YOLO")
st.write("Upload an MRI scan to detect tumors.")

# File uploader (single image)
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load image
    image = PIL.Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    # Run inference
    results = model(image)

    # Extract predictions
    r = results[0]
    boxes = r.boxes
    predictions = []
    for box in boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])
        predictions.append((label, conf))

    # Show results without color styling
    if predictions:
        for label, conf in predictions:
            if "No Tumor" in label.lower():
                st.markdown(f"### ✅ No Tumor Detected ({conf:.2f} confidence)")
            else:
                st.markdown(f"### ❌ Tumor Detected: {label} ({conf:.2f} confidence)")

        # Confidence bar chart
        labels, confs = zip(*predictions)
        fig, ax = plt.subplots()
        ax.bar(labels, confs)  # default color
        ax.set_ylim([0, 1])
        ax.set_ylabel("Confidence")
        st.pyplot(fig)
    else:
        st.markdown("### ✅ No Tumor Detected")

    # Show annotated image
    annotated_path = "result.jpg"
    r.save(filename=annotated_path)
    st.image(annotated_path, caption="Detection Result", use_container_width=True)
