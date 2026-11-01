
import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
from collections import Counter
from io import BytesIO

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Apple AI | Disease Detection",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "runs"
    / "detect"
    / "runs"
    / "apple_disease"
    / "weights"
    / "best.pt"
)

CLASS_NAMES = ["BLOTCH", "HEALTHY", "ROT", "SCAB"]

# --------------------------------------------------
# 2. CUSTOM PROFESSIONAL CSS
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #0b1120;
    color: #f1f5f9;
}

[data-testid="stSidebar"] {
    background-color: #111c2e;
    border-right: 1px solid #26364d;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

h1, h2, h3 {
    color: #f8fafc !important;
}

p, label {
    color: #cbd5e1;
}

.hero {
    background: linear-gradient(120deg, #142c3d, #123d35);
    border: 1px solid #28634f;
    padding: 28px;
    border-radius: 18px;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 34px;
    margin-bottom: 8px;
}

.hero p {
    color: #b8d9ce;
    font-size: 15px;
    margin-bottom: 0;
}

.metric-card {
    background: #142033;
    border: 1px solid #2a3b52;
    border-radius: 14px;
    padding: 20px;
    min-height: 115px;
}

.metric-label {
    color: #94a3b8;
    font-size: 13px;
    margin-bottom: 10px;
}

.metric-value {
    color: #f8fafc;
    font-size: 28px;
    font-weight: 700;
}

.metric-note {
    color: #4ade80;
    font-size: 12px;
    margin-top: 5px;
}

.section-heading {
    font-size: 21px;
    font-weight: 650;
    margin-top: 25px;
    margin-bottom: 14px;
    color: #f8fafc;
}

.info-card {
    background: #142033;
    border: 1px solid #2a3b52;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 10px;
}

.stButton > button {
    background: linear-gradient(90deg, #16a34a, #059669);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 20px;
    font-weight: 650;
    width: 100%;
}

.stButton > button:hover {
    background: #15803d;
    color: white;
    border: none;
}

.stDownloadButton > button {
    border-radius: 9px;
}

div[data-testid="stFileUploader"] {
    background: #142033;
    border: 1px dashed #3b526b;
    padding: 12px;
    border-radius: 12px;
}

hr {
    border-color: #26364d;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 3. LOAD YOLO MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


# --------------------------------------------------
# 4. DEVICE INFORMATION
# --------------------------------------------------

device = 0 if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
else:
    device_name = "CPU"


# --------------------------------------------------
# 5. SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 🍎 Apple AI")

    st.caption("INTELLIGENT FRUIT DISEASE DETECTION")

    st.divider()

    st.markdown("### Detection Settings")

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="Minimum confidence required to display a detection."
    )

    st.caption(
        "Higher confidence thresholds may filter out weaker detections."
    )

    st.divider()

    st.markdown("### Model Information")

    st.write("**Model:** YOLO11n")
    st.write("**Task:** Object Detection")
    st.write("**Classes:** 4")
    st.write(f"**Device:** {device_name}")

    st.divider()

    st.markdown("### Supported Classes")

    st.write("🟢 HEALTHY")
    st.write("🟠 BLOTCH")
    st.write("🔴 ROT")
    st.write("🟡 SCAB")

    st.divider()

    st.caption("Academic project prototype")
    st.caption("Built using Python, PyTorch and Streamlit")


# --------------------------------------------------
# 6. MAIN DASHBOARD HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>🍎 Apple Fruit Disease Detection</h1>
    <p>
        AI-powered apple health analysis using YOLO11.
        Upload an image and detect healthy or diseased apples.
    </p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 7. CHECK MODEL
# --------------------------------------------------

if not MODEL_PATH.exists():

    st.error(
        "Trained model not found. Please check the best.pt file path."
    )

    st.code(str(MODEL_PATH))
    st.stop()

try:
    model = load_model()

except Exception as error:
    st.error(f"Unable to load the YOLO model: {error}")
    st.stop()


# --------------------------------------------------
# 8. IMAGE UPLOAD
# --------------------------------------------------

st.markdown(
    '<div class="section-heading">📤 Upload Apple Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    help="Upload an image containing one or more apples."
)

if uploaded_file is not None:

    original_image = Image.open(uploaded_file).convert("RGB")

    st.success("Image uploaded successfully.")

    # --------------------------------------------------
    # 9. DETECTION BUTTON
    # --------------------------------------------------

    if st.button("🔍 Analyze Apple Image", use_container_width=True):

        with st.spinner("YOLO11 is analyzing your image..."):

            try:

                results = model.predict(
                    source=original_image,
                    conf=confidence,
                    imgsz=640,
                    device=device,
                    verbose=False
                )

                result = results[0]

                # Convert detection output to RGB
                annotated_array = result.plot()[:, :, ::-1]
                annotated_image = Image.fromarray(annotated_array)

                # Save annotated image to memory
                image_buffer = BytesIO()
                annotated_image.save(image_buffer, format="PNG")

                detections = []

                for box in result.boxes:

                    class_id = int(box.cls[0])
                    score = float(box.conf[0])

                    if isinstance(model.names, dict):
                        class_name = model.names[class_id]
                    else:
                        class_name = model.names[class_id]

                    detections.append({
                        "class": str(class_name).upper(),
                        "confidence": score
                    })

                st.session_state["apple_detection"] = {
                    "annotated": image_buffer.getvalue(),
                    "detections": detections,
                    "original": original_image.copy()
                }

            except Exception as error:

                st.error(f"Detection failed: {error}")


# --------------------------------------------------
# 10. DISPLAY DETECTION RESULTS
# --------------------------------------------------

if "apple_detection" in st.session_state:

    data = st.session_state["apple_detection"]

    detections = data["detections"]

    st.markdown("---")

    st.markdown(
        '<div class="section-heading">📊 Detection Overview</div>',
        unsafe_allow_html=True
    )

    total = len(detections)

    healthy_count = sum(
        1 for item in detections
        if item["class"] == "HEALTHY"
    )

    diseased_count = total - healthy_count

    if total > 0:
        average_confidence = (
            sum(item["confidence"] for item in detections) / total
        ) * 100
    else:
        average_confidence = 0

    # Four dashboard metric cards

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">TOTAL DETECTIONS</div>
            <div class="metric-value">{total}</div>
            <div class="metric-note">Objects detected</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">HEALTHY</div>
            <div class="metric-value">{healthy_count}</div>
            <div class="metric-note">Healthy-class detections</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">DISEASED</div>
            <div class="metric-value">{diseased_count}</div>
            <div class="metric-note">Non-healthy detections</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">AVG. CONFIDENCE</div>
            <div class="metric-value">{average_confidence:.1f}%</div>
            <div class="metric-note">Across detections</div>
        </div>
        """, unsafe_allow_html=True)

    # --------------------------------------------------
    # 11. IMAGE COMPARISON
    # --------------------------------------------------

    st.markdown(
        '<div class="section-heading">🖼️ Image Analysis</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("**Original Image**")

        st.image(
            data["original"],
            use_container_width=True
        )

    with col2:

        st.markdown("**YOLO11 Detection Output**")

        st.image(
            data["annotated"],
            use_container_width=True
        )

    st.download_button(
        label="⬇️ Download Annotated Image",
        data=data["annotated"],
        file_name="apple_disease_detection.png",
        mime="image/png",
        use_container_width=True
    )

    # --------------------------------------------------
    # 12. CLASS-WISE SUMMARY
    # --------------------------------------------------

    st.markdown(
        '<div class="section-heading">📋 Disease Classification</div>',
        unsafe_allow_html=True
    )

    if total == 0:

        st.warning(
            "No objects were detected at the selected confidence threshold. "
            "Try lowering the threshold or uploading another image."
        )

    else:

        class_counts = Counter(
            item["class"] for item in detections
        )

        for class_name in CLASS_NAMES:

            count = class_counts.get(class_name, 0)

            if count > 0:

                st.markdown(
                    f'<div class="info-card">'
                    f'<b>{class_name}</b>'
                    f'<span style="float:right;color:#4ade80;">'
                    f'{count} detection(s)</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        # Individual prediction details

        with st.expander("View individual detection details"):

            for index, item in enumerate(detections, start=1):

                st.write(
                    f"**Detection {index}:** {item['class']} "
                    f"| Confidence: {item['confidence'] * 100:.2f}%"
                )


# --------------------------------------------------
# 13. FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown("""
<div style="text-align:center; padding:15px; color:#94a3b8;">

    <b>Apple AI — Apple Fruit Disease Detection System</b>

    <br><br>

    Powered by YOLO11 | Python | PyTorch | Streamlit

    <br><br>

    Academic prototype. Predictions may be incorrect and
    should be verified by an agricultural expert.

</div>
""", unsafe_allow_html=True)
