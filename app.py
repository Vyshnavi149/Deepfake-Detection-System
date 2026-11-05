
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from datetime import datetime
from io import BytesIO

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="DEEPTRACE AI | Image Verification",
    page_icon="🔍",
    layout="wide"
)

IMG_SIZE = 128

# ---------------- PROFESSIONAL DESIGN ----------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #f7f8fc;
    color: #20283a;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.brand {
    font-size: 27px;
    font-weight: 800;
    letter-spacing: -1px;
    color: #17233c;
}

.brand span {
    color: #5269e8;
}

.tagline {
    color: #78839a;
    font-size: 13px;
    margin-top: -5px;
}

.hero {
    background: linear-gradient(115deg, #182642, #344d82);
    padding: 35px;
    border-radius: 20px;
    color: white;
    margin-top: 20px;
    margin-bottom: 28px;
}

.hero h1 {
    color: white;
    font-size: 35px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero p {
    color: #d7e0f5;
    font-size: 15px;
}

.section-title {
    font-size: 21px;
    font-weight: 800;
    color: #202b43;
    margin-bottom: 5px;
}

.muted {
    color: #8791a5;
    font-size: 13px;
}

.panel {
    background: white;
    border: 1px solid #e8ebf2;
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 18px;
}

.metric {
    background: white;
    border: 1px solid #e8ebf2;
    border-radius: 15px;
    padding: 20px;
}

.metric-label {
    color: #8590a5;
    font-size: 12px;
    font-weight: 600;
}

.metric-value {
    font-size: 27px;
    font-weight: 800;
    color: #263653;
    margin-top: 5px;
}

.result-real {
    background: #e9f8ef;
    border: 1px solid #bde8cc;
    padding: 22px;
    border-radius: 15px;
    color: #176b3c;
    text-align: center;
}

.result-fake {
    background: #fff0ed;
    border: 1px solid #f3c5bd;
    padding: 22px;
    border-radius: 15px;
    color: #a53628;
    text-align: center;
}

.stButton > button {
    background: #5269e8;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 24px;
    font-weight: 700;
    width: 100%;
}

.stButton > button:hover {
    background: #394fc5;
    color: white;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ---------------- MODEL LOADING ----------------

@st.cache_resource
def load_detection_model():
    return tf.keras.models.load_model(
        "models/deepfake_model.keras"
    )


# ---------------- SESSION HISTORY ----------------

if "history" not in st.session_state:
    st.session_state.history = []

if "result" not in st.session_state:
    st.session_state.result = None


# ---------------- HEADER ----------------

header_left, header_right = st.columns([3, 1])

with header_left:
    st.markdown("""
    <div class="brand">◈ DEEPTRACE<span> AI</span></div>
    <div class="tagline">INTELLIGENT IMAGE AUTHENTICITY ANALYSIS</div>
    """, unsafe_allow_html=True)

with header_right:
    st.markdown(
        "<p style='text-align:right;color:#71809a;font-size:13px;'>"
        "● SYSTEM READY &nbsp; | &nbsp; v1.0"
        "</p>",
        unsafe_allow_html=True
    )

st.divider()


# ---------------- HERO ----------------

st.markdown("""
<div class="hero">
    <h1>Verify what you see.</h1>
    <p>
    Analyse digital images using artificial intelligence
    and identify potential deepfake content.
    </p>
    <p>AI-powered image analysis · CNN · MobileNetV2</p>
</div>
""", unsafe_allow_html=True)


# ---------------- TOP METRICS ----------------

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="metric">
        <div class="metric-label">MODEL TEST ACCURACY</div>
        <div class="metric-value">77.32%</div>
        <div class="muted">Held-out test dataset</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="metric">
        <div class="metric-label">AI ARCHITECTURE</div>
        <div class="metric-value">CNN</div>
        <div class="muted">MobileNetV2 transfer learning</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="metric">
        <div class="metric-label">TRAINING DATASET</div>
        <div class="metric-value">12,890</div>
        <div class="muted">Real and fake images</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")


# ---------------- MAIN ANALYSIS AREA ----------------

left, right = st.columns([1.1, 0.9], gap="large")

with left:

    st.markdown(
        '<div class="section-title">01 / Image Investigation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="muted">Upload an image to begin the authenticity analysis.</p>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        help="Upload a JPG or PNG image."
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

        st.image(
            image,
            caption="Image submitted for analysis",
            width="stretch"
        )

        st.markdown(
            f'<div class="muted">FILE: {uploaded_file.name}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="muted">SIZE: {image.width} × {image.height} pixels</div>',
            unsafe_allow_html=True
        )

        if st.button("◈  ANALYSE IMAGE", width="stretch"):

            with st.spinner("AI model is analysing the image..."):

                try:
                    model = load_detection_model()

                    processed_image = image.resize(
                        (IMG_SIZE, IMG_SIZE)
                    )

                    image_array = np.array(
                        processed_image
                    ).astype("float32")

                    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
                        image_array
                    )

                    image_array = np.expand_dims(
                        image_array,
                        axis=0
                    )

                    prediction = float(
                        model.predict(
                            image_array,
                            verbose=0
                        )[0][0]
                    )

                    fake_score = prediction * 100
                    real_score = (1 - prediction) * 100

                    label = "FAKE" if prediction >= 0.5 else "REAL"

                    result = {
                        "file": uploaded_file.name,
                        "label": label,
                        "fake_score": fake_score,
                        "real_score": real_score,
                        "time": datetime.now().strftime(
                            "%d-%m-%Y %H:%M:%S"
                        )
                    }

                    st.session_state.result = result
                    st.session_state.history.insert(0, result)

                except Exception as e:
                    st.error(f"Analysis error: {e}")

    else:
        st.info("Upload an image to start the investigation.")


# ---------------- RESULT PANEL ----------------

with right:

    st.markdown(
        '<div class="section-title">02 / Analysis Report</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="muted">Your AI model prediction will appear here.</p>',
        unsafe_allow_html=True
    )

    result = st.session_state.result

    if result is not None:

        if result["label"] == "REAL":

            st.markdown("""
            <div class="result-real">
                <div style="font-size:13px;font-weight:700;">
                AUTHENTICITY CLASSIFICATION</div>
                <div style="font-size:38px;font-weight:800;">
                ✓ REAL IMAGE</div>
                <div>Model classified this image as real.</div>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class="result-fake">
                <div style="font-size:13px;font-weight:700;">
                AUTHENTICITY CLASSIFICATION</div>
                <div style="font-size:38px;font-weight:800;">
                ! FAKE IMAGE</div>
                <div>Model classified this image as potentially manipulated.</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        st.markdown("**Model score breakdown**")

        st.write(f"Fake score: {result['fake_score']:.2f}%")
        st.progress(result["fake_score"] / 100)

        st.write(f"Real score: {result['real_score']:.2f}%")
        st.progress(result["real_score"] / 100)

        st.caption(
            "These are model output scores, not guaranteed probabilities "
            "or proof of authenticity."
        )

        st.write("")

        report = f"""
DEEPTRACE AI - IMAGE ANALYSIS REPORT

File: {result['file']}
Date: {result['time']}

Prediction: {result['label']}
Fake model score: {result['fake_score']:.2f}%
Real model score: {result['real_score']:.2f}%

Model: MobileNetV2
Input size: 128 x 128
Test accuracy: 77.32%

DISCLAIMER:
This is an experimental AI prediction.
It is not proof that an image is genuine or manipulated.
"""

        st.download_button(
            "⬇ Download Analysis Report",
            data=report,
            file_name="deeptrace_analysis_report.txt",
            mime="text/plain",
            width="stretch"
        )

    else:

        st.markdown("""
        <div class="panel" style="text-align:center;padding:55px 15px;">
            <div style="font-size:42px;">◉</div>
            <h3 style="color:#34415b;">Awaiting Image Analysis</h3>
            <p style="color:#8993a7;font-size:13px;">
            Upload an image and click Analyse Image
            to generate your report.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ---------------- LOWER SECTIONS ----------------

st.write("")
st.divider()

tab1, tab2, tab3 = st.tabs([
    "◷  Detection History",
    "▥  Model Performance",
    "ⓘ  About the Project"
])


# ---------------- HISTORY ----------------

with tab1:

    st.markdown("### Previous Investigations")

    if st.session_state.history:

        for item in st.session_state.history[:10]:

            col1, col2, col3 = st.columns([2, 1, 2])

            with col1:
                st.write(item["file"])

            with col2:
                if item["label"] == "REAL":
                    st.success("REAL")
                else:
                    st.error("FAKE")

            with col3:
                st.caption(item["time"])

    else:
        st.info("No images have been analysed in this session yet.")


# ---------------- MODEL PERFORMANCE ----------------

with tab2:

    st.markdown("### Model Evaluation")

    p1, p2 = st.columns(2)

    with p1:
        st.metric("Test Accuracy", "77.32%")

    with p2:
        st.metric("Test Images", "1,865")

    st.markdown("**Confusion Matrix**")

    st.markdown("""
    | Actual / Predicted | Real | Fake |
    |:--|--:|--:|
    | Real | 613 | 270 |
    | Fake | 153 | 829 |
    """)

    st.markdown("**Technologies used**")

    st.write(
        "Python · TensorFlow · Keras · MobileNetV2 · "
        "OpenCV · NumPy · Streamlit"
    )

    st.caption(
        "Evaluation figures are from the model's held-out test set."
    )


# ---------------- ABOUT ----------------

with tab3:

    st.markdown("### About DEEPTRACE AI")

    st.write("""
    DEEPTRACE AI is an academic deepfake detection project
    developed to classify images as real or fake using deep learning.

    The system uses a Convolutional Neural Network approach
    with MobileNetV2 transfer learning.

    The uploaded image is resized, preprocessed and passed
    to the trained model. The model generates a score used
    to classify the image as REAL or FAKE.
    """)

    st.markdown("**Project specifications**")

    st.write("""
    - Project type: Student team project
    - Domain: Artificial Intelligence and Machine Learning
    - Model: MobileNetV2
    - Input: JPG and PNG images
    - Image dimensions: 128 × 128
    - Classes: Real and Fake
    - Interface: Streamlit
    """)

    st.warning(
        "This is an experimental academic prototype. "
        "It analyses images only and may make incorrect predictions, "
        "especially on images unlike its training dataset."
    )


# ---------------- FOOTER ----------------

st.write("")
st.divider()

st.markdown("""
<div style="text-align:center;padding:15px;color:#8a94a8;">
    <b style="color:#34415b;">DEEPTRACE AI</b><br>
    AI-Based Deepfake Detection System<br>
    <small>Academic PBL Project | Built with Python and TensorFlow</small>
</div>
""", unsafe_allow_html=True)

