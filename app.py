# ===============================================
# DEMO APP
# Deep Learning–Driven CCTV Surveillance
# ===============================================

import streamlit as st
from PIL import Image
import random

# ---------------------------------
# PAGE CONFIG
# ---------------------------------

st.set_page_config(
    page_title="CCTV Anomaly Detection Demo",
    page_icon="🎥",
    layout="wide"
)

# ---------------------------------
# TITLE
# ---------------------------------

st.title("🎥 Deep Learning–Driven CCTV Surveillance")
st.subheader("Anomalous Human Action Detection in Urban Environments")

st.write("Upload a CCTV frame or image to detect suspicious activity.")

# ---------------------------------
# FILE UPLOAD
# ---------------------------------

uploaded_file = st.file_uploader(
    "Upload CCTV Image",
    type=["jpg","png","jpeg"]
)

# ---------------------------------
# DEMO DESCRIPTION FUNCTION
# ---------------------------------

def generate_description():

    descriptions = [
        "A person walking normally on the street.",
        "Two people standing and talking.",
        "A vehicle passing through the road.",
        "A person running quickly which may indicate suspicious activity.",
        "A group of people gathered near a building entrance."
    ]

    return random.choice(descriptions)

# ---------------------------------
# DEMO ANOMALY DETECTION
# ---------------------------------

def detect_anomaly():

    results = ["Normal Activity","Anomalous Activity Detected"]
    return random.choice(results)

# ---------------------------------
# SHOW RESULT
# ---------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded CCTV Frame", use_column_width=True)

    with col2:

        st.subheader("AI Analysis")

        result = detect_anomaly()
        description = generate_description()

        if result == "Normal Activity":
            st.success(result)
        else:
            st.error(result)

        st.write("**Scene Description:**")
        st.write(description)

# ---------------------------------
# FOOTER
# ---------------------------------

st.markdown("---")
st.write("Demo Version – Model will be integrated later using UCF-Crime dataset.")
