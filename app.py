import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ---------------- Load Model ----------------
model = load_model("crime_detection_model.h5")
classes = ['NormalVideos','Fighting','Explosion']  # EXACT same as training

IMG_SIZE = 64
SEQUENCE_LENGTH = 30

# ---------------- Streamlit UI ----------------
st.title("💥 CCTV Surveillance Anomaly Detection")
st.write("Upload a video (mp4/avi) and get model prediction + description")

uploaded_file = st.file_uploader("Choose a video...", type=["mp4","avi"])

if uploaded_file is not None:
    # Save uploaded video temporarily
    with open("temp_video.mp4","wb") as f:
        f.write(uploaded_file.read())
    
    cap = cv2.VideoCapture("temp_video.mp4")
    frames = []
    while len(frames) < SEQUENCE_LENGTH:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame,(IMG_SIZE,IMG_SIZE))
        frame = frame / 255.0
        frames.append(frame)
    cap.release()
    
    if len(frames) == SEQUENCE_LENGTH:
        input_seq = np.expand_dims(np.array(frames), axis=0)
        pred = model.predict(input_seq)
        predicted_class = classes[np.argmax(pred)]
        st.success(f"Predicted Action: {predicted_class}")
        
        # Optional: AI description (requires GPT API key)
        # description = get_ai_description(predicted_class)
        # st.write(description)
    else:
        st.warning("Video too short, need at least 30 frames")
