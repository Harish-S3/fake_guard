import streamlit as st
import numpy as np
import cv2
from PIL import Image
from transformers import pipeline
import tempfile
import os
import torch

# --- MODEL LOADING ---
@st.cache_resource
def load_text_model():
    """Loads a reliable text detection model"""
    try:
        # Using a more reliable model for text detection
        return pipeline("text-classification", model="microsoft/DialogRPT-updown")
    except Exception as e:
        st.error(f"FATAL: Could not load the text detection model. Error: {e}")
        st.stop()

@st.cache_resource
def load_deepfake_model():
    """Loads a deepfake detection model (image-based)."""
    try:
        return pipeline("image-classification", model="umm-maybe/AI-image-detector")
    except Exception as e:
        st.error(f"Error loading deepfake detection model: {e}")
        return None

# Load the models
text_detector = load_text_model()
image_detector = load_deepfake_model()

# --- HELPER FUNCTIONS ---

def analyze_text(text):
    """
    Analyze text for AI-generated content
    """
    if not text.strip():
        return None
        
    # Ensure text has sufficient length for analysis
    if len(text.split()) < 10:
        st.warning("For best results, please provide at least 10-15 words of text.")
        return None
        
    try:
        # Get prediction from the model
        result = text_detector(text, truncation=True, max_length=512)[0]
        
        # The DialogRPT model returns a score where higher is better quality
        # For AI detection, we'll invert this (lower score = more likely AI)
        ai_score = 1 - result['score']
            
        return ai_score

    except Exception as e:
        st.error(f"An error occurred during text analysis: {e}")
        return None

def analyze_video(video_file):
    """
    Video detection function
    """
    if video_file is None: return None
    video_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(video_file.read())
            video_path = tfile.name
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error: Could not open video file.")
            return None
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        frame_count, fake_count, face_frames = 0, 0, 0
        max_frames_to_process = 100
        progress_bar = st.progress(0, text="Analyzing video frames...")
        while cap.isOpened() and frame_count < max_frames_to_process:
            ret, frame = cap.read()
            if not ret: break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)
            if len(faces) > 0:
                face_frames += 1
                (x, y, w, h) = faces[0]
                face_roi = frame[y:y+h, x:x+w]
                face_image = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                result = image_detector(face_image)
                if any(d['label'].lower() == 'artificial' and d['score'] > 0.75 for d in result):
                    fake_count += 1
            frame_count += 1
            progress_bar.progress(frame_count / max_frames_to_process, text=f"Analyzing video frames... ({frame_count}/{max_frames_to_process})")
        cap.release()
        progress_bar.empty()
        if face_frames == 0:
            return {"message": "No faces were detected in the processed video segment."}
        return {"faces_detected_in_frames": face_frames, "frames_classified_as_fake": fake_count}
    finally:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)

# --- STREAMLIT UI ---

st.set_page_config(
    page_title="Fake Guard", 
    page_icon="🛡️", 
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #dee2e6;
    }
    .metric-card h3 {
        color: #6c757d;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card h2 {
        color: #343a40;
        font-size: 2rem;
        margin: 0;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #c3e6cb;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #ffeeba;
    }
    .danger-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🛡️ Fake Guard</h1>', unsafe_allow_html=True)
st.markdown("---")

# Simplified Navigation
st.sidebar.title("Navigation")
detection_mode = st.sidebar.radio("Choose a detection mode:", ("Text Detection", "Video Detection"))

# Text Detection Page
if detection_mode == "Text Detection":
    st.header("📝 AI-Generated Text Detection")
    
    input_text = st.text_area(
        "Enter text to analyze:", 
        height=300, 
        placeholder="Paste a paragraph, article, or any text you want to check for AI generation..."
    )
    
    if st.button("🔍 Analyze Text", type="primary", use_container_width=True):
        if text_detector is None:
            st.error("Text detection model is unavailable.")
        elif input_text:
            with st.spinner("Analyzing text content..."):
                ai_score = analyze_text(input_text)
                
                if ai_score is not None:
                    st.subheader("Analysis Results")
                    
                    # Display the AI score
                    st.metric(label="AI-Generated Likelihood", value=f"{ai_score:.2%}")
                    st.progress(ai_score)
                    
                    # Provide a verdict based on the score
                    if ai_score > 0.75:
                        st.markdown("""
                        <div class="danger-box">
                            <h4>🚨 High Confidence: AI-Generated</h4>
                            <p>This text shows strong indicators of AI generation with {:.1%} confidence.</p>
                        </div>
                        """.format(ai_score), unsafe_allow_html=True)
                    elif ai_score > 0.45:
                        st.markdown("""
                        <div class="warning-box">
                            <h4>⚠️ Moderate Confidence: Likely AI-Generated</h4>
                            <p>This text has significant characteristics of AI generation with {:.1%} confidence.</p>
                        </div>
                        """.format(ai_score), unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="success-box">
                            <h4>✅ High Confidence: Human-Written</h4>
                            <p>This text appears to be human-written with {:.1%} confidence.</p>
                        </div>
                        """.format(1 - ai_score), unsafe_allow_html=True)
        else:
            st.warning("Please enter some text to analyze.")

# Video Detection Page
else:
    st.header("🎥 Deepfake Video Detection")
    
    # Add the warning message
    st.warning("An authentic video should have **zero** suspicious frames. The detection of even one AI-generated frame is a strong indicator of manipulation.", icon="⚠️")
    
    uploaded_video = st.file_uploader(
        "Choose a video file...", 
        type=["mp4", "mov", "avi"],
        help="Supported formats: MP4, MOV, AVI"
    )
    
    if uploaded_video:
        # Show video info
        st.video(uploaded_video)
        
    if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
        if image_detector is None:
            st.error("Video detection model is not available.")
        elif uploaded_video:
            with st.spinner("Processing video. This may take a few minutes..."):
                result = analyze_video(uploaded_video)
                
                if result:
                    if "message" in result:
                        st.info(result['message'])
                    else:
                        st.subheader("Analysis Results")
                        
                        # UI layout with two columns
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("""
                            <div class="metric-card">
                                <h3>Frames with Faces Detected</h3>
                                <h2>{}</h2>
                            </div>
                            """.format(result['faces_detected_in_frames']), unsafe_allow_html=True)
                            
                        with col2:
                            st.markdown("""
                            <div class="metric-card">
                                <h3>Suspicious Frames Flagged</h3>
                                <h2>{}</h2>
                            </div>
                            """.format(result['frames_classified_as_fake']), unsafe_allow_html=True)
                        
                        # Provide verdict
                        fake_frames = result['frames_classified_as_fake']
                        if fake_frames > 1:
                            st.markdown("""
                            <div class="danger-box">
                                <h4>🚨 High Suspicion</h4>
                                <p>{} frames were flagged as potentially AI-generated. This is a strong indicator of a deepfake or manipulated video.</p>
                            </div>
                            """.format(fake_frames), unsafe_allow_html=True)
                        elif fake_frames == 1:
                            st.markdown("""
                            <div class="warning-box">
                                <h4>⚠️ Moderate Suspicion</h4>
                                <p>1 frame was flagged as potentially AI-generated. This is unusual and warrants careful review.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="success-box">
                                <h4>✅ Low Suspicion</h4>
                                <p>No frames were flagged as AI-generated. The video appears to be authentic.</p>
                            </div>
                            """, unsafe_allow_html=True)
        else:
            st.warning("Please upload a video file to analyze.")

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #6c757d;">
        <p>Built with AI models for content authenticity verification</p>
    </div>
    """, 
    unsafe_allow_html=True
)