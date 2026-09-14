import base64
import tempfile
import time
import cv2
import numpy as np
import streamlit as st
import supervision as sv
from ultralytics import YOLO

# ---------------------------------------------------------
# 1. Page Configuration & Custom UI Design
# ---------------------------------------------------------
st.set_page_config(
    page_title="Palm Tree Counter",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to convert local GIF/images to base64 for inline HTML rendering
def get_image_base64(image_file):
    try:
        with open(image_file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return ""

# Function to encode and apply local GIF background to the Sidebar
def set_sidebar_bg_local(image_file):
    b64_img = get_image_base64(image_file)
    if b64_img:
        st.markdown(
            f"""
            <style>
            [data-testid="stSidebar"] {{
                background-image: url("data:image/gif;base64,{b64_img}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            
            [data-testid="stSidebar"]::before {{
                content: "";
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(11, 15, 18, 0.75);
                z-index: -1;
            }}
            
            [data-testid="stSidebar"] * {{
                color: #ffffff !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"] { background-color: #0d1317; }
            </style>
            """,
            unsafe_allow_html=True
        )

# Load sidebar GIF background
set_sidebar_bg_local("sidebar_bg.gif")

# Load local pixel art GIF for header
palm_gif_b64 = get_image_base64("palm_tree.gif")

# Global UI Theme Styling & Animated Gradient Header Text
st.markdown("""
    <style>
    /* Global Main Canvas Linear Gradient */
    .stApp {
        background: linear-gradient(135deg, #0b0f12 0%, #112214 50%, #0b0f12 100%);
        background-attachment: fixed;
        color: #e0e0e0;
    }

    /* Header Flex Container */
    .header-container {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 5px;
    }

    /* Original Pixel Art GIF Styling */
    .header-palm-gif {
        height: 52px;
        width: auto;
        object-fit: contain;
    }

    /* Animated Color Gradient Title Text */
    .animated-title {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.3px;
        background: linear-gradient(90deg, #ffffff, #81c784, #a7f3d0, #00e676, #ffffff);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: textGradientShift 6s ease infinite;
    }

    /* Keyframes for Gradient Shift Animation */
    @keyframes textGradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .sub-header {
        color: #81c784;
        font-size: 14px;
        margin-bottom: 20px;
    }

    /* Custom Metric Cards */
    .metric-card {
        background-color: rgba(22, 28, 34, 0.85);
        border: 1px solid #2e7d32;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 10px;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #66bb6a;
    }

    .metric-label {
        font-size: 13px;
        color: #a5d6a7;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Button Styling */
    div.stButton > button {
        background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%) !important;
        color: white !important;
        font-size: 16px !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 12px !important;
        transition: all 0.3s ease;
    }
    
    div.stButton > button:hover {
        background: linear-gradient(135deg, #388e3c 0%, #2e7d32 100%) !important;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Sidebar Controls & Info Panel
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/palm-tree.png", width=70)
    st.markdown("## **PalmTracker AI**")
    st.caption("Mobile Object Detection System")
    st.markdown("---")

    st.markdown("### ⚙️ **Detection Settings**")
    conf_threshold = st.slider("Confidence Threshold", 0.05, 1.0, 0.25, 0.05)
    
    st.markdown("### 📱 **Mobile Performance**")
    frame_skip = st.slider("Frame Skip (FPS Boost)", 1, 5, 3, 1, help="Higher values reduce mobile lag over network tunnels.")
    
    process_resolution = st.selectbox(
        "Display Resolution",
        options=["320x180 (Ultra Fast)", "480x270 (Mobile Optimized)", "640x360 (Balanced)"],
        index=0
    )
    res_w, res_h = map(int, process_resolution.split(" ")[0].split("x"))

    st.markdown("---")
    st.markdown("### ℹ️ **System Status**")
    st.info("Engine: YOLOv8 + ByteTrack\n\nStatus: High-FPS Mode")

# ---------------------------------------------------------
# 3. Main Interface Header, User Guide & Team Profiles
# ---------------------------------------------------------
if palm_gif_b64:
    header_html = f"""
        <div class='header-container'>
            <img src='data:image/gif;base64,{palm_gif_b64}' class='header-palm-gif' />
            <span class='animated-title'>Palm Tree Detection & Counting</span>
        </div>
    """
else:
    header_html = """
        <div class='header-container'>
            <span class='animated-title'>🌴 Palm Tree Detection & Counting</span>
        </div>
    """

st.markdown(header_html, unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Aerial Video Analytics and Automated Counting System</p>", unsafe_allow_html=True)

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    with st.expander("📖 **About This System & User Guide**", expanded=False):
        st.markdown("""
        ### **Welcome to PalmTracker AI**
        This application is designed for real-time aerial object detection and computer vision tracking of palm trees using drone video footage.
        
        #### **Key Features:**
        * **Automated Counting:** Utilizes **YOLOv8** for real-time palm detection and **ByteTrack** for persistent tracking IDs.
        * **Mobile Optimized:** Downscales resolution and compresses frame bandwidth to ensure fluid performance on mobile devices over network tunnels.
        * **Custom Controls:** Fine-tune confidence thresholds and dynamic frame-skipping directly from the left sidebar.
        
        #### **How to Use:**
        1. *(Optional)* Adjust settings in the left sidebar.
        2. Upload an aerial video (`.mp4`, `.avi`, `.mov`) or use the default video.
        3. Click **🚀 Start Analytics Engine** to run detection and view metrics.
        """)

with col_exp2:
    with st.expander("👥 **Project Development Team**", expanded=False):
        st.markdown("### **Development Team**")
        
        # Row 1: Team Lead (Ian)
        team_row1 = st.columns([1, 2])
        with team_row1[0]:
            try:
                st.image("ian.jpg", use_container_width=True)
            except Exception:
                st.image("https://img.icons8.com/color/96/user-male-circle.png", width=75)
        with team_row1[1]:
            st.markdown("### **Ian Howard A. Labendia**")
            st.caption("🏆 **Project Lead & Lead Developer**")

        st.markdown("---")
        
        # Row 2: Team Members 2 to 5 in 4 grid columns
        m2, m3, m4, m5 = st.columns(4)
        
        with m2:
            try:
                st.image("member2.jpg", use_container_width=True)
            except Exception:
                st.image("https://img.icons8.com/color/96/user-male-circle.png", width=60)
            st.markdown("**Kaila Mae Chua**")
            st.caption("Developer")

        with m3:
            try:
                st.image("member3.jpg", use_container_width=True)
            except Exception:
                st.image("https://img.icons8.com/color/96/user-male-circle.png", width=60)
            st.markdown("**Eduardo Cupin Jr.**")
            st.caption("Developer")

        with m4:
            try:
                st.image("member4.jpg", use_container_width=True)
            except Exception:
                st.image("https://img.icons8.com/color/96/user-male-circle.png", width=60)
            st.markdown("**Rey Jan Alicante Bug-os**")
            st.caption("Developer")

        with m5:
            try:
                st.image("member5.jpg", use_container_width=True)
            except Exception:
                st.image("https://img.icons8.com/color/96/user-male-circle.png", width=60)
            st.markdown("**Justher Jhon Javier**")
            st.caption("Developer")

# Annotator Setup
custom_color = sv.Color(r=0, g=230, b=118)
box_annotator = sv.BoxAnnotator(thickness=2, color=custom_color)
label_annotator = sv.LabelAnnotator(
    text_scale=0.4, 
    text_thickness=1, 
    color=custom_color,
    text_color=sv.Color(r=0, g=0, b=0)
)

# Model Loader
@st.cache_resource
def load_yolo_model():
    for path in ["best.onnx", "best.pt"]:
        try:
            return YOLO(path)
        except Exception:
            continue
    return None

model = load_yolo_model()
if model is None:
    st.error("Error loading model: Neither 'best.onnx' nor 'best.pt' could be loaded. Please ensure weight files are present.")
    st.stop()

# ---------------------------------------------------------
# 4. Input & Control Layout
# ---------------------------------------------------------
uploaded_file = st.file_uploader("📂 Select Input Video Stream", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()
    video_path = tfile.name
else:
    video_path = "input_video (1).mp4"

col_btn1, col_btn2 = st.columns([2, 1])
with col_btn1:
    run_button = st.button("🚀 Start Analytics Engine", use_container_width=True)
with col_btn2:
    reset_button = st.button("🔄 Reset", use_container_width=True)

if reset_button:
    st.rerun()

# ---------------------------------------------------------
# 5. Live Dashboard & Streaming Layout (ONNX Shape Safe)
# ---------------------------------------------------------
if run_button:
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.markdown("### 📊 Live Analytics")
        metric_placeholder = st.empty()
        frame_placeholder = st.empty()
        status_placeholder = st.empty()
        
    with col_right:
        st.markdown("### 📹 Annotated Feed")
        video_placeholder = st.empty()

    cap = cv2.VideoCapture(video_path)
    unique_palm_ids = set()
    frame_count = 0

    while cap.isOpened():
        ret, raw_frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # 1. Skip frames dynamically based on sidebar setting
        if frame_count % frame_skip != 0:
            continue

        # 2. Resizing for display performance
        frame = cv2.resize(raw_frame, (res_w, res_h))

        # 3. YOLO Tracking (imgsz override removed for ONNX compatibility)
        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=conf_threshold,
            verbose=False
        )[0]

        detections = sv.Detections.from_ultralytics(results)

        if detections.tracker_id is not None:
            for tracker_id in detections.tracker_id:
                unique_palm_ids.add(int(tracker_id))

            labels = [f"Palm #{tid}" for tid in detections.tracker_id]
            frame = box_annotator.annotate(scene=frame, detections=detections)
            frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        # 4. Low-latency JPEG Compression (Quality 30 for low bandwidth)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)

        # 5. Live UI Updates
        video_placeholder.image(buffer.tobytes(), use_container_width=True)
        
        metric_placeholder.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(unique_palm_ids)}</div>
                <div class="metric-label">Total Palms Identified</div>
            </div>
        """, unsafe_allow_html=True)
        
        frame_placeholder.markdown(f"""
            <div class="metric-card" style="border-color: #424242;">
                <div class="metric-value" style="color: #9e9e9e;">{frame_count}</div>
                <div class="metric-label">Processed Frames</div>
            </div>
        """, unsafe_allow_html=True)

        status_placeholder.caption("🟢 **Status:** Processing Live Stream...")

    cap.release()
    status_placeholder.success("✅ **Status:** Stream Completed Successfully!")