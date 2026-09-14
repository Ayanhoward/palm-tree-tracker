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

# Helper function to convert local images to base64 for inline HTML
def get_image_base64(image_file):
    try:
        with open(image_file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return ""

# Function to encode and apply local GIF background to Sidebar
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

# Load sidebar background
set_sidebar_bg_local("sidebar_bg.gif")

# Load local pixel art GIF for header
palm_gif_b64 = get_image_base64("palm_tree.gif")

# Global Theme Styling
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
    st.info("Engine: PyTorch YOLOv8 + ByteTrack\n\nStatus: High-FPS Dynamic Mode")

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
        1. Adjust detection thresholds in the left sidebar if needed.
        2. Upload an aerial video (`.mp4`, `.avi`, `.mov`).
        3. Click **🚀 Start Analytics Engine** to run real-time inference.
        """)

with col_exp2:
    with st.expander("👥 **Project Development Team**", expanded=False):
        st.markdown("### **Development Team**")
        
        # CSS Grid for Team Alignment
        st.markdown("""
            <style>
            .team-container {
                display: flex;
                flex-direction: column;
                gap: 12px;
            }
            .lead-card {
                display: flex;
                align-items: center;
                gap: 15px;
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                padding: 10px;
            }
            .lead-img {
                width: 65px;
                height: 65px;
                border-radius: 50%;
                object-fit: cover;
            }
            .team-grid {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 8px;
            }
            .member-card {
                display: flex;
                flex-direction: column;
                align-items: center;
                text-align: center;
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 10px;
                padding: 8px 4px;
                height: 100%;
            }
            .member-img {
                width: 100%;
                aspect-ratio: 1 / 1;
                border-radius: 8px;
                object-fit: cover;
                margin-bottom: 6px;
            }
            .member-name {
                font-size: 0.78rem;
                font-weight: 700;
                color: #ffffff;
                line-height: 1.15;
                margin-bottom: 4px;
                min-height: 2.3em;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .member-role {
                font-size: 0.68rem;
                color: #81c784;
            }
            </style>
        """, unsafe_allow_html=True)

        ian_b64 = get_image_base64("ian.jpg")
        m2_b64 = get_image_base64("member2.jpg")
        m3_b64 = get_image_base64("member3.jpg")
        m4_b64 = get_image_base64("member4.jpg")
        m5_b64 = get_image_base64("member5.jpg")

        fallback_user = "https://img.icons8.com/color/96/user-male-circle.png"

        img_ian = f"data:image/jpeg;base64,{ian_b64}" if ian_b64 else fallback_user
        img_m2 = f"data:image/jpeg;base64,{m2_b64}" if m2_b64 else fallback_user
        img_m3 = f"data:image/jpeg;base64,{m3_b64}" if m3_b64 else fallback_user
        img_m4 = f"data:image/jpeg;base64,{m4_b64}" if m4_b64 else fallback_user
        img_m5 = f"data:image/jpeg;base64,{m5_b64}" if m5_b64 else fallback_user

        st.markdown(f"""
            <div class="team-container">
                <div class="lead-card">
                    <img src="{img_ian}" class="lead-img" />
                    <div>
                        <div style="font-size: 1.05rem; font-weight: bold; color: #fff;">Ian Howard A. Labendia</div>
                        <div style="font-size: 0.8rem; color: #81c784;">🏆 Project Lead & Lead Developer</div>
                    </div>
                </div>

                <div class="team-grid">
                    <div class="member-card">
                        <img src="{img_m2}" class="member-img" />
                        <div class="member-name">Kaila Mae Chua</div>
                        <div class="member-role">Developer</div>
                    </div>
                    <div class="member-card">
                        <img src="{img_m3}" class="member-img" />
                        <div class="member-name">Eduardo Cupin Jr.</div>
                        <div class="member-role">Developer</div>
                    </div>
                    <div class="member-card">
                        <img src="{img_m4}" class="member-img" />
                        <div class="member-name">Rey Jan Alicante Bug-os</div>
                        <div class="member-role">Developer</div>
                    </div>
                    <div class="member-card">
                        <img src="{img_m5}" class="member-img" />
                        <div class="member-name">Justher Jhon Javier</div>
                        <div class="member-role">Developer</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# Supervision Annotators
custom_color = sv.Color(r=0, g=230, b=118)
box_annotator = sv.BoxAnnotator(thickness=2, color=custom_color)
label_annotator = sv.LabelAnnotator(
    text_scale=0.4, 
    text_thickness=1, 
    color=custom_color,
    text_color=sv.Color(r=0, g=0, b=0)
)

# Load YOLO Model directly using PyTorch weights best.pt
@st.cache_resource
def load_yolo_model():
    try:
        return YOLO("best.pt")
    except Exception as e:
        st.error(f"Error loading model weights 'best.pt': {e}")
        return None

model = load_yolo_model()
if model is None:
    st.stop()

# ---------------------------------------------------------
# 4. Input & Control Layout
# ---------------------------------------------------------
uploaded_file = st.file_uploader("📂 Select Input Video Stream", type=["mp4", "avi", "mov"])

col_btn1, col_btn2 = st.columns([2, 1])
with col_btn1:
    run_button = st.button("🚀 Start Analytics Engine", use_container_width=True)
with col_btn2:
    reset_button = st.button("🔄 Reset", use_container_width=True)

if reset_button:
    st.rerun()

# ---------------------------------------------------------
# 5. Live Dashboard & Streaming Layout
# ---------------------------------------------------------
if run_button:
    if uploaded_file is None:
        st.warning("⚠️ Please upload a video file before starting the analytics engine.")
    else:
        # Save temporary video file locally
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()
        video_path = tfile.name

        col_left, col_right = st.columns([1, 2])
        
        with col_left:
            st.markdown("### 📊 Live Analytics")
            metric_placeholder = st.empty()
            frame_placeholder = st.empty()
            status_placeholder = st.empty()
            
        with col_right:
            st.markdown("### 📹 Annotated Feed")
            video_placeholder = st.empty()

        # OpenCV with low buffer size for low latency
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        unique_palm_ids = set()
        frame_count = 0

        while cap.isOpened():
            ret, raw_frame = cap.read()
            if not ret:
                break

            frame_count += 1
            
            # Skip frames for smoother streaming
            if frame_count % frame_skip != 0:
                continue

            # Resize frame early to save compute resources
            frame = cv2.resize(raw_frame, (res_w, res_h))

            # YOLO High-FPS Tracking
            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=conf_threshold,
                imgsz=320,
                verbose=False
            )[0]

            detections = sv.Detections.from_ultralytics(results)

            if detections.tracker_id is not None:
                for tracker_id in detections.tracker_id:
                    unique_palm_ids.add(int(tracker_id))

                labels = [f"Palm #{tid}" for tid in detections.tracker_id]
                frame = box_annotator.annotate(scene=frame, detections=detections)
                frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

            # Low latency compressed JPEG encoding
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 35]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)

            # Update Streamlit UI
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