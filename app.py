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
    page_title="Palm Tree Tracker",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_image_base64(image_file):
    try:
        with open(image_file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return ""

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
                top: 0; left: 0; width: 100%; height: 100%;
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

# Global UI Theme Styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0b0f12 0%, #112214 50%, #0b0f12 100%);
        background-attachment: fixed;
        color: #e0e0e0;
    }

    .header-container {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 5px;
    }

    .header-palm-gif {
        height: 52px;
        width: auto;
        object-fit: contain;
    }

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

    .camera-feature-card {
        background: rgba(18, 30, 22, 0.7);
        border: 1px solid #2e7d32;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 15px;
    }

    .badge-coconut {
        background-color: #1b5e20;
        color: #00e676;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        margin-right: 6px;
    }

    .badge-palm {
        background-color: #2e7d32;
        color: #a7f3d0;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }

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
    st.markdown("## **Palm Tree Tracker**")
    st.caption("AI-Powered Detection System")
    st.markdown("---")

    st.markdown("### ⚙️ **Detection Settings**")
    conf_threshold = st.slider("Confidence Threshold", 0.05, 1.0, 0.50, 0.05)
    
    st.markdown("### 📱 **Mobile Performance**")
    frame_skip = st.slider("Frame Skip (FPS Boost)", 1, 5, 3, 1)
    
    process_resolution = st.selectbox(
        "Display Resolution",
        options=["320x180 (Ultra Fast)", "480x270 (Mobile Optimized)", "640x360 (Balanced)"],
        index=0
    )
    res_w, res_h = map(int, process_resolution.split(" ")[0].split("x"))

    st.markdown("---")
    st.markdown("### ℹ️ **System Status**")
    st.info("Engine: YOLOv8 + ByteTrack\n\nStatus: Live Tracking Active")

# ---------------------------------------------------------
# 3. Main Interface Header, User Guide & Team Profiles
# ---------------------------------------------------------
if palm_gif_b64:
    header_html = f"""
        <div class='header-container'>
            <img src='data:image/gif;base64,{palm_gif_b64}' class='header-palm-gif' />
            <span class='animated-title'>Palm Tree Tracker</span>
        </div>
    """
else:
    header_html = """
        <div class='header-container'>
            <span class='animated-title'>🌴 Palm Tree Tracker</span>
        </div>
    """

st.markdown(header_html, unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Real-Time Aerial & Mobile Tree Analytics System</p>", unsafe_allow_html=True)

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    with st.expander("📖 **About This System & User Guide**", expanded=False):
        st.markdown("""
        ### **Welcome to Palm Tree Tracker**
        This application is designed for real-time aerial and mobile object detection to track and analyze tree species.
        
        #### **Key Features:**
        * **Automated Identification:** Leverages **YOLOv8** to track palm and coconut trees.
        * **Species Identification Mode:** Upload photos or capture via camera to identify tree species instantly.
        * **Mobile Optimized:** High-FPS throughput and real-time analytics.
        """)

with col_exp2:
    with st.expander("👥 **Project Development Team**", expanded=False):
        st.markdown("### **Development Team**")
        
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
    st.error("Error loading model: Neither 'best.onnx' nor 'best.pt' could be loaded.")
    st.stop()

# ---------------------------------------------------------
# 4. Input & Control Layout
# ---------------------------------------------------------
input_mode = st.radio(
    "📂 **Select Operation Mode:**",
    ["📹 Aerial Stream Analysis", "📸 Photo & Camera Tree Classifier"],
    horizontal=True
)

video_path = None

if input_mode == "📹 Aerial Stream Analysis":
    uploaded_file = st.file_uploader("📂 Upload Video Stream File", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()
        video_path = tfile.name
    else:
        video_path = "input_video (1).mp4"

elif input_mode == "📸 Photo & Camera Tree Classifier":
    st.markdown("""
        <div class="camera-feature-card">
            <h4 style="margin:0 0 8px 0; color:#81c784;">🔍 Tree Identification Feature</h4>
            <p style="margin:0 0 10px 0; font-size:13px; color:#c8e6c9;">
                Upload a photo or take a live camera shot to classify and count species in real time:
            </p>
            <span class="badge-coconut">🥥 Coconut Tree</span>
            <span class="badge-palm">🌴 Palm Tree</span>
        </div>
    """, unsafe_allow_html=True)
    
    photo_input_type = st.radio(
        "Choose Photo Input Source:",
        ["🖼️ Upload Photo File", "📷 Take Photo with Camera"],
        horizontal=True
    )
    
    image_bytes = None
    if photo_input_type == "🖼️ Upload Photo File":
        uploaded_img = st.file_uploader("Upload an Image File", type=["jpg", "jpeg", "png"])
        if uploaded_img is not None:
            image_bytes = uploaded_img.read()
    else:
        camera_shot = st.camera_input("Take a photo to identify tree species")
        if camera_shot is not None:
            image_bytes = camera_shot.read()

col_btn1, col_btn2 = st.columns([2, 1])
with col_btn1:
    run_button = st.button("🚀 Start Analytics Engine", use_container_width=True)
with col_btn2:
    reset_button = st.button("🔄 Reset", use_container_width=True)

if reset_button:
    st.rerun()

# ---------------------------------------------------------
# 5. Live Dashboard & Processing
# ---------------------------------------------------------
# Photo & Camera Classifier Output
if input_mode == "📸 Photo & Camera Tree Classifier" and image_bytes is not None:
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    results = model.predict(source=frame, conf=conf_threshold, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    class_names = model.names if hasattr(model, 'names') else {0: 'coconut_tree', 1: 'palm_tree'}
    
    coconut_count = 0
    palm_count = 0
    
    filtered_classes = []
    filtered_boxes = []
    filtered_conf = []
    labels = []
    
    if detections.class_id is not None and detections.confidence is not None:
        for box, cid, conf in zip(detections.xyxy, detections.class_id, detections.confidence):
            # Enforce strict confidence and strict name matching to block false positives on faces/objects
            if conf < max(conf_threshold, 0.60):
                continue
                
            c_name = class_names.get(int(cid), "").lower()
            
            if "coconut" in c_name:
                coconut_count += 1
                labels.append(f"Coconut Tree ({conf:.2f})")
                filtered_classes.append(cid)
                filtered_boxes.append(box)
                filtered_conf.append(conf)
            elif "palm" in c_name:
                palm_count += 1
                labels.append(f"Palm Tree ({conf:.2f})")
                filtered_classes.append(cid)
                filtered_boxes.append(box)
                filtered_conf.append(conf)
            
        if filtered_boxes:
            detections.xyxy = np.array(filtered_boxes)
            detections.class_id = np.array(filtered_classes)
            detections.confidence = np.array(filtered_conf)
            
            frame = box_annotator.annotate(scene=frame, detections=detections)
            frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
        else:
            detections.xyxy = np.empty((0, 4))
            detections.class_id = np.array([], dtype=int)
            detections.confidence = np.array([], dtype=float)
        
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{coconut_count}</div>
                <div class="metric-label">🥥 Coconut Trees Detected</div>
            </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{palm_count}</div>
                <div class="metric-label">🌴 Palm Trees Detected</div>
            </div>
        """, unsafe_allow_html=True)

    if coconut_count == 0 and palm_count == 0:
        st.warning("⚠️ No coconut or palm tree detected in this image.")

    st.image(frame, channels="BGR", caption="Classification Result", use_container_width=True)

# Video Stream Mode Output
elif input_mode == "📹 Aerial Stream Analysis" and run_button:
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.markdown("### 📊 Live Analytics")
        total_trees_metric = st.empty()
        frame_placeholder = st.empty()
        status_placeholder = st.empty()
        
    with col_right:
        st.markdown("### 📹 Annotated Feed")
        video_placeholder = st.empty()

    cap = cv2.VideoCapture(video_path)
    unique_tree_ids = set()
    frame_count = 0

    class_names = model.names if hasattr(model, 'names') else {0: 'palm_tree'}

    while cap.isOpened():
        ret, raw_frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        if frame_count % frame_skip != 0:
            continue

        frame = cv2.resize(raw_frame, (res_w, res_h))

        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=conf_threshold,
            verbose=False
        )[0]

        detections = sv.Detections.from_ultralytics(results)

        if detections.tracker_id is not None and detections.class_id is not None:
            labels = []
            for tracker_id, class_id in zip(detections.tracker_id, detections.class_id):
                c_name = class_names.get(int(class_id), "Palm Tree")
                unique_tree_ids.add(int(tracker_id))
                labels.append(f"{c_name} #{tracker_id}")

            frame = box_annotator.annotate(scene=frame, detections=detections)
            frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)

        video_placeholder.image(buffer.tobytes(), use_container_width=True)
        
        total_trees_metric.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(unique_tree_ids)}</div>
                <div class="metric-label">🌴 Total Palm Trees Tracked</div>
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