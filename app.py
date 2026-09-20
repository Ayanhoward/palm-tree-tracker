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

# Load assets if present
set_sidebar_bg_local("sidebar_bg.gif")
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

    .badge-palm {
        background-color: #1b5e20;
        color: #00e676;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        margin-right: 6px;
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
    st.caption("AI-Powered Tree Analytics System")
    st.markdown("---")

    st.markdown("### 🤖 **Model Version**")
    model_choice = st.selectbox(
        "Select Model Weights",
        ["yolov11n.pt (YOLOv11 - Fast & Accurate)", "best.pt / best.onnx (Custom Trained)"]
    )
    selected_model_file = "yolov11n.pt" if "yolov11n" in model_choice else "best.onnx"

    st.markdown("### ⚙️ **Detection Settings**")
    conf_threshold = st.slider("Confidence Threshold", 0.05, 1.0, 0.40, 0.05)
    
    st.markdown("### 📱 **Mobile Performance**")
    frame_skip = st.slider("Frame Skip (FPS Boost)", 1, 5, 3, 1)
    
    process_resolution = st.selectbox(
        "Display Resolution",
        options=["320x180 (Ultra Fast)", "480x270 (Mobile Optimized)", "640x360 (Balanced)"],
        index=0
    )
    res_w, res_h = map(int, process_resolution.split(" ")[0].split("x"))

    st.markdown("---")
    st.markdown("### 📥 **Download Demo Samples**")
    st.caption("Download test files to try the app:")

    with st.expander("🖼️ Sample Images"):
        samples = [
            ("Sample 1", "sample1.jpg"),
            ("Sample 2", "sample2.jpg"),
            ("Sample 3", "sample3.png"),
            ("Sample 4", "sample4.jpg")
        ]
        for label, filename in samples:
            try:
                with open(filename, "rb") as f:
                    mime_type = "image/png" if filename.endswith(".png") else "image/jpeg"
                    st.download_button(
                        label=f"Download {label}",
                        data=f,
                        file_name=filename,
                        mime=mime_type,
                        use_container_width=True,
                        key=f"dl_{filename}"
                    )
            except FileNotFoundError:
                pass

    with st.expander("📹 Sample Videos"):
        videos = [
            ("Default Video", "palm tree.mp4"),
            ("Palm Tree Video 2", "palm tree 2.mp4"),
            ("Palm Tree Video 3", "palm tree 3.mp4")
        ]
        for label, filename in videos:
            try:
                with open(filename, "rb") as f:
                    st.download_button(
                        label=f"Download {label}",
                        data=f,
                        file_name=filename,
                        mime="video/mp4",
                        use_container_width=True,
                        key=f"dl_{filename}"
                    )
            except FileNotFoundError:
                pass

    st.markdown("---")
    st.markdown("### ℹ️ **System Status**")
    st.info("Engine: YOLOv11 + ByteTrack\n\nStatus: Live Tracking Active")

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
        This application is designed for real-time aerial and mobile object detection to track and analyze palm trees.
        
        #### **Key Features:**
        * **Automated Identification:** Leverages state-of-the-art **YOLOv11** model weights.
        * **Palm Tree Counting:** Detects and counts palm trees accurately.
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

# Model Loader supporting YOLOv11 and custom models
@st.cache_resource
def load_yolo_model(model_filename):
    for path in [model_filename, "yolov11n.pt", "best.onnx", "best.pt", "yolov8n.pt"]:
        if os.path.exists(path):
            try:
                return YOLO(path)
            except Exception:
                continue
    # Fallback to downloading/loading yolov11n directly
    try:
        return YOLO("yolov11n.pt")
    except Exception:
        return None

import os
model = load_yolo_model(selected_model_file)
if model is None:
    st.error("Error loading model: Please make sure 'yolov11n.pt' or 'best.pt' is available in your directory.")
    st.stop()

# Extract class names dictionary if model has names
model_names = model.names if hasattr(model, 'names') else {0: "Palm Tree"}

# ---------------------------------------------------------
# 4. Input & Control Layout
# ---------------------------------------------------------
input_mode = st.radio(
    "📂 **Select Operation Mode:**",
    ["📹 Aerial Stream Analysis", "📸 Photo & Camera Tree Classifier", "📷 Live Camera Streaming"],
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
        video_path = "palm tree.mp4"

elif input_mode == "📸 Photo & Camera Tree Classifier":
    st.markdown("""
        <div class="camera-feature-card">
            <h4 style="margin:0 0 8px 0; color:#81c784;">🔍 Palm Tree Identification</h4>
            <p style="margin:0 0 10px 0; font-size:13px; color:#c8e6c9;">
                Upload a photo or take a live camera shot to identify and count palm trees:
            </p>
            <span class="badge-palm">🌴 Palm Tree Tracker</span>
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
        camera_shot = st.camera_input("Take a photo to identify palm trees")
        if camera_shot is not None:
            image_bytes = camera_shot.read()

elif input_mode == "📷 Live Camera Streaming":
    st.markdown("""
        <div class="camera-feature-card">
            <h4 style="margin:0 0 8px 0; color:#81c784;">📹 Real-Time Web Camera Feed</h4>
            <p style="margin:0 0 10px 0; font-size:13px; color:#c8e6c9;">
                Use your device camera for real-time video detection and counting.
            </p>
        </div>
    """, unsafe_allow_html=True)

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
if input_mode == "📸 Photo & Camera Tree Classifier" and image_bytes is not None:
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    results = model.predict(source=frame, conf=conf_threshold, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    tree_count = 0
    filtered_classes = []
    filtered_boxes = []
    filtered_conf = []
    labels = []
    
    if detections.class_id is not None and detections.confidence is not None:
        for box, cid, conf in zip(detections.xyxy, detections.class_id, detections.confidence):
            if conf < conf_threshold:
                continue
                
            tree_count += 1
            class_name = model_names.get(int(cid), "Palm Tree")
            labels.append(f"{class_name} ({conf:.2f})")
            filtered_classes.append(cid)
            filtered_boxes.append(box)
            filtered_conf.append(conf)
            
        if filtered_boxes:
            detections.xyxy = np.array(filtered_boxes)
            detections.class_id = np.array(filtered_classes, dtype=int)
            detections.confidence = np.array(filtered_conf)
            
            frame = box_annotator.annotate(scene=frame, detections=detections)
            frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
        else:
            detections.xyxy = np.empty((0, 4))
            detections.class_id = np.array([], dtype=int)
            detections.confidence = np.array([], dtype=float)
        
    m_col1 = st.columns(1)[0]
    with m_col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{tree_count}</div>
                <div class="metric-label">🌴 Total Palm Trees Detected</div>
            </div>
        """, unsafe_allow_html=True)

    if tree_count == 0:
        st.warning("⚠️ No palm trees detected in this image. Try lowering the confidence threshold or uploading a clearer photo.")

    st.image(frame, channels="BGR", caption="Classification Result", use_container_width=True)

elif input_mode == "📷 Live Camera Streaming" and run_button:
    st.markdown("### 🔴 Live Web Camera Feed Running...")
    cam_placeholder = st.empty()
    metric_placeholder = st.empty()
    
    cap = cv2.VideoCapture(0)
    unique_tree_ids = set()
    frame_count = 0
    
    while cap.isOpened():
        ret, raw_frame = cap.read()
        if not ret:
            st.error("Failed to access web camera.")
            break
            
        frame_count += 1
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
            for tracker_id, cid in zip(detections.tracker_id, detections.class_id):
                unique_tree_ids.add(int(tracker_id))
                class_name = model_names.get(int(cid), "Palm Tree")
                labels.append(f"{class_name} #{tracker_id}")
                
            frame = box_annotator.annotate(scene=frame, detections=detections)
            frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
            
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        
        cam_placeholder.image(buffer.tobytes(), use_container_width=True)
        metric_placeholder.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(unique_tree_ids)}</div>
                <div class="metric-label">🌴 Total Unique Palm Trees Tracked (Live Camera)</div>
            </div>
        """, unsafe_allow_html=True)
        
    cap.release()

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
            for tracker_id, cid in zip(detections.tracker_id, detections.class_id):
                unique_tree_ids.add(int(tracker_id))
                class_name = model_names.get(int(cid), "Palm Tree")
                labels.append(f"{class_name} #{tracker_id}")

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

        status_placeholder.caption("🟢 **Status:** Processing Live Stream with YOLOv11...")

    cap.release()
    status_placeholder.success("✅ **Status:** Stream Completed Successfully!")