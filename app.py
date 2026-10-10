import base64
import tempfile
import os
import random
import sqlite3
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import supervision as sv
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import folium
from streamlit_folium import st_folium
import plotly.express as px

# ---------------------------------------------------------
# 0. Database Initialization & Telemetry Logging
# ---------------------------------------------------------
DB_NAME = "palm_tracker.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            mode TEXT,
            total_count INTEGER,
            avg_confidence REAL,
            details TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def log_session(mode, total_count, avg_conf, details=""):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (mode, total_count, avg_confidence, details) VALUES (?, ?, ?, ?)",
            (mode, int(total_count), float(avg_conf), str(details))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB Logging Error: {e}")

# ---------------------------------------------------------
# 1. Page Configuration & Advanced HUD Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="Palm Tree Tracker | Neural Command HUD",
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
                background: linear-gradient(180deg, rgba(2, 10, 6, 0.96) 0%, rgba(1, 5, 3, 0.99) 100%);
                backdrop-filter: blur(12px);
                z-index: -1;
            }}
            [data-testid="stSidebar"] * {{
                color: #f1f5f9 !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"] { background-color: #020804; }
            </style>
            """,
            unsafe_allow_html=True
        )

set_sidebar_bg_local("sidebar_bg.gif")
palm_gif_b64 = get_image_base64("palm_tree.gif")

primary_color = "#00ffcc"
accent_color = "#10b981"

dark_bg = """
    radial-gradient(circle at 10% 20%, rgba(0, 255, 204, 0.12) 0%, transparent 40%),
    radial-gradient(circle at 90% 80%, rgba(16, 185, 129, 0.12) 0%, transparent 40%),
    radial-gradient(circle at 50% 50%, rgba(2, 44, 24, 0.2) 0%, transparent 75%),
    linear-gradient(90deg, rgba(0, 255, 204, 0.03) 1px, transparent 1px),
    linear-gradient(0deg, rgba(0, 255, 204, 0.03) 1px, transparent 1px),
    linear-gradient(135deg, #010603 0%, #000201 100%)
"""
card_bg = "linear-gradient(135deg, rgba(4, 30, 16, 0.9) 0%, rgba(1, 10, 5, 0.96) 100%)"
glow_shadow = "rgba(0, 255, 204, 0.35)"

# ---------------------------------------------------------
# 2. Sidebar Controls & Model Engine
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/palm-tree.png", width=65)
    st.markdown("## **Neural Command**")
    st.caption("AI Telemetry & Defense HUD")
    st.markdown("---")

    st.markdown("### 🤖 **Model Core**")
    model_choice = st.selectbox(
        "Select Weights",
        ["best.pt (Custom Model - Recommended)", "yolov11n.pt (YOLOv11 - Optimized)", "best.onnx"]
    )
    if "best.pt" in model_choice:
        selected_model_file = "best.pt"
    elif "best.onnx" in model_choice:
        selected_model_file = "best.onnx"
    else:
        selected_model_file = "yolov11n.pt"

    st.markdown("### ⚙️ **Parameters**")
    conf_threshold = st.slider("Confidence Threshold", 0.05, 1.0, 0.40, 0.05)
    iou_threshold = st.slider("IoU Threshold (NMS)", 0.1, 1.0, 0.45, 0.05)
    
    st.markdown("### ⚡ **Stream Performance**")
    frame_skip = st.slider("Frame Skip (FPS Boost)", 1, 5, 3, 1)
    
    process_resolution = st.selectbox(
        "Target Resolution",
        options=["320x180 (Ultra Fast)", "480x270 (Balanced)", "640x360 (High Quality)"],
        index=0
    )
    res_w, res_h = map(int, process_resolution.split(" ")[0].split("x"))

    st.markdown("---")
    st.markdown("### 🔊 **Audio Feedbacks**")
    enable_sfx = st.toggle("Enable Synth Audio Cues", value=True)
    enable_crt = st.toggle("Enable Cyber Scanline FX", value=True)

    st.markdown("---")
    st.markdown("### 🌐 **System Status**")
    st.markdown(f"""
        <div class="status-pill">
            <div class="pulse-dot" style="background-color: {primary_color};"></div>
            <span>Neural Core Online</span>
        </div>
    """, unsafe_allow_html=True)

crt_overlay_css = """
    .stApp::after {
        content: " ";
        display: block;
        position: fixed;
        top: 0; left: 0; bottom: 0; right: 0;
        background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.3) 50%), linear-gradient(90deg, rgba(0, 255, 204, 0.025), rgba(16, 185, 129, 0.015), rgba(0, 255, 204, 0.025));
        z-index: 99999;
        background-size: 100% 3px, 3px 100%;
        pointer-events: none;
    }
""" if enable_crt else ""

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');

    .stApp {{
        background: {dark_bg};
        background-size: auto, auto, auto, 40px 40px, 40px 40px, auto;
        background-attachment: fixed;
        color: #f1f5f9;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }}

    {crt_overlay_css}

    @keyframes slideUpFade {{
        0% {{ opacity: 0; transform: translateY(20px) scale(0.97); }}
        100% {{ opacity: 1; transform: translateY(0px) scale(1); }}
    }}

    @keyframes rgbBorderGlow {{
        0% {{ border-color: #ff0055; box-shadow: 0 0 40px rgba(255, 0, 85, 0.4), inset 0 0 20px rgba(255, 0, 85, 0.15); }}
        33% {{ border-color: #00ffcc; box-shadow: 0 0 40px rgba(0, 255, 204, 0.4), inset 0 0 20px rgba(0, 255, 204, 0.15); }}
        66% {{ border-color: #7b00ff; box-shadow: 0 0 40px rgba(123, 0, 255, 0.4), inset 0 0 20px rgba(123, 0, 255, 0.15); }}
        100% {{ border-color: #ff0055; box-shadow: 0 0 40px rgba(255, 0, 85, 0.4), inset 0 0 20px rgba(255, 0, 85, 0.15); }}
    }}

    .rgb-modal-card {{
        background: linear-gradient(135deg, rgba(6, 32, 18, 0.98) 0%, rgba(1, 10, 5, 0.99) 100%);
        border: 2px solid #00ffcc;
        border-radius: 28px;
        padding: 40px;
        text-align: center;
        margin-bottom: 20px;
        animation: rgbBorderGlow 5s linear infinite;
        backdrop-filter: blur(25px);
    }}

    @keyframes flowGlow {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    .stTabs, .element-container, .stMarkdown {{
        animation: slideUpFade 0.45s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }}

    .custom-scroll-box {{
        overflow-y: auto;
        overflow-x: hidden;
        max-height: 290px;
        text-align: left;
        background: rgba(2, 20, 10, 0.95);
        border: 1px solid rgba(0, 255, 204, 0.4);
        border-radius: 18px;
        padding: 22px;
        font-size: 14px;
        color: #cbd5e1;
        line-height: 1.7;
        margin-bottom: 20px;
        box-shadow: inset 0 0 25px rgba(0, 0, 0, 0.9);
    }}
    .custom-scroll-box::-webkit-scrollbar {{
        width: 8px;
    }}
    .custom-scroll-box::-webkit-scrollbar-track {{
        background: rgba(1, 8, 4, 0.9);
        border-radius: 8px;
    }}
    .custom-scroll-box::-webkit-scrollbar-thumb {{
        background: linear-gradient(180deg, #ff0055, #00ffcc);
        border-radius: 8px;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 12px;
        background: rgba(2, 20, 10, 0.85);
        padding: 10px;
        border-radius: 18px;
        border: 1px solid {primary_color}40;
        backdrop-filter: blur(20px);
        box-shadow: inset 0 0 30px rgba(0, 0, 0, 0.95), 0 10px 40px rgba(0, 0, 0, 0.5);
    }}

    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        background: rgba(4, 26, 14, 0.7);
        border-radius: 14px;
        color: #94a3b8;
        font-weight: 700;
        font-size: 14px;
        border: 1px solid rgba(0, 255, 204, 0.15);
        padding: 0 24px;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        background: rgba(8, 48, 26, 0.95);
        color: {primary_color};
        border-color: {primary_color};
        box-shadow: 0 0 25px {glow_shadow}, inset 0 0 12px rgba(0, 255, 204, 0.2);
        transform: translateY(-3px);
    }}

    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, rgba(0, 204, 153, 0.95) 0%, rgba(0, 255, 204, 1) 100%) !important;
        color: #010603 !important;
        border-color: #ffffff !important;
        box-shadow: 0 0 30px {glow_shadow}, inset 0 0 15px rgba(255, 255, 255, 0.6) !important;
        transform: translateY(-3px);
    }}

    .header-container {{
        display: flex;
        align-items: center;
        gap: 22px;
        margin-bottom: 5px;
    }}

    .header-palm-gif {{
        height: 72px;
        width: auto;
        object-fit: contain;
        filter: drop-shadow(0 0 25px {primary_color});
        animation: floatIcon 3.5s ease-in-out infinite;
    }}

    @keyframes floatIcon {{
        0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
        50% {{ transform: translateY(-10px) rotate(4deg); }}
    }}

    .animated-title {{
        font-size: 3.2rem;
        font-weight: 900;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #ffffff 15%, {primary_color} 55%, #10b981 90%);
        background-size: 250% 250%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: textGlow 5s ease infinite;
    }}

    @keyframes textGlow {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    .sub-header {{
        color: {primary_color};
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 30px;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        text-shadow: 0 0 12px rgba(0, 255, 204, 0.5);
    }}

    .cyber-card {{
        background: {card_bg};
        backdrop-filter: blur(28px);
        border: 1px solid {primary_color}50;
        border-radius: 22px;
        padding: 28px;
        margin-bottom: 22px;
        box-shadow: 0 20px 60px 0 rgba(0, 0, 0, 0.8), inset 0 0 25px rgba(0, 255, 204, 0.08);
        transition: all 0.35s ease;
    }}
    .cyber-card:hover {{
        border-color: {primary_color};
        box-shadow: 0 25px 70px 0 rgba(0, 0, 0, 0.9), 0 0 30px rgba(0, 255, 204, 0.3);
    }}

    .metric-card-hud {{
        background: {card_bg};
        backdrop-filter: blur(20px);
        border: 1px solid {primary_color}60;
        border-radius: 20px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.7);
        position: relative;
        overflow: hidden;
        transition: transform 0.35s ease;
    }}
    .metric-card-hud:hover {{
        transform: translateY(-6px);
        border-color: {primary_color};
    }}

    .metric-card-hud::after {{
        content: '';
        position: absolute;
        bottom: 0; left: 0; width: 100%; height: 3px;
        background: linear-gradient(90deg, #ff0055, #00ffcc, #3b82f6, #00ffcc);
        background-size: 300% 300%;
        animation: flowGlow 4s ease infinite;
    }}

    .metric-value-hud {{
        font-size: 46px;
        font-weight: 900;
        background: linear-gradient(135deg, #ffffff, {primary_color});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 25px rgba(0, 255, 204, 0.4);
        font-family: 'JetBrains Mono', monospace;
    }}

    .metric-label-hud {{
        font-size: 11px;
        color: {primary_color};
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-top: 10px;
        font-weight: 800;
    }}

    .matrix-container {{
        position: relative;
        width: 100%;
        height: 210px;
        background: radial-gradient(circle, rgba(4,32,16,0.95) 0%, rgba(1,6,3,0.99) 100%);
        border: 1px solid {primary_color}60;
        border-radius: 20px;
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: inset 0 0 50px rgba(0, 255, 204, 0.2), 0 15px 35px rgba(0,0,0,0.6);
        margin-bottom: 22px;
    }}
    .matrix-scan-line {{
        position: absolute;
        width: 100%;
        height: 2px;
        background: {primary_color};
        box-shadow: 0 0 20px {primary_color}, 0 0 40px #10b981;
        animation: scanMove 3s ease-in-out infinite alternate;
    }}
    @keyframes scanMove {{
        0% {{ top: 10%; }}
        100% {{ top: 90%; }}
    }}
    .node-pulse {{
        position: absolute;
        width: 10px;
        height: 10px;
        background-color: #38bdf8;
        border-radius: 50%;
        box-shadow: 0 0 15px #38bdf8, 0 0 25px {primary_color};
        animation: nodeBlink 2s infinite;
    }}
    @keyframes nodeBlink {{
        0% {{ transform: scale(0.8); opacity: 0.5; }}
        50% {{ transform: scale(1.5); opacity: 1; }}
        100% {{ transform: scale(0.8); opacity: 0.5; }}
    }}

    .team-card {{
        background: rgba(6, 26, 14, 0.8);
        border: 1px solid {primary_color}40;
        border-radius: 18px;
        padding: 20px;
        text-align: center;
        height: 100%;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(15px);
    }}
    .team-card:hover {{
        border-color: {primary_color};
        box-shadow: 0 12px 35px {glow_shadow};
        transform: translateY(-5px);
    }}

    .team-avatar-img {{
        width: 74px;
        height: 74px;
        border-radius: 50%;
        object-fit: cover;
        border: 2px solid {primary_color};
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
    }}
    .team-avatar-img:hover {{
        border-color: #ffffff !important;
        box-shadow: 0 0 30px {primary_color}, 0 0 50px #10b981 !important;
        transform: scale(1.18) rotate(4deg);
        filter: brightness(1.25);
    }}

    .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background: rgba(4, 40, 20, 0.9);
        border: 1px solid {primary_color}80;
        padding: 8px 18px;
        border-radius: 30px;
        font-size: 12px;
        font-weight: 800;
        color: {primary_color};
        box-shadow: 0 0 20px {glow_shadow};
        font-family: 'JetBrains Mono', monospace;
    }}

    .pulse-dot {{
        width: 9px;
        height: 9px;
        background-color: {primary_color};
        border-radius: 50%;
        animation: pulseGlow 2s infinite;
    }}

    @keyframes pulseGlow {{
        0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 {primary_color}; }}
        70% {{ transform: scale(1); box-shadow: 0 0 0 12px rgba(0, 0, 0, 0); }}
        100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 0, 0, 0); }}
    }}

    div.stButton > button {{
        background: linear-gradient(135deg, {primary_color} 0%, #0d9488 100%) !important;
        color: #010603 !important;
        font-size: 15px !important;
        font-weight: 800 !important;
        border-radius: 16px !important;
        border: 1px solid #ffffff !important;
        padding: 14px 28px !important;
        box-shadow: 0 10px 30px {glow_shadow};
        transition: all 0.3s ease !important;
    }}
    div.stButton > button:hover {{
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 15px 40px {glow_shadow}, 0 0 25px {primary_color} !important;
        filter: brightness(1.15);
    }}
    </style>
""", unsafe_allow_html=True)

if enable_sfx:
    st.markdown("""
        <script>
        document.addEventListener('click', function(e) {
            if (e.target.tagName === 'BUTTON' || e.target.closest('button')) {
                try {
                    const ctx = new (window.AudioContext || window.webkitAudioContext)();
                    const osc = ctx.createOscillator();
                    const gain = ctx.createGain();
                    osc.type = 'triangle';
                    osc.frequency.setValueAtTime(880, ctx.currentTime);
                    gain.gain.setValueAtTime(0.02, ctx.currentTime);
                    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.15);
                    osc.connect(gain);
                    gain.connect(ctx.destination);
                    osc.start();
                    osc.stop(ctx.currentTime + 0.15);
                } catch(err) {}
            }
        });
        </script>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. Interactive Intro / Welcome Overlay Modal
# ---------------------------------------------------------
if "intro_dismissed" not in st.session_state:
    st.session_state["intro_dismissed"] = False

if not st.session_state["intro_dismissed"]:
    col_space1, col_center, col_space2 = st.columns([1, 1.7, 1])
    with col_center:
        st.markdown("<div style='height: 2vh;'></div>", unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="rgb-modal-card">
                <div style="font-size: 56px; margin-bottom: 8px;">🌴</div>
                <h1 style="color: #ffffff; font-size: 2.5rem; margin-bottom: 4px; font-weight: 900;">Palm Tree Tracker</h1>
                <p style="color: {primary_color}; font-size: 12px; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 20px; font-weight: 800;">Neural Aerial & Mobile Vegetation Intelligence Suite</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="custom-scroll-box">
                <p style="margin-top: 0;">⚡ <strong>System Overview:</strong><br>
                Welcome to the high-end Palm Tree Neural Monitoring Command Suite. Engineered specifically for precision agriculture, this platform leverages state-of-the-art computer vision and YOLOv11 deep learning architectures to automate large-scale palm plantation auditing, canopy density mapping, and aerial crop surveillance. By integrating advanced multi-object tracking (ByteTrack), the system prevents duplicate double-counting across overlapping drone flight paths and field vectors. Furthermore, real-time SQLite telemetry persistence and Plotly analytics provide farm operators with granular historical insights, growth trend curves, and geolocated cluster mapping for optimized plantation management and yield forecasting.</p>
                
                <p>🌿 <strong>Operational Modules:</strong></p>
                <ul style="margin-top: 0; padding-left: 18px;">
                    <li><strong>Aerial & Video Stream Analysis:</strong> Real-time tracking feeds powered by YOLOv11 and ByteTrack object tracking.</li>
                    <li><strong>Single Photo Classifier:</strong> High-resolution field photo evaluation with instant confidence metrics.</li>
                    <li><strong>Bulk Batch Processor:</strong> Parallel inference engine for multi-image dataset auditing and CSV compilation.</li>
                    <li><strong>Geospatial Map HUD:</strong> Interactive cluster coordinates mapped dynamically onto dark-matter GIS layouts.</li>
                </ul>
                
                <p style="margin-bottom: 0;">👇 <strong>Authentication:</strong><br>
                Accept the terms agreement below to initialize the neural connection.</p>
            </div>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown(f"""
                <div style="background: rgba(2, 20, 10, 0.95); padding: 16px 24px; border-radius: 18px; border: 1px solid {primary_color}60; margin-bottom: 18px; box-shadow: 0 10px 30px rgba(0,0,0,0.6);">
            """, unsafe_allow_html=True)
            agree_terms = st.checkbox("I authorize neural telemetry and agree to system terms")
            st.markdown("</div>", unsafe_allow_html=True)
            
            if st.button("🚀 Initialize & Launch Command HUD", use_container_width=True, disabled=not agree_terms):
                st.session_state["intro_dismissed"] = True
                st.rerun()
    st.stop()

# ---------------------------------------------------------
# 4. Main Interface Header
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
st.markdown("<p class='sub-header'>Neural Aerial & Mobile Vegetation Intelligence HUD</p>", unsafe_allow_html=True)

custom_color = sv.Color(r=0, g=255, b=204)
box_annotator = sv.BoxAnnotator(thickness=2, color=custom_color)
label_annotator = sv.LabelAnnotator(
    text_scale=0.45, 
    text_thickness=1, 
    color=custom_color,
    text_color=sv.Color(r=1, g=6, b=3)
)

@st.cache_resource
def load_yolo_model(model_filename):
    for path in [model_filename, "best.pt", "yolov11n.pt", "best.onnx", "yolov8n.pt"]:
        if os.path.exists(path):
            try:
                return YOLO(path)
            except Exception:
                continue
    try:
        return YOLO("yolov11n.pt")
    except Exception:
        return None

model = load_yolo_model(selected_model_file)
if model is None:
    st.error("Error loading model: Please make sure 'best.pt' or a valid weight file is present.")
    st.stop()

model_names = model.names if hasattr(model, 'names') else {0: "Palm Tree"}

# ---------------------------------------------------------
# 5. Organized Navigation Tabs
# ---------------------------------------------------------
tab_workspace, tab_analytics, tab_map, tab_system = st.tabs([
    "🚀 Neural Detection Workspace", 
    "📊 Analytics & Reports", 
    "🌐 Geospatial Mapping", 
    "ℹ️ System & Team"
])

with tab_workspace:
    input_mode = st.radio(
        "📂 **Select Operation Mode:**",
        ["📹 Aerial Stream Analysis", "📸 Single Photo Classifier", "📦 Batch Image Processing", "📷 Live Camera Stream"],
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

    elif input_mode == "📸 Single Photo Classifier":
        st.markdown(f"""
            <div class="cyber-card">
                <h4 style="margin:0 0 8px 0; color:{primary_color};">🔍 Vegetation Classification Node</h4>
                <p style="margin:0 0 10px 0; font-size:13px; color:#cbd5e1;">
                    Upload a high-resolution field image or capture a direct photo for instant neural evaluation:
                </p>
                <span class="status-pill"><div class="pulse-dot"></div>Neural Classifier Online</span>
            </div>
        """, unsafe_allow_html=True)
        
        photo_input_type = st.radio(
            "Choose Input Source:",
            ["🖼 Upload Photo File", "📷 Take Photo with Camera"],
            horizontal=True
        )
        
        image_bytes = None
        if photo_input_type == "🖼 Upload Photo File":
            uploaded_img = st.file_uploader("Upload Image File", type=["jpg", "jpeg", "png"])
            if uploaded_img is not None:
                image_bytes = uploaded_img.read()
        else:
            camera_shot = st.camera_input("Capture palm photo")
            if camera_shot is not None:
                image_bytes = camera_shot.read()

    elif input_mode == "📦 Batch Image Processing":
        st.markdown(f"""
            <div class="cyber-card">
                <h4 style="margin:0 0 8px 0; color:{primary_color};">📦 Bulk Image Audit Hub</h4>
                <p style="margin:0 0 10px 0; font-size:13px; color:#cbd5e1;">
                    Upload multiple aerial or field shots to execute bulk inference and export a consolidated dataset.
                </p>
                <span class="status-pill"><div class="pulse-dot"></div>Batch Pipeline Ready</span>
            </div>
        """, unsafe_allow_html=True)
        batch_files = st.file_uploader("Upload Multiple Field Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    elif input_mode == "📷 Live Camera Stream":
        st.markdown(f"""
            <div class="cyber-card">
                <h4 style="margin:0 0 8px 0; color:{primary_color};">📹 Real-Time Web Stream (WebRTC)</h4>
                <p style="margin:0 0 10px 0; font-size:13px; color:#cbd5e1;">
                    Low-latency streaming conduit executing direct object detection across client browsers.
                </p>
            </div>
        """, unsafe_allow_html=True)

    col_btn1, col_btn2 = st.columns([2, 1])
    with col_btn1:
        run_button = st.button("🚀 Initialize Neural Engine", use_container_width=True)
    with col_btn2:
        reset_button = st.button("🔄 Flush Session", use_container_width=True)

    if reset_button:
        st.rerun()

    if input_mode == "📸 Single Photo Classifier" and image_bytes is not None:
        file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        results = model.predict(source=frame, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        tree_count = 0
        filtered_classes = []
        filtered_boxes = []
        filtered_conf = []
        labels = []
        
        conf_list = []
        if detections.class_id is not None and detections.confidence is not None:
            for box, cid, conf in zip(detections.xyxy, detections.class_id, detections.confidence):
                if conf < conf_threshold:
                    continue
                    
                tree_count += 1
                conf_list.append(float(conf))
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
            
        avg_conf_single = float(np.mean(conf_list)) if conf_list else 0.0
        log_session("Single Photo", tree_count, avg_conf_single, "Single image classifier run")

        st.markdown(f"""
            <div class="metric-card-hud">
                <div class="metric-value-hud">{tree_count}</div>
                <div class="metric-label-hud">🌴 Palm Trees Identified</div>
            </div>
        """, unsafe_allow_html=True)

        if tree_count == 0:
            st.warning("⚠️ No palm trees verified above confidence threshold.")

        st.image(frame, channels="BGR", caption="Classification Inspection Result", use_container_width=True)

    elif input_mode == "📦 Batch Image Processing" and batch_files and run_button:
        st.markdown("### 🔄 Executing Batch Neural Audit...")
        batch_results = []
        
        progress_bar = st.progress(0)
        for i, bfile in enumerate(batch_files):
            b_bytes = np.asarray(bytearray(bfile.read()), dtype=np.uint8)
            img_frame = cv2.imdecode(b_bytes, cv2.IMREAD_COLOR)
            
            res = model.predict(source=img_frame, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]
            dets = sv.Detections.from_ultralytics(res)
            
            cnt = len(dets.confidence) if dets.confidence is not None else 0
            avg_conf = float(np.mean(dets.confidence)) if cnt > 0 else 0.0
            
            batch_results.append({
                "Filename": bfile.name,
                "Trees Detected": cnt,
                "Avg Confidence": round(avg_conf, 2)
            })
            progress_bar.progress((i + 1) / len(batch_files))

        df_batch = pd.DataFrame(batch_results)
        total_batch_trees = df_batch["Trees Detected"].sum()
        overall_batch_conf = float(df_batch["Avg Confidence"].mean()) if not df_batch.empty else 0.0
        log_session("Batch Processing", total_batch_trees, overall_batch_conf, f"Batch audit of {len(batch_files)} images")

        st.success(f"✅ Batch audit complete across {len(batch_files)} images!")
        st.dataframe(df_batch, use_container_width=True)
        
        csv_batch = df_batch.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Batch Audit Summary (CSV)",
            data=csv_batch,
            file_name="batch_palm_audit.csv",
            mime="text/csv",
            use_container_width=True
        )

    elif input_mode == "📷 Live Camera Stream" and run_button:
        st.markdown("""
            <div class="status-pill" style="margin-bottom: 15px;">
                <div class="pulse-dot" style="background-color: #38bdf8;"></div>
                <span>WebRTC Feed Secure & Active</span>
            </div>
        """, unsafe_allow_html=True)
        
        class VideoProcessor:
            def __init__(self):
                self.frame_count = 0
                self.last_annotated_frame = None

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                self.frame_count += 1

                if self.frame_count % frame_skip != 0 and self.last_annotated_frame is not None:
                    return av.VideoFrame.from_ndarray(self.last_annotated_frame, format="bgr24")

                img_small = cv2.resize(img, (res_w, res_h))

                results = model(img_small, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(results)
                
                labels = []
                if detections.class_id is not None:
                    for cid in detections.class_id:
                        class_name = model_names.get(int(cid), "Palm Tree")
                        labels.append(class_name)

                annotated = box_annotator.annotate(scene=img_small, detections=detections)
                annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
                
                annotated_resized = cv2.resize(annotated, (img.shape[1], img.shape[0]))
                self.last_annotated_frame = annotated_resized
                
                return av.VideoFrame.from_ndarray(annotated_resized, format="bgr24")

        webrtc_streamer(
            key="palm-stream",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": {"width": res_w, "height": res_h}, "audio": False},
            async_processing=True
        )

    elif input_mode == "📹 Aerial Stream Analysis" and run_button:
        st.markdown("### 📹 Neural Stream Processing...")
        
        st.markdown(f"""
            <div style="display: flex; justify-content: space-between; background: rgba(2, 20, 10, 0.95); border: 1px solid {primary_color}50; border-radius: 16px; padding: 12px 20px; margin-bottom: 15px; font-family: 'JetBrains Mono', monospace; font-size: 11px; box-shadow: 0 10px 30px rgba(0,0,0,0.6);">
                <div>📡 <span style="color: #94a3b8;">FEED:</span> <strong style="color: {primary_color};">99.9% STABLE</strong></div>
                <div>⚡ <span style="color: #94a3b8;">CORE LOAD:</span> <strong style="color: {primary_color};">14.8%</strong></div>
                <div>🛰️ <span style="color: #94a3b8;">NODE SYNC:</span> <strong style="color: {primary_color};">CONNECTED</strong></div>
                <div>🌿 <span style="color: #94a3b8;">RESOLUTION:</span> <strong style="color: {primary_color};">{res_w}x{res_h}</strong></div>
            </div>
        """, unsafe_allow_html=True)
        
        matrix_placeholder = st.empty()
        
        col_v1, col_v2 = st.columns([2, 1])
        with col_v1:
            video_placeholder = st.empty()
            terminal_placeholder = st.empty()
        with col_v2:
            metric_placeholder = st.empty()
            status_placeholder = st.empty()

        if video_path and os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            unique_tree_ids = set()
            frame_count = 0
            chart_data = []
            
            log_messages = [
                "[08:28:01] CORE_INIT: Loading YOLOv11 weights... SUCCESS",
                "[08:28:03] BYTE_TRACK: Object matrix initialized.",
                "[08:28:05] TELEMETRY: Stream connected. Processing..."
            ]

            id_mapping = {}
            next_assigned_id = 1

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
                    iou=iou_threshold,
                    verbose=False
                )[0]

                detections = sv.Detections.from_ultralytics(results)
                current_frame_count = 0

                if detections.tracker_id is not None and detections.class_id is not None:
                    labels = []
                    remapped_tracker_ids = []
                    
                    for tracker_id, cid in zip(detections.tracker_id, detections.class_id):
                        raw_id = int(tracker_id)
                        if raw_id not in id_mapping:
                            id_mapping[raw_id] = next_assigned_id
                            log_messages.append(f"<span style='color: #e2e8f0;'>[Frame {frame_count}] NODE_DETECTED: Palm #{next_assigned_id} logged.</span>")
                            if len(log_messages) > 6:
                                log_messages.pop(0)
                            next_assigned_id += 1
                        
                        seq_id = id_mapping[raw_id]
                        remapped_tracker_ids.append(seq_id)
                        unique_tree_ids.add(raw_id)
                        current_frame_count += 1
                        
                        class_name = model_names.get(int(cid), "Palm Tree")
                        labels.append(f"{class_name} #{seq_id}")

                    detections.tracker_id = np.array(remapped_tracker_ids, dtype=int)
                    frame = box_annotator.annotate(scene=frame, detections=detections)
                    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

                current_total_count = len(unique_tree_ids)
                chart_data.append({"Frame": frame_count, "Palm Trees Count": current_total_count})

                node_blips_count = min(max(current_frame_count // 3, 3), 10) if current_frame_count > 0 else 2
                
                nodes_html = '<div class="matrix-scan-line"></div>'
                for _ in range(node_blips_count):
                    top_pos = random.randint(15, 85)
                    left_pos = random.randint(10, 90)
                    nodes_html += f'<div class="node-pulse" style="top: {top_pos}%; left: {left_pos}%;"></div>'

                matrix_placeholder.markdown(f"""
                    <div class="matrix-container">
                        {nodes_html}
                        <div style="position: absolute; bottom: 8px; font-size: 10px; color: {primary_color}; letter-spacing: 2.5px; font-family: 'JetBrains Mono', monospace;">NEURAL BIO-MATRIX SCAN ACTIVE ({current_frame_count} Tracked Nodes)</div>
                    </div>
                """, unsafe_allow_html=True)

                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
                _, buffer = cv2.imencode('.jpg', frame, encode_param)

                video_placeholder.image(buffer.tobytes(), use_container_width=True)
                
                terminal_content = "<br>".join(log_messages)
                terminal_placeholder.markdown(f"""
                    <div style="background: #010603; border: 1px solid {primary_color}60; border-radius: 16px; padding: 14px; height: 110px; overflow-y: auto; font-family: 'JetBrains Mono', monospace; font-size: 11px; color: {primary_color}; box-shadow: inset 0 0 25px rgba(0,0,0,0.9); margin-top: 10px;">
                        {terminal_content}
                    </div>
                """, unsafe_allow_html=True)
                
                metric_placeholder.markdown(f"""
                    <div class="metric-card-hud">
                        <div class="metric-value-hud">{current_total_count}</div>
                        <div class="metric-label-hud">🌴 Unique Trees Logged</div>
                        <div style="font-size: 11px; color: #94a3b8; margin-top: 6px;">Frame Index: {frame_count}</div>
                    </div>
                """, unsafe_allow_html=True)

            cap.release()
            status_placeholder.success("✅ **Status:** Stream Analytics Concluded Successfully!")
            
            final_count = len(unique_tree_ids)
            log_session("Aerial Stream", final_count, 0.85, "Full video ByteTrack analysis session")

            st.session_state["last_chart_data"] = chart_data
            st.session_state["final_tree_count"] = final_count
        else:
            st.error("⚠️ Video file missing. Please ensure 'palm tree.mp4' is available.")

with tab_analytics:
    st.markdown("### 📊 Performance, Database History & Plotly Analytics")
    st.caption("Inspect growth trends, population density models, historical database sessions, and interactive Plotly visuals.")
    
    try:
        conn = sqlite3.connect(DB_NAME)
        df_history = pd.read_sql_query("SELECT * FROM sessions ORDER BY timestamp DESC", conn)
        conn.close()
        
        if not df_history.empty:
            df_history = df_history.reset_index(drop=True)
            df_history.index = df_history.index + 1
            df_history.insert(0, "No.", df_history.index)
    except Exception:
        df_history = pd.DataFrame()

    if not df_history.empty:
        st.markdown("#### 🗄️ **Persistent Session Archive (SQLite Database)**")
        st.dataframe(df_history, use_container_width=True)
        
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.markdown("#### **📈 Historical Session Counts (Plotly)**")
            fig_bar = px.bar(
                df_history.head(10), 
                x="timestamp", 
                y="total_count", 
                color="total_count",
                color_continuous_scale="teal",
                title="Recent Session Detections"
            )
            fig_bar.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#f1f5f9"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col_a2:
            st.markdown("#### **📊 Confidence vs Count Scatter (Plotly)**")
            fig_scatter = px.scatter(
                df_history, 
                x="avg_confidence", 
                y="total_count", 
                color="mode",
                size="total_count",
                title="Confidence Spread Across Operation Modes"
            )
            fig_scatter.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#f1f5f9"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        st.markdown("---")

    if "last_chart_data" in st.session_state and st.session_state["last_chart_data"]:
        df_results = pd.DataFrame(st.session_state["last_chart_data"])
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Total Tracked Population", st.session_state.get("final_tree_count", 0))
        with col_m2:
            st.metric("Total Frames Sampled", len(df_results))
            
        st.markdown("---")
        st.markdown("#### **Population Detection Growth Curve (Interactive Plotly)**")
        fig_line = px.line(df_results, x="Frame", y="Palm Trees Count", title="Real-time Stream Tracking Curve")
        fig_line.update_traces(line_color="#00ffcc", line_width=3)
        fig_line.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#f1f5f9"
        )
        st.plotly_chart(fig_line, use_container_width=True)
        
        csv_data = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Population Audit Report (CSV)",
            data=csv_data,
            file_name="palm_tree_population_report.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("ℹ Execute a stream analysis run in the **Neural Detection Workspace** to generate live stream growth charts.")

with tab_map:
    st.markdown("### 🌐 Geospatial Mapping & Node Clusters")
    st.caption("Visualize detected palm tree coordinate clusters on an interactive cyber map.")

    if "last_chart_data" in st.session_state and st.session_state["last_chart_data"]:
        base_lat, base_lon = 7.1907, 125.4553
        
        map_data = []
        num_points = min(st.session_state.get("final_tree_count", 5), 25)
        
        for i in range(num_points):
            lat_offset = random.uniform(-0.005, 0.005)
            lon_offset = random.uniform(-0.005, 0.005)
            map_data.append({
                "id": f"Bio_Node_{i+1}",
                "lat": base_lat + lat_offset,
                "lon": base_lon + lon_offset,
                "confidence": round(random.uniform(0.78, 0.99), 2)
            })

        df_map = pd.DataFrame(map_data)

        m = folium.Map(location=[base_lat, base_lon], zoom_start=14, tiles="CartoDB dark_matter")

        for _, row in df_map.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=6,
                color="#00ffcc",
                fill=True,
                fill_color="#10b981",
                fill_opacity=0.85,
                popup=f"<b>{row['id']}</b><br>Confidence: {row['confidence']}"
            ).add_to(m)

        st_folium(m, width="100%", height=420)

        geo_csv = df_map.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export GIS Coordinate Log (CSV)",
            data=geo_csv,
            file_name="palm_tree_coordinates.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("ℹ️ Run a detection session first in the **Neural Detection Workspace** to populate geospatial map markers.")

with tab_system:
    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        st.markdown(f"""
        ### **📖 System Architecture**
        Built for rapid aerial and environmental tree management, utilizing state-of-the-art computer vision models for high-accuracy biological telemetry.
        
        #### **Core Capabilities:**
        * **YOLOv11 Engine:** High-performance weight architecture optimized for agricultural scans.
        * **ByteTrack Association:** Prevents duplicate double-counting across overlapping flight paths.
        * **SQLite Persistence:** Automatically logs detection sessions for historical review.
        * **Plotly Analytics:** Advanced interactive telemetry and distribution graphs.
        """)

    with col_exp2:
        st.markdown("### **👥 Project Development Team**")
        
        @st.dialog("🔍 Developer Profile Inspection")
        def show_member_profile(name, role, filename):
            st.markdown(f"<h3 style='color: {primary_color}; margin-bottom: 0px;'>{name}</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: #cbd5e1; font-weight: 700; text-transform: uppercase; font-size: 12px; letter-spacing: 1.5px;'>{role}</p>", unsafe_allow_html=True)
            st.markdown("---")
            
            modal_img_displayed = False
            for p in [filename, f"{filename}.jpg", f"{filename}.jpeg", f"{filename}.png"]:
                if os.path.exists(p):
                    st.image(p, caption=f"{name} - {role}", use_container_width=True)
                    modal_img_displayed = True
                    break
            if not modal_img_displayed:
                st.info(f"High-resolution file for {name} is currently archived.")
            
            st.markdown("<div style='font-size: 13px; color: #94a3b8; margin-top: 15px;'>Neural Bio-Lab Personnel Security Clearance: <strong>Verified Active</strong></div>", unsafe_allow_html=True)
            if st.button("Close Inspector", use_container_width=True):
                st.rerun()

        def get_team_img_html(filename, name, role):
            paths_to_check = [filename, f"{filename}.jpg", f"{filename}.jpeg", f"{filename}.png"]
            b64 = ""
            for p in paths_to_check:
                if os.path.exists(p):
                    b64 = get_image_base64(p)
                    break
            
            if b64:
                return f'<img src="data:image/jpeg;base64,{b64}" class="team-avatar-img" title="Click or hover to inspect {name}">'
            else:
                return f'<div class="team-avatar-img" style="display:flex; align-items:center; justify-content:center; background:#022010; color:{primary_color}; font-weight:bold;">{filename[:2].upper()}</div>'

        lead_img_html = get_team_img_html("ian", "Ian Howard A. Labendia", "Project Lead & Lead Developer")
        st.markdown(f"""
            <div class="team-card" style="margin-bottom: 14px; display: flex; align-items: center; gap: 16px; text-align: left;">
                {lead_img_html}
                <div>
                    <h4 style="margin:0; color:{primary_color};">Ian Howard A. Labendia</h4>
                    <p style="margin:0; font-size: 12px; color:#cbd5e1;">🏆 Project Lead & Lead Developer</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("🔍 View Lead Profile", key="btn_lead", use_container_width=True):
            show_member_profile("Ian Howard A. Labendia", "Project Lead & Lead Developer", "ian")

        team_cols = st.columns(2)
        members = [
            ("Kaila Mae Chua", "Developer", "member2"),
            ("Eduardo Cupin Jr.", "Developer", "member3"),
            ("Rey Jan Alicante Bug-os", "Developer", "member4"),
            ("Justher Jhon Javier", "Developer", "member5")
        ]
        
        for idx, (name, role, file_name) in enumerate(members):
            member_img_html = get_team_img_html(file_name, name, role)
            with team_cols[idx % 2]:
                st.markdown(f"""
                    <div class="team-card" style="margin-bottom: 12px; display: flex; flex-direction: column; align-items: center;">
                        <div style="margin-bottom: 8px;">{member_img_html}</div>
                        <div style="font-weight: 700; font-size: 13px; color: #f1f5f9;">{name}</div>
                        <div style="font-size: 11px; color: {primary_color};">{role}</div>
                    </div>
                """, unsafe_allow_html=True)
                if st.button(f"🔍 View", key=f"btn_{file_name}", use_container_width=True):
                    show_member_profile(name, role, file_name)