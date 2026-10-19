import os
import random
import time
from datetime import datetime, timedelta

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Assignment 3: Face Application",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stButton>button {
            border-radius: 8px;
            font-weight: 500;
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid #e2e8f0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# EMOTIONS and PALETTE following EfficientFace
EMOTIONS = ['0_Neutral', '1_Hapiness', '2_Sadness', '3_Surprise', '4_Fear', '5_Disgust', '6_Anger']
PALETTE = {
    "0_Neutral": "#64748b",
    "1_Hapiness": "#f59e0b",
    "2_Sadness": "#3b82f6",
    "3_Surprise": "#ec4899",
    "4_Fear": "#8b5cf6",
    "5_Disgust": "#10b981",
    "6_Anger": "#ef4444",
}

def hex_to_bgr(hex_str):
    hex_str = hex_str.lstrip('#')
    r, g, b = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)

PALETTE_BGR = {emo: hex_to_bgr(color) for emo, color in PALETTE.items()}

IMAGE_DIR = "./captured_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# ---------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------
@st.cache_resource
def load_onnx_model():
    model_path = "models/efficient_face.onnx"
    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    return session, input_name


@st.cache_resource
def load_face_detector():
    mp_face_detection = mp.solutions.face_detection
    detector = mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5,
    )
    return detector


session, input_name = load_onnx_model()
face_detector = load_face_detector()

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def preprocess_face(face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb).resize((224, 224))
    img_data = np.array(face_pil, dtype=np.float32) / 255.0

    mean = np.array([0.57535914, 0.44928582, 0.40079932], dtype=np.float32)
    std = np.array([0.20735591, 0.18981615, 0.18132027], dtype=np.float32)

    img_data = (img_data - mean) / std
    img_data = np.transpose(img_data, (2, 0, 1))
    return np.expand_dims(img_data, axis=0)


# ---------------------------------------------------------
# Session State
# ---------------------------------------------------------
if "tracking_active" not in st.session_state:
    st.session_state.tracking_active = False
if "records" not in st.session_state:
    st.session_state.records = []
if "session_start_time" not in st.session_state:
    st.session_state.session_start_time = None
if "last_plot_time" not in st.session_state:
    st.session_state.last_plot_time = 0.0

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    else:
        st.info("💡 Place `logo.png` in your project folder to display your logo here.")

    st.markdown("### Controls & Settings")
    session_date = st.date_input("Date", datetime.now().date())
    
    duration_minutes = st.number_input(
        "Max Session Duration (Minutes)", min_value=1.0, max_value=60.0, value=10.0, step=1.0
    )
    camera_index = st.number_input(
        "Camera Device Index", min_value=0, max_value=4, value=0, step=1
    )
    
    sampling_interval = 1.0
    st.caption("⏱ Sampling Interval: **1.0 Second**")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "▶ Start",
            type="primary",
            disabled=st.session_state.tracking_active,
            use_container_width=True,
        ):
            st.session_state.tracking_active = True
            st.session_state.session_start_time = time.time()
            st.session_state.last_plot_time = 0.0
            st.rerun()
    with col2:
        if st.button(
            "⏹ Stop",
            disabled=not st.session_state.tracking_active,
            use_container_width=True,
        ):
            st.session_state.tracking_active = False
            st.rerun()

    if st.button("🗑 Reset Session Data", use_container_width=True):
        st.session_state.records = []
        st.rerun()

# ---------------------------------------------------------
# Main Page Layout
# ---------------------------------------------------------
st.title("Face Application Analytics Dashboard")

status_placeholder = st.empty()
col_cam, col_graph = st.columns([1.0, 1.4], gap="medium")

with col_cam:
    cam_placeholder = st.empty()

with col_graph:
    chart_1m_placeholder = st.empty()
    chart_5m_placeholder = st.empty()
    chart_10m_placeholder = st.empty()

st.subheader("Session Log & Report")
table_placeholder = st.empty()
download_placeholder = st.empty()

# ---------------------------------------------------------
# Render Analytics Visuals
# ---------------------------------------------------------
def create_window_figure(df, minutes, show_legend=False):
    window_seconds = minutes * 60.0
    max_elapsed = df["Elapsed (s)"].max()
    min_x = max(0.0, max_elapsed - window_seconds)
    max_x = max(window_seconds, max_elapsed)

    fig = px.line(
        df,
        x="Elapsed (s)",
        y=EMOTIONS,
        color_discrete_map=PALETTE,
        labels={"value": "Probability (%)", "variable": "Emotion", "Elapsed (s)": "Time (s)"},
        range_y=[0, 105],
        range_x=[min_x, max_x],
        template="plotly_white",
        title=f"Last {minutes} Minute{'s' if minutes > 1 else ''}"
    )
    
    layout_args = dict(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        margin=dict(l=15, r=15, t=35, b=15),
        height=240,
        font=dict(color="#334155"),
        xaxis=dict(gridcolor="#e2e8f0"),
        yaxis=dict(gridcolor="#e2e8f0"),
        showlegend=show_legend
    )
    
    if show_legend:
        layout_args["legend"] = dict(
            orientation="h",
            yanchor="bottom",
            y=-0.45,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
        )

    fig.update_layout(**layout_args)
    return fig


def render_dashboard_visuals(force_plot=False):
    df = pd.DataFrame(st.session_state.records)

    if not df.empty:
        current_t = time.time()
        if force_plot or (current_t - st.session_state.get("last_plot_time", 0.0) >= 1.0):
            fig_1m = create_window_figure(df, 1, show_legend=False)
            fig_5m = create_window_figure(df, 5, show_legend=False)
            fig_10m = create_window_figure(df, 10, show_legend=True)

            chart_1m_placeholder.plotly_chart(fig_1m, use_container_width=True, key=f"plot_1m_{random.randint(0, 1e6)}")
            chart_5m_placeholder.plotly_chart(fig_5m, use_container_width=True, key=f"plot_5m_{random.randint(0, 1e6)}")
            chart_10m_placeholder.plotly_chart(fig_10m, use_container_width=True, key=f"plot_10m_{random.randint(0, 1e6)}")
            
            st.session_state.last_plot_time = current_t

        cols_to_show = ["Elapsed (s)", "Timestamp", "Dominant Emotion", "Confidence (%)"] + EMOTIONS
        available_cols = [c for c in cols_to_show if c in df.columns]
        
        table_placeholder.dataframe(
            df.tail(20)[available_cols],
            use_container_width=True
        )

        csv_data = df.to_csv(index=False).encode("utf-8")
        download_placeholder.download_button(
            label="📥 Download Results (CSV)",
            data=csv_data,
            file_name=f"emotion_log_{session_date}.csv",
            mime="text/csv",
            key="csv_download_btn" + str(random.randint(0, 1e6))
        )
    else:
        chart_1m_placeholder.info("Click **Start** to begin capturing and logging data.")
        chart_5m_placeholder.empty()
        chart_10m_placeholder.empty()
        table_placeholder.caption("No records logged for this session yet.")


render_dashboard_visuals(force_plot=True)

# ---------------------------------------------------------
# Live Tracking Loop
# ---------------------------------------------------------
if st.session_state.tracking_active:
    cap = cv2.VideoCapture(int(camera_index), cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(int(camera_index))

    max_duration_seconds = duration_minutes * 60
    last_sample_time = 0.0

    try:
        while st.session_state.tracking_active and cap.isOpened():
            elapsed = time.time() - st.session_state.session_start_time

            if elapsed >= max_duration_seconds:
                st.session_state.tracking_active = False
                status_placeholder.warning(
                    f"Session finished: reached {duration_minutes} minute limit."
                )
                break

            rem_sec = max(0, int(max_duration_seconds - elapsed))
            status_placeholder.info(
                f"● **Recording Active** | Elapsed: {str(timedelta(seconds=int(elapsed)))} | "
                f"Remaining: {str(timedelta(seconds=rem_sec))}"
            )

            ret, frame = cap.read()
            if not ret:
                status_placeholder.error(
                    "Could not capture frame from webcam. Please verify camera index."
                )
                break

            frame = cv2.flip(frame, 1)
            h_frame, w_frame, _ = frame.shape

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection_results = face_detector.process(frame_rgb)

            current_time = time.time()
            is_sample_tick = (current_time - last_sample_time) >= sampling_interval
            found_faces = False

            if detection_results.detections:
                found_faces = True
                for detection in detection_results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w_frame)
                    y = int(bbox.ymin * h_frame)
                    w = int(bbox.width * w_frame)
                    h = int(bbox.height * h_frame)

                    x, y = max(0, x), max(0, y)
                    w = min(w, w_frame - x)
                    h = min(h, h_frame - y)

                    face_roi = frame[y : y + h, x : x + w]
                    if face_roi.size == 0:
                        continue

                    input_tensor = preprocess_face(face_roi)
                    outputs = session.run(None, {input_name: input_tensor})
                    probs = softmax(outputs[0][0])
                    pred_idx = int(np.argmax(probs))
                    pred_emotion = EMOTIONS[pred_idx]
                    confidence = float(probs[pred_idx] * 100)

                    box_color = PALETTE_BGR.get(pred_emotion, (235, 99, 37))

                    label = f"{pred_emotion}: {confidence:.2f}%"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                    cv2.putText(
                        frame,
                        label,
                        (x, y - 10 if y - 10 > 10 else y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        box_color,
                        2,
                    )

                    if is_sample_tick:
                        timestamp_str = datetime.now().strftime("%H:%M:%S")
                        elapsed_sec = round(elapsed, 2)
                        
                        img_filename = f"capture_{int(time.time() * 1000)}_{pred_emotion}.jpg"
                        img_path = os.path.join(IMAGE_DIR, img_filename)
                        cv2.imwrite(img_path, face_roi)

                        row = {
                            "Date": str(session_date),
                            "Elapsed (s)": elapsed_sec,
                            "Timestamp": timestamp_str,
                            "Dominant Emotion": pred_emotion,
                            "Confidence (%)": round(confidence, 4),
                            "Image Path": img_path,
                        }
                        
                        for idx, emo in enumerate(EMOTIONS):
                            row[emo] = round(float(probs[idx]) * 100, 4)

                        st.session_state.records.append(row)
                        last_sample_time = current_time

            cam_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cam_placeholder.image(cam_display, channels="RGB", use_container_width=True)

            if is_sample_tick and found_faces:
                render_dashboard_visuals(force_plot=True)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        render_dashboard_visuals(force_plot=True)