import os
import platform
import random
import time
from datetime import datetime, timedelta
 
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from models.EfficientFace import efficient_face
from plotly.subplots import make_subplots

##################
# TODO: ONNX Runtime Option (Commented by default)
# If you wish to use ONNX runtime instead of PyTorch checkpoints:
# 1. Uncomment the import below:
#    import onnxruntime as ort
# 2. Set USE_ONNX = True below.
# 3. Uncomment load_onnx_model and the ONNX branch in predict_batch.
##################
# import onnxruntime as ort
 
# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Assignment 3: Face Application",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
st.markdown(
    """
    <style>
        .stButton>button { border-radius: 8px; font-weight: 500; }
        [data-testid="stSidebar"] { border-right: 1px solid #e2e8f0; }
        [data-testid="stMetricValue"] { font-size: 1.5rem; }
        .emo-chip {
            display:inline-block; padding:2px 10px; border-radius:999px;
            font-weight:600; font-size:0.95rem; color:#fff;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
 
##################
# TODO: Checkpoint & Model Settings
# By default, USE_ONNX is set to False to load the latest PyTorch checkpoint (.pth.tar or .ckpt).
# If you want to use the ONNX model instead:
#   1. Set USE_ONNX = True
#   2. Uncomment the ONNX functions below
##################
USE_ONNX = False
CHECKPOINT_DIR = "checkpoint"
ONNX_MODEL_PATH = "models/efficient_face.onnx"
 
# Shown in the sidebar next to the file's own mtime. If a NameError mentions a
# placeholder or a function that isn't in the source any more, the running file is
# an older copy — check this stamp against the one you expect before debugging.
APP_BUILD = "2026-10-19c · two-column layout + CSV export"
 
# Label order must match the EfficientFace / AffectNet-7 head.
EMOTIONS = [
    "0_Neutral",
    "1_Happiness",
    "2_Sadness",
    "3_Surprise",
    "4_Fear",
    "5_Disgust",
    "6_Anger",
]
 
# ---------------------------------------------------------
# Palette
#
# Six categorical hues + "ink" for Neutral (the de-emphasis/baseline class).
# Validated with the data-viz palette checker on both surfaces:
#   light  #fcfcfb -> lightness band PASS, chroma PASS, adjacent CVD 6.9 (warn band),
#                     normal-vision 27.5 PASS, 3 slots < 3:1 contrast (warn)
#   dark   #1a1a19 -> all checks PASS, adjacent CVD 6.5 (warn band)
# The CVD/contrast warnings are discharged by secondary encoding, which this
# dashboard ships: every highlighted line carries a direct label + its own marker
# symbol, and the full numeric table is always on the page.
#
# The original palette (slate/amber/blue/pink/violet/emerald/red) FAILED the
# chroma floor (#64748b read as gray) and, with all 7 drawn at once, failed CVD
# separation (violet vs blue deutan dE 1.3) and the normal-vision floor
# (red vs pink dE 11.4). Hence the emphasis form below: at most 3 hues at a time.
# ---------------------------------------------------------
THEMES = {
    "Light": dict(
        surface="#fcfcfb", plane="#f9f9f7", ink="#0b0b0b", ink2="#52514e",
        muted="#898781", grid="#e1e0d9", axis="#c3c2b7", context="#c3c2b7",
        template="plotly_white",
    ),
    "Dark": dict(
        surface="#1a1a19", plane="#0d0d0d", ink="#f0efec", ink2="#c3c2b7",
        muted="#898781", grid="#2c2c2a", axis="#383835", context="#4a4a46",
        template="plotly_dark",
    ),
}
 
SERIES_COLORS = {
    "Light": {
        "0_Neutral": "#0b0b0b",
        "1_Happiness": "#eda100",
        "2_Sadness": "#2a78d6",
        "3_Surprise": "#e87ba4",
        "4_Fear": "#6250d6",
        "5_Disgust": "#1baf7a",
        "6_Anger": "#e34948",
    },
    "Dark": {
        "0_Neutral": "#f0efec",
        "1_Happiness": "#c98500",
        "2_Sadness": "#3987e5",
        "3_Surprise": "#d55181",
        "4_Fear": "#9085e9",
        "5_Disgust": "#199e70",
        "6_Anger": "#e66767",
    },
}
 
# Composite encoding: hue x shape, so identity survives colour-vision deficiency.
SERIES_SYMBOL = {
    "0_Neutral": "circle",
    "1_Happiness": "star",
    "2_Sadness": "square",
    "3_Surprise": "diamond",
    "4_Fear": "triangle-up",
    "5_Disgust": "cross",
    "6_Anger": "x",
}
 
SHORT_NAME = {e: e.split("_", 1)[1] for e in EMOTIONS}
 
 
def hex_to_bgr(hex_str: str):
    hex_str = hex_str.lstrip("#")
    r, g, b = (int(hex_str[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)
 
 
# ---------------------------------------------------------
# Cached resources
# ---------------------------------------------------------
##################
# TODO: ONNX Model Loader (Commented by default)
# Uncomment the block below if you wish to run inference with ONNX Runtime:
##################
# @st.cache_resource(show_spinner="Loading ONNX expression model…")
# def load_onnx_model(model_path: str):
#     opts = ort.SessionOptions()
#     opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
#     opts.intra_op_num_threads = max(1, (os.cpu_count() or 2))
#     opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
#     available = ort.get_available_providers()
#     preferred = [p for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider") if p in available]
#     providers = preferred + ["CPUExecutionProvider"]
#     sess = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
#     return sess, sess.get_inputs()[0].name, sess.get_providers()[0]


def get_latest_checkpoint(checkpoint_dir: str = "checkpoint") -> str:
    """Find the most recently modified checkpoint (.pth.tar, .ckpt, .pth) in directory."""
    if not os.path.isdir(checkpoint_dir):
        return None
    valid_exts = (".pth.tar", ".ckpt", ".pth", ".tar")
    candidates = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if any(f.endswith(ext) for ext in valid_exts) and not f.startswith(".")
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


@st.cache_resource(show_spinner="Loading PyTorch expression model from checkpoint…")
def load_pytorch_model(checkpoint_path: str):
    """
    Load EfficientFace model architecture and restore weights from checkpoint.
    Supports checkpoints saved from main.py, PyTorch DataParallel, and PyTorch Lightning (.ckpt).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = efficient_face(num_classes=len(EMOTIONS))

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Strip DataParallel ('module.') or PyTorch Lightning ('model.') prefixes
    cleaned_dict = {}
    for k, v in state_dict.items():
        clean_k = k
        while clean_k.startswith("module.") or clean_k.startswith("model."):
            if clean_k.startswith("module."):
                clean_k = clean_k[7:]
            elif clean_k.startswith("model."):
                clean_k = clean_k[6:]
        cleaned_dict[clean_k] = v

    model.load_state_dict(cleaned_dict, strict=False)
    model.to(device)
    model.eval()
    return model, device


@st.cache_resource
def load_face_detector(min_confidence: float = 0.5):
    return mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=min_confidence
    )


face_detector = load_face_detector()

if USE_ONNX:
    ##################
    # TODO: Load ONNX Model (Commented by default)
    # Uncomment lines below if using ONNX runtime:
    ##################
    # if not os.path.exists(ONNX_MODEL_PATH):
    #     st.error(f"ONNX model not found at `{ONNX_MODEL_PATH}`. Place `efficient_face.onnx` there and reload.")
    #     st.stop()
    # session, input_name, active_provider = load_onnx_model(ONNX_MODEL_PATH)
    st.error("USE_ONNX is True but ONNX code is commented. Please uncomment the ONNX loader code to proceed.")
    st.stop()
else:
    latest_checkpoint = get_latest_checkpoint(CHECKPOINT_DIR)
    if not latest_checkpoint:
        st.error(
            f"No checkpoint found in `{CHECKPOINT_DIR}`. "
            "Please train the model using `train.py` or place a `.pth.tar` / `.ckpt` file in the checkpoint directory."
        )
        st.stop()
    model, device = load_pytorch_model(latest_checkpoint)
    active_provider = f"PyTorch ({device}) · {os.path.basename(latest_checkpoint)}" 
 
# ---------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------
# Fold the mean/std normalisation into one multiply-add so preprocessing is a
# single pass over the pixels instead of divide -> subtract -> divide.
_MEAN = np.array([0.57535914, 0.44928582, 0.40079932], dtype=np.float32)
_STD = np.array([0.20735591, 0.18981615, 0.18132027], dtype=np.float32)
_SCALE = (1.0 / (255.0 * _STD)).astype(np.float32)
_SHIFT = (-_MEAN / _STD).astype(np.float32)
INPUT_SIZE = 224
 
 
def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)
 
 
def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    """BGR crop -> normalised CHW float32. cv2 only; no PIL round-trip."""
    interp = cv2.INTER_AREA if face_bgr.shape[0] > INPUT_SIZE else cv2.INTER_LINEAR
    resized = cv2.resize(face_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=interp)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32, copy=False) * _SCALE + _SHIFT
    return np.ascontiguousarray(arr.transpose(2, 0, 1))
 
 
def predict_batch(crops) -> np.ndarray:
    """Predict emotion probabilities for a batch of cropped face images."""
    batch = np.stack([preprocess_face(c) for c in crops], axis=0)

    if USE_ONNX:
        ##################
        # TODO: ONNX Batch Inference (Commented by default)
        # Uncomment lines below if using ONNX runtime:
        ##################
        # logits = session.run(None, {input_name: batch})[0]
        # return softmax(np.asarray(logits, dtype=np.float32))
        pass

    # PyTorch Checkpoint Inference (Default)
    tensor_batch = torch.from_numpy(batch).to(device)
    with torch.no_grad():
        logits = model(tensor_batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs
 
 
def expand_box(x, y, w, h, frame_w, frame_h, margin=0.15):
    """EfficientFace was trained on loosely cropped faces; add a little context."""
    mx, my = int(w * margin), int(h * margin)
    x0 = max(0, x - mx)
    y0 = max(0, y - my)
    x1 = min(frame_w, x + w + mx)
    y1 = min(frame_h, y + h + my)
    return x0, y0, x1 - x0, y1 - y0
 
 
def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    return inter / float(aw * ah + bw * bh - inter)
 
 
def detect_faces(frame_bgr: np.ndarray, detect_width: int):
    """Run MediaPipe on a downscaled frame and map boxes back to full resolution."""
    h, w = frame_bgr.shape[:2]
    if w > detect_width:
        scale = detect_width / float(w)
        small = cv2.resize(frame_bgr, (detect_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = frame_bgr
 
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = face_detector.process(rgb)
    if not results.detections:
        return []
 
    boxes = []
    for det in results.detections:
        rb = det.location_data.relative_bounding_box
        x = int(rb.xmin * w)
        y = int(rb.ymin * h)
        bw = int(rb.width * w)
        bh = int(rb.height * h)
        # Clamp while preserving the far edge (the original clamped x then reused w).
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + bw), min(h, y + bh)
        if x1 - x0 > 20 and y1 - y0 > 20:
            boxes.append((x0, y0, x1 - x0, y1 - y0))
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)  # largest face first
    return boxes
 
 
# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------
_defaults = {
    "tracking_active": False,
    "records": [],
    "session_start_time": None,
    "clock_offset": 0.0,
    "render_seq": 0,
    "fps": 0.0,
    "latest": None,
    "last_export": None,
    # carried across runs, since a widget change restarts the script mid-capture
    "last_infer": 0.0,
    "last_sample": 0.0,
    "cached": [],
}
for key, value in _defaults.items():
    st.session_state.setdefault(key, value)
 
 
# ---------------------------------------------------------
# Export
# ---------------------------------------------------------
EXPORT_DIR = "exports"
 
 
def records_csv() -> bytes:
    return pd.DataFrame(st.session_state.records).to_csv(index=False).encode("utf-8")
 
 
def autosave_csv():
    """Write the log to disk whenever a session ends.
 
    The browser download is the primary route, but it only exists while the tab is
    open — an autosave means a session is never lost to a closed tab or a crash.
    """
    if not st.session_state.records:
        return None
    os.makedirs(EXPORT_DIR, exist_ok=True)
    base = os.path.join(EXPORT_DIR, f"emotion_log_{datetime.now():%Y%m%d_%H%M%S}")
    path, i = f"{base}.csv", 1
    while os.path.exists(path):      # stop/start inside the same second
        path, i = f"{base}_{i}.csv", i + 1
    pd.DataFrame(st.session_state.records).to_csv(path, index=False)
    st.session_state.last_export = os.path.abspath(path)
    return path
 
 
# ---------------------------------------------------------
# Camera (held open across reruns — reopening it each burst would cost about a second)
# ---------------------------------------------------------
def open_camera(index: int):
    backend = {
        "Darwin": cv2.CAP_AVFOUNDATION,
        "Windows": cv2.CAP_DSHOW,
        "Linux": cv2.CAP_V4L2,
    }.get(platform.system(), cv2.CAP_ANY)
    cap = cv2.VideoCapture(int(index), backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(int(index))
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always grab the newest frame
    return cap
 
 
def release_camera():
    cap = st.session_state.pop("cap", None)
    if cap is not None:
        cap.release()
    st.session_state["cap_index"] = None
 
 
def get_camera(index: int):
    cap = st.session_state.get("cap")
    if cap is not None and st.session_state.get("cap_index") == int(index) and cap.isOpened():
        return cap
    release_camera()
    cap = open_camera(index)
    st.session_state["cap"] = cap
    st.session_state["cap_index"] = int(index)
    return cap
 
# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
 
    st.markdown("### Controls & settings")
    session_date = st.date_input("Date", datetime.now().date())
    duration_minutes = st.number_input(
        "Max session duration (minutes)", min_value=1.0, max_value=120.0, value=10.0, step=1.0
    )
    camera_index = st.number_input("Camera device index", min_value=0, max_value=8, value=0, step=1)
    sampling_interval = st.select_slider(
        "Sampling interval (s)", options=[0.5, 1.0, 2.0, 5.0], value=1.0,
        help="How often a row is written to the log.",
    )
 
    with st.expander("Performance", expanded=False):
        infer_hz = st.slider(
            "Inference rate (Hz)", 1, 30, 8,
            help="Expression model runs this often; in between, the last label is reused "
                 "for the overlay. The single biggest speed lever.",
        )
        detect_width = st.select_slider(
            "Detection resolution (px wide)", options=[320, 480, 640, 960], value=480,
            help="MediaPipe runs on a downscaled frame; boxes are mapped back up.",
        )
        preview_width = st.select_slider(
            "Preview width (px)", options=[480, 640, 800, 1280], value=640
        )
        jpeg_quality = st.slider("Preview JPEG quality", 40, 95, 75)
        refresh_seconds = st.select_slider(
            "Chart refresh (s)", options=[0.5, 1.0, 2.0, 5.0], value=1.0,
            help="How often the charts redraw while capturing. The camera preview "
                 "always runs at full rate; raising this leaves more CPU for capture.",
        )
        log_all_faces = st.checkbox(
            "Log every detected face", value=False,
            help="Off: only the largest face is logged, so one row per sample tick.",
        )
        max_records = st.number_input(
            "Max rows kept in memory", min_value=500, max_value=100_000, value=20_000, step=500
        )
 
    st.markdown("---")
    st.markdown("### Chart")
    theme_name = st.radio("Theme", ["Light", "Dark"], horizontal=True, index=0)
    window_label = st.radio(
        "Main chart window", ["1 min", "5 min", "10 min", "Session"], horizontal=True, index=2,
        help="Only the large chart follows this. The 1- and 5-minute panels below "
             "it are fixed, so the three together read as 1 / 5 / 10 minutes.",
    )
    highlight = st.multiselect(
        "Highlight (max 3)", EMOTIONS, default=[], format_func=lambda e: SHORT_NAME[e],
        help="Leave empty to auto-highlight the three strongest emotions in the window. "
             "Unhighlighted series stay in context grey.",
    )
    if len(highlight) > 3:
        st.warning("Showing the first 3 — more than three hues at once stops being readable.")
        highlight = highlight[:3]
    smooth_n = st.slider(
        "Smoothing (samples)", 1, 15, 5,
        help="Rolling mean over the probability series. 1 = raw. Frame-to-frame "
             "expression probabilities are noisy; a short mean makes the trend readable.",
    )
 
    st.markdown("---")
    c1, c2 = st.columns(2)
    start_clicked = c1.button(
        "▶ Start", type="primary", disabled=st.session_state.tracking_active,
        use_container_width=True, key="btn_start",
    )
    stop_clicked = c2.button(
        "⏹ Stop", disabled=not st.session_state.tracking_active,
        use_container_width=True, key="btn_stop",
    )
    reset_clicked = st.button("🗑 Reset session data", use_container_width=True, key="btn_reset")
 
    st.markdown("---")
    st.markdown("### Export")
    # Filled in after the button handling below, so it reflects this click.
    export_slot = st.container()
 
    try:
        _saved = datetime.fromtimestamp(os.path.getmtime(__file__)).strftime("%d %b %H:%M")
    except OSError:
        _saved = "unknown"
    st.caption(f"Execution provider: `{active_provider}`  \nBuild: {APP_BUILD}  \nFile saved: {_saved}")
 
if reset_clicked:
    st.session_state.records = []
    st.session_state.clock_offset = 0.0
    st.session_state.latest = None
    st.session_state.cached = []
    st.session_state.last_export = None
    st.session_state.tracking_active = False
    release_camera()
    st.rerun()

if stop_clicked:
    st.session_state.tracking_active = False
    st.session_state.cached = []
    release_camera()
    autosave_csv()
    st.rerun()

if start_clicked:
    last_elapsed = st.session_state.records[-1]["Elapsed (s)"] if st.session_state.records else 0.0
    st.session_state.clock_offset = float(last_elapsed)
    st.session_state.session_start_time = time.time()
    st.session_state.last_infer = 0.0
    st.session_state.last_sample = 0.0
    st.session_state.cached = []
    st.session_state.tracking_active = True
    st.rerun()
 
with export_slot:
    _n_rows = len(st.session_state.records)
    if st.session_state.tracking_active:
        # The button's payload is fixed when the element is created, and the sidebar
        # is built before the capture loop runs — so a button offered mid-session
        # would quietly hand over a snapshot from before the session started.
        st.button("⬇ Download CSV", disabled=True, use_container_width=True,
                  key="btn_export_disabled")
        st.caption(f"{_n_rows:,} rows logged · press **Stop** to export.")
    elif _n_rows:
        st.download_button(
            f"⬇ Download CSV ({_n_rows:,} rows)", data=records_csv(),
            file_name=f"emotion_log_{session_date}.csv", mime="text/csv",
            use_container_width=True, key="btn_export",
        )
    else:
        st.button("⬇ Download CSV", disabled=True, use_container_width=True,
                  key="btn_export_disabled")
        st.caption("Nothing logged yet.")
    if st.session_state.last_export:
        st.caption(f"Auto-saved to `{st.session_state.last_export}`")
 
THEME = THEMES[theme_name]
COLORS = SERIES_COLORS[theme_name]
COLORS_BGR = {e: hex_to_bgr(c) for e, c in COLORS.items()}
 
WINDOW_SECONDS = {"1 min": 60.0, "5 min": 300.0, "10 min": 600.0, "Session": None}[window_label]
 
 
# ---------------------------------------------------------
# Layout
# ---------------------------------------------------------
st.title("Face Application Analytics Dashboard")
 
status_placeholder = st.empty()
# st.empty() not st.container(): re-entering a container APPENDS, so the KPI row
# would stack up a fresh copy on every redraw.
kpi_placeholder = st.empty()
 
# Two columns: the camera and the per-class views on the left, every time series
# on the right. Keeping all three line charts in one column lets the eye compare
# them on a shared vertical axis instead of jumping across the page.
col_left, col_right = st.columns([1.0, 1.2], gap="medium")
 
cam_placeholder = col_left.empty()
facets_placeholder = col_left.empty()
dist_placeholder = col_left.empty()
 
main_chart_placeholder = col_right.empty()
chart_1m_placeholder = col_right.empty()
chart_5m_placeholder = col_right.empty()
 
 
# ---------------------------------------------------------
# Chart builders
# ---------------------------------------------------------
def window_slice(df: pd.DataFrame, seconds) -> pd.DataFrame:
    """Slice to the visible window BEFORE plotting.
 
    The original set range_x but still handed Plotly the whole history, so every
    redraw shipped the full session (3 charts x 7 series x N points) to the browser.
    """
    if df.empty or seconds is None:
        return df
    cutoff = df["Elapsed (s)"].iloc[-1] - seconds
    return df[df["Elapsed (s)"] > cutoff]  # half-open, so "1 min" is exactly 60 s of samples
 
 
def smooth_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean on the probability columns only.
 
    Applied to the WHOLE frame before slicing, so the left edge of the 1-minute
    panel is smoothed against the samples just before it rather than restarting.
    """
    if smooth_n <= 1 or df.empty:
        return df
    out = df.copy()
    out[EMOTIONS] = out[EMOTIONS].rolling(int(smooth_n), min_periods=1).mean()
    return out
 
 
def pick_highlight(df_win: pd.DataFrame):
    if highlight:
        return list(highlight)
    if df_win.empty:
        return [EMOTIONS[0]]
    return list(df_win[EMOTIONS].mean().sort_values(ascending=False).head(3).index)
 
 
def base_layout(fig, height, title=None, legend=False):
    fig.update_layout(
        template=THEME["template"],
        paper_bgcolor=THEME["surface"],
        plot_bgcolor=THEME["surface"],
        margin=dict(l=12, r=72, t=38 if title else 12, b=12),
        height=height,
        font=dict(color=THEME["ink2"], family='system-ui, -apple-system, "Segoe UI", sans-serif'),
        title=dict(text=title, font=dict(size=14, color=THEME["ink"]), x=0, xanchor="left")
        if title else None,
        showlegend=legend,
        hovermode="x unified" if legend else "closest",
        uirevision="keep",  # keeps zoom/pan across redraws
    )
    fig.update_xaxes(gridcolor=THEME["grid"], zeroline=False, linecolor=THEME["axis"],
                     tickfont=dict(color=THEME["muted"], size=11))
    fig.update_yaxes(gridcolor=THEME["grid"], zeroline=False, linecolor=THEME["axis"],
                     tickfont=dict(color=THEME["muted"], size=11))
    return fig
 
 
def emphasis_figure(df_win: pd.DataFrame, focus, title: str, height: int = 360):
    """Trend over time, emphasis form: focus emotions in hue, the rest in context grey.
 
    Seven simultaneous hues failed the colour-vision separation gate; one to three
    hues plus grey context is both faster to draw and readable by everyone.
 
    `focus` is passed in rather than recomputed per panel so the main chart and the
    1/5-minute panels highlight the SAME emotions in the same hues — colour follows
    the entity, never the panel's own ranking.
    """
    fig = go.Figure()
    x = df_win["Elapsed (s)"].to_numpy()
 
    for emo in EMOTIONS:
        if emo in focus:
            continue
        fig.add_trace(go.Scattergl(
            x=x, y=df_win[emo].to_numpy(), name=SHORT_NAME[emo], mode="lines",
            line=dict(color=THEME["context"], width=1),
            opacity=0.55, hoverinfo="skip", showlegend=False,
        ))
 
    # Waypoint markers, not a marker on every sample: the symbol is there to carry
    # identity for colour-vision-deficient readers, and 300 glyphs bury the line.
    n = len(x)
    step = max(1, n // (12 if height >= 320 else 6))
    wp = np.arange(0, n, step)
 
    ends = []
    for emo in focus:
        y = df_win[emo].to_numpy()
        fig.add_trace(go.Scattergl(
            x=x, y=y, name=SHORT_NAME[emo], mode="lines",
            line=dict(color=COLORS[emo], width=2),
            hovertemplate="%{y:.1f}%<extra>" + SHORT_NAME[emo] + "</extra>",
        ))
        fig.add_trace(go.Scattergl(
            x=x[wp], y=y[wp], mode="markers", showlegend=False, hoverinfo="skip",
            marker=dict(symbol=SERIES_SYMBOL[emo], size=9, color=COLORS[emo],
                        line=dict(width=2, color=THEME["surface"])),
        ))
        if n:
            ends.append((emo, float(y[-1])))
 
    # Direct labels at the line ends, nudged apart when two series finish together.
    # The nudge is in pixels, so convert via this panel's own px-per-percent.
    px_per_unit = max(1e-6, (height - 50) / 105.0)
    annotations, placed = [], []
    for emo, yv in sorted(ends, key=lambda t: -t[1]):
        shift = 0
        for prev_y, prev_shift in placed:
            if abs((prev_y + prev_shift / px_per_unit) - (yv + shift / px_per_unit)) < 6:
                shift = prev_shift - 15
        placed.append((yv, shift))
        annotations.append(dict(
            x=x[-1], y=yv, xref="x", yref="y", text=f" {SHORT_NAME[emo]}", yshift=shift,
            showarrow=False, xanchor="left", align="left",
            font=dict(color=COLORS[emo], size=12),
        ))
 
    base_layout(fig, height, title=title, legend=False)
    fig.update_layout(annotations=annotations)
    fig.update_yaxes(range=[0, 105], ticksuffix="%")
    fig.update_xaxes(title=None, ticksuffix="s")
    return fig
 
 
def small_multiples_figure(df_win: pd.DataFrame, rows: int = 4, cols: int = 2,
                           height: int = 420):
    """All seven classes, one hue, faceted — the readable way to show every class.
 
    4x2 rather than 2x4: this now sits in the narrower left column, where two
    facets per row keeps each one wide enough for its shape to be readable.
    """
    fig = make_subplots(
        rows=rows, cols=cols, shared_yaxes=True,
        subplot_titles=[SHORT_NAME[e] for e in EMOTIONS] + [""] * (rows * cols - len(EMOTIONS)),
        vertical_spacing=0.09, horizontal_spacing=0.06,
    )
    x = df_win["Elapsed (s)"].to_numpy()
    for i, emo in enumerate(EMOTIONS):
        r, c = divmod(i, cols)
        fig.add_trace(
            go.Scattergl(x=x, y=df_win[emo].to_numpy(), mode="lines",
                         line=dict(color=COLORS["2_Sadness"], width=1.5),
                         fill="tozeroy", fillcolor="rgba(42,120,214,0.12)",
                         hovertemplate="%{y:.1f}%<extra></extra>", showlegend=False),
            row=r + 1, col=c + 1,
        )
    base_layout(fig, height, title="Every class, same scale")
    fig.update_yaxes(range=[0, 105], ticksuffix="%")
    fig.update_xaxes(showticklabels=False)
    for ann in fig.layout.annotations:
        ann.font.size = 11
        ann.font.color = THEME["ink2"]
    return fig
 
 
def distribution_figure(df_win: pd.DataFrame):
    """Share of samples by dominant class — a magnitude job, so one sequential hue."""
    counts = df_win["Dominant Emotion"].value_counts()
    share = (counts / counts.sum() * 100).sort_values()
    ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#184f95"]
    vals = share.to_numpy()
    vmax = vals.max() if len(vals) else 1.0
    fill = [ramp[min(len(ramp) - 1, int(round(v / vmax * (len(ramp) - 1))))] for v in vals]
 
    fig = go.Figure(go.Bar(
        x=vals, y=[SHORT_NAME.get(i, i) for i in share.index], orientation="h",
        marker=dict(color=fill, line=dict(width=2, color=THEME["surface"])),
        text=[f"{v:.0f}%" for v in vals], textposition="outside",
        textfont=dict(color=THEME["ink2"], size=11),
        hovertemplate="%{y}: %{x:.1f}% of samples<extra></extra>",
    ))
    base_layout(fig, 260, title="Dominant-class share")
    fig.update_xaxes(range=[0, 108], showticklabels=False, showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig
 
 
# ---------------------------------------------------------
# Rendering
# ---------------------------------------------------------
def render_kpis(df: pd.DataFrame, elapsed: float):
    latest = st.session_state.latest
    with kpi_placeholder.container():
        k1, k2, k3, k4, k5 = st.columns(5)
        if latest:
            k1.markdown(
                f'**Current**<br><span class="emo-chip" style="background:{COLORS[latest["emotion"]]}">'
                f'{SHORT_NAME[latest["emotion"]]}</span>',
                unsafe_allow_html=True,
            )
            k2.metric("Confidence", f"{latest['confidence']:.0f}%")
        else:
            k1.markdown("**Current**<br>—", unsafe_allow_html=True)
            k2.metric("Confidence", "—")
        k3.metric("Elapsed", str(timedelta(seconds=int(elapsed))))
        k4.metric("Samples", f"{len(df):,}")
        k5.metric("Capture FPS", f"{st.session_state.fps:.1f}")
 
 
# Streamlit registers every element id once per script run, and the capture loop
# below is a SINGLE long run that redraws many times. So each redraw inside the
# loop gets its own suffix. Deterministic and ordered, unlike the original's
# random.randint keys — and outside the loop (suffix "") the keys are stable, so
# the idle dashboard keeps its zoom.
def _next_suffix() -> str:
    st.session_state.render_seq += 1
    return f"_{st.session_state.render_seq}"
 
 
def render_charts(df: pd.DataFrame, with_detail: bool):
    if df.empty:
        main_chart_placeholder.info("Press **Start** to begin capturing.")
        return

    full = smooth_frame(df)
    main_win = window_slice(full, WINDOW_SECONDS)
    focus = pick_highlight(main_win)

    # ตัด parameter key ทิ้งทั้งหมด ปล่อยให้ placeholder จัดการ DOM slot เอง
    main_chart_placeholder.plotly_chart(
        emphasis_figure(main_win, focus, f"Expression probability — {window_label.lower()}"),
        use_container_width=True,
    )
    chart_1m_placeholder.plotly_chart(
        emphasis_figure(window_slice(full, 60.0), focus, "Last 1 minute", height=260),
        use_container_width=True,
    )
    chart_5m_placeholder.plotly_chart(
        emphasis_figure(window_slice(full, 300.0), focus, "Last 5 minutes", height=260),
        use_container_width=True,
    )

    if with_detail:
        facets_placeholder.plotly_chart(
            small_multiples_figure(main_win),
            use_container_width=True,
        )
        dist_placeholder.plotly_chart(
            distribution_figure(main_win),
            use_container_width=True,
        )
def render_all(df: pd.DataFrame, elapsed: float, with_detail: bool = True):
    render_kpis(df, elapsed)
    render_charts(df, with_detail)
 
 
records_df = pd.DataFrame(st.session_state.records)
if not st.session_state.tracking_active:
    render_all(records_df, st.session_state.clock_offset, with_detail=True)
 
# ---------------------------------------------------------
# Capture
#
# ONE long-running script run that streams frames into the placeholders. Streamlit
# pushes each placeholder write to the browser as it happens, so the preview stays
# live and the dashboard stays on screen for the whole session.
#
# (An earlier attempt split this into ~1 s bursts each ending in st.rerun(). That
# fixed the duplicate-key crash but broke the thing that matters: st.rerun() tears
# down the element tree, so the dashboard blanked on every cycle and the preview
# froze on the last frame before each teardown. The keys are what had to change,
# not the loop — hence _next_suffix() above.)
# ---------------------------------------------------------
if st.session_state.tracking_active:
    cap = get_camera(camera_index)
    if not cap.isOpened():
        st.session_state.tracking_active = False
        release_camera()
        status_placeholder.error(f"Could not open camera index {camera_index}.")
        render_all(records_df, st.session_state.clock_offset, with_detail=True)
        st.stop()
 
    max_duration = duration_minutes * 60.0
    infer_interval = 1.0 / float(infer_hz)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
 
    cached = st.session_state.cached
    loop_start = time.time()
    frames = 0
    finished_msg = None
    last_status = 0.0
    last_chart = 0.0
    last_detail = 0.0
    last_plotted_n = -1
 
    while st.session_state.tracking_active:
        now = time.time()
        elapsed = st.session_state.clock_offset + (now - st.session_state.session_start_time)

        time.sleep(0.001)
        if (now - st.session_state.session_start_time) >= max_duration:
            st.session_state.tracking_active = False
            finished_msg = ("warning", f"Session finished: reached the {duration_minutes:g} minute limit.")
            break
 
        ok, frame = cap.read()
        if not ok:
            st.session_state.tracking_active = False
            finished_msg = ("error", "Lost the camera feed. Check the device index.")
            break
 
        frames += 1
        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]
 
        boxes = detect_faces(frame, detect_width)
 
        if boxes and (now - st.session_state.last_infer) >= infer_interval:
            crops, kept = [], []
            for box in boxes:
                ex, ey, ew, eh = expand_box(*box, fw, fh)
                crop = frame[ey:ey + eh, ex:ex + ew]
                if crop.size:
                    crops.append(crop)
                    kept.append(box)
            if crops:
                probs_batch = predict_batch(crops)      # one batched ONNX call
                cached = list(zip(kept, probs_batch))
                st.session_state.last_infer = now
        elif cached and boxes:
            # Reuse the last prediction for the nearest box so the overlay stays
            # glued to the face between inference ticks.
            refreshed = []
            for box in boxes:
                best = max(cached, key=lambda c: iou(box, c[0]), default=None)
                if best and iou(box, best[0]) > 0.3:
                    refreshed.append((box, best[1]))
            cached = refreshed
        elif not boxes:
            cached = []
 
        # Persist every frame, not just at the end: pressing Stop interrupts this
        # loop with a RerunException, so anything after it never executes.
        st.session_state.cached = cached
 
        is_sample_tick = (now - st.session_state.last_sample) >= sampling_interval
 
        for rank, (box, probs) in enumerate(cached):
            x, y, w, h = box
            idx = int(np.argmax(probs))
            emo = EMOTIONS[idx]
            conf = float(probs[idx] * 100.0)
            color = COLORS_BGR[emo]
 
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{SHORT_NAME[emo]} {conf:.0f}%"
            ty = y - 8 if y - 8 > 14 else y + h + 20
            cv2.putText(frame, label, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                        cv2.LINE_AA)
 
            if rank == 0:
                st.session_state.latest = {"emotion": emo, "confidence": conf}
 
            if is_sample_tick and (log_all_faces or rank == 0):
                ##################
                # TODO: Student Contribution - Custom Image Capture for Fine-Tuning
                # Uncomment the block below to save cropped face images into ./captured_images/
                # during webcam monitoring. You can use these captured images to fine-tune your model in train.py!
                #
                # img_dir = "./captured_images"
                # os.makedirs(img_dir, exist_ok=True)
                # crop_face = frame[max(0, y):min(fh, y + h), max(0, x):min(fw, x + w)]
                # if crop_face.size > 0:
                #     img_filename = f"capture_{int(time.time() * 1000)}_{emo}.jpg"
                #     cv2.imwrite(os.path.join(img_dir, img_filename), crop_face)
                ##################
                row = {
                    "Date": str(session_date),
                    "Elapsed (s)": round(elapsed, 2),
                    "Timestamp": datetime.now().strftime("%H:%M:%S"),
                    "Face": rank,
                    "Dominant Emotion": emo,
                    "Confidence (%)": round(conf, 2),
                }
                row.update({e: round(float(probs[i]) * 100.0, 2) for i, e in enumerate(EMOTIONS)})
                st.session_state.records.append(row)
 
        if is_sample_tick:
            st.session_state.last_sample = now
            if len(st.session_state.records) > max_records:
                st.session_state.records = st.session_state.records[-int(max_records):]
 
        # Preview: downscale + JPEG-encode ourselves. st.image on a raw 720p array
        # PNG-encodes it every frame, which cost more than the model did.
        if fw > preview_width:
            scale = preview_width / float(fw)
            preview = cv2.resize(frame, (preview_width, int(fh * scale)),
                                 interpolation=cv2.INTER_AREA)
        else:
            preview = frame
        ok_jpg, buf = cv2.imencode(".jpg", preview, encode_params)
        if ok_jpg:
            cam_placeholder.image(buf.tobytes(), use_container_width=True)
 
        # ---- throttled dashboard updates, still inside the one long run ----
        if now - last_status >= 1.0:
            st.session_state.fps = frames / max(1e-6, now - loop_start)
            remaining = max(0, int(max_duration - (now - st.session_state.session_start_time)))
            status_placeholder.info(
                f"● **Recording** · elapsed {timedelta(seconds=int(elapsed))} · "
                f"remaining {timedelta(seconds=remaining)} · "
                f"{st.session_state.fps:.1f} fps · faces {len(cached)}"
            )
            last_status = now
 
        n_records = len(st.session_state.records)
        if n_records != last_plotted_n and (now - last_chart) >= float(refresh_seconds):
            df_live = pd.DataFrame(st.session_state.records)
            # The facets + distribution are the expensive half; slower cadence.
            with_detail = (now - last_detail) >= 5.0
            sfx = _next_suffix()
            render_kpis(df_live, elapsed)
            render_charts(df_live, with_detail)
            if with_detail:
                last_detail = now
            last_chart = now
            last_plotted_n = n_records
 
    # ---- the loop ended on its own (duration limit or a dead camera) ----
    st.session_state.cached = cached
    release_camera()
    saved_to = autosave_csv()
    if finished_msg:
        getattr(status_placeholder, finished_msg[0])(finished_msg[1])
    if saved_to:
        st.toast(f"Saved {len(st.session_state.records):,} rows to {saved_to}")
    final_df = pd.DataFrame(st.session_state.records)
    render_all(final_df, st.session_state.clock_offset, with_detail=True)