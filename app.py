"""
Traffic Sign Recognition — Streamlit App
Supports: Custom CNN & MobileNetV2 (GTSRB 43-class)
"""

import streamlit as st
import numpy as np
from PIL import Image
import io
import os
import json
from pathlib import Path

from src.class_mapping import decode_prediction_index

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrafficLens · Sign Recognition",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Class Names ──────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing veh over 3.5 tons", "Right-of-way at intersection",
    "Priority road", "Yield", "Stop", "No vehicles", "Veh > 3.5 tons prohibited",
    "No entry", "General caution", "Dangerous curve left", "Dangerous curve right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End speed + passing limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left", "Keep right",
    "Keep left", "Roundabout mandatory", "End of no passing",
    "End no passing veh > 3.5 tons"
]

CLASS_ICONS = {
    "Speed": "🚗", "No passing": "🚫", "Right-of-way": "⚠️", "Priority": "🔶",
    "Yield": "⚠️", "Stop": "🛑", "No vehicles": "🚫", "No entry": "⛔",
    "General caution": "⚠️", "Dangerous": "⚠️", "Bumpy": "🔺", "Slippery": "🌊",
    "Road work": "🚧", "Traffic signals": "🚦", "Pedestrians": "🚶",
    "Children": "👧", "Bicycles": "🚲", "Wild animals": "🦌",
    "Turn right": "↪️", "Turn left": "↩️", "Ahead only": "⬆️",
    "Go straight": "⬆️", "Keep right": "➡️", "Keep left": "⬅️",
    "Roundabout": "🔄", "End": "✅", "Beware": "🌨️", "Double curve": "〰️",
}

def get_icon(class_name):
    for key, icon in CLASS_ICONS.items():
        if key.lower() in class_name.lower():
            return icon
    return "🚸"

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* Root Variables */
:root {
    --bg: #0A0A0F;
    --surface: #13131A;
    --surface2: #1C1C28;
    --border: #2A2A3D;
    --accent: #FF4545;
    --accent2: #FFB020;
    --accent3: #22D3EE;
    --text: #F0F0F5;
    --muted: #6B6B80;
    --success: #22C55E;
    --font-head: 'Syne', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
}

/* Global Reset */
.stApp { background: var(--bg); color: var(--text); font-family: var(--font-head); }
.main .block-container { padding: 2rem 2.5rem; max-width: 1200px; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div { padding: 1.5rem 1rem; }

/* Hero Header */
.hero {
    background: linear-gradient(135deg, #0A0A0F 0%, #13131A 50%, #1a0a0f 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(255,69,69,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 150px; height: 150px;
    background: radial-gradient(circle, rgba(34,211,238,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-label {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.2em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 800;
    color: var(--text);
    line-height: 1.1;
    margin: 0 0 0.6rem 0;
    letter-spacing: -0.02em;
}
.hero h1 span { color: var(--accent); }
.hero-sub { color: var(--muted); font-size: 1rem; font-family: var(--font-mono); font-weight: 400; }

/* Upload Zone */
.upload-zone {
    border: 2px dashed var(--border);
    border-radius: 14px;
    padding: 3rem 2rem;
    text-align: center;
    background: var(--surface);
    transition: border-color 0.3s;
}
.upload-zone:hover { border-color: var(--accent3); }

/* Result Card */
.result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.8rem;
    position: relative;
    overflow: hidden;
}
.result-card.top-result {
    border-color: var(--accent);
    background: linear-gradient(135deg, var(--surface) 0%, rgba(255,69,69,0.05) 100%);
}
.result-rank {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.result-name {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text);
    margin: 0.2rem 0 0.8rem 0;
    letter-spacing: -0.01em;
}
.result-confidence {
    font-family: var(--font-mono);
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--accent);
}
.result-confidence.high { color: var(--success); }
.result-confidence.mid  { color: var(--accent2); }
.result-confidence.low  { color: var(--accent); }

/* Confidence Bar */
.conf-bar-bg {
    background: var(--border);
    border-radius: 4px;
    height: 6px;
    margin-top: 0.8rem;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--accent3), var(--success));
    transition: width 0.5s ease;
}

/* Top-5 Pills */
.top5-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    font-family: var(--font-head);
}
.top5-label { font-size: 0.9rem; color: var(--text); font-weight: 600; }
.top5-score {
    font-family: var(--font-mono);
    font-size: 0.85rem;
    color: var(--accent3);
    font-weight: 500;
}

/* Stats Pills */
.stat-pill {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
.stat-value { font-family: var(--font-mono); font-size: 1.2rem; font-weight: 700; color: var(--accent3); }
.stat-label { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.12em; }

/* Model badge */
.model-badge {
    display: inline-block;
    background: rgba(34,211,238,0.1);
    border: 1px solid rgba(34,211,238,0.3);
    color: var(--accent3);
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.05em;
}
.model-badge.cnn { background: rgba(255,69,69,0.1); border-color: rgba(255,69,69,0.3); color: var(--accent); }

/* Section headings */
.section-head {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-head::after { content: ''; flex: 1; height: 1px; background: var(--border); }

/* Streamlit overrides */
.stSelectbox > div > div { background: var(--surface2) !important; border-color: var(--border) !important; color: var(--text) !important; font-family: var(--font-head) !important; }
.stFileUploader > div { background: var(--surface) !important; border-color: var(--border) !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }
div[data-testid="stImage"] img { border-radius: 12px; }
.stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--font-head) !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    padding: 0.5rem 1.5rem !important;
}
.stButton > button:hover { background: #cc3333 !important; }

/* Info box */
.info-box {
    background: rgba(34,211,238,0.07);
    border: 1px solid rgba(34,211,238,0.2);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.88rem;
    color: var(--accent3);
    margin-bottom: 1rem;
    font-family: var(--font-mono);
}

/* Category tag */
.cat-tag {
    display: inline-block;
    background: rgba(255,176,32,0.15);
    border: 1px solid rgba(255,176,32,0.3);
    color: var(--accent2);
    border-radius: 6px;
    padding: 0.15rem 0.6rem;
    font-size: 0.72rem;
    font-family: var(--font-mono);
    font-weight: 500;
    margin-top: 0.4rem;
}

/* Divider */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_choice):
    try:
        import tensorflow as tf
        output_dir = Path("outputs")
        if model_choice == "Custom CNN":
            path = output_dir / "custom_cnn_best.keras"
        else:
            path = output_dir / "mobilenet_best.keras"

        if not path.exists():
            return None, f"Model file not found: `{path}`\nPlease train the model first by running `task8_traffic_signs.py`."
        model = tf.keras.models.load_model(str(path))
        return model, None
    except ImportError:
        return None, "TensorFlow not installed. Run: `pip install tensorflow`"
    except Exception as e:
        return None, str(e)


def preprocess_image(img: Image.Image, size=(32, 32)) -> np.ndarray:
    img = img.convert("RGB").resize(size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def get_category(class_name: str) -> str:
    if "Speed limit" in class_name or "End of speed" in class_name or "End speed" in class_name:
        return "Speed Regulation"
    elif "No passing" in class_name or "No vehicles" in class_name or "No entry" in class_name or "prohibited" in class_name:
        return "Prohibition"
    elif "caution" in class_name or "Dangerous" in class_name or "Bumpy" in class_name or "Slippery" in class_name or "narrows" in class_name or "animals" in class_name or "Children" in class_name or "Pedestrians" in class_name or "Bicycles" in class_name or "ice" in class_name or "curve" in class_name or "work" in class_name or "signals" in class_name:
        return "Warning"
    elif "right" in class_name.lower() or "left" in class_name.lower() or "Ahead" in class_name or "straight" in class_name or "Keep" in class_name or "Roundabout" in class_name:
        return "Mandatory Direction"
    elif "Right-of-way" in class_name or "Priority" in class_name or "Yield" in class_name or "Stop" in class_name:
        return "Priority / Right-of-Way"
    elif "End" in class_name:
        return "End of Restriction"
    return "Other"


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='margin-bottom:1.5rem;'>
        <div style='font-family:var(--font-mono);font-size:0.65rem;letter-spacing:0.2em;color:var(--muted);text-transform:uppercase;margin-bottom:0.4rem;'>System</div>
        <div style='font-size:1.3rem;font-weight:800;color:var(--text);font-family:var(--font-head);'>TrafficLens</div>
        <div style='font-family:var(--font-mono);font-size:0.75rem;color:var(--muted);margin-top:0.2rem;'>v1.0 · GTSRB · 43 Classes</div>
    </div>
    <hr style='border-color:#2A2A3D;margin:1rem 0;'>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-head">Model Selection</div>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Architecture",
        ["Custom CNN", "MobileNetV2 (Transfer Learning)"],
        label_visibility="collapsed",
    )

    st.markdown('<hr style="border-color:#2A2A3D;margin:1rem 0;">', unsafe_allow_html=True)
    st.markdown('<div class="section-head">Options</div>', unsafe_allow_html=True)

    top_k = st.slider("Top-K Predictions", min_value=3, max_value=10, value=5)
    show_preview = st.toggle("Show image preview", value=True)
    show_raw     = st.toggle("Show raw probabilities", value=False)

    st.markdown('<hr style="border-color:#2A2A3D;margin:1rem 0;">', unsafe_allow_html=True)

    # Load metrics if available
    metrics_path = Path("outputs/metrics.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        st.markdown('<div class="section-head">Trained Performance</div>', unsafe_allow_html=True)
        key = "custom_cnn" if "CNN" in model_choice else "mobilenet_v2"
        m   = metrics.get(key, {})
        if m:
            st.markdown(f"""
            <div class="stat-pill"><div class="stat-value">{m.get('accuracy',0)*100:.1f}%</div><div class="stat-label">Val Accuracy</div></div>
            <div class="stat-pill"><div class="stat-value">{m.get('loss',0):.4f}</div><div class="stat-label">Val Loss</div></div>
            <div class="stat-pill"><div class="stat-value">{metrics.get('params',{}).get(key,0):,}</div><div class="stat-label">Parameters</div></div>
            """, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#2A2A3D;margin:1rem 0;">', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:var(--font-mono);font-size:0.7rem;color:var(--muted);line-height:1.7;'>
    Dataset: GTSRB<br>
    Classes: 43 traffic signs<br>
    Input: 32×32 RGB<br>
    Framework: TensorFlow/Keras
    </div>
    """, unsafe_allow_html=True)


# ─── Main Content ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-label">🚦 AI · Computer Vision · Real-time Inference</div>
    <h1>Traffic<span>Lens</span></h1>
    <div class="hero-sub">German Traffic Sign Recognition · 43-Class Classifier · GTSRB Benchmark</div>
</div>
""", unsafe_allow_html=True)

# ─── Load Model ───────────────────────────────────────────────────────────────
with st.spinner("Loading model weights..."):
    model, model_error = load_model(model_choice)

if model_error:
    badge_cls = "cnn" if "CNN" in model_choice else ""
    st.markdown(f"""
    <div class="info-box">
    ⚠️ Model not loaded · {model_error}
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="result-card">
    <div class="result-rank">Demo Mode</div>
    <div style="color:var(--muted);font-family:var(--font-mono);font-size:0.88rem;line-height:1.8;">
    To use this app:<br>
    1. Train the model by running <code style="color:var(--accent3);">task8_traffic_signs.py</code><br>
    2. Place trained <code style="color:var(--accent3);">.keras</code> files in the <code style="color:var(--accent3);">outputs/</code> folder<br>
    3. Restart this app<br><br>
    The app will then perform live inference on any uploaded traffic sign image.
    </div>
    </div>
    """, unsafe_allow_html=True)
    model_loaded = False
else:
    model_loaded = True
    badge_cls = "cnn" if "CNN" in model_choice else ""
    st.markdown(f"""
    <div class="info-box" style="background:rgba(34,200,94,0.07);border-color:rgba(34,200,94,0.2);color:#22C55E;">
    ✓ Model loaded · <span class="model-badge {"cnn" if "CNN" in model_choice else ""}">{model_choice}</span>
    </div>
    """, unsafe_allow_html=True)

# ─── Upload & Inference ────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.6], gap="large")

with col_left:
    st.markdown('<div class="section-head">Upload Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop a traffic sign image",
        type=["png", "jpg", "jpeg", "bmp", "ppm"],
        label_visibility="collapsed",
    )

    if uploaded and show_preview:
        img = Image.open(uploaded)
        st.image(img, caption=f"{uploaded.name}  ·  {img.size[0]}×{img.size[1]}px", use_container_width=True)

        st.markdown(f"""
        <div style="font-family:var(--font-mono);font-size:0.75rem;color:var(--muted);margin-top:0.5rem;line-height:1.8;">
        Mode: {img.mode}<br>
        Size: {img.size[0]} × {img.size[1]} px<br>
        File: {uploaded.name}
        </div>
        """, unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-head">Recognition Result</div>', unsafe_allow_html=True)

    if not uploaded:
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size:3rem;margin-bottom:1rem;">🚦</div>
            <div style="color:var(--muted);font-family:var(--font-mono);font-size:0.85rem;line-height:1.8;">
            Upload a traffic sign image<br>
            to begin recognition<br><br>
            <span style="color:var(--border);">Supported: PNG · JPG · BMP · PPM</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif not model_loaded:
        st.markdown("""
        <div class="result-card">
            <div class="result-rank">⚠ Waiting for Model</div>
            <div style="color:var(--muted);font-family:var(--font-mono);font-size:0.88rem;">
            Train and load model weights to run inference.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        img  = Image.open(uploaded)
        inp  = preprocess_image(img)

        with st.spinner("Running inference..."):
            preds = model.predict(inp, verbose=0)[0]

        top_indices = np.argsort(preds)[::-1][:top_k]
        top_probs   = preds[top_indices]

        pred_class = decode_prediction_index(int(top_indices[0]))
        pred_conf  = float(top_probs[0])
        pred_name  = CLASS_NAMES[pred_class]
        pred_icon  = get_icon(pred_name)
        pred_cat   = get_category(pred_name)

        # Confidence color
        if pred_conf >= 0.85:   conf_cls = "high"
        elif pred_conf >= 0.60: conf_cls = "mid"
        else:                   conf_cls = "low"

        bar_w = int(pred_conf * 100)

        st.markdown(f"""
        <div class="result-card top-result">
            <div class="result-rank">Top Prediction · Class #{pred_class:02d}</div>
            <div style="font-size:2.5rem;margin:0.4rem 0;">{pred_icon}</div>
            <div class="result-name">{pred_name}</div>
            <div class="cat-tag">{pred_cat}</div>
            <div style="margin-top:1.2rem;">
                <div style="display:flex;justify-content:space-between;align-items:baseline;">
                    <span style="font-family:var(--font-mono);font-size:0.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.15em;">Confidence</span>
                    <span class="result-confidence {conf_cls}">{pred_conf*100:.1f}%</span>
                </div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{bar_w}%;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Top-K list
        st.markdown(f'<div class="section-head" style="margin-top:1.2rem;">Top {top_k} Candidates</div>', unsafe_allow_html=True)
        for rank, (raw_idx, prob) in enumerate(zip(top_indices, top_probs)):
            cid   = decode_prediction_index(int(raw_idx))
            icon  = get_icon(CLASS_NAMES[cid])
            name  = CLASS_NAMES[cid]
            w     = int(prob * 100)
            alpha = max(0.4, 1.0 - rank * 0.15)
            st.markdown(f"""
            <div class="top5-item" style="opacity:{alpha};">
                <div style="display:flex;align-items:center;gap:0.6rem;">
                    <span style="font-family:var(--font-mono);font-size:0.7rem;color:var(--muted);min-width:1.2rem;">#{rank+1}</span>
                    <span style="font-size:1.1rem;">{icon}</span>
                    <span class="top5-label">{name}</span>
                </div>
                <span class="top5-score">{prob*100:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)

        # Raw probabilities table
        if show_raw:
            st.markdown('<div class="section-head" style="margin-top:1.2rem;">Raw Probabilities</div>', unsafe_allow_html=True)
            import pandas as pd
            raw_df = pd.DataFrame({
                "Class": [CLASS_NAMES[decode_prediction_index(int(i))] for i in top_indices],
                "GTSRB ID": [decode_prediction_index(int(i)) for i in top_indices],
                "Model index": list(top_indices),
                "Probability": [f"{p:.6f}" for p in top_probs],
                "Confidence %": [f"{p*100:.2f}%" for p in top_probs],
            })
            st.dataframe(raw_df, use_container_width=True, hide_index=True)

# ─── Class Reference ──────────────────────────────────────────────────────────
st.markdown('<hr>', unsafe_allow_html=True)
with st.expander("📋 All 43 GTSRB Classes", expanded=False):
    st.markdown('<div class="section-head">Class Reference</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, name in enumerate(CLASS_NAMES):
        icon = get_icon(name)
        cat  = get_category(name)
        with cols[i % 3]:
            st.markdown(f"""
            <div class="top5-item" style="margin-bottom:0.4rem;">
                <div style="display:flex;align-items:center;gap:0.5rem;">
                    <span style="font-family:var(--font-mono);font-size:0.65rem;color:var(--muted);min-width:1.4rem;">{i:02d}</span>
                    <span>{icon}</span>
                    <span style="font-size:0.8rem;font-weight:600;color:var(--text);">{name}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding:1.5rem 0;border-top:1px solid #2A2A3D;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem;">
    <div style="font-family:var(--font-mono);font-size:0.72rem;color:var(--muted);">
        TrafficLens · GTSRB Traffic Sign Recognition · TensorFlow / Keras
    </div>
    <div style="font-family:var(--font-mono);font-size:0.72rem;color:var(--muted);">
        Custom CNN &amp; MobileNetV2 · 43 Classes · 32×32 Input
    </div>
</div>
""", unsafe_allow_html=True)