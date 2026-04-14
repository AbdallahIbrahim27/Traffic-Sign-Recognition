"""
Traffic Sign Recognition — Streamlit App
Supports: Custom CNN ONLY (GTSRB 43-class)
"""

import streamlit as st
import numpy as np
from PIL import Image
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

# ─── SINGLE definition of get_icon ────────────────────────────────────────────
def get_icon(class_name: str) -> str:
    for key, icon in CLASS_ICONS.items():
        if key.lower() in class_name.lower():
            return icon
    return "🚸"


def get_category(name: str) -> str:
    if "Speed" in name:
        return "Speed Regulation"
    if "Stop" in name or "Yield" in name:
        return "Priority"
    if "No" in name:
        return "Prohibition"
    return "Other"


# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

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

.stApp { background: var(--bg); color: var(--text); font-family: var(--font-head); }

.hero {
    background: linear-gradient(135deg, #0A0A0F 0%, #13131A 50%, #1a0a0f 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
}

.hero-label {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.2em;
    color: var(--accent);
}

.hero h1 {
    font-size: 2.8rem;
    font-weight: 800;
}

.hero h1 span { color: var(--accent); }

.section-head {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: var(--muted);
    margin-bottom: 1rem;
}

.upload-zone {
    border: 2px dashed var(--border);
    border-radius: 14px;
    padding: 3rem;
    text-align: center;
}

.result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.8rem;
}

.result-card.top-result {
    border-color: var(--accent);
}

.result-name {
    font-size: 1.5rem;
    font-weight: 700;
}

.result-confidence {
    font-family: var(--font-mono);
    font-size: 2rem;
    color: var(--accent);
}

.conf-bar-bg {
    background: var(--border);
    height: 6px;
    border-radius: 4px;
    margin-top: 0.8rem;
}

.conf-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent3), var(--success));
}

.info-box {
    background: rgba(34,211,238,0.07);
    border: 1px solid rgba(34,211,238,0.2);
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}

.model-badge {
    display: inline-block;
    background: rgba(255,69,69,0.1);
    border: 1px solid rgba(255,69,69,0.3);
    color: var(--accent);
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-family: var(--font-mono);
}

hr { border-color: var(--border); }
</style>
""", unsafe_allow_html=True)

# ─── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        import tensorflow as tf
        path = Path("models/custom_cnn_best.keras")

        if not path.exists():
            return None, f"Model file not found: `{path}`"

        model = tf.keras.models.load_model(str(path))
        return model, None

    except ImportError:
        return None, "TensorFlow not installed. Run: pip install tensorflow"
    except Exception as e:
        return None, str(e)


def preprocess_image(img: Image.Image, size=(32, 32)):
    img = img.convert("RGB").resize(size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🚦 TrafficLens")
    st.markdown("**Custom CNN Only Mode**")

    metrics_path = Path("outputs/metrics.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

        m = metrics.get("custom_cnn", {})
        if m:
            st.metric("Val Accuracy", f"{m.get('accuracy', 0) * 100:.1f}%")
            st.metric("Val Loss", f"{m.get('loss', 0):.4f}")

    st.markdown("---")
    st.markdown("GTSRB Dataset · 43 Classes")

# ─── Main UI ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-label">🚦 AI · Traffic Sign Recognition</div>
    <h1>Traffic<span>Lens</span></h1>
</div>
""", unsafe_allow_html=True)

# ─── Load Model ───────────────────────────────────────────────────────────────
model, error = load_model()

if error:
    st.markdown(f"<div class='info-box'>⚠️ {error}</div>", unsafe_allow_html=True)
    st.stop()

st.markdown("""
<div class="info-box">
✓ Model Loaded · <span class="model-badge">Custom CNN</span>
</div>
""", unsafe_allow_html=True)

# ─── Upload ───────────────────────────────────────────────────────────────────
st.markdown("### Upload Image")

uploaded = st.file_uploader(
    "Upload traffic sign",
    type=["png", "jpg", "jpeg", "bmp", "ppm"]
)

if not uploaded:
    st.markdown("""
    <div class="upload-zone">
        🚦 Upload an image to start prediction
    </div>
    """, unsafe_allow_html=True)
    st.stop()

img = Image.open(uploaded)
st.image(img, width=250)

# ─── Prediction ───────────────────────────────────────────────────────────────
inp = preprocess_image(img)

with st.spinner("Predicting..."):
    preds = model.predict(inp, verbose=0)[0]

best_idx = int(np.argmax(preds))
best_prob = float(preds[best_idx])

# ─── Safe decode with None guard ──────────────────────────────────────────────
class_id = decode_prediction_index(best_idx)

if class_id is None:
    st.error(f"Could not decode prediction index: {best_idx}. Check your class mapping.")
    st.stop()

if class_id < 0 or class_id >= len(CLASS_NAMES):
    st.error(f"Decoded class_id={class_id} is out of range (0–{len(CLASS_NAMES)-1}).")
    st.stop()

class_name = CLASS_NAMES[class_id]
icon = get_icon(class_name)
category = get_category(class_name)

st.markdown(f"""
<div class="result-card top-result">
    <div style="font-size:2rem">{icon}</div>
    <div class="result-name">{class_name}</div>
    <div>{category}</div>
    <div class="result-confidence">{best_prob * 100:.2f}%</div>
    <div class="conf-bar-bg">
        <div class="conf-bar-fill" style="width:{best_prob * 100}%"></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<hr>
<div style="display:flex;justify-content:space-between;">
    <div>TrafficLens · GTSRB · TensorFlow</div>
    <div>Abdallah Ibrahim · AI Engineer</div>
</div>
""", unsafe_allow_html=True)