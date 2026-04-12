# 🚦 Traffic Sign Recognition — GTSRB

> **Industry-Level CNN Project** | Python · Keras · TensorFlow · OpenCV

---

## Overview

End-to-end deep learning pipeline for classifying **43 classes** of traffic signs using the GTSRB (German Traffic Sign Recognition Benchmark) dataset from Kaggle.

**Bonus features completed:**
- ✅ Data Augmentation (rotation, zoom, shift, shear, brightness — no horizontal flip!)
- ✅ Custom CNN from scratch
- ✅ MobileNet-style model with depthwise separable convolutions
- ✅ Full model comparison with accuracy, loss, parameters, and per-class analysis

---

## Results

| Model | Val Accuracy | Val Loss | Parameters |
|---|---|---|---|
| **Custom CNN** | **94.2%** | 0.208 | 333,899 |
| **MobileNet Style** | **96.3%** | 0.144 | 3,489,832 |

🏆 **MobileNet-style wins** by +2.1% accuracy — more parameters + depthwise separable convolutions capture richer features.

---

## Architecture

### Custom CNN (3-Block)
```
Input (32×32×3)
→ Block 1: Conv32 → BN → Conv32 → BN → MaxPool → Dropout(0.25)
→ Block 2: Conv64 → BN → Conv64 → BN → MaxPool → Dropout(0.25)
→ Block 3: Conv128 → BN → Conv128 → BN → MaxPool → Dropout(0.30)
→ GlobalAveragePooling → Dense(256) → Dropout(0.4) → Softmax(43)
```

### MobileNet-Style (Depthwise Separable)
```
Input (32×32×3)
→ Conv32 → BN
→ DepthwiseConv → PointwiseConv(64) → BN → MaxPool → Dropout
→ DepthwiseConv → PointwiseConv(128) → BN → MaxPool → Dropout
→ DepthwiseConv → PointwiseConv(256) → BN → MaxPool → Dropout
→ GlobalAveragePooling → Dense(512) → Dropout(0.4) → Softmax(43)
```

---

## Data Augmentation (Bonus ✅)

```python
ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.20,
    width_shift_range=0.10,
    height_shift_range=0.10,
    shear_range=0.10,
    brightness_range=[0.7, 1.3],
    horizontal_flip=False,   # ← Traffic signs are NOT symmetric!
    fill_mode="nearest",
)
```

Key insight: `horizontal_flip=False` is **critical** — a mirrored "Turn Right" sign becomes a "Turn Left" sign!

---

## Project Structure

```
traffic-sign-recognition/
├── src/
│   ├── train.py          # Main training script (CNN + MobileNet)
│   ├── models.py         # Model definitions + factory function
│   ├── preprocessing.py  # CLAHE preprocessing + augmentation utilities
│   └── predict.py        # Inference on single images or folders
├── outputs/
│   ├── results_dashboard.png     # Full training results visualization
│   ├── augmentation_examples.png # Augmentation grid
│   ├── sample_signs.png          # All 43 GTSRB classes
│   └── metrics.json              # Saved model metrics
└── README.md
```

---

## Quick Start

### 1. Download Dataset
```bash
# Kaggle CLI
kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
unzip gtsrb-german-traffic-sign.zip -d data/
```

### 2. Install Dependencies
```bash
pip install tensorflow keras opencv-python scikit-learn matplotlib seaborn
```

### 3. Train
```bash
python src/train.py
```

### 4. Predict
```bash
python src/predict.py --model outputs/custom_cnn_best.keras --image my_sign.jpg
```

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| `GlobalAveragePooling2D` instead of `Flatten` | Reduces parameters, adds regularization |
| `BatchNormalization` after every Conv | Stabilizes training, allows higher LR |
| `EarlyStopping(patience=8)` | Prevents overfitting on GTSRB |
| `ReduceLROnPlateau(factor=0.5)` | Fine-tunes learning rate automatically |
| No horizontal flip | Traffic sign semantics are asymmetric |
| CLAHE preprocessing | Enhances contrast for dark/nighttime signs |

---

## Evaluation

- **Accuracy** and **loss** on validation split (20%)
- **Confusion matrix** — identifies commonly confused sign pairs
- **Per-class accuracy** — highlights hardest classes (Speed 20, Double Curve)
- **Classification report** — precision, recall, F1 per class

---

*Dataset: [GTSRB on Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)*

## Made By Abdullah Ibrahim 