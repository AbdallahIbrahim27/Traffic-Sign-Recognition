# GTSRB Traffic Sign Classifier — Custom CNN

> **Deep Learning · Computer Vision · TensorFlow/Keras · 43 Sign Classes**

A custom Convolutional Neural Network built from scratch to classify all 43 traffic sign categories in the German Traffic Sign Recognition Benchmark (GTSRB). No pre-trained backbones — every weight is learned from the dataset.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Custom CNN Architecture](#2-custom-cnn-architecture)
3. [Dataset](#3-dataset)
4. [Training Procedure](#4-training-procedure)
5. [Usage Instructions](#5-usage-instructions)
6. [Project Structure](#6-project-structure)
7. [Expected Performance](#7-expected-performance)

---

## 1. Project Overview

This project implements a lightweight three-block CNN for real-world traffic sign recognition, targeting the 43-class GTSRB benchmark. The network is designed to run on CPU hardware while still achieving >95% test accuracy — a deliberate constraint that keeps the model practical for embedded or edge deployments.

The codebase is organized across five modules:

| Module | Responsibility |
|---|---|
| `models.py` | Custom CNN and MobileNetV2 model definitions with a factory interface |
| `preprocessing.py` | Image loading, CLAHE enhancement, normalization, and augmentation pipelines |
| `train.py` | End-to-end training loop with callbacks, checkpointing, and metric logging |
| `predict.py` | Inference engine for single images or entire folders with Top-5 confidence output |
| `class_mapping.py` | Decoding layer that maps Keras softmax indices to canonical GTSRB class IDs 0–42 |

> **Scope note:** This README focuses exclusively on `build_custom_cnn`. The MobileNetV2 transfer-learning path shares the same preprocessing and training infrastructure but is not covered here.

---

## 2. Custom CNN Architecture

The network follows a VGG-inspired double-convolution pattern compressed to three spatial stages, with modern regularization at every stage.

### 2.1 Layer-by-Layer Breakdown

```
Input (32 × 32 × 3)
│
├── Block 1
│   ├── Conv2D(32, 3×3, padding=same) → BatchNorm → ReLU
│   ├── Conv2D(32, 3×3, padding=same) → BatchNorm → ReLU
│   ├── MaxPooling2D(2×2)
│   └── Dropout(0.3)                         → 16 × 16 × 32
│
├── Block 2
│   ├── Conv2D(64, 3×3, padding=same) → BatchNorm → ReLU
│   ├── Conv2D(64, 3×3, padding=same) → BatchNorm → ReLU
│   ├── MaxPooling2D(2×2)
│   └── Dropout(0.3)                         → 8 × 8 × 64
│
├── Block 3
│   ├── Conv2D(128, 3×3, padding=same) → BatchNorm → ReLU
│   ├── Conv2D(128, 3×3, padding=same) → BatchNorm → ReLU
│   ├── MaxPooling2D(2×2)
│   └── Dropout(0.3)                         → 4 × 4 × 128
│
├── GlobalAveragePooling2D                   → 128
├── Dense(256) → BatchNorm → ReLU
├── Dropout(0.5)
└── Dense(43, softmax)                       → class probabilities
```

### 2.2 Detailed Layer Table

| Stage | Operation | Output Shape | Key Parameters |
|---|---|---|---|
| Input | `keras.Input` | 32 × 32 × 3 | — |
| Block 1 | Conv2D × 2 + BN | 32 × 32 × 32 | kernel 3×3, padding=same, ReLU |
| Block 1 | MaxPooling + Dropout | 16 × 16 × 32 | pool 2×2, drop=0.3 |
| Block 2 | Conv2D × 2 + BN | 16 × 16 × 64 | kernel 3×3, padding=same, ReLU |
| Block 2 | MaxPooling + Dropout | 8 × 8 × 64 | pool 2×2, drop=0.3 |
| Block 3 | Conv2D × 2 + BN | 8 × 8 × 128 | kernel 3×3, padding=same, ReLU |
| Block 3 | MaxPooling + Dropout | 4 × 4 × 128 | pool 2×2, drop=0.3 |
| Head | GlobalAveragePooling2D | 128 | replaces Flatten — no spatial params |
| Head | Dense + BN + Dropout | 256 | ReLU, drop=0.5 |
| Output | Dense (softmax) | 43 | categorical cross-entropy target |

### 2.3 Design Rationale

**ReLU activation**
ReLU is used in all convolutional and dense layers. It avoids the vanishing-gradient problem common in deeper networks, is computationally inexpensive, and outperforms sigmoid/tanh on image classification tasks. The final layer uses softmax to produce a valid probability distribution over 43 classes.

**Batch Normalization after every conv pair**
BatchNormalization normalizes activations to zero-mean unit-variance within each mini-batch. This allows higher learning rates, acts as a mild regularizer by adding stochastic noise during training, and reduces sensitivity to weight initialization. BN is placed before pooling and dropout in each block.

**Progressive filter doubling: 32 → 64 → 128**
Each successive block doubles filters while halving spatial resolution via MaxPooling. Early layers learn low-level edges and textures at fine resolution; later layers learn semantic shapes (circles, triangles, numerals) at coarser resolution. Doubling filters compensates for the spatial information lost to pooling.

**Two convolutions per block**
Two stacked 3×3 convolutions have an effective receptive field of 5×5 with fewer parameters than a single 5×5 convolution, and insert an extra non-linearity — a technique validated in VGG and related architectures.

**GlobalAveragePooling2D instead of Flatten**
After Block 3 the spatial maps are 4×4×128. `GlobalAveragePooling2D` reduces each feature map to a single scalar instead of flattening to a 2048-element vector. This halves the parameter count in the dense head, naturally regularizes against spatial overfitting, and makes the network invariant to small translations — important for signs captured at varying distances.

**Dropout rates: 0.3 (conv) vs 0.5 (head)**
The higher dropout rate at the dense head is intentional: the fully connected layer has more capacity for overfitting since it has no spatial inductive bias.

### 2.4 Parameter Summary

| Component | Approx. Parameters |
|---|---|
| Block 1 (conv + BN) | ~19,000 |
| Block 2 (conv + BN) | ~74,000 |
| Block 3 (conv + BN) | ~295,000 |
| Dense head | ~66,000 |
| **Total** | **~464,000** |

The compact size is by design: GTSRB images are 32×32 px, leaving little spatial resolution to exploit with deeper or wider networks. The model is fully trainable on CPU within a few hours.

---

## 3. Dataset

### 3.1 Source

The German Traffic Sign Recognition Benchmark (GTSRB) is publicly available on Kaggle:

```
https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
```

The dataset contains 50,000+ images across 43 traffic sign classes captured from real German roads under varying lighting, weather, and occlusion conditions. The official split provides separate `Train/` and `Test/` directories, each subdivided into folders `0`–`42` corresponding to GTSRB class IDs.

### 3.2 Class Distribution

GTSRB is significantly imbalanced. Speed-limit signs (classes 0–8) have ~8,000 training examples each, while rare classes such as *Dangerous curve left* (class 19) have fewer than 300. This imbalance is addressed through aggressive augmentation and, optionally, class-weight balancing during training.

### 3.3 Preprocessing Pipeline

Each image passes through the following deterministic steps in `preprocessing.py`:

| Step | Operation | Purpose |
|---|---|---|
| 1 | `cv2.resize` to 32×32 | Standardize spatial dimensions |
| 2 | BGR → LAB color conversion | Isolate luminance channel for contrast enhancement |
| 3 | CLAHE on L channel (`clipLimit=2.0`, `tileGridSize=(4,4)`) | Compensate for under/over-exposed sign images |
| 4 | LAB → BGR reconversion | Restore color information after equalization |
| 5 | Cast to `float32`, divide by 255 | Normalize pixel values to [0, 1] |

CLAHE (Contrast Limited Adaptive Histogram Equalization) operates at tile level rather than globally, preventing over-amplification of noise in uniform regions — a common failure mode with standard histogram equalization on small sign images.

> **Inference alignment:** `preprocess_for_inference()` skips CLAHE and applies only resize + RGB conversion + normalization to match the simpler path used by `ImageDataGenerator` at training time. This avoids a train/inference preprocessing mismatch.

### 3.4 Data Augmentation

The training generator applies stochastic augmentations at load time, producing a different transformed version of each image each epoch:

| Parameter | Value | Rationale |
|---|---|---|
| `rotation_range` | ±15° | Signs are mounted at slight angles |
| `zoom_range` | 0.25 | Simulate signs at different capture distances |
| `width_shift_range` | 0.12 | Horizontal centering variation |
| `height_shift_range` | 0.12 | Vertical centering variation |
| `shear_range` | 0.12 | Perspective distortion from oblique angles |
| `brightness_range` | [0.6, 1.4] | Day/night and shadow variation |
| `channel_shift_range` | 20.0 | Simulate different white-balance / lighting hues |
| `horizontal_flip` | **False** | Traffic signs are NOT horizontally symmetric |

> ⚠️ **`horizontal_flip=False` is a correctness constraint, not a hyperparameter.** Sign classes such as *Turn right ahead* (class 33) and *Turn left ahead* (class 34) are mirror images of each other. Enabling flipping would silently corrupt labels and degrade accuracy on directional signs.

### 3.5 Validation Split

The `Train/` directory is split 80/20 using `ImageDataGenerator` with `validation_split=0.2` and a fixed random seed. The `Test/` directory is held out entirely for final evaluation and is never seen during training or hyperparameter tuning.

---

## 4. Training Procedure

### 4.1 Loss Function

**Categorical Cross-Entropy:**

```
L = − Σ y_i · log(ŷ_i)    for i = 0 … 42
```

where `y_i` is the one-hot true label and `ŷ_i` is the softmax output. One-hot encoding is produced automatically by `flow_from_directory(class_mode="categorical")`.

### 4.2 Optimizer

**Adam** (Adaptive Moment Estimation):

| Hyperparameter | Value |
|---|---|
| Learning rate | `1e-3` |
| Beta_1 | `0.9` (default) |
| Beta_2 | `0.999` (default) |
| Epsilon | `1e-7` (default) |
| Weight decay | none (regularization via Dropout + BN) |

Adam is chosen over SGD because per-parameter adaptive learning rates reach convergence in fewer epochs — important when training from random initialization on 43 classes.

### 4.3 Callbacks

| Callback | Configuration | Effect |
|---|---|---|
| `ModelCheckpoint` | `monitor=val_accuracy`, `save_best_only=True` | Persists the epoch with highest validation accuracy |
| `EarlyStopping` | `patience=10`, `restore_best_weights=True` | Halts if `val_accuracy` stagnates for 10 epochs |
| `ReduceLROnPlateau` | `factor=0.5`, `patience=5`, `min_lr=1e-7` | Halves learning rate on plateau to escape local minima |

### 4.4 Hyperparameters

| Hyperparameter | Default | Notes |
|---|---|---|
| Batch size | 32 | Fits in CPU RAM; increase to 64 on GPU |
| Max epochs | 50 | EarlyStopping typically fires at 20–35 |
| Input size | 32×32×3 | Matches GTSRB native resolution |
| Dropout (conv blocks) | 0.3 | Tunable via `build_custom_cnn(dropout_rate=...)` |
| Dropout (dense head) | 0.5 | Fixed — higher rate intentional |
| Dense units | 256 | 128 shows degradation on minority classes |

### 4.5 Class Index Decoding

`flow_from_directory` sorts folder names alphabetically as strings, not numerically. This means folder `"10"` sorts before `"2"`, causing a mismatch between model output index and GTSRB class ID. `class_mapping.py` resolves this transparently:

1. If `outputs/class_order.json` exists and flags `indices_are_gtsrb_ids=True`, the identity mapping is used.
2. Otherwise, the legacy alphabetical lookup table (`_LEGACY_INDEX_TO_GTSRB`) is applied.
3. `decode_prediction_index(model_index)` always returns the correct GTSRB label 0–42.

---

## 5. Usage Instructions

### 5.1 Requirements

| Package | Version | Role |
|---|---|---|
| `tensorflow` | ≥ 2.10 | Model, training, inference |
| `opencv-python` | ≥ 4.5 | Image loading and CLAHE preprocessing |
| `numpy` | ≥ 1.21 | Array operations |
| `matplotlib` | ≥ 3.5 | Augmentation visualization (optional) |

```bash
pip install tensorflow opencv-python numpy matplotlib
```

### 5.2 Dataset Setup

Download GTSRB from Kaggle and arrange the directory as:

```
data/
  Train/
    0/   1/   2/   ...   42/
  Test/
    0/   1/   2/   ...   42/
```

### 5.3 Training

```bash
python train.py \
  --model custom_cnn \
  --data_dir data/ \
  --epochs 50 \
  --batch_size 32
```

On completion, the best checkpoint is saved to `outputs/custom_cnn_best.keras` and `outputs/class_order.json` is written so the prediction pipeline decodes labels correctly.

### 5.4 Single-Image Inference

```bash
python predict.py \
  --model outputs/custom_cnn_best.keras \
  --image path/to/sign.png
```

**Expected output:**
```
  Prediction: Speed limit (50km/h)
  Confidence: 97.3%

  Top-5:
   97.3%  Speed limit (50km/h)
    1.8%  Speed limit (30km/h)
    0.5%  Speed limit (60km/h)
    0.2%  Speed limit (80km/h)
    0.1%  End of speed limit (80km/h)
```

### 5.5 Batch Folder Inference

```bash
python predict.py \
  --model outputs/custom_cnn_best.keras \
  --folder path/to/test_images/
```

### 5.6 Programmatic API

**Build the model:**
```python
from src.models import get_model

# Default configuration
model = get_model("custom_cnn")
model.summary()

# Increase regularization for a low-data scenario
model = get_model("custom_cnn", dropout_rate=0.4)
```

**Run inference:**
```python
from src.predict import predict_image

result = predict_image("outputs/custom_cnn_best.keras", "sign.png")
print(result["class_name"], f"{result['confidence']*100:.1f}%")

# Full result dict:
# {
#   "class_id":   2,
#   "class_name": "Speed limit (50km/h)",
#   "confidence": 0.973,
#   "top5":       [(2, "Speed limit (50km/h)", 0.973), ...]
# }
```

**Reset class decoder cache (after retraining):**
```python
from src.class_mapping import reset_decoder_cache
reset_decoder_cache()
```

### 5.7 Augmentation Visualization

```python
from src.preprocessing import visualize_augmentations, preprocess_image
import cv2

img = preprocess_image(cv2.imread("sign.png"))
visualize_augmentations(img, n_aug=8, save_path="augmentation_grid.png")
```

---

## 6. Project Structure

```
gtsrb-classifier/
├── src/
│   ├── models.py           # CNN & MobileNetV2 definitions + factory
│   ├── preprocessing.py    # CLAHE pipeline, augmentation, dataset loader
│   ├── predict.py          # Inference CLI (single image + folder)
│   └── class_mapping.py    # Softmax index → GTSRB class ID decoder
├── train.py                # Training entry point
├── data/
│   ├── Train/              # GTSRB training set (downloaded separately)
│   └── Test/               # GTSRB test set (held out)
├── outputs/
│   ├── custom_cnn_best.keras
│   └── class_order.json
└── README.md
```

---

## 7. Expected Performance

| Metric | Typical Value |
|---|---|
| Test Accuracy | ≥ 95% |
| Convergence | 20–35 epochs |
| Inference time (CPU) | ~4 ms / image |
| Model size (`.keras`) | ~6 MB |

> **Reproducibility note:** Accuracy varies by ±1–2% across random seeds due to stochastic augmentation and dropout. For publication results, run at least 5 seeds and report mean ± std.

---

## GTSRB Class Reference (0–42)

<details>
<summary>Click to expand all 43 classes</summary>

| ID | Class Name | ID | Class Name |
|---|---|---|---|
| 0 | Speed limit (20km/h) | 22 | Bumpy road |
| 1 | Speed limit (30km/h) | 23 | Slippery road |
| 2 | Speed limit (50km/h) | 24 | Road narrows on the right |
| 3 | Speed limit (60km/h) | 25 | Road work |
| 4 | Speed limit (70km/h) | 26 | Traffic signals |
| 5 | Speed limit (80km/h) | 27 | Pedestrians |
| 6 | End of speed limit (80km/h) | 28 | Children crossing |
| 7 | Speed limit (100km/h) | 29 | Bicycles crossing |
| 8 | Speed limit (120km/h) | 30 | Beware of ice/snow |
| 9 | No passing | 31 | Wild animals crossing |
| 10 | No passing veh over 3.5 tons | 32 | End speed + passing limits |
| 11 | Right-of-way at intersection | 33 | Turn right ahead |
| 12 | Priority road | 34 | Turn left ahead |
| 13 | Yield | 35 | Ahead only |
| 14 | Stop | 36 | Go straight or right |
| 15 | No vehicles | 37 | Go straight or left |
| 16 | Veh > 3.5 tons prohibited | 38 | Keep right |
| 17 | No entry | 39 | Keep left |
| 18 | General caution | 40 | Roundabout mandatory |
| 19 | Dangerous curve left | 41 | End of no passing |
| 20 | Dangerous curve right | 42 | End no passing veh > 3.5 tons |
| 21 | Double curve | | |

</details>
