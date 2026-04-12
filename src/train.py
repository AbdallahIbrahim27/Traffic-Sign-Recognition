"""
Task 8: Traffic Sign Recognition
Dataset: GTSRB (German Traffic Sign Recognition Benchmark)
Models: Custom CNN vs MobileNetV2 (Transfer Learning)
Bonus: Data Augmentation + Model Comparison

HOW TO USE:
  1. Download the GTSRB dataset from Kaggle:
     https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
  2. Extract it so your folder structure looks like:
       GTSRB/
         Train/
           0/   (Speed limit 20km/h images)
           1/   (Speed limit 30km/h images)
           ...
           42/
         Test/
           images/
           GT-final_test.csv
  3. Set DATA_DIR below to the path of your GTSRB/Train folder.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
import json

# ─── Config ───────────────────────────────────────────────────────────────────
IMG_SIZE    = (32, 32)
BATCH_SIZE  = 64
EPOCHS      = 30
NUM_CLASSES = 43          # GTSRB has 43 classes
SEED        = 42
OUTPUT_DIR  = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# ★  SET THIS TO YOUR GTSRB TRAIN FOLDER  ★
# Example (Windows): DATA_DIR = Path("C:/datasets/GTSRB/Train")
# Example (Linux/Mac): DATA_DIR = Path("/home/user/datasets/GTSRB/Train")
DATA_DIR = Path(r"C:\Users\Boudy\Desktop\Traffic Sign Recognition\data\raw\Train")
# ──────────────────────────────────────────────────────────────────────────────

# GTSRB class names (43 classes)
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

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ─── Validate Data Directory ──────────────────────────────────────────────────
print("=" * 60)
print("  TRAFFIC SIGN RECOGNITION — GTSRB")
print("=" * 60)

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"\n[ERROR] Dataset directory not found: {DATA_DIR}\n"
        "  Please download the GTSRB dataset from:\n"
        "  https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign\n"
        "  Then set DATA_DIR at the top of this script to the 'Train' folder path."
    )

print(f"\n  ✓ Dataset found at: {DATA_DIR.resolve()}")

# ─── Data Generators with Augmentation ───────────────────────────────────────
print("\n[1/5] Setting up data generators with augmentation...")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    brightness_range=[0.7, 1.3],
    horizontal_flip=False,   # Traffic signs are NOT symmetric
    fill_mode="nearest",
    validation_split=0.2,
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
)

# ─── Load Data from Directory ─────────────────────────────────────────────────
# Explicit class order: Keras otherwise sorts folder names alphabetically
# ("0","1","10",...) so softmax index ≠ GTSRB class id 0..42.
CLASS_SUBDIRS = [str(i) for i in range(NUM_CLASSES)]

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    seed=SEED,
    shuffle=True,
    classes=CLASS_SUBDIRS,
)

val_gen = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    seed=SEED,
    shuffle=False,
    classes=CLASS_SUBDIRS,
)

# Inference uses this to know if model indices already match GTSRB ids
(OUTPUT_DIR / "class_order.json").write_text(
    json.dumps({"indices_are_gtsrb_ids": True, "folders": CLASS_SUBDIRS}, indent=2),
    encoding="utf-8",
)

print(f"  → Train samples : {train_gen.samples:,}")
print(f"  → Val samples   : {val_gen.samples:,}")
print(f"  → Image shape   : {IMG_SIZE + (3,)} | Classes: {NUM_CLASSES}")

# Preload validation data into memory for fast evaluation / confusion matrix
print("  → Loading validation set into memory...")
val_gen.reset()
X_val_list, y_val_list = [], []
for _ in range(len(val_gen)):
    xb, yb = next(val_gen)
    X_val_list.append(xb)
    y_val_list.append(yb)

X_val     = np.concatenate(X_val_list, axis=0)
y_val_cat = np.concatenate(y_val_list, axis=0)
y_val     = np.argmax(y_val_cat, axis=1)   # integer labels for metrics
print(f"  → Val array shape: {X_val.shape}")

# ─── Model 1: Custom CNN ──────────────────────────────────────────────────────
print("\n[2/5] Building Custom CNN...")

def build_custom_cnn(input_shape=(32, 32, 3), num_classes=43):
    inputs = keras.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # Classifier
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="CustomCNN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

cnn_model = build_custom_cnn()
cnn_model.summary()
print(f"  → Total parameters: {cnn_model.count_params():,}")

# ─── Model 2: MobileNetV2 (Transfer Learning) ─────────────────────────────────
print("\n[3/5] Building MobileNetV2 Transfer Learning model...")

def build_mobilenet(input_shape=(32, 32, 3), num_classes=43):
    # MobileNetV2 performs better at higher resolutions; upsample internally
    inputs = keras.Input(shape=input_shape)
    x = layers.Resizing(96, 96)(inputs)
    x = layers.Lambda(
        lambda t: tf.keras.applications.mobilenet_v2.preprocess_input(t * 255)
    )(x)

    base = MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=False,
        weights="imagenet",
    )
    # Freeze all except last 30 layers (fine-tuning)
    for layer in base.layers[:-30]:
        layer.trainable = False

    x = base(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="MobileNetV2_FineTuned")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

mobilenet_model = build_mobilenet()
mobilenet_model.summary()

# ─── Callbacks ────────────────────────────────────────────────────────────────
def get_callbacks(name):
    return [
        EarlyStopping(patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        ModelCheckpoint(
            OUTPUT_DIR / f"{name}_best.keras",
            save_best_only=True, verbose=0
        ),
    ]

# ─── Training ─────────────────────────────────────────────────────────────────
print("\n[4/5] Training models...\n")

print("─" * 40)
print("  Training Custom CNN")
print("─" * 40)
cnn_history = cnn_model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=(X_val, y_val_cat),
    callbacks=get_callbacks("custom_cnn"),
    verbose=1,
)

print("\n" + "─" * 40)
print("  Training MobileNetV2")
print("─" * 40)
mobilenet_history = mobilenet_model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=(X_val, y_val_cat),
    callbacks=get_callbacks("mobilenet"),
    verbose=1,
)

# ─── Evaluation ───────────────────────────────────────────────────────────────
print("\n[5/5] Evaluating models...\n")

cnn_loss, cnn_acc = cnn_model.evaluate(X_val, y_val_cat, verbose=0)
mob_loss, mob_acc = mobilenet_model.evaluate(X_val, y_val_cat, verbose=0)
cnn_preds         = np.argmax(cnn_model.predict(X_val, verbose=0), axis=1)
mob_preds         = np.argmax(mobilenet_model.predict(X_val, verbose=0), axis=1)

print(f"  Custom CNN     → Accuracy: {cnn_acc*100:.2f}%  Loss: {cnn_loss:.4f}")
print(f"  MobileNetV2    → Accuracy: {mob_acc*100:.2f}%  Loss: {mob_loss:.4f}")

# Save metrics
metrics = {
    "custom_cnn":   {"accuracy": round(float(cnn_acc), 4), "loss": round(float(cnn_loss), 4)},
    "mobilenet_v2": {"accuracy": round(float(mob_acc), 4), "loss": round(float(mob_loss), 4)},
    "params": {
        "custom_cnn":   cnn_model.count_params(),
        "mobilenet_v2": mobilenet_model.count_params(),
    }
}
with open(OUTPUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ─── Visualization ────────────────────────────────────────────────────────────
plt.style.use("dark_background")
GOLD  = "#F5A623"
GREEN = "#4CAF50"
BLUE  = "#2196F3"
RED   = "#F44336"
WHITE = "#FFFFFF"
GRAY  = "#888888"

fig = plt.figure(figsize=(22, 20), facecolor="#0D0D0D")
fig.suptitle(
    "Traffic Sign Recognition — GTSRB\nCustom CNN vs MobileNetV2",
    fontsize=22, fontweight="bold", color=WHITE, y=0.98
)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── 1. Training Accuracy
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(cnn_history.history["accuracy"],           color=BLUE, lw=2, label="CNN Train")
ax1.plot(cnn_history.history["val_accuracy"],       color=BLUE, lw=2, ls="--", label="CNN Val")
ax1.plot(mobilenet_history.history["accuracy"],     color=GOLD, lw=2, label="MobileNet Train")
ax1.plot(mobilenet_history.history["val_accuracy"], color=GOLD, lw=2, ls="--", label="MobileNet Val")
ax1.set_title("Training Accuracy", color=WHITE, fontweight="bold")
ax1.set_xlabel("Epoch", color=GRAY); ax1.set_ylabel("Accuracy", color=GRAY)
ax1.legend(fontsize=8); ax1.set_facecolor("#1A1A1A"); ax1.grid(alpha=0.2)
ax1.tick_params(colors=GRAY)

# ── 2. Training Loss
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(cnn_history.history["loss"],           color=BLUE, lw=2, label="CNN Train")
ax2.plot(cnn_history.history["val_loss"],       color=BLUE, lw=2, ls="--", label="CNN Val")
ax2.plot(mobilenet_history.history["loss"],     color=GOLD, lw=2, label="MobileNet Train")
ax2.plot(mobilenet_history.history["val_loss"], color=GOLD, lw=2, ls="--", label="MobileNet Val")
ax2.set_title("Training Loss", color=WHITE, fontweight="bold")
ax2.set_xlabel("Epoch", color=GRAY); ax2.set_ylabel("Loss", color=GRAY)
ax2.legend(fontsize=8); ax2.set_facecolor("#1A1A1A"); ax2.grid(alpha=0.2)
ax2.tick_params(colors=GRAY)

# ── 3. Model Comparison Bar Chart
ax3 = fig.add_subplot(gs[0, 2])
models_names = ["Custom CNN", "MobileNetV2"]
accs         = [cnn_acc * 100, mob_acc * 100]
bars         = ax3.bar(models_names, accs, color=[BLUE, GOLD], width=0.5, edgecolor="none")
for bar, acc in zip(bars, accs):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f"{acc:.1f}%", ha="center", va="bottom", color=WHITE, fontweight="bold")
ax3.set_title("Model Accuracy Comparison", color=WHITE, fontweight="bold")
ax3.set_ylabel("Validation Accuracy (%)", color=GRAY)
ax3.set_ylim(0, 110); ax3.set_facecolor("#1A1A1A"); ax3.grid(axis="y", alpha=0.2)
ax3.tick_params(colors=GRAY)

# ── 4. Confusion Matrix — CNN (top 15 classes)
ax4 = fig.add_subplot(gs[1, :2])
cm_cnn     = confusion_matrix(y_val, cnn_preds)
top_classes = np.argsort(np.bincount(y_val))[-15:]
cm_sub      = cm_cnn[np.ix_(top_classes, top_classes)]
sns.heatmap(
    cm_sub, ax=ax4, cmap="Blues", annot=True, fmt="d", linewidths=0.3,
    xticklabels=[CLASS_NAMES[i][:12] for i in top_classes],
    yticklabels=[CLASS_NAMES[i][:12] for i in top_classes],
    cbar_kws={"shrink": 0.8}
)
ax4.set_title("Confusion Matrix — Custom CNN (Top 15 Classes)", color=WHITE, fontweight="bold")
ax4.tick_params(colors=GRAY, labelsize=7)
ax4.set_facecolor("#1A1A1A")

# ── 5. Confusion Matrix — MobileNet (top 8 classes)
ax5 = fig.add_subplot(gs[1, 2])
cm_mob     = confusion_matrix(y_val, mob_preds)
cm_mob_sub = cm_mob[np.ix_(top_classes[:8], top_classes[:8])]
sns.heatmap(
    cm_mob_sub, ax=ax5, cmap="YlOrBr", annot=True, fmt="d", linewidths=0.3,
    xticklabels=[CLASS_NAMES[i][:8] for i in top_classes[:8]],
    yticklabels=[CLASS_NAMES[i][:8] for i in top_classes[:8]],
    cbar_kws={"shrink": 0.8}
)
ax5.set_title("MobileNetV2\nConfusion (Top 8)", color=WHITE, fontweight="bold")
ax5.tick_params(colors=GRAY, labelsize=6)

# ── 6. Model Stats Comparison
ax6 = fig.add_subplot(gs[2, 0])
categories = ["Parameters\n(M)", "Val Acc\n(%)", "Val Loss\n(×10)"]
cnn_vals   = [cnn_model.count_params() / 1e6, cnn_acc * 100, cnn_loss * 10]
mob_vals   = [mobilenet_model.count_params() / 1e6, mob_acc * 100, mob_loss * 10]
x = np.arange(len(categories))
w = 0.35
ax6.bar(x - w / 2, cnn_vals, w, label="Custom CNN",  color=BLUE, alpha=0.85)
ax6.bar(x + w / 2, mob_vals, w, label="MobileNetV2", color=GOLD, alpha=0.85)
ax6.set_xticks(x); ax6.set_xticklabels(categories, color=GRAY, fontsize=9)
ax6.set_title("Model Stats", color=WHITE, fontweight="bold")
ax6.legend(fontsize=8); ax6.set_facecolor("#1A1A1A"); ax6.grid(axis="y", alpha=0.2)
ax6.tick_params(colors=GRAY)

# ── 7. Per-class accuracy CNN (top 5 / bottom 5)
ax7 = fig.add_subplot(gs[2, 1:])
per_class_acc = []
for c in range(NUM_CLASSES):
    mask = (y_val == c)
    if mask.sum() > 0:
        per_class_acc.append((c, (cnn_preds[mask] == c).mean() * 100))

per_class_acc.sort(key=lambda x: x[1])
bottom5  = per_class_acc[:5]
top5     = per_class_acc[-5:]
display  = bottom5 + top5
cls_labels = [CLASS_NAMES[c][:20] for c, _ in display]
cls_accs   = [a for _, a in display]
colors_bar = [RED] * 5 + [GREEN] * 5

bars = ax7.barh(cls_labels, cls_accs, color=colors_bar, edgecolor="none", height=0.6)
ax7.axvline(50, color=WHITE, ls="--", alpha=0.4, lw=1)
ax7.set_xlim(0, 115)
for bar, acc in zip(bars, cls_accs):
    ax7.text(acc + 1, bar.get_y() + bar.get_height() / 2,
             f"{acc:.0f}%", va="center", color=WHITE, fontsize=8)
ax7.set_title("Per-Class Accuracy — CNN\n(🔴 Bottom 5  |  🟢 Top 5)", color=WHITE, fontweight="bold")
ax7.set_xlabel("Accuracy (%)", color=GRAY)
ax7.set_facecolor("#1A1A1A"); ax7.grid(axis="x", alpha=0.2)
ax7.tick_params(colors=GRAY, labelsize=8)

plt.savefig(OUTPUT_DIR / "results_dashboard.png", dpi=150, bbox_inches="tight",
            facecolor="#0D0D0D")
print(f"\n  → Dashboard saved: outputs/results_dashboard.png")

# ─── Classification Report ────────────────────────────────────────────────────
print("\n── Custom CNN Classification Report (sampled classes) ──")
sampled = np.random.choice(NUM_CLASSES, size=10, replace=False)
mask    = np.isin(y_val, sampled)
print(classification_report(
    y_val[mask], cnn_preds[mask],
    target_names=[CLASS_NAMES[i] for i in sorted(sampled)],
    zero_division=0
))

# ─── Final Summary ────────────────────────────────────────────────────────────
print("\n── Final Summary ──────────────────────────────────────────")
print(f"  Custom CNN     → {cnn_acc*100:.2f}% accuracy | {cnn_model.count_params():,} params")
print(f"  MobileNetV2    → {mob_acc*100:.2f}% accuracy | {mobilenet_model.count_params():,} params")
winner = "MobileNetV2" if mob_acc > cnn_acc else "Custom CNN"
print(f"  🏆 Winner: {winner}")
print(f"\n  Outputs saved in: ./outputs/")
print("=" * 60)