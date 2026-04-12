import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import tensorflow as tf
from pathlib import Path

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

# ── Load saved model
print("Loading saved CNN model...")
cnn_model = tf.keras.models.load_model("outputs/custom_cnn_best.keras")
print("Model loaded!")

# ── Reload validation data (same split as train.py)
print("Reloading validation data...")
from tensorflow.keras.utils import image_dataset_from_directory

val_data = tf.keras.utils.image_dataset_from_directory(
    "data/raw/Train",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(32, 32),
    batch_size=64,
    label_mode="int",
    class_names=[str(i) for i in range(43)],
)

# Collect all val labels and predictions
y_val, y_pred = [], []
for images, labels in val_data:
    images = tf.cast(images, tf.float32) / 255.0
    preds  = np.argmax(cnn_model.predict(images, verbose=0), axis=1)
    y_val.extend(labels.numpy())
    y_pred.extend(preds)

y_val  = np.array(y_val)
y_pred = np.array(y_pred)

# ── Full classification report
print("\n── Custom CNN — Full Classification Report ──\n")
print(classification_report(
    y_val, y_pred,
    labels=list(range(43)),
    target_names=CLASS_NAMES,
    zero_division=0
))