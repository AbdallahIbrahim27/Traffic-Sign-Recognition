"""
models.py — Custom CNN + MobileNetV2 Transfer Learning definitions
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2


NUM_CLASSES = 43


# ─── Custom CNN ───────────────────────────────────────────────────────────────

def build_custom_cnn(input_shape=(32, 32, 3), num_classes=NUM_CLASSES,
                     dropout_rate=0.3) -> keras.Model:
    """
    3-block CNN architecture with BatchNormalization + GlobalAveragePooling.
    Designed specifically for 32×32 traffic sign images.

    Architecture:
        Input (32,32,3)
        → Block1: Conv32→Conv32→MaxPool→Dropout
        → Block2: Conv64→Conv64→MaxPool→Dropout
        → Block3: Conv128→Conv128→MaxPool→Dropout
        → GAP → Dense(256) → BN → Dropout → Softmax(43)
    """
    inputs = keras.Input(shape=input_shape)

    # ── Block 1
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)

    # ── Block 2
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)

    # ── Block 3
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)

    # ── Classifier head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="CustomCNN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─── MobileNetV2 Transfer Learning ───────────────────────────────────────────

def build_mobilenet(input_shape=(32, 32, 3), num_classes=NUM_CLASSES,
                    fine_tune_layers=30) -> keras.Model:
    """
    MobileNetV2 with ImageNet weights + fine-tuning.

    Strategy:
        • Freeze all layers except the last `fine_tune_layers`
        • Upsample 32×32 → 96×96 (min recommended for MobileNet)
        • Custom classification head
    """
    inputs = keras.Input(shape=input_shape)

    # Upsample for MobileNet
    x = layers.Resizing(96, 96)(inputs)
    x = layers.Lambda(
        lambda t: tf.keras.applications.mobilenet_v2.preprocess_input(t * 255.0)
    )(x)

    base = MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=False,
        weights="imagenet",
    )

    # Freeze base except last N layers
    base.trainable = True
    for layer in base.layers[:-fine_tune_layers]:
        layer.trainable = False

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="MobileNetV2_FineTuned")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─── Model factory ────────────────────────────────────────────────────────────

def get_model(name: str, **kwargs) -> keras.Model:
    """
    Factory function.

    Usage:
        model = get_model("custom_cnn")
        model = get_model("mobilenet", fine_tune_layers=50)
    """
    registry = {
        "custom_cnn": build_custom_cnn,
        "mobilenet":  build_mobilenet,
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(registry)}")
    return registry[name](**kwargs)


def compare_models():
    """Print a side-by-side parameter comparison."""
    cnn = build_custom_cnn()
    mob = build_mobilenet()

    print("\n" + "=" * 50)
    print("  Model Comparison")
    print("=" * 50)
    print(f"  {'Model':<25} {'Params':>12}")
    print("-" * 40)
    print(f"  {'Custom CNN':<25} {cnn.count_params():>12,}")
    print(f"  {'MobileNetV2 (fine-tuned)':<25} {mob.count_params():>12,}")
    print("=" * 50 + "\n")
