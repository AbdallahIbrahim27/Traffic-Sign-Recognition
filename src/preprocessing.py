"""
preprocessing.py — GTSRB Data Preprocessing & Augmentation Utilities
"""

import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMG_SIZE = (32, 32)

# ─── Image Preprocessing ──────────────────────────────────────────────────────

def preprocess_for_inference(img: np.ndarray, target_size=IMG_SIZE) -> np.ndarray:
    """
    Match Keras ImageDataGenerator training in train.py:
    RGB, resize, float32 in [0, 1]. OpenCV loads BGR — convert before resize.
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, target_size)
    return rgb.astype(np.float32) / 255.0


def preprocess_image(img: np.ndarray, target_size=IMG_SIZE) -> np.ndarray:
    """
    Full preprocessing pipeline for a single traffic sign image:
      1. Resize to target size
      2. Convert to LAB color space
      3. Apply CLAHE (contrast-limited adaptive histogram equalization) on L channel
      4. Convert back to BGR → normalize to [0, 1]
    """
    img = cv2.resize(img, target_size)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l_eq  = clahe.apply(l)

    lab_eq = cv2.merge([l_eq, a, b])
    img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return img_eq.astype(np.float32) / 255.0


def preprocess_batch(images: np.ndarray, target_size=IMG_SIZE) -> np.ndarray:
    """Preprocess a numpy batch of images (N, H, W, C)."""
    return np.array([
        preprocess_image((img * 255).astype(np.uint8), target_size)
        for img in images
    ])


# ─── Augmentation Pipelines ───────────────────────────────────────────────────

def get_train_augmentor() -> ImageDataGenerator:
    """
    Aggressive augmentation for GTSRB training.
    NOTE: horizontal_flip=False — traffic signs are NOT horizontally symmetric!
    """
    return ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.25,
        width_shift_range=0.12,
        height_shift_range=0.12,
        shear_range=0.12,
        brightness_range=[0.6, 1.4],
        channel_shift_range=20.0,   # simulate different lighting conditions
        horizontal_flip=False,
        fill_mode="nearest",
    )


def get_val_augmentor() -> ImageDataGenerator:
    """Minimal augmentor for validation — only rescaling."""
    return ImageDataGenerator()   # images already normalized in preprocess_image


# ─── GTSRB Dataset Loader (real dataset) ─────────────────────────────────────

def load_gtsrb_from_dir(data_dir: str, split="train"):
    """
    Load GTSRB dataset from the standard Kaggle directory structure:

        data_dir/
          Train/
            0/   ← class folders
            1/
            ...
            42/
          Test/
            ...

    Returns:
        X: np.ndarray  (N, 32, 32, 3)  float32 in [0, 1]
        y: np.ndarray  (N,)             int labels
    """
    base = Path(data_dir) / split.capitalize()
    if not base.exists():
        raise FileNotFoundError(
            f"GTSRB directory not found: {base}\n"
            "Download from: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"
        )

    images, labels = [], []
    for class_dir in sorted(base.iterdir()):
        if not class_dir.is_dir():
            continue
        class_id = int(class_dir.name)
        for img_path in class_dir.glob("*.png"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            images.append(preprocess_image(img))
            labels.append(class_id)

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    print(f"  Loaded {len(X):,} images from {base} ({len(np.unique(y))} classes)")
    return X, y


# ─── Augmentation Visualization ───────────────────────────────────────────────

def visualize_augmentations(image: np.ndarray, n_aug=8, save_path=None):
    """Show original image alongside n_aug augmented versions."""
    import matplotlib.pyplot as plt

    aug = get_train_augmentor()
    img_batch  = image[np.newaxis]                  # (1, H, W, C)
    aug_images = [image]
    gen        = aug.flow(img_batch, batch_size=1, shuffle=False)
    for _ in range(n_aug):
        aug_images.append(next(gen)[0])

    fig, axes = plt.subplots(1, n_aug + 1, figsize=(2 * (n_aug + 1), 2.5),
                              facecolor="#111")
    titles = ["Original"] + [f"Aug {i+1}" for i in range(n_aug)]
    for ax, img, title in zip(axes, aug_images, titles):
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title, fontsize=7, color="white")
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, facecolor="#111", bbox_inches="tight")
    return fig
