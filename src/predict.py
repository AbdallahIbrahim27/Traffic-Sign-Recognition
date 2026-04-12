"""
predict.py — Run inference on single images or a folder of images
"""

import argparse
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

from src.class_mapping import decode_prediction_index
from src.preprocessing import preprocess_for_inference

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


def predict_image(model_path: str, image_path: str) -> dict:
    """
    Load a saved model and predict a single image.

    Returns:
        {
            "class_id":    int,
            "class_name":  str,
            "confidence":  float,
            "top5":        [(class_id, class_name, confidence), ...]
        }
    """
    model = tf.keras.models.load_model(model_path)
    img   = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    processed = preprocess_for_inference(img)[np.newaxis]  # (1, 32, 32, 3) RGB [0,1]
    probs     = model.predict(processed, verbose=0)[0]
    top5_raw  = np.argsort(probs)[::-1][:5]

    top1_gtsrb = decode_prediction_index(int(top5_raw[0]))
    return {
        "class_id":   top1_gtsrb,
        "class_name": CLASS_NAMES[top1_gtsrb],
        "confidence": float(probs[top5_raw[0]]),
        "top5": [
            (decode_prediction_index(int(i)), CLASS_NAMES[decode_prediction_index(int(i))], float(probs[i]))
            for i in top5_raw
        ],
    }


def predict_folder(model_path: str, folder: str) -> list[dict]:
    """Predict all images in a folder."""
    model  = tf.keras.models.load_model(model_path)
    folder = Path(folder)
    results = []

    for img_path in sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg")):
        img  = cv2.imread(str(img_path))
        if img is None:
            continue
        proc  = preprocess_for_inference(img)[np.newaxis]
        probs = model.predict(proc, verbose=0)[0]
        raw   = int(np.argmax(probs))
        top1  = decode_prediction_index(raw)
        results.append({
            "file":       img_path.name,
            "class_id":   top1,
            "class_name": CLASS_NAMES[top1],
            "confidence": float(probs[raw]),
        })
        print(f"  {img_path.name:<30} → {CLASS_NAMES[top1]:<35} ({probs[raw]*100:.1f}%)")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Sign Inference")
    parser.add_argument("--model",  required=True, help="Path to saved .keras model")
    parser.add_argument("--image",  default=None,  help="Single image to predict")
    parser.add_argument("--folder", default=None,  help="Folder of images")
    args = parser.parse_args()

    if args.image:
        result = predict_image(args.model, args.image)
        print(f"\n  Prediction: {result['class_name']}")
        print(f"  Confidence: {result['confidence']*100:.1f}%")
        print("\n  Top-5:")
        for cid, cname, conf in result["top5"]:
            print(f"    {conf*100:5.1f}%  {cname}")

    elif args.folder:
        results = predict_folder(args.model, args.folder)
        print(f"\n  Processed {len(results)} images.")
    else:
        print("Provide --image or --folder.")
