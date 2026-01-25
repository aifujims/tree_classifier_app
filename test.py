import os
import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input


# Model paths
STUMP_MODEL_PATH = "stump_classifier.keras"          # normal / stump
XMAS_MODEL_PATH = "xmas_classifier.keras"            # normal / xmas
ROOT_MODEL_PATH = "root_classifier.keras"            # root_no / root_yes
TRUNK_MODEL_PATH = "trunk_classifier.keras"          # thin / thick
DEADTREE_MODEL_PATH = "deadtree_classifier.keras"    # normal / deadtree


# class order
CLASS_NAMES = {
    "stump": ["normal", "stump"],
    "xmas": ["normal", "xmas"],
    "root": ["root_no", "root_yes"],
    "trunk": ["thin", "thick"],
    "deadtree": ["normal", "deadtree"],
}


# which label is treated as YES
POSITIVE_LABEL = {
    "stump": "stump",
    "xmas": "xmas",
    "root": "root_yes",
    "trunk": "thick",
    "deadtree": "deadtree",
}


# Utils
def preprocess_image(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"画像の読み込みに失敗しました: {img_path}")
    img = cv2.resize(img, (224, 224))
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x.astype(np.float32))
    return x


def load_models() -> dict:
    paths = {
        "stump": STUMP_MODEL_PATH,
        "xmas": XMAS_MODEL_PATH,
        "root": ROOT_MODEL_PATH,
        "trunk": TRUNK_MODEL_PATH,
        "deadtree": DEADTREE_MODEL_PATH,
    }
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing models: " + ", ".join(missing))

    return {k: load_model(v) for k, v in paths.items()}


def iter_images(path: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    if path.is_file():
        return [path]
    if path.is_dir():
        return [p for p in sorted(path.rglob("*")) if p.is_file() and p.suffix.lower() in exts]
    raise FileNotFoundError(f"Not found: {path}")


def prob_yes(model, x: np.ndarray, class_names: list[str], positive_label: str) -> float:
    """
    Return probability of positive_label, robust to:
      - sigmoid binary output: (1,) or (1,1)
      - softmax binary output: (2,) or (1,2)
    """
    y = model.predict(x, verbose=0)
    y = np.array(y)

    # sigmoid: size 1
    if y.size == 1:
        p1 = float(y.reshape(-1)[0])  # prob of class_names[1]
        if positive_label == class_names[1]:
            return p1
        if positive_label == class_names[0]:
            return 1.0 - p1
        raise ValueError(f"positive_label not in class_names: {positive_label} not in {class_names}")

    # softmax: size == num_classes (here typically 2)
    probs = y.reshape(-1)
    if probs.size != len(class_names):
        raise ValueError(f"Unexpected output size: {probs.size}, expected {len(class_names)} for {class_names}")

    try:
        idx = class_names.index(positive_label)
    except ValueError:
        raise ValueError(f"positive_label not in class_names: {positive_label} not in {class_names}") from None

    return float(probs[idx])


def predict_yesno_and_conf(model, x: np.ndarray, key: str, threshold: float) -> tuple[str, float]:
    """
    Returns: (YES/NO, conf)

    conf is the probability of the predicted side:
      - if YES: conf = P(YES)
      - if NO : conf = P(NO) = 1 - P(YES)
    This matches UI intuition better.
    """
    p_yes = prob_yes(model, x, CLASS_NAMES[key], POSITIVE_LABEL[key])
    if p_yes >= threshold:
        return "YES", p_yes
    return "NO", 1.0 - p_yes


# Main
def main():
    parser = argparse.ArgumentParser(
        description="TreeClassifier test runner (print YES/NO + confidence for each model)."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="test_images",
        help="Image file or directory (default: test_images)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for YES/NO (default: 0.5)",
    )
    parser.add_argument(
        "--show-shape",
        action="store_true",
        help="Print each model output_shape once (debug)",
    )
    args = parser.parse_args()

    target = Path(args.path)
    images = iter_images(target)
    if not images:
        print("No images found.")
        sys.exit(0)

    models = load_models()

    if args.show_shape:
        print("=== model output_shape (debug) ===")
        for k, m in models.items():
            print(f"{k:8s}: {m.output_shape}")
        print("=================================")
        print()

    for img_path in images:
        try:
            x = preprocess_image(str(img_path))

            stump_yn, stump_conf = predict_yesno_and_conf(models["stump"], x, "stump", args.threshold)
            xmas_yn, xmas_conf = predict_yesno_and_conf(models["xmas"], x, "xmas", args.threshold)
            root_yn, root_conf = predict_yesno_and_conf(models["root"], x, "root", args.threshold)
            trunk_yn, trunk_conf = predict_yesno_and_conf(models["trunk"], x, "trunk", args.threshold)
            dead_yn, dead_conf = predict_yesno_and_conf(models["deadtree"], x, "deadtree", args.threshold)

            print(f"Image: {img_path}")
            print(f"stump     : {stump_yn:>3} (conf={stump_conf:.2f})")
            print(f"xmas      : {xmas_yn:>3} (conf={xmas_conf:.2f})")
            print(f"root      : {root_yn:>3} (conf={root_conf:.2f})")
            print(f"trunk     : {trunk_yn:>3} (conf={trunk_conf:.2f})")
            print(f"deadtree  : {dead_yn:>3} (conf={dead_conf:.2f})")
            print()

        except Exception as e:
            print(f"Image: {img_path}")
            print(f"ERROR: {e}")
            print()


if __name__ == "__main__":
    main()

