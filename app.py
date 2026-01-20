import os
import numpy as np
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2


app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(app.root_path, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

STUMP_MODEL_PATH = "stump_classifier.keras"   # stump / normal
XMAS_MODEL_PATH = "xmas_classifier.keras"   # xmas / normal
ROOT_MODEL_PATH = "root_classifier.keras"    # root_yes / root_no
TRUNK_MODEL_PATH = "trunk_classifier.keras"  # thick / thin
DEADTREE_MODEL_PATH = "deadtree_classifier.keras"   # deadtree / normal

stump_model = load_model(STUMP_MODEL_PATH) if os.path.exists(STUMP_MODEL_PATH) else None
xmas_model = load_model(XMAS_MODEL_PATH) if os.path.exists(XMAS_MODEL_PATH) else None
root_model = load_model(ROOT_MODEL_PATH) if os.path.exists(ROOT_MODEL_PATH) else None
trunk_model = load_model(TRUNK_MODEL_PATH) if os.path.exists(TRUNK_MODEL_PATH) else None
deadtree_model = load_model(DEADTREE_MODEL_PATH) if os.path.exists(DEADTREE_MODEL_PATH) else None

STUMP_CLASSES = ["normal", "stump"]
XMAS_CLASSES = ["normal", "xmas"]
ROOT_CLASSES = ["root_no", "root_yes"]
TRUNK_CLASSES = ["thin", "thick"]
DEADTREE_CLASSES = ["normal", "deadtree"]


def preprocess_image(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("画像の読み込みに失敗しました。")
    img = cv2.resize(img, (224, 224))
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    return x


def predict_label(model, x: np.ndarray, class_names: list[str], threshold: float = 0.5) -> tuple[str, float]:
    y = model.predict(x, verbose=0)
    y = np.array(y)
    
    # sigmoid
    if (y.ndim == 2 and y.shape[1] == 1) or (y.ndim == 1 and y.shape[0] == 1):
        p1 = float(y.reshape(-1)[0])
        idx = 1 if p1 >= threshold else 0
        conf = p1 if idx == 1 else (1.0 - p1)
        return class_names[idx], conf
    
    # softmax(※現時点では使用なし)
    probs = y[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    return class_names[idx], conf


def ensure_models():
    missing = []
    if stump_model is None: missing.append(STUMP_MODEL_PATH)
    if xmas_model is None: missing.append(XMAS_MODEL_PATH)
    if root_model is None: missing.append(ROOT_MODEL_PATH)
    if trunk_model is None: missing.append(TRUNK_MODEL_PATH)
    if deadtree_model is None: missing.append(DEADTREE_MODEL_PATH)
    if missing:
        raise FileNotFoundError("Missing models: " + ", ".join(missing))


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", result=None)

    file = request.files.get("file")
    if not file or file.filename == "":
        return render_template("index.html", result={"error": "画像ファイルが選択されていません。"})

    filename = secure_filename(file.filename)
    if not filename.lower().endswith(".png"):
        return render_template(
            "index.html",
            result={"error": "PNG形式の画像をアップロードしてください。"}
        )
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    uploaded_url = url_for("static", filename=f"uploads/{filename}")

    try:
        ensure_models()

        x = preprocess_image(save_path)

        # stump判定
        stump_label, _ = predict_label(stump_model, x, STUMP_CLASSES)

        if stump_label == "stump":
            result = {
                "is_stump": True,
                "is_xmas": None,
                "stump_label": "切り株",
                "xmas_label": None,
                "root_label": None,
                "trunk_label": None,
                "leaves_label": None,
                "uploaded_url": uploaded_url,
                "error": None,
            }
            return render_template("index.html", result=result)

        # xmas判定
        xmas_label, _ = predict_label(xmas_model, x, XMAS_CLASSES)

        if xmas_label == "xmas":
            result = {
                "is_stump": False,
                "is_xmas": True,
                "stump_label": None,
                "xmas_label": "クリスマスツリー",
                "root_label": None,
                "trunk_label": None,
                "leaves_label": None,
                "uploaded_url": uploaded_url,
                "error": None,
            }
            return render_template("index.html", result=result)

        # normal の場合だけ root + trunk + leaves 判定
        root_label, _ = predict_label(root_model, x, ROOT_CLASSES)
        trunk_label, _ = predict_label(trunk_model, x, TRUNK_CLASSES)
        leaves_label, _ = predict_label(deadtree_model, x, DEADTREE_CLASSES)

        root_jp = "あり" if root_label == "root_yes" else "なし"
        trunk_jp = "太め" if trunk_label == "thick" else "細め"
        leaves_jp = "なし" if leaves_label == "deadtree" else "あり"

        result = {
            "is_stump": False,
            "is_xmas": False,
            "stump_label": None,
            "xmas_label": None,
            "root_label": root_jp,
            "trunk_label": trunk_jp,
            "leaves_label": leaves_jp,
            "uploaded_url": uploaded_url,
            "error": None,
        }
        return render_template("index.html", result=result)

    except Exception as e:
        return render_template(
            "index.html",
            result={"error": f"エラーメッセージ: {e}", "uploaded_url": uploaded_url}
        )

if __name__ == "__main__":
    app.run(debug=True)
