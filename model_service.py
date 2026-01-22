import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")      # GPU探しを抑制（ログ減）
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")     # TFログ抑制

import numpy as np
import cv2
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "")          # 例: stump_classifier.keras
CLASS_NAMES = os.environ.get("CLASS_NAMES", "")        # 例: normal,stump
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))  # 2値sigmoid用
API_KEY = os.environ.get("API_KEY", "")                # 任意（空なら認証なし）

def _resolve_model_path(p: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.join(app.root_path, p)

model_abs_path = _resolve_model_path(MODEL_PATH)
if not MODEL_PATH or not os.path.exists(model_abs_path):
    raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH} (abs: {model_abs_path})")

classes = [c.strip() for c in CLASS_NAMES.split(",") if c.strip()]
if len(classes) < 2:
    raise ValueError("CLASS_NAMES must be like 'normal,stump'")

# ★ 1サービス=1モデル：起動時に1回だけロード（workers=1推奨）
model = load_model(model_abs_path)

def preprocess_image_bytes(file_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("画像の読み込みに失敗しました。")
    img = cv2.resize(img, (224, 224))
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    return x

def predict_label(x: np.ndarray):
    y = model.predict(x, verbose=0)
    y = np.array(y)

    # sigmoid（二値想定）
    if (y.ndim == 2 and y.shape[1] == 1) or (y.ndim == 1 and y.shape[0] == 1):
        p1 = float(y.reshape(-1)[0])
        idx = 1 if p1 >= THRESHOLD else 0
        conf = p1 if idx == 1 else (1.0 - p1)
        return classes[idx], conf

    # softmax
    probs = y[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    return classes[idx], conf

@app.get("/health")
def health():
    return "ok", 200

@app.post("/predict")
def predict():
    if API_KEY and request.headers.get("X-API-KEY") != API_KEY:
        return jsonify({"error": "unauthorized"}), 401

    f = request.files.get("file")
    if not f:
        return jsonify({"error": "file is required"}), 400

    x = preprocess_image_bytes(f.read())
    label, conf = predict_label(x)
    return jsonify({"label": label, "conf": conf})
