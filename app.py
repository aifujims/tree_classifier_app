import os
import requests
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(app.root_path, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ====== Renderの環境変数で入れる（後で設定） ======
STUMP_URL = os.environ.get("STUMP_URL", "")       # 例: http://stump:10000/predict など
XMAS_URL = os.environ.get("XMAS_URL", "")
ROOT_URL = os.environ.get("ROOT_URL", "")
TRUNK_URL = os.environ.get("TRUNK_URL", "")
DEADTREE_URL = os.environ.get("DEADTREE_URL", "")
API_KEY = os.environ.get("API_KEY", "")           # model側と揃える（空でもOK）

def call_model(url: str, img_path: str) -> tuple[str, float]:
    if not url:
        raise ValueError("モデルサービスURLが未設定です（環境変数を確認してください）。")

    headers = {}
    if API_KEY:
        headers["X-API-KEY"] = API_KEY

    with open(img_path, "rb") as f:
        files = {"file": (os.path.basename(img_path), f, "image/png")}
        r = requests.post(url, files=files, headers=headers, timeout=(5, 240))  # 接続5秒/応答240秒
    r.raise_for_status()
    data = r.json()
    if "label" not in data:
        raise RuntimeError(f"予期しない応答: {data}")
    return data["label"], float(data.get("conf", 0.0))

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", result=None)

    file = request.files.get("file")
    if not file or file.filename == "":
        return render_template("index.html", result={"error": "画像ファイルが選択されていません。"})

    filename = secure_filename(file.filename)
    if not filename.lower().endswith(".png"):
        return render_template("index.html", result={"error": "PNG形式の画像をアップロードしてください。"})

    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)
    uploaded_url = url_for("static", filename=f"uploads/{filename}")

    try:
        # 1) stump
        stump_label, _ = call_model(STUMP_URL, save_path)
        if stump_label == "stump":
            return render_template("index.html", result={
                "is_stump": True,
                "is_xmas": None,
                "stump_label": "切り株",
                "xmas_label": None,
                "root_label": None,
                "trunk_label": None,
                "leaves_label": None,
                "uploaded_url": uploaded_url,
                "error": None,
            })

        # 2) xmas
        xmas_label, _ = call_model(XMAS_URL, save_path)
        if xmas_label == "xmas":
            return render_template("index.html", result={
                "is_stump": False,
                "is_xmas": True,
                "stump_label": None,
                "xmas_label": "クリスマスツリー",
                "root_label": None,
                "trunk_label": None,
                "leaves_label": None,
                "uploaded_url": uploaded_url,
                "error": None,
            })

        # 3) normal → root + trunk + leaves
        root_label, _ = call_model(ROOT_URL, save_path)
        trunk_label, _ = call_model(TRUNK_URL, save_path)
        leaves_label, _ = call_model(DEADTREE_URL, save_path)

        root_jp = "あり" if root_label == "root_yes" else "なし"
        trunk_jp = "太め" if trunk_label == "thick" else "細め"
        leaves_jp = "なし" if leaves_label == "deadtree" else "あり"

        return render_template("index.html", result={
            "is_stump": False,
            "is_xmas": False,
            "stump_label": None,
            "xmas_label": None,
            "root_label": root_jp,
            "trunk_label": trunk_jp,
            "leaves_label": leaves_jp,
            "uploaded_url": uploaded_url,
            "error": None,
        })

    except Exception as e:
        return render_template("index.html", result={
            "error": f"エラーメッセージ: {e}",
            "uploaded_url": uploaded_url
        })

if __name__ == "__main__":
    app.run(debug=True)
