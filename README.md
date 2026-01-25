# Tree Classifier App

木の画像をアップロードすると、種類や特徴を判定する  
Flask + TensorFlow を用いた画像分類Webアプリです。

---

## 概要

このアプリでは、アップロードされた木の画像に対して以下の順序で判定を行います。

1. 切り株かどうかを判定  
   - 切り株と判定された場合は、ここで処理が終了
2. クリスマスツリーかどうかを判定  
   - クリスマスツリーと判定された場合は、ここで処理が終了
3. 上記いずれにも該当しない「通常の木」の場合のみ、以下の特徴を判定します
   - 根が見えているかどうか
   - 幹が太めか細めか
   - 葉があるかどうか（枯れ木かどうか）

---

## 使用技術

- Python
- Flask
- TensorFlow / Keras
- OpenCV
- VGG16（特徴抽出）
- WSL（開発環境）

---

## 使い方

### 1. 仮想環境を有効化
```bash
source .venv/bin/activate
```

### 2. アプリを起動
```bash
python app.py
```

### 3. ブラウザでアクセス
http://127.0.0.1:5000

### 4. 画像をアップロード
PNG形式の画像をアップロードすると、判定結果が表示されます。

---

## ディレクトリ構成
tree_classifier_app/
├── app.py
├── test.py
├── test_images/
├── templates/
│   └── index.html
├── static/
│   ├── stylesheet.css
│   ├── img/
│   └── uploads/
├── *.keras
└── README.md

app.py：Flaskアプリ本体
test.py：精度検証用スクリプト
test_images/：検証用画像
static/uploads/：アップロード画像の一時保存
*.keras：学習済みモデル