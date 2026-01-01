from flask import Flask, request, jsonify
from ultralytics import YOLO
import base64
import cv2
import numpy as np
import os
import uuid
import yaml

app = Flask(__name__)

# ---------------- 配置 ----------------
MODEL_PATH = "yolov8m.pt"
TRAINED_MODEL_PATH = "yolov8m_aimlab.pt"
DATASET_DIR = "dataset"
IMG_SIZE = 640
EPOCHS = 1

CLASS_NAMES = ["aimlab_blueball"]  # 你的 Label Studio 类别
label_name_to_id = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# 加载模型
if os.path.exists(TRAINED_MODEL_PATH):
    model = YOLO(TRAINED_MODEL_PATH)
else:
    model = YOLO(MODEL_PATH)

# ---------------- 健康检查 ----------------
@app.route("/predict/health", methods=["GET"])
def health():
    return jsonify({"status":"ok"})

@app.route("/health", methods=["GET"])
def health_alt():
    return jsonify({"status":"ok"})

# ---------------- /setup ----------------
@app.route("/setup", methods=["POST"])
def setup():
    return jsonify({
        "instructions": "YOLOv8 object detection backend",
        "labels": [{"name": CLASS_NAMES[0], "id": 0}],
        "task": "object_detection"
    })

# ---------------- /predict ----------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    image_b64 = data.get("image")
    if not image_b64:
        return jsonify({"error": "No image provided"}), 400

    image_bytes = base64.b64decode(image_b64.split(",")[-1])
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    H, W = img.shape[:2]

    results = model(img)
    predictions = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0]
            x = float((x1+x2)/2/W)
            y = float((y1+y2)/2/H)
            width = float((x2-x1)/W)
            height = float((y2-y1)/H)
            predictions.append({
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "label": str(cls_id)
            })
    return jsonify(predictions)

# ---------------- /train ----------------
@app.route("/train", methods=["POST"])
def train():
    data = request.json
    annotations = data.get("annotations", [])

    images_dir = os.path.join(DATASET_DIR, "images")
    labels_dir = os.path.join(DATASET_DIR, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for item in annotations:
        img_b64 = item.get("image")
        anns = item.get("annotations", [])
        if not img_b64 or not anns:
            continue

        # 保存图片
        img_bytes = base64.b64decode(img_b64.split(",")[-1])
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img_name = f"{uuid.uuid4().hex}.jpg"
        img_path = os.path.join(images_dir, img_name)
        cv2.imwrite(img_path, img)
        H, W = img.shape[:2]

        # 保存 YOLO 标签
        label_path = os.path.join(labels_dir, img_name.replace(".jpg", ".txt"))
        with open(label_path, "w") as f:
            for ann in anns:
                cls_name = ann["label"]
                cls_id = label_name_to_id.get(cls_name, 0)
                x = ann["x"]
                y = ann["y"]
                width = ann["width"]
                height = ann["height"]
                f.write(f"{cls_id} {x} {y} {width} {height}\n")

    # 创建 YOLO 数据集 yaml
    dataset_yaml = os.path.join(DATASET_DIR, "dataset.yaml")
    data_yaml = {
        "train": images_dir,
        "val": images_dir,
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES
    }
    with open(dataset_yaml, "w") as f:
        yaml.dump(data_yaml, f)

    # 增量训练
    model.train(data=dataset_yaml, epochs=EPOCHS, imgsz=IMG_SIZE, exist_ok=True)
    model.save(TRAINED_MODEL_PATH)

    return jsonify({"status": "ok", "message": "Model fine-tuned successfully"})

# ---------------- 主程序 ----------------
if __name__ == "__main__":
    print("ML Backend running on http://localhost:9090")
    app.run(host="0.0.0.0", port=9090)
