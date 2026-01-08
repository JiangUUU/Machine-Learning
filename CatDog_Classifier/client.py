from config import MODEL_PATH, TESTSET_PATH
import os
import sys
from PIL import Image
import random
import torch
import torch.nn.functional as F
from torchvision import transforms
import timm
from PyQt6.QtWidgets import (
    QApplication, QLabel, QPushButton, QVBoxLayout, QWidget,
    QHBoxLayout, QSpinBox, QFormLayout, QMessageBox
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

device = "cuda" if torch.cuda.is_available() else "cpu"

def sample_test_images(cls_name, num):
    # load image from local test set
    if cls_name not in ["cats", "dogs"]:
        raise ValueError("cls_name must be 'cats' or 'dogs'")

    folder = os.path.join(TESTSET_PATH, cls_name)
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Test folder not found: {folder}")

    all_files = [os.path.join(folder, f) for f in os.listdir(folder)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not all_files:
        raise FileNotFoundError(f"No images found in {folder}")

    selected = random.sample(all_files, min(num, len(all_files)))
    return selected


model = timm.create_model("vit_base_patch16_224",
                          pretrained=False, num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]
        pred_idx = probs.argmax()
    return "cats" if pred_idx == 0 else "dogs", probs[pred_idx]


class ImageViewer(QWidget):
    def __init__(self, image_paths):
        super().__init__()
        self.image_paths = image_paths
        self.index = 0

        self.setMinimumSize(600, 500)
        self.setWindowTitle("Cats vs Dogs Classifier")

        self.label_img = QLabel()
        self.label_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_img.setScaledContents(True)
        self.label_img.setMinimumSize(400, 300)

        self.label_result = QLabel()
        self.label_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_result.setStyleSheet("font-size: 18px; font-weight: bold;")

        btn_prev = QPushButton("← Previous")
        btn_next = QPushButton("Next →")
        btn_prev.clicked.connect(self.prev_image)
        btn_next.clicked.connect(self.next_image)

        h_layout = QHBoxLayout()
        h_layout.addWidget(btn_prev)
        h_layout.addWidget(btn_next)

        layout = QVBoxLayout()
        layout.addWidget(self.label_img)
        layout.addWidget(self.label_result)
        layout.addLayout(h_layout)
        self.setLayout(layout)

        self.update_image()

    def update_image(self):
        if not self.image_paths:
            self.label_img.setText("No images to display.")
            self.label_result.setText("")
            return

        path = self.image_paths[self.index]
        pixmap = QPixmap(path)
        self.label_img.setPixmap(pixmap)

        pred, prob = predict_image(path)
        self.label_result.setText(f"Prediction: {pred} ({prob*100:.2f}%)")

    def next_image(self):
        self.index = (self.index + 1) % len(self.image_paths)
        self.update_image()

    def prev_image(self):
        self.index = (self.index - 1) % len(self.image_paths)
        self.update_image()


class SamplerUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sample from Test Set")
        self.setMinimumSize(400, 200)

        self.spin_cats = QSpinBox()
        self.spin_cats.setRange(0, 100)
        self.spin_cats.setValue(5)

        self.spin_dogs = QSpinBox()
        self.spin_dogs.setRange(0, 100)
        self.spin_dogs.setValue(5)

        btn_sample = QPushButton("Sample & Classify")
        btn_sample.clicked.connect(self.start_sampling)

        form_layout = QFormLayout()
        form_layout.addRow("Number of Cats:", self.spin_cats)
        form_layout.addRow("Number of Dogs:", self.spin_dogs)
        form_layout.addRow(btn_sample)

        self.setLayout(form_layout)

    def start_sampling(self):
        cat_num = self.spin_cats.value()
        dog_num = self.spin_dogs.value()

        if cat_num == 0 and dog_num == 0:
            QMessageBox.warning(
                self, "Warning", "Please select at least one image!")
            return

        all_images = []
        try:
            if cat_num > 0:
                cat_paths = sample_test_images("cats", cat_num)
                all_images.extend(cat_paths)
            if dog_num > 0:
                dog_paths = sample_test_images("dogs", dog_num)
                all_images.extend(dog_paths)
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to load test images:\n{str(e)}")
            return

        if not all_images:
            QMessageBox.warning(
                self, "Warning", "No images found in test set!")
            return

        # 打乱顺序（可选，让猫狗混合显示）
        random.shuffle(all_images)

        self.viewer = ImageViewer(all_images)
        self.viewer.show()
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SamplerUI()
    window.show()
    app.exec()
