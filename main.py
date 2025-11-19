# Image Processing with Deep Learning  -  https://github.com/alizangeneh

import sys
import os

import cv2
import numpy as np
from PIL import Image

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

# اختیاری / بر اساس نیاز:
# pip install rembg torch torchvision realesrgan
try:
    from rembg import remove as rembg_remove
except ImportError:
    rembg_remove = None

try:
    import torch
    from realesrgan import RealESRGAN
except ImportError:
    RealESRGAN = None
    torch = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

FACE_PROTO = os.path.join(MODELS_DIR, "deploy.prototxt")
FACE_MODEL = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
ESRGAN_WEIGHTS = os.path.join(MODELS_DIR, "RealESRGAN_x4plus.pth")


class ImageLabel(QLabel):
    """Label برای نمایش تصویر + drag & drop"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setText("Drop an image here\nor use the Open button")
        self.setStyleSheet(
            "QLabel { border: 2px dashed #888; color: #aaa; font-size: 14px; }"
        )

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isfile(path):
                self.parent().load_image(path)
                break


class ImageToolApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Smart Image Tool - Deep Learning Edition")
        self.resize(900, 600)

        self.image_label = ImageLabel(self)
        self.current_image = None
        self.current_image_has_alpha = False

        open_btn = QPushButton("Open")
        save_btn = QPushButton("Save As...")

        compress_btn = QPushButton("⭐ Smart Image Compressor")
        sr_btn = QPushButton("Quality Booster (Super-Resolution)")
        bg_btn = QPushButton("⭐ Background Removal")
        face_blur_btn = QPushButton("⭐ Face Blurring Privacy Tool")

        open_btn.clicked.connect(self.open_image_dialog)
        save_btn.clicked.connect(self.save_image_dialog)

        compress_btn.clicked.connect(self.smart_compress)
        sr_btn.clicked.connect(self.quality_boost)
        bg_btn.clicked.connect(self.remove_background)
        face_blur_btn.clicked.connect(self.blur_faces)

        top_buttons = QHBoxLayout()
        top_buttons.addWidget(open_btn)
        top_buttons.addWidget(save_btn)
        top_buttons.addStretch()

        actions_layout = QVBoxLayout()
        actions_layout.addWidget(compress_btn)
        actions_layout.addWidget(sr_btn)
        actions_layout.addWidget(bg_btn)
        actions_layout.addWidget(face_blur_btn)
        actions_layout.addStretch()

        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        left_layout.addLayout(top_buttons)
        left_layout.addWidget(self.image_label)

        main_layout.addLayout(left_layout, stretch=3)
        main_layout.addLayout(actions_layout, stretch=1)

        self.setLayout(main_layout)

        self.face_net = None
        self.sr_model = None

    def set_image(self, img: np.ndarray, has_alpha: bool = False):
        if img is None:
            return

        self.current_image = img
        self.current_image_has_alpha = has_alpha

        if has_alpha:
            if img.shape[2] == 4:
                img_rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            else:
                img_rgba = img
            h, w, ch = img_rgba.shape
            bytes_per_line = ch * w
            q_img = QImage(
                img_rgba.data, w, h, bytes_per_line, QImage.Format_RGBA8888
            )
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(
                img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888
            )

        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )
        self.image_label.setStyleSheet("QLabel { border: none; }")
        self.image_label.setText("")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_image is not None:
            self.set_image(self.current_image, self.current_image_has_alpha)

    def open_image_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if file_path:
            self.load_image(file_path)

    def load_image(self, path: str):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            QMessageBox.warning(self, "Error", "Cannot load this image.")
            return

        has_alpha = (img.ndim == 3 and img.shape[2] == 4)
        self.set_image(img, has_alpha)

    def save_image_dialog(self):
        if self.current_image is None:
            QMessageBox.information(self, "Info", "No image to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image As",
            "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;WEBP (*.webp);;BMP (*.bmp)",
        )
        if not file_path:
            return

        ext = os.path.splitext(file_path)[1].lower()
        img_to_save = self.current_image

        if self.current_image_has_alpha and ext in [".jpg", ".jpeg", ".bmp"]:
            bg = np.ones_like(self.current_image[:, :, :3]) * 255
            alpha = self.current_image[:, :, 3:4] / 255.0
            img_to_save = (alpha * self.current_image[:, :, :3] + (1 - alpha) * bg).astype(
                np.uint8
            )

        if ext in [".jpg", ".jpeg"]:
            cv2.imwrite(file_path, img_to_save, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        else:
            cv2.imwrite(file_path, img_to_save)

        QMessageBox.information(self, "Saved", f"Image saved to:\n{file_path}")

    def smart_compress(self):
        if self.current_image is None:
            QMessageBox.information(self, "Info", "Load an image first.")
            return

        img = self.current_image.copy()
        max_dim = 2048
        h, w = img.shape[:2]
        scale = min(1.0, max_dim / max(h, w))
        if scale < 1.0:
            img = cv2.resize(
                img,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        self.set_image(img, has_alpha=self.current_image_has_alpha)
        QMessageBox.information(
            self,
            "Smart Compress",
            "Image resized to a reasonable resolution.\n"
            "Save as JPEG with quality 90 for good compression.",
        )

    def load_sr_model(self):
        if self.sr_model is not None:
            return

        if RealESRGAN is None or torch is None:
            QMessageBox.warning(
                self,
                "Error",
                "RealESRGAN / torch not installed.\n"
                "Install with: pip install torch torchvision realesrgan",
            )
            return

        if not os.path.isfile(ESRGAN_WEIGHTS):
            QMessageBox.warning(
                self,
                "Error",
                f"ESRGAN weights not found:\n{ESRGAN_WEIGHTS}",
            )
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model = RealESRGAN(device, scale=4)
            model.load_weights(ESRGAN_WEIGHTS, download=False)
            self.sr_model = model
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def quality_boost(self):
        if self.current_image is None:
            QMessageBox.information(self, "Info", "Load an image first.")
            return

        self.load_sr_model()
        if self.sr_model is None:
            return

        if self.current_image_has_alpha:
            img = self.current_image
            bg = np.ones_like(img[:, :, :3]) * 255
            alpha = img[:, :, 3:4] / 255.0
            rgb = (alpha * img[:, :, :3] + (1 - alpha) * bg).astype(np.uint8)
        else:
            rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(rgb)

        try:
            with torch.no_grad():
                sr_img = self.sr_model.predict(pil_img)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return

        sr_np = np.array(sr_img)
        self.set_image(cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR), has_alpha=False)

        QMessageBox.information(
            self, "Quality Booster", "Super-resolution applied using RealESRGAN (x4)."
        )

    def remove_background(self):
        if self.current_image is None:
            QMessageBox.information(self, "Info", "Load an image first.")
            return

        if rembg_remove is None:
            QMessageBox.warning(
                self,
                "Error",
                "rembg is not installed.\nInstall with: pip install rembg",
            )
            return

        if self.current_image_has_alpha:
            pil_img = Image.fromarray(cv2.cvtColor(self.current_image, cv2.COLOR_BGRA2RGBA))
        else:
            pil_img = Image.fromarray(cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB))

        try:
            out = rembg_remove(pil_img)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return

        bgra = cv2.cvtColor(np.array(out), cv2.COLOR_RGBA2BGRA)
        self.set_image(bgra, has_alpha=True)
        QMessageBox.information(
            self,
            "Background Removal",
            "Background removed using a deep learning model (rembg).",
        )

    def load_face_net(self):
        if self.face_net is not None:
            return

        if not (os.path.isfile(FACE_PROTO) and os.path.isfile(FACE_MODEL)):
            QMessageBox.warning(
                self,
                "Error",
                f"Face detection model not found:\n{FACE_PROTO}\n{FACE_MODEL}",
            )
            return

        try:
            self.face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def blur_faces(self):
        if self.current_image is None:
            QMessageBox.information(self, "Info", "Load an image first.")
            return

        self.load_face_net()
        if self.face_net is None:
            return

        if self.current_image_has_alpha:
            img_bgr = cv2.cvtColor(self.current_image, cv2.COLOR_BGRA2BGR)
        else:
            img_bgr = self.current_image.copy()

        (h, w) = img_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img_bgr, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
        )

        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        output = img_bgr.copy()

        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf < 0.5:
                continue

            (x1, y1, x2, y2) = (
                detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            ).astype(int)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            face = output[y1:y2, x1:x2]
            if face.size == 0:
                continue

            k = max(15, (min(face.shape[:2]) // 3) | 1)
            output[y1:y2, x1:x2] = cv2.GaussianBlur(face, (k, k), 0)

        if self.current_image_has_alpha:
            alpha = self.current_image[:, :, 3:4]
            out_bgra = cv2.cvtColor(output, cv2.COLOR_BGR2BGRA)
            out_bgra[:, :, 3:4] = alpha
            self.set_image(out_bgra, has_alpha=True)
        else:
            self.set_image(output, has_alpha=False)

        QMessageBox.information(self, "Face Blurring", "Faces blurred successfully.")


def main():
    app = QApplication(sys.argv)
    win = ImageToolApp()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
