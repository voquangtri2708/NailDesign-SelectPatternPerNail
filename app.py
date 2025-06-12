import tkinter as tk
from tkinter import filedialog, colorchooser
from PIL import Image, ImageTk
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
import os

load_dotenv()
key_roboflow = os.getenv("ROBLOFLOW_API_KEY")
if not key_roboflow:
    raise ValueError("Vui lòng đặt biến môi trường ROBLOFLOW_API_KEY với giá trị API key của bạn.")
# Đảm bảo rằng bạn đã cài đặt các thư viện cần thiết:
# pip install opencv-python numpy pillow inference-sdk python-dotenv    

# Khởi tạo client với URL và API key
# Bạn cần thay thế giá trị này bằng API key của bạn
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=key_roboflow
)


class NailColorPatternApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nail Segmentation - Color or Pattern")

        self.original_image = None
        self.display_image = None
        self.polygons = []

        self.alpha = 0.6
        self.selected_color = (0, 0, 255)  # Red by default (BGR)
        self.pattern_image = None
        self.use_pattern = False

        # UI
        self.canvas = tk.Canvas(root, width=500, height=500)
        self.canvas.pack()

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="📁 Chọn Ảnh & Segment", command=self.load_and_segment).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="🎨 Chọn Màu", command=self.choose_color).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="🖼️ Chọn Họa Tiết", command=self.load_pattern).pack(side=tk.LEFT, padx=5)

        alpha_frame = tk.Frame(root)
        alpha_frame.pack(pady=5)

        tk.Label(alpha_frame, text="Độ trong suốt (Alpha): ").pack(side=tk.LEFT)
        self.alpha_scale = tk.Scale(alpha_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.update_alpha)
        self.alpha_scale.set(int(self.alpha * 100))
        self.alpha_scale.pack(side=tk.LEFT)

    def load_and_segment(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if not file_path:
            return
        self.original_image = cv2.imread(file_path)
        if self.original_image is None:
            print("Không đọc được ảnh")
            return

        result = CLIENT.infer(file_path, model_id="nail_segmentation-hejrk-tvknk/1")
        self.polygons = []
        for pred in result['predictions']:
            points = pred['points']
            pts = [(int(p['x']), int(p['y'])) for p in points]
            pts = cv2.convexHull(np.array(pts)).squeeze()
            if pts.ndim == 1:
                pts = [tuple(pts)]
            else:
                pts = [tuple(pt) for pt in pts]
            self.polygons.append(pts)

        self.display_image = self.original_image.copy()
        self.show_image(self.display_image)

    def choose_color(self):
        color_rgb, _ = colorchooser.askcolor(title="Chọn màu móng")
        if color_rgb is None:
            return
        self.selected_color = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))  # RGB → BGR
        self.use_pattern = False
        self.apply_overlay()

    def load_pattern(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if not file_path:
            return
        pattern = cv2.imread(file_path)
        if pattern is None:
            print("Không đọc được họa tiết")
            return
        self.pattern_image = pattern
        self.use_pattern = True
        self.apply_overlay()

    def update_alpha(self, val):
        self.alpha = int(val) / 100
        self.apply_overlay()

    def apply_overlay(self):
        if self.original_image is None or not self.polygons:
            return

        overlay = self.original_image.copy()

        for poly in self.polygons:
            pts = np.array(poly, np.int32).reshape((-1, 1, 2))
            x, y, w, h = cv2.boundingRect(pts)

            if self.use_pattern and self.pattern_image is not None:
                # Resize pattern to fit the nail region
                resized_pattern = cv2.resize(self.pattern_image, (w, h))
                mask = np.zeros((h, w), dtype=np.uint8)
                local_pts = pts - [x, y]
                cv2.fillPoly(mask, [local_pts], 255)

                roi = overlay[y:y+h, x:x+w]
                bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
                fg = cv2.bitwise_and(resized_pattern, resized_pattern, mask=mask)
                combined = cv2.add(bg, fg)
                overlay[y:y+h, x:x+w] = combined
            else:
                # Fill nail region with solid color
                cv2.fillPoly(overlay, [pts], self.selected_color)

        self.display_image = cv2.addWeighted(overlay, self.alpha, self.original_image, 1 - self.alpha, 0)
        self.show_image(self.display_image)

    def show_image(self, img_bgr):
        h, w = img_bgr.shape[:2]
        max_dim = 500
        scale = max_dim / max(h, w)
        img_resized = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))

        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        self.tk_img = ImageTk.PhotoImage(img_pil)

        self.canvas.config(width=img_pil.width, height=img_pil.height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

if __name__ == "__main__":
    root = tk.Tk()
    app = NailColorPatternApp(root)
    root.mainloop()
# This code is a simple GUI application for nail segmentation and coloring using Roboflow's inference SDK.
# It allows users to load an image, segment nails, choose a color or pattern, and apply overlays with adjustable transparency.
# The application uses OpenCV for image processing and Tkinter for the GUI.
# Ensure you have the required libraries installed: tkinter, opencv-python, numpy, pillow, and roboflow.
# Make sure to replace the API key and URL with your own Roboflow credentials.
# The application supports loading images, selecting colors, and applying patterns to segmented nail regions.
# Note: The Roboflow model ID should be replaced with your own model ID for nail segmentation.
# Ensure you have the required libraries installed: tkinter, opencv-python, numpy, pillow, and roboflow.
# Make sure to replace the API key and URL with your own Roboflow credentials.  
# The application supports loading images, selecting colors, and applying patterns to segmented nail regions.