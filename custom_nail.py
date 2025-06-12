import tkinter as tk
import cv2
import os
from tkinter import filedialog, colorchooser
from PIL import Image, ImageTk
import numpy as np
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv


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


class NailDesignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nail Design - Select Pattern per Nail")

        self.original_image = None
        self.display_image = None
        self.polygons = []
        self.pattern_images = {}
        self.nail_colors = {}
        self.nail_alphas = {}
        self.selected_color = (0, 0, 255)
        self.selected_alpha = 0.6
        self.alpha = 0.6
        self.use_pattern = False
        self.mode = tk.StringVar(value="all")
        self.selected_nail = None

        self.canvas = tk.Canvas(root)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        mode_frame = tk.Frame(root)
        mode_frame.pack(pady=5)
        tk.Radiobutton(mode_frame, text="Áp dụng tất cả", variable=self.mode, value="all", command=self.on_mode_change).pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="Tùy chỉnh từng ngón", variable=self.mode, value="custom", command=self.on_mode_change).pack(side=tk.LEFT)

        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(pady=10)

        self.btn_choose_color = tk.Button(self.btn_frame, text="🎨 Chọn Màu", command=self.choose_color)
        self.btn_choose_color.pack(side=tk.LEFT, padx=5)
        self.btn_choose_pattern = tk.Button(self.btn_frame, text="🖼️ Chọn Họa Tiết", command=self.choose_pattern_all)
        self.btn_choose_pattern.pack(side=tk.LEFT, padx=5)
        self.btn_save = tk.Button(self.btn_frame, text="💾 Lưu Ảnh", command=self.save_image)
        self.btn_save.pack(side=tk.LEFT, padx=5)
        self.btn_load = tk.Button(self.btn_frame, text="📁 Chọn Ảnh & Segment", command=self.load_and_segment)
        self.btn_load.pack(side=tk.LEFT, padx=5)

        self.alpha_frame = tk.Frame(root)
        self.alpha_frame.pack(pady=5)
        self.lbl_alpha = tk.Label(self.alpha_frame, text="Alpha:")
        self.lbl_alpha.pack(side=tk.LEFT)
        self.alpha_scale = tk.Scale(self.alpha_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.update_alpha)
        self.alpha_scale.set(int(self.alpha * 100))
        self.alpha_scale.pack(side=tk.LEFT)

        self.on_mode_change()  # Khởi tạo trạng thái ban đầu

    def on_mode_change(self):
        if self.mode.get() == "custom":
            # Trả về ảnh gốc khi chuyển sang custom
            if self.original_image is not None:
                self.display_image = self.original_image.copy()
                self.show_image(self.display_image)
            # Ẩn các option áp dụng tất cả
            self.btn_choose_color.pack_forget()
            self.btn_choose_pattern.pack_forget()
            self.alpha_frame.pack_forget()
        else:
            # Hiện lại các option khi chọn "áp dụng tất cả"
            self.btn_choose_color.pack(side=tk.LEFT, padx=5)
            self.btn_choose_pattern.pack(side=tk.LEFT, padx=5)
            self.alpha_frame.pack(pady=5)
            self.apply_overlay()

    def load_and_segment(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if not file_path:
            return
        self.original_image = cv2.imread(file_path)
        if self.original_image is None:
            print("Lỗi đọc ảnh.")
            return

        result = CLIENT.infer(file_path, model_id="nail_segmentation-hejrk-tvknk/1")
        self.polygons = []
        for pred in result['predictions']:
            pts = [(int(p['x']), int(p['y'])) for p in pred['points']]
            pts = cv2.convexHull(np.array(pts)).squeeze()
            pts = [tuple(pts)] if pts.ndim == 1 else [tuple(p) for p in pts]
            self.polygons.append(pts)

        self.pattern_images = {}  # reset họa tiết
        self.display_image = self.original_image.copy()
        self.apply_overlay()

    def choose_color(self):
        color, _ = colorchooser.askcolor()
        if color:
            bgr = (int(color[2]), int(color[1]), int(color[0]))
            if self.mode.get() == "all":
                self.selected_color = bgr
                self.use_pattern = False
            elif self.mode.get() == "custom" and self.selected_nail is not None:
                self.nail_colors[self.selected_nail] = bgr
                self.pattern_images.pop(self.selected_nail, None)
            self.apply_overlay()

    def update_alpha(self, val):
        alpha_val = int(val) / 100
        if self.mode.get() == "all":
            self.alpha = alpha_val
        elif self.mode.get() == "custom" and self.selected_nail is not None:
            self.nail_alphas[self.selected_nail] = alpha_val
        self.apply_overlay()

    def on_canvas_click(self, event):
        if not self.polygons or self.mode.get() != "custom":
            return
        x, y = int(event.x / self.resize_scale), int(event.y / self.resize_scale)
        for idx, poly in enumerate(self.polygons):
            if cv2.pointPolygonTest(np.array(poly, dtype=np.int32), (x, y), False) >= 0:
                self.selected_nail = idx
                self.show_custom_options(idx, event.x_root, event.y_root)
                break

    def choose_pattern_all(self):
        path = filedialog.askopenfilename(title="Chọn họa tiết cho tất cả móng", filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if not path:
            return
        img = cv2.imread(path)
        if img is not None:
            self.pattern_images["all"] = img
            self.use_pattern = True
            self.apply_overlay()

    def show_custom_options(self, idx, x_root, y_root):
        top = tk.Toplevel(self.root)
        top.title(f"Tùy chỉnh móng {idx+1}")
        top.geometry(f"{350}x{250}+{x_root}+{y_root}")  # Kích thước cố định lớn hơn

        tk.Button(top, text="Chọn màu", command=lambda: [self.choose_color(), top.destroy()]).pack(pady=10)
        tk.Button(top, text="Chọn họa tiết", command=lambda: [self.assign_pattern_to_nail(idx), top.destroy()]).pack(pady=10)
        tk.Label(top, text="Alpha:").pack()
        scale = tk.Scale(top, from_=0, to=100, orient=tk.HORIZONTAL)
        scale.set(int(self.nail_alphas.get(idx, self.alpha) * 100))
        scale.pack(fill=tk.X, padx=20)
        def on_alpha_change(val):
            self.set_custom_alpha(idx, int(val))
        scale.config(command=on_alpha_change)
        # Không cần nút OK nữa

    def set_custom_alpha(self, idx, val):
        self.nail_alphas[idx] = val / 100
        self.apply_overlay()

    def assign_pattern_to_nail(self, idx):
        path = filedialog.askopenfilename(title=f"Chọn họa tiết cho móng {idx+1}", filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if not path:
            return
        img = cv2.imread(path)
        if img is not None:
            self.pattern_images[idx] = img
            self.use_pattern = True
            self.apply_overlay()

    def apply_overlay(self):
        if self.original_image is None or not self.polygons:
            return

        overlay = self.original_image.copy()
        for idx, poly in enumerate(self.polygons):
            pts = np.array(poly, np.int32).reshape((-1, 1, 2))
            x, y, w, h = cv2.boundingRect(pts)

            if self.mode.get() == "all":
                color = self.selected_color
                alpha = self.alpha
                pattern = self.pattern_images.get("all")
            else:
                # Nếu ngón này chưa có chỉnh gì thì bỏ qua, giữ nguyên ảnh gốc
                if idx not in self.nail_colors and idx not in self.pattern_images and idx not in self.nail_alphas:
                    continue
                color = self.nail_colors.get(idx, self.selected_color)
                alpha = self.nail_alphas.get(idx, self.alpha)
                pattern = self.pattern_images.get(idx)

            if pattern is not None:
                resized = cv2.resize(pattern, (w, h))
                mask = np.zeros((h, w), dtype=np.uint8)
                local_pts = pts - [x, y]
                cv2.fillPoly(mask, [local_pts], 255)
                roi = overlay[y:y+h, x:x+w]
                bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
                fg = cv2.bitwise_and(resized, resized, mask=mask)
                combined = cv2.add(bg, fg)
                overlay[y:y+h, x:x+w] = combined
            else:
                cv2.fillPoly(overlay, [pts], color)

            if self.mode.get() == "custom":
                nail_img = overlay[y:y+h, x:x+w].copy()
                orig_img = self.original_image[y:y+h, x:x+w]
                overlay[y:y+h, x:x+w] = cv2.addWeighted(nail_img, alpha, orig_img, 1 - alpha, 0)

        if self.mode.get() == "all":
            self.display_image = cv2.addWeighted(overlay, self.alpha, self.original_image, 1 - self.alpha, 0)
        else:
            self.display_image = overlay
        self.show_image(self.display_image)

    def show_image(self, img_bgr):
        h, w = img_bgr.shape[:2]
        scale = min(500 / max(h, w), 1.0)
        self.resize_scale = scale

        img_resized = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        self.tk_img = ImageTk.PhotoImage(img_pil)

        self.canvas.config(width=img_pil.width, height=img_pil.height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

    def save_image(self):
        if self.display_image is None:
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if file_path:
            cv2.imwrite(file_path, self.display_image)
            print("Đã lưu ảnh:", file_path)

if __name__ == "__main__":
    root = tk.Tk()
    app = NailDesignApp(root)
    root.mainloop()
