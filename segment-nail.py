from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np

# Load ảnh gốc
image_path = 'n.jpg'  # đổi thành ảnh thật của bạn
img = cv2.imread(image_path)

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="AC1B0JvfpzCNKv5tLBt5"
)

result = CLIENT.infer(inference_input=image_path, model_id="nail_segmentation-hejrk-tvknk/1")
for pred in result['predictions']:
    points = pred['points']
    pts = np.array([(int(p['x']), int(p['y'])) for p in points], dtype=np.int32)
    pts = cv2.convexHull(pts)  # Tạo hình kín nếu cần

    # Tô vùng segment (ví dụ màu xanh lá nhạt)
    cv2.fillPoly(img, [pts], color=(0, 255, 0))  # Tô polygon

    # Vẽ viền ngoài để nổi bật
    # cv2.polylines(img, [pts], isClosed=True, color=(0, 100, 0), thickness=2)

    # Hiển thị độ tin cậy (confidence) với font lớn hơn
    text = f"{pred['confidence']:.2f}"
    text_position = (pts[0][0][0], pts[0][0][1])  # Lấy tọa độ đầu tiên trong polygon
    cv2.putText(img, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)


# Hiển thị ảnh
# cv2.imshow('Nail Segmentation', img)
resize_factor = 0.3  # hoặc 0.3 tùy bạn
resized = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)
cv2.imshow('Nail Segmentation', resized)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Hoặc lưu lại
# cv2.imwrite('output.jpg', img)
# import requests