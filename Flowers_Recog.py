import os
import keras
from keras.models import load_model
import streamlit as st 
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

st.header('CNN Model - Flowers')
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']  # 5 lớp

# Tải mô hình đã huấn luyện
model = load_model('D:/AI/Python/Trituenhantao/homework3/homework3/CNN_Flowers.keras')

# Hàm tìm bounding box quanh vật thể lớn nhất (bông hoa)
def find_largest_contour_bbox(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        return (x, y), (x + w, y + h)
    return (50, 50), (150, 150)  # fallback nếu không phát hiện

# Hàm phân loại ảnh và vẽ bounding box quanh hoa
def classify_images(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "Không thể đọc ảnh. Vui lòng kiểm tra định dạng hoặc đường dẫn."

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Tiền xử lý ảnh
    input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    input_array = tf.keras.utils.img_to_array(input_image)
    input_array = tf.expand_dims(input_array, 0)

    # Dự đoán
    predictions = model.predict(input_array)
    if predictions.shape[1] != len(flower_names):
        return f"Mô hình trả về {predictions.shape[1]} lớp, nhưng danh sách flower_names có {len(flower_names)} lớp."

    result = tf.nn.softmax(predictions[0])
    pred_index = np.argmax(result)
    confidence = np.max(result)
    label = flower_names[pred_index]

    outcome = f'Đây là : {label} (độ chính xác dự đoán là : {np.round(confidence * 100, 2)}%)'

    # Xác định bounding box quanh hoa
    start_point, end_point = find_largest_contour_bbox(image_rgb)
    color = (255, 0, 0)
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Vẽ khung và nhãn
    cv2.rectangle(image_rgb, start_point, end_point, color, thickness)
    cv2.putText(image_rgb, label, (start_point[0], start_point[1] - 10), font, 1, color, 2, cv2.LINE_AA)

    pil_img = Image.fromarray(image_rgb)
    st.image(pil_img, caption='Ảnh với bounding box & tên hoa', use_container_width=True)

    return outcome

# Giao diện tải ảnh
uploaded_file = st.file_uploader('Vui lòng tải lên 1 hình ảnh')
if uploaded_file is not None:
    os.makedirs('Samples', exist_ok=True)
    save_path = os.path.join('Samples', uploaded_file.name)
    with open(save_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, width=200)
    result = classify_images(save_path)
    st.markdown(result)
